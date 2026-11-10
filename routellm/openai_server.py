"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import json
import logging
import math
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal

import fastapi
import shortuuid
import uvicorn
from fastapi import Request
from fastapi.concurrency import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routellm.audit import Auditor, build_audit
from routellm.audit_http import Trail, settle
from routellm.auth import PrincipalResolver, bearer_token, build_auth, current_principal
from routellm.capabilities import Capabilities, side_capabilities, union
from routellm.comparison import COMPARE_HEADER
from routellm.config import layer_summary, load_config
from routellm.controller import HOP_HEADER, Controller, RoutingError
from routellm.decision import build_decision
from routellm.egress_gate import build_egress_gates
from routellm.endpoints import Endpoint, EndpointRegistry, Tier
from routellm.hints import TYPESAFE_INSTALL_HINT
from routellm.middleware.intent_model_selector import (
    IntentModelMapping,
    IntentModelSelector,
)
from routellm.policy import PolicyDenied, build_policy, check_decision_classes
from routellm.policy_routing import build_policy_routing, current_attributes
from routellm.request_class import (
    RequestClassRefusal,
    check_policy_principals,
    current_request_class,
    declare,
)
from routellm.requirements import _prompt_text
from routellm.resilience import EndpointsCooling
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import (
    SERVER_SECTIONS,
    build_cache_config,
    build_comparison_config,
    build_embedding_endpoint,
    build_quality_manager,
    build_resilience_config,
    build_transforms,
)
from routellm.transcription import TranscriptionError
from routellm.transform import TransformError

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None
#: The loaded `policy:` file, or None when the config names none.
POLICY = None

#: `created` for every listed model: the moment this process came up.
#: The listing is derived from the config, so it never changes while the
#: process lives.
SERVER_START = int(time.time())

#: Endpoint names the implicit `default` tier is built under. Hyphens are
#: reserved for the model-name grammar, so these carry none.
IMPLICIT_STRONG = "implicit_strong"
IMPLICIT_WEAK = "implicit_weak"

#: The pair this server has always routed `router-<r>-<thr>` against.
#: Used only when the caller names no pair and the config defines no
#: `default` tier, so the legacy flat form keeps answering. It is never
#: registered as an endpoint and never advertised on `/v1/models`.
LEGACY_STRONG = "gpt-4-1106-preview"
LEGACY_WEAK = "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"


def legacy_pair() -> tuple[str, str]:
    """Return the historic flat pair, warning that it was assumed.

    Reached only when neither model flag was given and the config
    defines no `default` tier. The warning goes out at WARNING so it is
    visible without `--verbose`: routing against a pair the caller never
    named is a fact they should see.

    Returns
    -------
    tuple[str, str]
        The `(strong, weak)` model names to hand the controller.
    """
    logging.warning(
        "No --strong-model/--weak-model and no 'default' tier in --config; "
        "routing 'router-<router>-<threshold>' against the historic pair "
        "%s / %s. Pass both flags, or define a 'default' tier, to choose "
        "the pair yourself.",
        LEGACY_STRONG,
        LEGACY_WEAK,
    )
    return LEGACY_STRONG, LEGACY_WEAK


def build_registry(file_config: dict | None, config_path: str | None = None) -> EndpointRegistry:
    """Build the endpoint registry the server routes against.

    The merged config is the single source for endpoints and tiers, and
    it is discovered rather than named: `--config` is only its highest
    layer. When that config carries no `default` tier and both
    `--strong-model` and `--weak-model` are given, the two are wrapped
    as endpoints and joined into an implicit `default` tier running
    `--routers[0]` at `--default-threshold`, so a flat command line
    still answers a request addressed to `default`.

    A config `default` tier wins, and the flags are then left to the
    flat pair the controller keeps. Neither flag and no config
    `default` derives nothing: no tier is invented and none is
    advertised, and the legacy flat form routes against the historic
    pair instead.

    Parameters
    ----------
    file_config : dict, optional
        The merged config as `load_config` returns it.
    config_path : str, optional
        The file that config was read from. A relative `quality_from:`
        is resolved against its directory, never against the CWD, so a
        server started from elsewhere reads the sidecar the operator
        wrote next to the config.

    Returns
    -------
    EndpointRegistry
        Registry holding the configured endpoints and tiers, plus the
        implicit `default` tier when one was derived.
    """
    registry = EndpointRegistry.from_config(file_config or {}, config_path=config_path)

    if registry.has_tier("default"):
        logging.info("default tier: from --config")
        if args.strong_model or args.weak_model:
            logging.info(
                "--strong-model/--weak-model ignored for the 'default' tier: "
                "the config defines one. They remain the flat pair."
            )
        return registry

    # Argparse has already rejected exactly one of the two, so either
    # both are set or neither is.
    if not args.strong_model:
        return registry

    added: dict[str, Endpoint] = {}

    # A flag may name a configured endpoint, which already carries its
    # own base URL and credential; only a raw model name needs wrapping.
    sides = {}
    for side, flag, wrapper in (
        ("strong", args.strong_model, IMPLICIT_STRONG),
        ("weak", args.weak_model, IMPLICIT_WEAK),
    ):
        if flag in registry.names():
            sides[side] = flag
            continue
        if registry.strict_registry:
            known = ", ".join(registry.names()) or "<none>"
            raise ValueError(
                f"--{side}-model {flag!r} is not a registry endpoint and "
                f"strict_registry forbids raw model names. Configured "
                f"endpoints: {known}"
            )
        added[wrapper] = Endpoint(name=wrapper, model=flag)
        sides[side] = wrapper

    # Added in place: the registry keeps every setting it was built
    # with, none of which this function has to list.
    default = Tier(
        name="default",
        router=args.routers[0] if args.routers else None,
        threshold=args.default_threshold,
        strong=sides["strong"],
        weak=sides["weak"],
    )
    registry.extend(added, {"default": default})

    logging.info(
        "default tier: implicit, %s/%s at %s over %s",
        args.strong_model,
        args.weak_model,
        args.default_threshold,
        args.routers[0] if args.routers else "<none>",
    )
    return registry


#: Detector backends the `intents:` section may ask for.
INTENT_DETECTORS = ("jev", "litellm")

#: Told to the operator when the typesafe extension is not importable.
#: The install line is shared with the controller, which raises on a
#: tier routing over `jev`.
TYPESAFE_DETECTOR_HINT = (
    f"intents.detector: jev is unavailable. {TYPESAFE_INSTALL_HINT} "
    "Or set intents.detector to litellm."
)


def build_intents(
    file_config: dict | None,
    registry: EndpointRegistry,
) -> IntentModelSelector | None:
    """Build the intent middleware the server routes tiers with.

    Reads the `intents:` key: a `detector` of `jev` or `litellm`, the
    `model` that detector classifies with, a `confidence_floor` the
    Jev detector honours, per-intent `descriptions`, and a `tiers` map
    from intent label to tier name.

    Both detectors need one `IntentModelMapping` per intent, carrying
    its description: the litellm path writes them into its own
    classification prompt, the Jev path turns them into the criteria of
    its single Choice question. The pair on each mapping is never
    consulted: every mapping and the default carry no pair at all, so
    `get_model_pair` returns None and the request walks the tier tree
    the intent chose rather than bypassing it with a flat pair.

    `routellm_typesafe` is imported here rather than at module level,
    so a server whose config names no Jev detector never needs the
    extension installed.

    Parameters
    ----------
    file_config : dict, optional
        The merged config as `load_config` returns it.
    registry : EndpointRegistry
        Registry whose tiers the mapping is checked against.

    Returns
    -------
    IntentModelSelector or None
        The middleware to hand the controller, or None when the config
        carries no `intents:` section.

    Raises
    ------
    ValueError
        If the detector is unknown, the `tiers` mapping is missing or
        empty, or an intent maps to a tier the registry does not carry.
    ImportError
        If `detector: jev` is asked for and the typesafe extension is
        not installed. The message names the install command.
    """
    spec = (file_config or {}).get("intents") or {}
    if not spec:
        return None

    detector_name = spec.get("detector", "litellm")
    if detector_name not in INTENT_DETECTORS:
        raise ValueError(
            f"Unknown intents.detector: {detector_name}. "
            f"Configured detectors: {', '.join(INTENT_DETECTORS)}"
        )

    intent_tiers = dict(spec.get("tiers") or {})
    if not intent_tiers:
        raise ValueError(
            "intents: needs a non-empty `tiers` mapping. Without one the "
            "classifier would run on every request and choose nothing."
        )

    for intent, tier in intent_tiers.items():
        if not registry.has_tier(tier):
            known = ", ".join(registry.tier_names()) or "<none>"
            raise ValueError(
                f"Intent {intent!r} maps to unknown tier {tier!r}. Configured tiers: {known}"
            )

    descriptions = dict(spec.get("descriptions") or {})
    model = spec.get("model")

    # No mapping carries a pair: this selector decides a tier and
    # nothing else, and `default_model_pair=None` keeps `get_model_pair`
    # from bypassing the very tree the tier was chosen to enter.
    mappings = [
        IntentModelMapping(
            intent=intent,
            model_pair=None,
            description=descriptions.get(intent, ""),
        )
        for intent in intent_tiers
    ]

    if detector_name == "litellm":
        if not model:
            logging.info(
                "intents.model is unset: intent detection is disabled. Set it to a "
                "registry endpoint name to classify requests."
            )
            return None
        return IntentModelSelector(
            intent_mappings=mappings,
            default_model_pair=None,
            intent_endpoint=_registry_endpoint(registry, model, "intents.model"),
            intent_tiers=intent_tiers,
        )

    # The Jev upstream is a registry endpoint too, so it carries reach
    # and jurisdiction tags like any model call. Checked before the
    # import so a bad config fails the same way with or without the
    # extension installed.
    upstream = _registry_endpoint(registry, spec.get("upstream"), "intents.upstream")

    try:
        from routellm_typesafe.intent_detector import JevIntentDetector
    except ImportError as exc:
        raise ImportError(TYPESAFE_DETECTOR_HINT) from exc

    detector = JevIntentDetector(
        intent_mappings=mappings,
        model=model,
        confidence_floor=spec.get("confidence_floor", 0.5),
        descriptions=descriptions,
        base_url=upstream.api_base,
        upstream=upstream.name,
    )
    return IntentModelSelector(
        intent_mappings=mappings,
        default_model_pair=None,
        intent_detector=detector,
        intent_tiers=intent_tiers,
    )


def _registry_endpoint(registry: EndpointRegistry, name: str | None, key: str):
    """Return the registry endpoint `name`, or raise naming `key`.

    A tier is refused with its own message: classifying needs one
    concrete call target, and a tier is resolved per request.
    """
    if not name:
        raise ValueError(f"{key} must name a registry endpoint; it has no default.")
    if registry.has_tier(name):
        raise ValueError(f"{key}: {name!r} is a tier; name a registry endpoint instead.")
    if name not in registry.names():
        known = ", ".join(registry.names()) or "<none>"
        raise ValueError(
            f"{key}: {name!r} is not a registry endpoint. Configured endpoints: {known}"
        )
    return registry.get(name)


def build_router_config(file_config: dict | None) -> dict | None:
    """Return the router config, with the keys other owners claim removed.

    `endpoints:` and `tiers:` belong to the registry, `intents:` to the
    intent middleware, and `quality_from:` / `quality_from_override:`
    to the sidecar merge. Everything else stays router config. Leaving
    any of them in hands an unknown key to every configured router.

    Parameters
    ----------
    file_config : dict, optional
        The loaded YAML config, or None when `--config` was not given.

    Returns
    -------
    dict or None
        The remaining config, or None when nothing is left, which keeps
        the router defaults.
    """
    router_config = dict(file_config or {})
    for key in (
        "endpoints",
        "tiers",
        "intents",
        "decision",
        "quality_from",
        "quality_from_override",
        "strict_registry",
        "policy",
        "egress_gates",
        "policy_routing",
        *SERVER_SECTIONS,
    ):
        router_config.pop(key, None)
    return router_config or None


def payable_bases_for(endpoints, default_base):
    """Return the base URLs this server's config authorised to charge.

    The scope is read off the registry the server actually routes on,
    so `pay: true` cannot mean one thing to routing and another to the
    wallet. Only endpoints carrying that flag contribute; a config that
    sets it nowhere authorises nothing, and the installed session signs
    no payment at all.

    Parameters
    ----------
    endpoints : EndpointRegistry
        The registry built from the merged config.
    default_base : str or None
        The server's `--base-url`, used by a payable endpoint that
        names no `api_base` of its own.

    Returns
    -------
    list[str]
        The authorised base URLs, empty when no endpoint asked to pay.
    """
    return endpoints.payable_bases(default_base=default_base)


def payment_limits_for(endpoints, max_payment, default_base):
    """Return the per-payment caps this server will enforce.

    Two layers meet here. `--max-payment` is a process-wide ceiling no
    endpoint may exceed, and an endpoint's own `max_payment:` may
    lower it further but never raise it. They are read off the same
    registry the server routes on, so a cap cannot mean one thing to
    routing and another to the wallet.

    Both layers absent leaves the payment library's own per-payment
    default standing. That is deliberately not the same as disabling
    spend controls: an unconfigured deployment keeps a ceiling.

    Parameters
    ----------
    endpoints : EndpointRegistry
        The registry built from the merged config.
    max_payment : str or None
        The `--max-payment` ceiling, as written.
    default_base : str or None
        The server's `--base-url`, used by a payable endpoint that
        names no `api_base` of its own.

    Returns
    -------
    PaymentLimits
        The caps, ready to hand to the session and the controller.
    """
    from routellm.payment.limits import PaymentLimits

    return PaymentLimits(
        global_cap=max_payment,
        per_base=endpoints.payment_caps(default_base=default_base),
    )


def payment_budget_for(payment_budget):
    """Return the cumulative spend ledger this server will enforce.

    A per-payment cap bounds one payment. It says nothing about how
    many payments there are, so any number of requests each under the
    cap spend a multiple of it. `--payment-budget` is the total, and
    one ledger serves both seams a payment can be authorised at.

    What it counts is the *authorised* amount, never a settled one:
    nothing has settled on chain at signing, so a provider that
    challenges but never settles still consumes budget. The count
    therefore under-spends rather than over-spends.

    It is process-wide and held in memory, exactly as `--max-payment`
    is. A restart clears it and the whole budget is available again.

    Parameters
    ----------
    payment_budget : str or None
        The `--payment-budget` total, as written. None means the
        operator set none, which leaves the total unbounded and per-
        payment caps as the only limit. That is not the same as "$0",
        which is a deliberate budget of nothing.

    Returns
    -------
    PaymentBudget or None
        The ledger, or None when no budget was set.
    """
    if payment_budget is None:
        return None

    from routellm.payment.limits import PaymentBudget

    return PaymentBudget(payment_budget)


#: Resolver for `auth:`; None leaves the server unauthenticated.
AUTH: PrincipalResolver | None = None
AUDIT: Auditor | None = None

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@asynccontextmanager
async def lifespan(app):
    global CONTROLLER

    # The config is discovered, not named: system, user, project, then
    # ROUTELLM_CONFIG and `--config`, each merged onto the last. The
    # layers that were actually read are logged so a surprising value
    # has a file to blame.
    loaded = load_config(
        explicit=args.config,
        skip_user=True if args.no_user_config else None,
    )
    logging.info("config layers %s", layer_summary(loaded))

    # `endpoints:` and `tiers:` belong to the registry and `intents:`
    # to the intent middleware; the rest of the merged mapping stays
    # router config, so all three are popped out before the handoff.
    # The winning layer's path resolves a relative `quality_from:`
    # against the file that set it, never against the CWD.
    #
    # The sidecar merge happens inside EndpointRegistry.from_config, so
    # every consumer that builds a registry -- this server and both
    # pairing CLIs -- sees the same measured numbers. Applying it here
    # instead would leave the CLIs explaining a different ordering than
    # the one the server actually routes on.
    file_config = loaded.data
    # `--strict-registry` is the flag form of `strict_registry: true`;
    # either turns it on, and the flag cannot turn a config's on off.
    if args.strict_registry:
        file_config["strict_registry"] = True
    config_origin = str(loaded.layers[-1].path) if loaded.layers else None
    # A policy implies governance and strict_registry, so it loads before
    # the registry is built; a relative path resolves against the file
    # that set it. Any load error stops startup.
    global POLICY, AUDIT
    POLICY = build_policy(file_config, loaded.resolve_path)
    # A loaded policy requires an audit trail: no audit, no startup.
    AUDIT = build_audit(file_config, POLICY, loaded.resolve_path)
    endpoints = build_registry(file_config, config_origin)
    intents = build_intents(file_config, endpoints)
    decision = build_decision(file_config, endpoints)
    if POLICY is not None and decision is not None:
        check_decision_classes(POLICY, decision.config.data_classes)
    router_config = build_router_config(file_config)

    # Neither flag and no tier to route into leaves the legacy flat form
    # with nothing to pair; fall back to the pair this server has always
    # used rather than refusing requests it used to answer.
    strong_model, weak_model = args.strong_model, args.weak_model
    # Strict mode never invents it: the pair is raw model names, so the
    # controller refuses to start without a `default` tier or two
    # registry endpoints.
    if not strong_model and not endpoints.has_tier("default") and not endpoints.strict_registry:
        strong_model, weak_model = legacy_pair()

    # Paying a 402 needs the challenge headers, the body and a replay of
    # the request, none of which survive litellm's exception mapper. So
    # the gateway is installed underneath litellm, on its shared async
    # session, and only when a provider and a wallet key are both
    # present -- otherwise litellm keeps its own client untouched.
    #
    # That session is process-global, so the flag alone would make every
    # upstream payable. It is installed after the registry is built
    # precisely so the scope can come from the config: only the base
    # URLs of endpoints carrying `pay: true`, and nothing when none do.
    from routellm.payment.transport import maybe_install_payment_session

    #
    # Authorisation is only half of it. `pay: true` says who may
    # charge, never how much, so the same registry also supplies the
    # per-payment caps -- the `--max-payment` ceiling and each
    # endpoint's own lower one -- and they go to both seams a 402 can
    # be paid at: the session's transport here, and the controller's
    # own retry below.
    limits = payment_limits_for(endpoints, max_payment=args.max_payment, default_base=args.base_url)

    # One ledger, built here and handed to both seams. Two would be
    # two budgets, and the process would spend twice what was allowed.
    budget = payment_budget_for(args.payment_budget)

    gateway = maybe_install_payment_session(
        provider=args.payment_provider,
        wallet_key=os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", ""),
        payable_bases=payable_bases_for(endpoints, args.base_url),
        limits=limits,
        budget=budget,
    )

    global AUTH
    AUTH = build_auth(file_config, config_origin)
    if POLICY is not None:
        check_policy_principals(POLICY, AUTH)
    if AUTH is None and args.host not in LOOPBACK_HOSTS:
        logging.warning(
            "listening on %s with no `auth:` configured: anyone who can reach this port can use it",
            args.host,
        )

    CONTROLLER = Controller(
        routers=args.routers,
        config=router_config,
        strong_model=strong_model,
        weak_model=weak_model,
        endpoints=endpoints,
        api_base=args.base_url,
        api_key=args.api_key,
        progress_bar=True,
        payment_gateway=gateway,
        payment_limits=limits,
        payment_budget=budget,
        default_router=args.routers[0] if args.routers else None,
        default_threshold=args.default_threshold,
        middleware=[intents] if intents else None,
        decision=decision,
        cache_config=build_cache_config(file_config, config_origin),
        resilience_config=build_resilience_config(file_config),
        comparison_config=build_comparison_config(file_config),
        quality_manager=build_quality_manager(file_config, config_origin),
        strict_registry=endpoints.strict_registry,
        policy=POLICY,
        egress_gates=build_egress_gates(file_config),
        policy_routing=build_policy_routing(file_config, POLICY),
        embedding_endpoint=build_embedding_endpoint(file_config),
        allow_remote_embeddings=args.allow_remote_embeddings,
        transforms=build_transforms(file_config),
    )
    if AUDIT is not None:
        AUDIT.resolve = _endpoint_facts
    yield
    CONTROLLER = None
    AUTH = None
    if AUDIT is not None:
        AUDIT.sink.close()
    AUDIT = None


app = fastapi.FastAPI(lifespan=lifespan)


class Unauthorized(Exception):
    """No bearer key, or one that maps to no principal."""


def require_principal(http_request: Request):
    """Authenticate the caller and expose it as `request.state.principal`.

    A dependency, so it runs before the body is validated. With auth off
    it does nothing and `request.state.principal` is None. The 401 is the
    same for a missing and an unknown key, and never echoes the key.
    """
    http_request.state.principal = None
    resolver = AUTH
    if resolver is None:
        return None
    token = bearer_token(http_request.headers.get("authorization"))
    principal = resolver.resolve(token) if token else None
    if principal is None:
        raise Unauthorized
    http_request.state.principal = principal
    current_principal.set(principal)
    return principal


def _endpoint_facts(name: str) -> tuple[list[str], str | None]:
    """Reach labels and jurisdiction of endpoint `name`, for audit events."""
    endpoint = CONTROLLER.endpoints.resolve(name)
    return sorted(endpoint.reach), endpoint.jur


def _early_trail(request: Request) -> "Trail | None":
    """A trail for a request refused before its handler ran; None off the audited routes."""
    path = request.url.path
    if request.method != "POST" or path not in ("/v1/chat/completions", "/v1/compare"):
        return None
    return Trail.start(AUDIT, request, None, "compare" if path == "/v1/compare" else "chat")


@app.exception_handler(Unauthorized)
async def _unauthorized(request: Request, _exc: Unauthorized) -> JSONResponse:
    response = JSONResponse(
        {"object": "error", "message": "Missing or invalid bearer key"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )
    return settle(_early_trail(request), response, "refused", 401, "unauthorized")


@app.exception_handler(RequestValidationError)
async def _invalid_request(request: Request, exc: RequestValidationError):
    from fastapi.exception_handlers import request_validation_exception_handler

    response = await request_validation_exception_handler(request, exc)
    return settle(_early_trail(request), response, "refused", 422, "invalid_request")


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class ChatCompletionRequest(BaseModel):
    """The chat completion body this server accepts.

    Agent clients send the full OpenAI chat shapes: nested `tools`,
    `tool_choice` as a string or an object, messages with `tool_calls`,
    `role: "tool"` turns, null or absent `content`, content parts
    (text, image_url, input_audio) and `response_format` of type `json_schema`. The
    nested shapes belong to the provider, so they are typed as open
    dicts and forwarded untouched.

    Only the fields named here are forwarded. Anything else in a body is
    dropped, which is deliberate: a parameter this server does not know
    may be one a routed provider rejects. The allowlist is this class.
    """

    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: str | list[dict[str, Any]]
    # Every optional field defaults to None so an unset one stays unset.
    # A default value here would be manufactured into the litellm call,
    # and a provider that does not accept the parameter rejects the whole
    # request over a value the client never sent.
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    # { "type": "json_object" } for json mode, or a `json_schema` object.
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    user: str | None = None
    # Read for `data_class` and `operation` only, then dropped: never forwarded.
    metadata: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


def _has_images(messages: Any) -> bool:
    """Whether any message carries an image part."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in ("image_url", "input_image")
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


def _has_audio(messages: Any) -> bool:
    """Whether any message carries an audio part."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in ("input_audio", "audio_url")
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


def request_summary(request: "ChatCompletionRequest") -> str:
    """One log line describing a request, without any of its content.

    Prompts are the user's data and stay out of the logs: the line holds
    the model name, counts and flags only.
    """
    messages = request.messages
    count = len(messages) if isinstance(messages, list) else 1
    return (
        f"Received request: model={request.model} messages={count} "
        f"tools={bool(request.tools)} images={_has_images(messages)} "
        f"audio={_has_audio(messages)} "
        f"stream={bool(request.stream)}"
    )


def _error_body(request_id: str) -> dict:
    """The only thing a client learns of an upstream failure."""
    return {
        "message": f"Upstream request failed (request id {request_id})",
        "type": "server_error",
        "request_id": request_id,
    }


def _cooling_response(exc: EndpointsCooling, request_id: str) -> JSONResponse:
    """Answer a request whose every candidate is cooling after a rate limit.

    503 with `Retry-After` set to the soonest cooldown end, whole
    seconds rounded up, at least 1. The body is generic; the endpoint
    names and timings stay in the server log under the request id.
    """
    logging.getLogger(__name__).warning(f"Request failed (request id {request_id}): {exc}")
    retry_after = max(1, math.ceil(exc.retry_after))
    return JSONResponse(
        {
            "object": "error",
            "message": f"All upstreams are rate-limited; retry later (request id {request_id})",
            "type": "server_error",
            "request_id": request_id,
        },
        status_code=503,
        headers={"Retry-After": str(retry_after)},
    )


#: Request header naming the language of any audio in the request.
TRANSCRIBE_LANGUAGE_HEADER = "X-RouteLLM-Transcribe-Language"


def _transcription_failure(exc: TranscriptionError, request_id: str) -> JSONResponse:
    """Answer a failed transcription: a generic message and the request id.

    What the transcriber said stays in the server log under the same id.
    """
    logging.getLogger(__name__).error(
        f"Transcription failed (request id {request_id}): {exc.detail}"
    )
    kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return JSONResponse(
        {
            "object": "error",
            "message": f"Audio transcription failed: {exc.public} (request id {request_id})",
            "type": kind,
            "request_id": request_id,
        },
        status_code=exc.status_code,
    )


async def stream_response(response, request_id: str | None = None, done=None) -> AsyncGenerator:
    """Frame a chunk stream as server-sent events.

    A stream that fails after its first chunk has already sent a 200, so
    the status cannot carry the failure. An `error` event does, and the
    stream then ends without `[DONE]`: a clean terminator would let a
    client take a truncated answer for a complete one.

    The event is generic: a message and a request id. The exception
    itself (which can name the upstream and echo its text) is logged
    server-side under the same id.

    `done(outcome, reason, tokens_in, tokens_out)`, when given, is called
    once as the stream ends: `served`, or `failed` for an upstream error
    or a caller that went away.
    """
    request_id = request_id or shortuuid.uuid()
    outcome, reason, usage = "failed", "client_disconnect", None
    try:
        async for chunk in response:
            usage = getattr(chunk, "usage", None) or usage
            yield f"data: {chunk.model_dump_json()}\n\n"
        outcome, reason = "served", None
    except Exception as e:
        outcome, reason = "failed", "stream_error"
        logging.getLogger(__name__).error(
            f"Stream failed after its first chunk (request id {request_id}): "
            f"{type(e).__name__}: {e!s}"
        )
        yield f"data: {json.dumps({'error': _error_body(request_id)})}\n\n"
        return
    finally:
        # A client that disconnects cancels this generator mid-stream;
        # without this the upstream connection (and the provider's
        # generation) would run on until garbage collection.
        try:
            closer = getattr(response, "aclose", None)
            if closer is not None:
                await closer()
        finally:
            if done is not None:
                done(
                    outcome,
                    reason,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                )
    yield "data: [DONE]\n\n"


class LoopDetected(Exception):
    """A request that already carries the hop header."""


def _declare(request: "ChatCompletionRequest", http_request: Request):
    """The request's validated declaration, attached to its context; or a refusal."""
    classifier = CONTROLLER.decision.config.data_classes if CONTROLLER.decision else ()
    principal = http_request.state.principal
    floor = None
    router = CONTROLLER.policy_routing
    if router is not None:
        # The selected routing policy's floor supplies a class a request
        # did not declare; a policy the caller may not select is refused.
        try:
            floor = router.floor(router.attributes(http_request.headers), principal)
        except PolicyDenied as exc:
            raise RequestClassRefusal(exc.status, str(exc)) from None
    rc = declare(http_request.headers, request.metadata, principal, POLICY, classifier, floor)
    http_request.state.request_class = rc
    current_request_class.set(rc)
    return rc


def _attributes(http_request: Request):
    """The request's routing attributes, attached to its context; None without policy routing.

    Read from headers only and never forwarded: no header is copied upstream.
    """
    router = CONTROLLER.policy_routing
    if router is None:
        return None
    attrs = router.attributes(http_request.headers)
    http_request.state.attributes = attrs
    current_attributes.set(attrs)
    return attrs


def _class_refusal(exc: RequestClassRefusal) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=str(exc)).model_dump(), status_code=exc.status)


def refuse_loops(http_request: Request) -> None:
    """Refuse a request that has already been through a router.

    A dependency, not a check in the handler, so it runs before the body
    is validated: a loop is refused as a loop, whatever the body is.
    """
    if HOP_HEADER in http_request.headers:
        raise LoopDetected


@app.exception_handler(LoopDetected)
async def _loop_detected(request: Request, _exc: LoopDetected) -> JSONResponse:
    response = JSONResponse(
        ErrorResponse(message=f"Loop detected: request already carries {HOP_HEADER}").model_dump(),
        status_code=508,
    )
    return settle(_early_trail(request), response, "refused", 508, "loop_detected")


@app.post(
    "/v1/chat/completions",
    dependencies=[fastapi.Depends(require_principal), fastapi.Depends(refuse_loops)],
)
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    """Route one chat completion and return it in OpenAI's shape.

    The model field carries the routing parameters in any of four
    forms: `<tier>:router-<r>-<thr>`, `router-<r>-<thr>`,
    `router-<tier>`, or a bare `<tier>`. A tiered request walks the
    tier tree, one router call per level.

    A non-streamed response carries the decision path under a top-level
    `routellm` key, which OpenAI clients ignore. A streamed response
    carries it in the server logs only, since its chunks follow the SSE
    shape the client parses.
    """
    request_id = shortuuid.uuid()
    logging.info(f"{request_summary(request)} request_id={request_id}")
    trail = Trail.start(
        AUDIT,
        http_request,
        request_id,
        "chat",
        prompt=_prompt_text(request.messages)
        if isinstance(request.messages, list)
        else request.messages,
        streaming=bool(request.stream),
    )

    # Only what the client actually sent is forwarded. `exclude_unset`
    # drops the fields the body never named, `exclude_none` drops the
    # ones it named as null, and `model` and `messages` are passed
    # explicitly because routing rewrites the first and reads the second.
    # `stream` stays in: the server reads it to pick the response shape,
    # but litellm is what has to produce the chunks.
    forwarded = request.model_dump(exclude_unset=True, exclude_none=True)
    for handled in ("model", "messages", "metadata"):
        forwarded.pop(handled, None)

    try:
        declared = _declare(request, http_request)
    except RequestClassRefusal as e:
        return settle(trail, _class_refusal(e), "refused", e.status, "class_refused")
    if trail is not None:
        trail.declared(declared)

    language = http_request.headers.get(TRANSCRIBE_LANGUAGE_HEADER)

    try:
        res = await CONTROLLER.acompletion(
            model=request.model,
            messages=request.messages,
            transcribe_language=language,
            request_id=request_id,
            principal=http_request.state.principal,
            declared_class=declared.data_class,
            attributes=_attributes(http_request),
            compare=http_request.headers.get(COMPARE_HEADER),
            **forwarded,
        )
    except PolicyDenied as e:
        response = JSONResponse(ErrorResponse(message=str(e)).model_dump(), status_code=e.status)
        return settle(trail, response, "refused", e.status, "policy_denied")
    except RoutingError as e:
        response = JSONResponse(ErrorResponse(message=str(e)).model_dump(), status_code=400)
        return settle(trail, response, "refused", 400, "routing_error")
    except TranscriptionError as e:
        return settle(
            trail, _transcription_failure(e, request_id), "failed", e.status_code, "transcription"
        )
    except EndpointsCooling as e:
        return settle(trail, _cooling_response(e, request_id), "failed", 503, "endpoints_cooling")
    except TransformError as e:
        # Nothing was sent upstream. The reason is ours, not an
        # upstream's, so the client may read it; the cause stays here.
        logging.getLogger(__name__).error(
            f"Request refused (request id {request_id}): {e!s}", exc_info=e.__cause__
        )
        response = JSONResponse(
            {
                "object": "error",
                "message": f"Request refused: {e.reason} (request id {request_id})",
                "type": "server_error",
                "code": "transform_failed",
                "request_id": request_id,
            },
            status_code=500,
        )
        return settle(trail, response, "failed", 500, "transform_failed")
    except Exception as e:
        # Whatever an upstream raised can name it and echo its text: the
        # client gets a generic message and an id; the detail stays here.
        logging.getLogger(__name__).error(
            f"Request failed (request id {request_id}): {type(e).__name__}: {e!s}"
        )
        response = JSONResponse({"object": "error", **_error_body(request_id)}, status_code=502)
        return settle(trail, response, "failed", 502, "provider_error")

    logging.info(CONTROLLER.model_counts)

    hidden = getattr(res, "_hidden_params", None)
    path = hidden.get("routellm_path") if isinstance(hidden, dict) else None
    logging.info(f"Routing path: {path}")

    if request.stream:
        stream = StreamingResponse(
            content=stream_response(
                res, request_id, trail.stream_done(res) if trail is not None else None
            ),
            media_type="text/event-stream",
        )
        # The route is final once the stream is open. The decision is
        # written before the first byte; a failed write refuses the stream.
        settled = settle(trail, stream, "served", 200, phase="decision", res=res)
        if settled is not stream:
            await res.aclose()
        return settled

    body = res.model_dump()
    if path is not None:
        body["routellm"] = {"path": path}
    return settle(trail, JSONResponse(content=body), "served", 200, res=res)


def _compare_prompt(request: "ChatCompletionRequest") -> str | None:
    """The text the audit digest covers: the last message, as the router reads it."""
    if isinstance(request.messages, list):
        return _prompt_text(request.messages)
    return request.messages


class CompareRequest(ChatCompletionRequest):
    """A chat request addressed to several models at once.

    `models` is a list of endpoint names, or the name of a profile or
    experiment. `model` is not used. `tier` applies that tier's privacy:
    a local-only tier may only be compared across `local` endpoints.
    `measure_ttft` streams each model upstream to time its first token;
    the response is still one JSON body.
    """

    model: str | None = None
    models: list[str] | str
    tier: str | None = None
    measure_ttft: bool | None = None


@app.post(
    "/v1/compare",
    dependencies=[fastapi.Depends(require_principal), fastapi.Depends(refuse_loops)],
)
async def create_comparison(request: CompareRequest, http_request: Request):
    """Run one request on every named model and return the results side by side.

    Present only when `comparison.enabled` and `comparison.endpoint`
    are both on. Each result carries the model's output, an `error`
    code when it failed, its metrics (TTFT, latency, tokens per
    second, token counts, cost) and a judge `score` when a judge is
    configured. One model failing never fails the request. Non-
    streaming only.
    """
    request_id = shortuuid.uuid()
    trail = Trail.start(AUDIT, http_request, request_id, "compare", _compare_prompt(request))
    manager = CONTROLLER.comparison
    if manager is None or not manager.config.endpoint:
        response = JSONResponse(ErrorResponse(message="Not found").model_dump(), status_code=404)
        return settle(trail, response, "refused", 404, "not_found")

    if request.stream:
        response = JSONResponse(
            ErrorResponse(message="/v1/compare does not stream").model_dump(),
            status_code=400,
        )
        return settle(trail, response, "refused", 400, "compare_stream")

    try:
        declared = _declare(request, http_request)
    except RequestClassRefusal as e:
        return settle(trail, _class_refusal(e), "refused", e.status, "class_refused")
    if trail is not None:
        trail.declared(declared)
        trail.record.entry_tier = request.tier
        trail.record.targets = (
            [request.models] if isinstance(request.models, str) else list(request.models)
        )

    forwarded = request.model_dump(exclude_unset=True, exclude_none=True)
    for handled in ("model", "models", "tier", "measure_ttft", "metadata"):
        forwarded.pop(handled, None)

    try:
        # The class every target is held to: the strictest of the selected
        # routing policy's floor, the declared class and the decision's;
        # and the policy itself, whose tag lists apply to every target.
        data_class, route_policy = await CONTROLLER.compare_scope(
            forwarded,
            declared.data_class,
            _attributes(http_request),
            http_request.state.principal,
            tier=request.tier,
        )
        if trail is not None:
            trail.record.effective_class, trail.record.policy = data_class, route_policy
        body = await manager.compare(
            request.models,
            forwarded,
            tier=request.tier,
            measure_ttft=request.measure_ttft,
            data_class=data_class,
            route_policy=route_policy,
        )
    except PolicyDenied as e:
        response = JSONResponse(ErrorResponse(message=str(e)).model_dump(), status_code=e.status)
        return settle(trail, response, "refused", e.status, "policy_denied")
    except (RoutingError, ValueError) as e:
        response = JSONResponse(ErrorResponse(message=str(e)).model_dump(), status_code=400)
        return settle(trail, response, "refused", 400, "routing_error")
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Compare failed (request id {request_id}): {type(e).__name__}: {e!s}"
        )
        response = JSONResponse({"object": "error", **_error_body(request_id)}, status_code=502)
        return settle(trail, response, "failed", 502, "provider_error")
    return settle(trail, JSONResponse(content=body), "served", 200)


@app.get("/v1/models", dependencies=[fastapi.Depends(require_principal)])
async def list_models():
    """List every model name this server answers to.

    Two families of id: one per configured tier, addressable bare or
    as `router-<tier>`, and one `router-<name>-<default_threshold>` per
    loaded router. The legacy flat form carries a threshold, and only a
    concrete one is a usable id, so each router is listed at the
    server's `--default-threshold`; any other threshold still routes.

    Returns
    -------
    JSONResponse
        An OpenAI model list, `data` sorted by id.
    """
    tier_names = set(CONTROLLER.endpoints.tier_names())
    ids = set(tier_names)
    ids.update(f"router-{name}-{CONTROLLER.default_threshold}" for name in CONTROLLER.routers)

    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": SERVER_START,
                    "owned_by": "routellm",
                    "routellm": _routellm_extension(model_id, tier_names),
                }
                for model_id in sorted(ids)
            ],
        }
    )


def _routellm_extension(model_id: str, tier_names: set) -> dict:
    """Return the additive `routellm` object for one model entry.

    A tier id carries the union of what every reachable leaf can do,
    read from the controller's cached index, plus the keys no leaf
    knows. A `router-<name>-<thr>` id carries `kind: router`, and the
    flat pair's capabilities when the controller has one.

    The standard OpenAI keys keep their exact values and order; this
    extension is purely additive, and unknown keys are ignored by every
    OpenAI client.

    Parameters
    ----------
    model_id : str
        The id this entry lists.
    tier_names : set[str]
        Every configured tier name.

    Returns
    -------
    dict
        The extension object.
    """
    if model_id in tier_names:
        caps = CONTROLLER._tier_caps.get(model_id, Capabilities())
        return {
            "kind": "tier",
            "capabilities": caps.model_dump(exclude_none=True),
            "unknown": caps.unknown_fields(),
        }

    extension = {"kind": "router"}

    pair = CONTROLLER.default_model_pair
    if pair is not None:
        caps = union(
            [
                side_capabilities(
                    str(side),
                    CONTROLLER.endpoints,
                    CONTROLLER._tier_caps,
                    CONTROLLER._catalog_records,
                )
                for side in (pair.strong, pair.weak)
                if side is not None
            ]
        )
        extension["capabilities"] = caps.model_dump(exclude_none=True)
        extension["unknown"] = caps.unknown_fields()

    return extension


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})


parser = argparse.ArgumentParser(description="An OpenAI-compatible API server for LLM routing.")
parser.add_argument(
    "--verbose",
    action="store_true",
)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help=(
        "Explicit config file, merged last over the discovered chain "
        "(system, user, project, ROUTELLM_CONFIG). Run "
        "`python -m routellm.config paths` to see the chain."
    ),
)
parser.add_argument(
    "--no-user-config",
    action="store_true",
    help=(
        "Skip the user (XDG) config layer; the system, project, "
        "ROUTELLM_CONFIG and --config layers still apply. Same as "
        "ROUTELLM_NO_USER_CONFIG=1."
    ),
)
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--host",
    help=(
        "Interface to bind. Defaults to 127.0.0.1: unless the config has an "
        "`auth:` section the server is unauthenticated, so anything wider "
        "exposes it to that network (and logs a warning)."
    ),
    type=str,
    default="127.0.0.1",
)
parser.add_argument(
    "--routers",
    nargs="+",
    type=str,
    default=["random"],
    choices=list(ROUTER_CLS.keys()),
)
parser.add_argument(
    "--base-url",
    help="The base URL used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument(
    "--api-key",
    help="The API key used for all LLM requests",
    type=str,
    default=None,
)
parser.add_argument(
    "--strong-model",
    help=(
        "Endpoint name from the config, or a raw model name. Pass with "
        "--weak-model to derive a 'default' tier when the config defines none."
    ),
    type=str,
    default=None,
)
parser.add_argument(
    "--weak-model",
    help="Endpoint name from the config, or a raw model name. See --strong-model.",
    type=str,
    default=None,
)
parser.add_argument(
    "--strict-registry",
    action="store_true",
    help=(
        "Refuse every model name that is not a registry endpoint: raw "
        "--strong-model/--weak-model values, the legacy default pair, and "
        "raw targets in traffic rules, canary and load balancers fail at "
        "startup. Same as `strict_registry: true` in the config."
    ),
)
parser.add_argument(
    "--allow-remote-embeddings",
    action="store_true",
    help=(
        "When the `mf` or `sw_ranking` router or the semantic cache is in use "
        "and the config names no embedding endpoint (`embeddings.endpoint`, or "
        "an endpoint called `embedding`), embed through OPENAI_BASE_URL / "
        "OPENAI_API_KEY instead of refusing to start. Logs a WARNING naming "
        "the host; no policy can gate that call."
    ),
)
parser.add_argument(
    "--default-threshold",
    help="Threshold for a tier level that names none and whose request carries none",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--payment-provider",
    default=None,
    choices=["x402"],
    help="Enable payment gateway (e.g. x402)",
)
parser.add_argument(
    "--wallet-key-env",
    default="ROUTELLM_WALLET_KEY",
    help="Env var holding wallet private key",
)
parser.add_argument(
    "--payment-budget",
    default=None,
    help=(
        "Total this process may spend across every payment, as money "
        '(e.g. "$1.00"). A per-payment limit bounds one payment; this '
        "bounds their sum. It counts the amount each payment is authorised "
        "to spend, which is conservative: a challenge that never settles "
        "still consumes budget. Held in memory, so a restart clears it. "
        'Unset means no cumulative limit; "$0" means spend nothing.'
    ),
)
parser.add_argument(
    "--max-payment",
    default=None,
    help=(
        'Process-wide ceiling on a single payment, as money (e.g. "$0.01"). '
        "No endpoint may exceed it; an endpoint's own `max_payment:` may only "
        "lower it. Unset keeps the payment library's own default ceiling."
    ),
)
args = parser.parse_args()

# A flat pair needs both sides. One alone would silently pair with the
# historic default for the other, which is never what the caller meant.
if bool(args.strong_model) != bool(args.weak_model):
    parser.error(
        "--strong-model and --weak-model must be given together: one alone "
        "cannot form a pair. Pass both, or neither and define a 'default' "
        "tier in --config."
    )

if args.verbose:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Launching server with routers:", args.routers)
    uvicorn.run(
        "routellm.openai_server:app",
        port=args.port,
        host=args.host,
        workers=args.workers,
    )
