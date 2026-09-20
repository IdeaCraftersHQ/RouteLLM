"""A server that provides OpenAI-compatible RESTful APIs.

It current only supports Chat Completions: https://platform.openai.com/docs/api-reference/chat)
"""

import argparse
import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routellm.config import load_config
from routellm.capabilities import Capabilities, side_capabilities, union
from routellm.controller import Controller, RoutingError
from routellm.endpoints import Endpoint, EndpointRegistry, Tier
from routellm.hints import TYPESAFE_INSTALL_HINT
from routellm.middleware.intent_model_selector import (
    IntentModelMapping,
    IntentModelSelector,
)
from routellm.routers.routers import ROUTER_CLS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None

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


def build_registry(
    file_config: Optional[dict], config_path: Optional[str] = None
) -> EndpointRegistry:
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
    registry = EndpointRegistry.from_config(
        file_config or {}, config_path=config_path
    )

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

    endpoints = {name: registry.get(name) for name in registry.names()}

    # A flag may name a configured endpoint, which already carries its
    # own base URL and credential; only a raw model name needs wrapping.
    sides = {}
    for side, flag, wrapper in (
        ("strong", args.strong_model, IMPLICIT_STRONG),
        ("weak", args.weak_model, IMPLICIT_WEAK),
    ):
        if flag in endpoints:
            sides[side] = flag
            continue
        endpoints[wrapper] = Endpoint(name=wrapper, model=flag)
        sides[side] = wrapper

    tiers = registry.tiers
    tiers["default"] = Tier(
        name="default",
        router=args.routers[0] if args.routers else None,
        threshold=args.default_threshold,
        strong=sides["strong"],
        weak=sides["weak"],
    )

    logging.info(
        "default tier: implicit, %s/%s at %s over %s",
        args.strong_model,
        args.weak_model,
        args.default_threshold,
        args.routers[0] if args.routers else "<none>",
    )
    return EndpointRegistry(endpoints, tiers)


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
    file_config: Optional[dict],
    registry: EndpointRegistry,
) -> Optional[IntentModelSelector]:
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
                f"Intent {intent!r} maps to unknown tier {tier!r}. "
                f"Configured tiers: {known}"
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
        return IntentModelSelector(
            intent_mappings=mappings,
            default_model_pair=None,
            intent_detection_model=model or "gpt-3.5-turbo",
            intent_tiers=intent_tiers,
        )

    try:
        from routellm_typesafe.intent_detector import JevIntentDetector
    except ImportError as exc:
        raise ImportError(TYPESAFE_DETECTOR_HINT) from exc

    detector = JevIntentDetector(
        intent_mappings=mappings,
        model=model,
        confidence_floor=spec.get("confidence_floor", 0.5),
        descriptions=descriptions,
    )
    return IntentModelSelector(
        intent_mappings=mappings,
        default_model_pair=None,
        intent_detector=detector,
        intent_tiers=intent_tiers,
    )



def build_router_config(file_config: Optional[dict]) -> Optional[dict]:
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
    for key in ("endpoints", "tiers", "intents", "quality_from",
                "quality_from_override"):
        router_config.pop(key, None)
    return router_config or None


@asynccontextmanager
async def lifespan(app):
    global CONTROLLER

    gateway = None
    if args.payment_provider == "x402":
        from routellm.payment.x402 import X402Adapter
        key = os.environ.get(args.wallet_key_env or "ROUTELLM_WALLET_KEY", "")
        if key:
            gateway = X402Adapter(private_key=key)

    # The config is discovered, not named: system, user, project, then
    # ROUTELLM_CONFIG and `--config`, each merged onto the last. The
    # layers that were actually read are logged so a surprising value
    # has a file to blame.
    loaded = load_config(explicit=args.config)
    if loaded.layers:
        logging.info(
            "config layers: %s",
            ", ".join(f"{entry.source}={entry.path}" for entry in loaded.layers),
        )
    else:
        logging.info("config layers: none found; built-in router defaults only")

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
    config_origin = str(loaded.layers[-1].path) if loaded.layers else None
    endpoints = build_registry(file_config, config_origin)
    intents = build_intents(file_config, endpoints)
    router_config = build_router_config(file_config)

    # Neither flag and no tier to route into leaves the legacy flat form
    # with nothing to pair; fall back to the pair this server has always
    # used rather than refusing requests it used to answer.
    strong_model, weak_model = args.strong_model, args.weak_model
    if not strong_model and not endpoints.has_tier("default"):
        strong_model, weak_model = legacy_pair()

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
        default_router=args.routers[0] if args.routers else None,
        default_threshold=args.default_threshold,
        middleware=[intents] if intents else None,
    )
    yield
    CONTROLLER = None


app = fastapi.FastAPI(lifespan=lifespan)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    # Every optional field defaults to None so an unset one stays unset.
    # A default value here would be manufactured into the litellm call,
    # and a provider that does not accept the parameter rejects the whole
    # request over a value the client never sent.
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, str]] = (
        None  # { "type": "json_object" } for json mode
    )
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


async def stream_response(response) -> AsyncGenerator:
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
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
    logging.info(f"Received request: {request}")

    # Only what the client actually sent is forwarded. `exclude_unset`
    # drops the fields the body never named, `exclude_none` drops the
    # ones it named as null, and `model` and `messages` are passed
    # explicitly because routing rewrites the first and reads the second.
    # `stream` stays in: the server reads it to pick the response shape,
    # but litellm is what has to produce the chunks.
    forwarded = request.model_dump(exclude_unset=True, exclude_none=True)
    for handled in ("model", "messages"):
        forwarded.pop(handled, None)

    try:
        res = await CONTROLLER.acompletion(
            model=request.model,
            messages=request.messages,
            **forwarded,
        )
    except RoutingError as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump(),
            status_code=400,
        )

    logging.info(CONTROLLER.model_counts)

    hidden = getattr(res, "_hidden_params", None)
    path = hidden.get("routellm_path") if isinstance(hidden, dict) else None
    logging.info(f"Routing path: {path}")

    if request.stream:
        return StreamingResponse(
            content=stream_response(res), media_type="text/event-stream"
        )

    body = res.model_dump()
    if path is not None:
        body["routellm"] = {"path": path}
    return JSONResponse(content=body)


@app.get("/v1/models")
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
    ids.update(
        f"router-{name}-{CONTROLLER.default_threshold}"
        for name in CONTROLLER.routers
    )

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


parser = argparse.ArgumentParser(
    description="An OpenAI-compatible API server for LLM routing."
)
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
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--host",
    help=(
        "Interface to bind. Defaults to 127.0.0.1: the server is "
        "unauthenticated, so anything wider exposes it to that network."
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
