"""Main controller for intelligent model routing.

Implements a router-based system that selects between strong and weak
models based on prompt difficulty. Supports caching, resilience,
quality management, and payment gateway integration.
"""

import asyncio
import dataclasses
import functools
import logging
import time
import uuid
from collections import defaultdict
from typing import Any

from litellm import acompletion, completion, stream_chunk_builder
from tqdm import tqdm

from routellm import requirements as requirements_module
from routellm import transcription
from routellm.audit import note as audit_note
from routellm.auth import Principal, current_principal
from routellm.caching import Cache, CacheConfig, cache_namespace, context_key, conversation_key
from routellm.capabilities import (
    build_tier_index,
    catalog_records,
    satisfies,
    side_capabilities,
    usable_sibling,
)
from routellm.class_tiers import (
    check_class_tiers,
    check_decision_endpoint,
    check_entry_tier,
    class_denial,
    route_error,
)
from routellm.comparison import ComparisonConfig, ComparisonManager
from routellm.decision import DecisionRefused, set_decision
from routellm.egress_gate import EgressGates
from routellm.endpoints import EndpointRegistry, Selector
from routellm.hints import TYPESAFE_INSTALL_HINT
from routellm.leaf_params import drop_kwargs, log_dropped
from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentChallenge
from routellm.policy import Policy, PolicyDenied
from routellm.policy_routing import PolicyRouter, RequestAttributes
from routellm.quality import QualityManager
from routellm.resilience import EndpointsCooling, Resilience, ResilienceConfig
from routellm.routers.embeddings import configure_embeddings, require_embeddings
from routellm.routers.routers import ROUTER_CLS
from routellm.routing import (
    forced_side,
    parse_model_name,
    resolve_level,
    resolve_tier,
    sibling_of,
)
from routellm.streaming import RoutedStream
from routellm.strict import check_pair, check_strict_registry
from routellm.traffic import NoAllowedTarget, TrafficManager
from routellm.transform import TransformRunner, context_for, forward_call
from routellm.transform_stream import RestoringIterator
from routellm.types import Middleware, ModelPair

logger = logging.getLogger(__name__)

# Default config for routers augmented using golden label data from GPT-4.
# Kept broadly in sync with config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "checkpoint_path": "routellm/sw_ranking_gpt4_augmented",
    },
    "mf": {
        "checkpoint_path": "routellm/mf_gpt4_augmented",
    },
    "bert": {
        "checkpoint_path": "routellm/bert_gpt4_augmented",
    },
    "causal_llm": {
        "checkpoint_path": "routellm/causal_llm_gpt4_augmented",
    },
}


#: Header set on every outbound upstream call. A router that receives a
#: request carrying it refuses it (HTTP 508): the request has already
#: been through a router, so routing it again would loop. The name is
#: fixed, not configurable: both ends of a hop must agree on it.
HOP_HEADER = "X-RouteLLM-Hop"


def outbound_headers(*sources: dict[str, str] | None) -> dict[str, str]:
    """Merge header dicts for one upstream call, ending with the hop header.

    Later sources win over earlier ones. The hop header is applied last
    and replaces any variant of it (header names are case-insensitive),
    so neither endpoint config nor a request can switch the guard off.

    Parameters
    ----------
    *sources : dict or None
        Header dicts, lowest precedence first; None is skipped.

    Returns
    -------
    dict
        The merged headers.
    """
    merged: dict[str, str] = {}
    for source in sources:
        if source:
            merged.update(source)
    merged = {k: v for k, v in merged.items() if k.lower() != HOP_HEADER.lower()}
    merged[HOP_HEADER] = "1"
    return merged


class RoutingError(Exception):
    """Raised when routing configuration or parameters are invalid."""

    pass


def _has_selector(registry: EndpointRegistry) -> bool:
    """Return whether any tier side is still an unresolved policy.

    Kept cheap so a registry of plain names never imports the optional
    models.dev client.
    """
    return any(
        isinstance(side, Selector)
        for tier in registry.tiers.values()
        for side in (tier.strong, tier.weak)
    )


def _needs_embedding(router_cls) -> bool:
    return bool(getattr(router_cls, "requires_embedding", False))


class Controller:
    """Main router controller for intelligent model routing.

    Orchestrates routing decisions, caching, resilience, quality management,
    and payment handling. Matches OpenAI API while supporting advanced
    routing features.
    """

    def __init__(
        self,
        routers: list[str],
        strong_model: str | None = None,
        weak_model: str | None = None,
        config: dict[str, dict[str, Any]] | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        progress_bar: bool = False,
        middleware: list[Middleware] | None = None,
        payment_gateway: PaymentGateway | None = None,
        payment_limits=None,
        payment_budget=None,
        resilience_config: ResilienceConfig | None = None,
        cache_config: CacheConfig | None = None,
        traffic_manager: TrafficManager | None = None,
        quality_manager: QualityManager | None = None,
        endpoints: EndpointRegistry | None = None,
        default_router: str | None = None,
        default_threshold: float = 0.5,
        comparison_config: ComparisonConfig | None = None,
        strict_registry: bool = False,
        decision=None,
        policy: Policy | None = None,
        egress_gates: EgressGates | None = None,
        policy_routing: PolicyRouter | None = None,
        embedding_endpoint: str | None = None,
        allow_remote_embeddings: bool = False,
        transforms: list | None = None,
    ):
        """Initialize controller with routers and configuration.

        Parameters
        ----------
        routers : list[str]
            List of router names to initialize (e.g., ["sw_ranking", "bert"]).
            Routers named by any tier are instantiated too.
        strong_model : str, optional
            Name of strong/expensive model (e.g., "gpt-4"). May be None
            when the registry carries a `default` tier.
        weak_model : str, optional
            Name of weak/cheap model (e.g., "gpt-3.5-turbo"). May be
            None when the registry carries a `default` tier.
        config : dict, optional
            Router-specific configurations. Uses GPT-4 augmented defaults
            if None.
        api_base : str, optional
            LiteLLM API base URL.
        api_key : str, optional
            LiteLLM API key.
        progress_bar : bool
            Show progress bar during router initialization. Default: False.
        middleware : list[Middleware], optional
            Custom middleware for prompt-based routing override.
        payment_limits : PaymentLimits, optional
            How much a single payment may be, process-wide and per
            endpoint. None caps nothing of ours, which leaves the
            gateway's own default per-payment ceiling standing --
            never an unbounded payment.
        payment_budget : PaymentBudget, optional
            What the wallet may spend in total, across every payment.
            A per-payment cap bounds one payment and says nothing
            about how many there are, so this counts the sum. The same
            ledger is shared with the session's transport, since both
            seams draw on one wallet. None leaves the total unbounded.
        payment_gateway : PaymentGateway, optional
            Gateway for 402 payment challenges.
        resilience_config : ResilienceConfig, optional
            Configuration for circuit breaker and retry logic.
        cache_config : CacheConfig, optional
            Configuration for completion caching.
        embedding_endpoint : str, optional
            Registry endpoint the `mf` and `sw_ranking` routers and the
            semantic cache embed prompts through; the endpoint named
            `embedding` when None. There is no default model: startup
            fails when one of them is in use and no endpoint exists.
        allow_remote_embeddings : bool
            Fall back to OPENAI_BASE_URL / OPENAI_API_KEY when no
            endpoint exists, with a WARNING naming the host. Default False.
        transforms : list, optional
            Request transforms (`routellm.transform.RequestTransform`), in
            order. Each rewrites the messages of every upstream call
            after the candidate is chosen and its inverse restores the
            reply; see `routellm.transform`. Empty leaves the request
            path untouched.
        traffic_manager : TrafficManager, optional
            Traffic management and load balancing.
        quality_manager : QualityManager, optional
            Quality and canary testing manager.
        endpoints : EndpointRegistry, optional
            Named endpoints with their own base URL and credential.
            Empty registry when None, which leaves every model name a
            raw litellm model name using `api_base` / `api_key`.
        default_router : str, optional
            Router used by a tier level that names none and whose
            request carries none. Falls back to `routers[0]`.
        default_threshold : float
            Threshold used under the same conditions. Default 0.5.
        strict_registry : bool, optional
            Refuse every model that is not a registry endpoint: the
            registry's `resolve` raises for unknown names and every
            name given here is checked at construction. Also on when
            the registry was built with `strict_registry: true`.
        comparison_config : ComparisonConfig, optional
            Shadow and compare settings. None, or a disabled config,
            builds no manager and the request path is untouched.
        policy : Policy, optional
            The loaded policy. With one, a request must address a tier
            its effective class allows (`PolicyDenied`, 403), and at
            construction every leaf reachable from a class's tiers must
            carry that class's reach and jurisdiction.
        egress_gates : EgressGates, optional
            Hard gates in front of tagged endpoints. At construction
            every selected endpoint's model ids are checked; per request
            each candidate passes them beside the class check.
        policy_routing : PolicyRouter, optional
            Policy routing. A request addressed to an entry tier goes to
            the tier its rules pick, and its effective class also counts
            its routing policy's default. Needs `policy`; validated
            against the registry, the policy and the decision step here.

        Raises
        ------
        EgressGateError
            If a selected endpoint's model id (or an id under a gate's
            `also_check`) breaks the gate's allow or deny list, or a gate
            is invalid against the registry and policy.
        PolicyError
            If a class names a tier the registry lacks, or a leaf
            reachable from its tiers is outside its reach or
            jurisdiction.
        ValueError
            If both models are None and no `default` tier is configured,
            a tier names a router that is not registered, a tier's
            policy resolves to no endpoint or to the same endpoint on
            both sides, or `strong_model`/`weak_model` names a tier.
        """
        self.endpoints = endpoints or EndpointRegistry()
        if strict_registry:
            self.endpoints.strict_registry = True

        # A tier side may be a policy rather than a name. Resolve every
        # one against the configured endpoints and the models.dev
        # catalog now, once, then re-validate: everything downstream
        # sees names only.
        if _has_selector(self.endpoints):
            from routellm.pairing import resolve_registry_pairings

            resolve_registry_pairings(self.endpoints)
            self.endpoints.revalidate()

        # Every tier side is a name now, so the capability fold sees
        # only leaves. Built once: a config change means a restart.
        self._catalog_records = catalog_records(self.endpoints)
        self._tier_caps = build_tier_index(self.endpoints, self._catalog_records)

        for label, name in (("strong_model", strong_model), ("weak_model", weak_model)):
            if name is not None and self.endpoints.has_tier(name):
                raise ValueError(
                    f"{label}={name!r} names a tier, not an endpoint; a flat "
                    "pair must name two endpoints. Use the tier as the "
                    "request model instead."
                )

        if strong_model is None and weak_model is None:
            if not self.endpoints.has_tier("default"):
                raise ValueError(
                    "Controller needs strong_model and weak_model, or a "
                    "registry carrying a 'default' tier."
                )
            self.default_model_pair = None
        else:
            self.default_model_pair = ModelPair(strong=strong_model, weak=weak_model)

        self.default_router = default_router or (routers[0] if routers else None)
        self.default_threshold = default_threshold
        self.routers = {}
        # Registry endpoint each hosted router sends prompts to.
        self._router_upstreams: dict[str, str] = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        # Stream traces being written off the event loop; held so a
        # task is not garbage collected before it finishes.
        self._trace_writes: set[asyncio.Future] = set()
        self.progress_bar = progress_bar
        self.middleware = middleware or []
        self.decision = decision
        if decision is not None:
            decision.bind(self._decision_transport)
        self.payment_gateway = payment_gateway
        self.payment_limits = payment_limits
        self.payment_budget = payment_budget
        self.resilience = Resilience(resilience_config)
        self.cache = Cache(cache_config)
        self.traffic_manager = traffic_manager or TrafficManager()
        self.quality_manager = quality_manager or QualityManager()

        # Strict mode: every name handed in from outside the registry is
        # refused now rather than reaching litellm on a request.
        if self.endpoints.strict_registry:
            check_strict_registry(
                self.endpoints,
                strong_model=strong_model,
                weak_model=weak_model,
                traffic_manager=self.traffic_manager,
                quality_manager=self.quality_manager,
                middleware=self.middleware,
            )

        # Class to tiers: every leaf a class's tiers can reach, by any
        # path, carries what the class allows, or the server never starts.
        self.policy = policy
        self.quality_manager.policy = policy
        if policy is not None:
            check_class_tiers(
                policy,
                self.endpoints,
                traffic_manager=self.traffic_manager,
                quality_manager=self.quality_manager,
                middleware=self.middleware,
            )

        # Policy routing: its rules, tiers and classes are checked now.
        self.policy_routing = policy_routing
        if policy_routing is not None:
            policy_routing.check(self.endpoints, policy, getattr(decision, "config", None))

        # Egress gates: model ids checked now, class and policy per request.
        self.egress_gates = egress_gates
        self.transforms = TransformRunner(transforms)
        if egress_gates is not None:
            egress_gates.check(
                self.endpoints,
                policy,
                decision_endpoint=getattr(getattr(decision, "config", None), "endpoint", None),
            )

        # The classifier reads every request before its class is known,
        # so its endpoint must pass every class that can result.
        if policy is not None and decision is not None:
            check_decision_endpoint(policy, self.endpoints, decision.config.endpoint)

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        configure_embeddings(
            self.endpoints,
            embedding_endpoint,
            allow_remote_embeddings,
            strict=policy is not None,
        )

        # A tier's router must exist too, but it is named in the config
        # file rather than on the command line, so it is checked here
        # against the live ROUTER_CLS: extensions register by discovery,
        # so a name unknown at import may be known by now.
        to_build = list(routers)
        for tier in self.endpoints.tiers.values():
            if tier.router is None or tier.router in to_build:
                continue
            if tier.router not in ROUTER_CLS:
                # `jev` is the one name an operator can reasonably
                # expect to work, so say where it comes from.
                hint = f" {TYPESAFE_INSTALL_HINT}" if tier.router == "jev" else ""
                raise ValueError(
                    f"Tier {tier.name!r} names unknown router "
                    f"{tier.router!r}. Registered routers: "
                    f"{', '.join(sorted(ROUTER_CLS)) or '<none>'}.{hint}"
                )
            to_build.append(tier.router)

        collisions = sorted(set(self.endpoints.tier_names()) & set(ROUTER_CLS))
        if collisions:
            raise ValueError(
                "Names are both a tier and a router: "
                f"{', '.join(collisions)}. Tier and router names must be "
                "distinct."
            )

        # Anything that embeds prompts needs an explicit endpoint, checked
        # before a router loads its datasets.
        embedders = [f"router {r!r}" for r in to_build if _needs_embedding(ROUTER_CLS[r])]
        if cache_config is not None and cache_config.enabled and cache_config.semantic_enabled:
            embedders.append("the semantic cache")
        self.embedding_endpoint = require_embeddings(embedders)

        router_pbar = None
        if self.progress_bar:
            router_pbar = tqdm(total=len(to_build), desc="Initializing routers")

        for router in to_build:
            router_config = dict(config.get(router, {}))
            router_cls = ROUTER_CLS[router]
            if getattr(router_cls, "requires_upstream", False):
                router_config.update(self._router_upstream(router, router_config))
                self._router_upstreams[router] = router_config["upstream"]
            self.routers[router] = router_cls(**router_config)
            if router_pbar:
                router_pbar.update(1)

        # Built last: it validates against the finished registry.
        self.comparison = (
            ComparisonManager(comparison_config, self)
            if comparison_config is not None and comparison_config.enabled
            else None
        )

    def _router_upstream(self, router: str, router_config: dict) -> dict[str, str | None]:
        """Resolve the registry endpoint a hosted router calls.

        A router that sends prompts to a hosted service sets
        `requires_upstream = True`; its config must then carry
        `upstream: <endpoint>`, so the call carries that endpoint's
        reach and jurisdiction tags like any model call and an egress
        gate can name it. The endpoint's `api_base` becomes the
        router's `base_url`.

        Raises
        ------
        ValueError
            If `upstream` is unset or names no registry endpoint.
        """
        name = router_config.pop("upstream", None)
        if not name:
            raise ValueError(
                f"Router {router!r} calls a hosted service and needs "
                f"`{router}.upstream: <registry endpoint>`; it has no default."
            )
        if name not in self.endpoints.names():
            raise ValueError(
                f"Router {router!r} upstream {name!r} is not a registry endpoint. "
                f"Configured endpoints: {', '.join(self.endpoints.names()) or '<none>'}"
            )
        return {"upstream": name, "base_url": self.endpoints.get(name).api_base}

    @property
    def model_pair(self) -> ModelPair:
        """Return the flat strong/weak pair the controller was built with.

        Returns
        -------
        ModelPair
            The configured pair.

        Raises
        ------
        RoutingError
            If the controller carries tiers only, which the evaluation
            and calibration entry points cannot route against.
        """
        if self.default_model_pair is None:
            raise RoutingError(
                "This controller routes through tiers and has no flat model "
                "pair; evals need --strong-model/--weak-model."
            )
        return self.default_model_pair

    async def _decision_transport(self, endpoint, messages, response_format, timeout_ms):
        """Call the decision endpoint like any leaf: registry, resilience, timeout."""
        model, api_base, api_key, extra = self._endpoint_call_params(endpoint)

        async def _call():
            kwargs = {**extra, "messages": messages, "response_format": response_format}
            kwargs.update(model=model, temperature=0, stream=False)
            self._no_inner_retries(kwargs)
            return await acompletion(api_base=api_base, api_key=api_key, **kwargs)

        res = await self.resilience.wrap_acompletion(endpoint, _call, timeout_ms=timeout_ms)
        return res.choices[0].message.content

    async def _decide(self, kwargs, declared_class, conversation_id):
        """Run the decision step; None when no `decision:` section is configured.

        Runs before the routing policy is selected, so the classifier's own
        call (its endpoint, which sees the prompt) is held only to the check
        made at startup (`check_decision_endpoint`): it cannot be judged
        against a policy's `deny_tags` or gate allow set, which are per
        request. Keep the classifier endpoint local, or on tags no policy denies.

        Raises
        ------
        RoutingError
            If the step failed and `on_error` is `refuse`.
        """
        if self.decision is None:
            return None
        try:
            decision = await self.decision.classify(
                kwargs["messages"],
                declared_class=declared_class,
                conversation_id=conversation_id,
                user=kwargs.get("user"),
            )
        except DecisionRefused as refusal:
            raise RoutingError(str(refusal)) from refusal
        set_decision(decision)
        return decision

    def _decide_sync(self, kwargs, declared_class, conversation_id):
        """`_decide` for the synchronous entry point, on a loop of its own."""
        if self.decision is None:
            return None
        coro = self._decide(kwargs, declared_class, conversation_id)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    @staticmethod
    def _request_class(declared_class, decision, effective=None):
        """Return the class a request is held to, the one source every consumer reads.

        `effective` (policy routing's strictest of floor, declared and
        decision) when policy routing ran; else the decision's class,
        else the declared one.
        """
        if effective is not None:
            return effective
        return decision.data_class if decision is not None else declared_class

    def _check_entry_tier(self, tier, declared_class, decision, reqs=None, effective=None):
        """Refuse a tier the request's effective class may not enter.

        Runs after the decision step, so the class is the strictest of
        the declared one and whatever the step concluded (or `effective`,
        when policy routing already combined them with its policy's
        default); before any
        routing, so nothing is called for a refused request.

        Returns
        -------
        Requirements or None
            `reqs` carrying the effective class under a loaded policy,
            which makes `_can_serve` hold every candidate to it; `reqs`
            unchanged without one.
        """
        if self.policy is None:
            return reqs
        effective = self._request_class(declared_class, decision, effective)
        check_entry_tier(self.policy, tier, effective)
        if reqs is None:
            return reqs
        return dataclasses.replace(reqs, data_class=effective)

    def _policy_route(self, tier, declared_class, decision, reqs, attributes, principal=None):
        """Apply policy routing; no-op without it.

        Returns
        -------
        tuple
            `(tier, effective class or None, reqs, Selection or None)`:
            the tier to enter (an entry tier is replaced by the rules'
            pick), the class combined from the routing policy's default,
            the declared class and the decision (None without policy
            routing, which leaves the class to `_check_entry_tier`),
            `reqs` carrying the policy's name, and the record for the path.

        Raises
        ------
        PolicyDenied
            A tier the routing policy does not allow and has no fallback for.
        """
        if self.policy_routing is None:
            return tier, None, reqs, None
        sel = self.policy_routing.select(
            tier, declared_class, decision, reqs, attributes, principal
        )
        if reqs is not None:
            reqs = dataclasses.replace(reqs, route_policy=sel.policy)
        return sel.tier, sel.effective, reqs, sel

    async def effective_class(self, kwargs, declared_class, conversation_id=None):
        """Return the class a request is held to: the decision's, else the declared one."""
        decision = await self._decide(kwargs, declared_class, conversation_id)
        return decision.data_class if decision is not None else declared_class

    async def compare_scope(
        self,
        kwargs,
        declared_class,
        attributes=None,
        principal=None,
        tier=None,
        conversation_id=None,
    ):
        """Return `(effective class, routing policy name)` for a `/v1/compare` request.

        Selects the routing policy the way chat does (principal-bound),
        and takes the strictest of the policy's floor, the declared class
        and the decision's. Every compare target is then held to that
        class and to the policy's `deny_tags` and gate allow set. Without
        policy routing it is `(decision's or declared class, None)`.

        An explicit `tier` is checked against the policy's tiers, never
        moved: it is a privacy scope for the comparison, not a route.

        Raises
        ------
        PolicyDenied
            A policy the caller may not select, a tier the policy does
            not allow, or a class the policy file does not list.
        """
        decision = await self._decide(kwargs, declared_class, conversation_id)
        if self.policy_routing is None:
            return (decision.data_class if decision is not None else declared_class), None
        entry = self.policy_routing.config.entry
        sel = self.policy_routing.select(
            None if tier in entry else tier, declared_class, decision, None, attributes, principal
        )
        return sel.effective, sel.policy

    def _aux_denial(
        self, name: str, data_class: str | None, route_policy: str | None = None
    ) -> str | None:
        """Return why a call carrying request content may not go to endpoint `name`, or None.

        The one check every auxiliary call passes (hosted routers, the
        embedding endpoint, the intent detector, the comparison judge):
        `policy:<class>:<reach|jur>` when the class may not reach it
        (only under a loaded policy and with a class), then
        `gate:<name>:<why>` when an egress gate refuses it. A request
        routed by a named route policy (`route_policy`) is also held to
        that policy's tag denials and its allowed set at the gates.
        """
        if self.policy is not None and data_class is not None:
            denial = class_denial(self.policy, self.endpoints, name, data_class)
            if denial is not None:
                return denial
        allowed = None
        if self.policy_routing is not None and route_policy is not None:
            denial = self.policy_routing.tag_denial(self.endpoints, name, route_policy)
            if denial is not None:
                return denial
            allowed = self.policy_routing.policies[route_policy]
        if self.egress_gates is not None:
            return self.egress_gates.denial(self.endpoints, name, data_class, route_policy=allowed)
        return None

    def _semantic_cache_allowed(
        self, data_class: str | None, route_policy: str | None = None
    ) -> bool:
        """Return whether this request may be embedded for the semantic cache.

        Semantic lookup and store embed the prompt and keep it beside
        its vector, so a class whose treatment is not `raw` never takes
        part, and neither does a class that may not reach the embedding
        endpoint. Without a class (no policy) nothing is withheld.
        """
        if not self.cache.config.semantic_enabled or data_class is None:
            return True
        if self.policy is not None:
            spec = self.policy.classes.get(data_class)
            if spec is None or spec.treatment != "raw":
                logger.info("semantic cache skipped: class %s is not raw", data_class)
                return False
        target = self.embedding_endpoint
        if target is None:
            return self.policy is None
        reason = self._aux_denial(target, data_class, route_policy)
        if reason is not None:
            logger.info("semantic cache skipped: %s", reason)
        return reason is None

    def _detector_gated(
        self, middleware, data_class: str | None, route_policy: str | None = None
    ) -> bool:
        """Return whether the endpoint a middleware sends prompts to is refused for this class.

        A middleware whose detector calls a registry endpoint reports it
        as `egress_endpoint`. When the class policy or an egress gate
        refuses that endpoint the middleware is not consulted at all, so
        the prompt never reaches the detector.
        """
        target = getattr(middleware, "egress_endpoint", None)
        if not target:
            return False
        reason = self._aux_denial(target, data_class, route_policy)
        if reason is not None:
            logger.info("middleware %s skipped: %s", type(middleware).__name__, reason)
        return reason is not None

    def _get_model_pair_for_prompt(
        self, prompt: str, data_class: str | None = None, route_policy: str | None = None
    ) -> ModelPair | None:
        """Get the model pair a middleware asks for, or None."""
        for m in self.middleware:
            if self._detector_gated(m, data_class, route_policy):
                continue
            pair = m.get_model_pair(prompt)
            if pair:
                return pair
        return None

    def _get_tier_for_prompt(
        self, prompt: str, data_class: str | None = None, route_policy: str | None = None
    ) -> str | None:
        """Get the tier a middleware asks the request to enter, or None.

        `get_tier` is optional on the middleware protocol, so it is
        looked up with `getattr`: middleware written before the hook
        carries only `get_model_pair` and is skipped here.

        Parameters
        ----------
        prompt : str
            The request's prompt.
        data_class : str, optional
            The request's effective class; a middleware whose detector
            endpoint a gate refuses for it is not asked.

        Returns
        -------
        str or None
            The first tier name a middleware names, or None when none
            does.
        """
        for m in self.middleware:
            hook = getattr(m, "get_tier", None)
            if hook is None or self._detector_gated(m, data_class, route_policy):
                continue
            tier = hook(prompt)
            if tier:
                return tier
        return None

    def _apply_intent_tier(
        self,
        prompt: str,
        tier: str | None,
        data_class: str | None = None,
        route_policy: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Let a middleware choose the tier this request enters.

        A tier named by a middleware replaces the addressed one only
        when the addressed tier accepts intent routing, which the tier
        named `default` does unless it says otherwise and every other
        tier does only by declaring `intent_routing: true`. A request
        that chose its tier by name is therefore never re-routed; the
        tier that was passed over is reported so the path can carry it.

        Parameters
        ----------
        prompt : str
            The request's prompt.
        tier : str, optional
            Tier addressed by the model name, if any.
        data_class : str, optional
            The request's effective class under a policy. A tier the
            class may not enter is not honoured, and is reported like
            one the addressed tier passed over.
        route_policy : str, optional
            The routing policy the request selected; a detector endpoint
            it denies by tag (or its gate allow set) is not consulted.

        Returns
        -------
        tuple[str or None, str or None, str or None]
            The tier to enter, where it came from (`"intent"`, or None
            when the addressed tier stands), and the tier a middleware
            named that was not honoured.

        Raises
        ------
        RoutingError
            If a middleware names a tier the registry does not carry.
        """
        if tier is None or not self.middleware:
            return tier, None, None

        chosen = self._get_tier_for_prompt(prompt, data_class, route_policy)
        if chosen is None:
            return tier, None, None

        if not self.endpoints.has_tier(chosen):
            known = ", ".join(self.endpoints.tier_names()) or "<none>"
            raise RoutingError(
                f"Middleware selected unknown tier: {chosen}. Configured tiers: {known}"
            )

        if not self.endpoints.get_tier(tier).accepts_intent_routing():
            logger.debug(
                "tier %s does not accept intent routing; ignoring %s",
                tier,
                chosen,
            )
            return tier, None, chosen

        # A class never leaves its own tiers because a detector said so:
        # the switch is ignored, the request stays where it entered.
        if self.policy is not None and data_class is not None:
            spec = self.policy.classes.get(data_class)
            if spec is None or chosen not in spec.tiers:
                logger.info("intent tier %s is outside class %s; ignored", chosen, data_class)
                return tier, None, chosen

        return chosen, "intent", None

    def _parse_model_name(self, model_name: str) -> tuple[str | None, str | None, float | None]:
        """Split a model name into a tier, a router, and a threshold.

        Parameters
        ----------
        model_name : str
            The request's model string, in any of the four forms
            `routellm.routing` documents.

        Returns
        -------
        tuple[str or None, str or None, float or None]
            The `(tier, router, threshold)` the name asks for.

        Raises
        ------
        RoutingError
            If the name is malformed or names an unconfigured tier.
        """
        return parse_model_name(model_name, self.endpoints, RoutingError)

    def _router_denial(
        self, router: str, data_class: str | None, route_policy: str | None = None
    ) -> str | None:
        """Return why a router may not see a prompt of this class, or None.

        A router that embeds sends the prompt to the embedding endpoint;
        a hosted router sends it to its `upstream`. Either must pass the
        class policy and any egress gate. An embedding call with no
        registry endpoint (`--allow-remote-embeddings`) carries no tags,
        so a request with a class is refused it. A request routed by a
        named route policy is also held to its tag denials.
        """
        instance = self.routers.get(router)
        if getattr(type(instance), "requires_embedding", False):
            target = self.embedding_endpoint
            if target is None:
                if data_class is not None and self.policy is not None:
                    return f"policy:{data_class}:reach"
            else:
                reason = self._aux_denial(target, data_class, route_policy)
                if reason is not None:
                    return reason
        upstream = self._router_upstreams.get(router)
        if upstream is not None:
            return self._aux_denial(upstream, data_class, route_policy)
        return None

    def _run_router(
        self,
        router: str,
        threshold: float,
        prompt: str,
        pair: ModelPair,
        data_class: str | None = None,
        route_policy: str | None = None,
    ) -> tuple[str, float | None] | tuple[str, float | None, dict[str, str]]:
        """Run one router over one pair and report what it picked.

        Delegates to `Router.route_with_score`, which scores the prompt
        once and keeps the pick and the score consistent. A router that
        overrides only `route` decides by its own means and reports no
        score, so the path records `win_rate` as None. Nothing about the
        router instance is mutated, so one instance is safe to drive
        from several requests.

        Parameters
        ----------
        router : str
            Router name, validated against the instantiated routers.
        threshold : float
            Decision threshold in [0, 1].
        prompt : str
            Prompt to score.
        pair : ModelPair
            The two sides this level chooses between.
        data_class : str, optional
            The request's effective class. A router whose embedding
            endpoint or upstream the class may not reach is not run: the
            weak side is taken and the path notes `gate:router`.
        route_policy : str, optional
            The routing policy the request selected; its tag denials and
            gate allow set apply to the router's embedding endpoint and
            upstream as well.

        Returns
        -------
        tuple[str, float or None] or tuple[str, float or None, dict]
            The side picked, and the win rate behind it, None when the
            router reports none. A skipped router adds a third item,
            the path-entry keys that say why.

        Raises
        ------
        RoutingError
            If the router is unknown or the threshold is out of range.
        """
        self._validate_router_threshold(router, threshold)
        instance = self.routers[router]

        denied = self._router_denial(router, data_class, route_policy)
        if denied is not None:
            logger.info("router %s skipped: %s", router, denied)
            return pair.weak, None, {"gate": "gate:router", "gate_reason": denied}

        # Routers predating the hook, and test doubles, may carry only
        # `route`; they decide for themselves and report no score.
        scored = getattr(instance, "route_with_score", None)
        if scored is None:
            return instance.route(prompt, threshold, pair), None

        return scored(prompt, threshold, pair)

    def _transcriber_for(self, kwargs: dict[str, Any], tier: str | None):
        """Return the transcriber to run on this request, or None.

        The transcriber is the one the ADDRESSED tier names (the global
        section for no tier), read before any intent re-routing, so the
        decision to send audio anywhere is made by the tier the client
        chose. None when the request carries no audio, nothing is
        configured, or the mode is `passthrough`.
        """
        if not transcription.has_audio(kwargs.get("messages")):
            return None
        config = self.endpoints.transcriber_for(tier)
        if config is None or not config.active:
            return None
        return config

    def _policy_reason(self, name: str, reqs) -> str | None:
        """Return why the request may not use `name`, or None.

        `policy:<class>:<reach|jur>` when its class may not (None without
        a loaded policy or a class on the request), then
        `gate:<name>:<why>` when an egress gate refuses it. A tier is
        judged by every leaf it can reach. The gate reads the class the
        controller put on `reqs` once the decision step has run, so a
        request is never judged on a class that is not final.
        """
        return self._aux_denial(
            name, getattr(reqs, "data_class", None), getattr(reqs, "route_policy", None)
        )

    def _can_serve(self, name: str, reqs) -> str | None:
        """Return the first requirement `name` cannot serve, or None.

        The request's data class comes first: under a policy, a leaf
        outside the class's reach or jurisdiction (or a tier with such a
        leaf) returns `policy:<class>:<reach|jur>`, whatever its
        capabilities; an egress gate's refusal returns
        `gate:<name>:<why>` the same way. This is the one check every tier side, forced
        side, sibling, pool member, balancer target and comparison
        target passes.

        A tier reads its cached union and is never strict: a tier is
        not an endpoint, and its union already assumes the best of
        every reachable leaf. See `capabilities.satisfies`.
        """
        denial = self._policy_reason(name, reqs)
        if denial is not None:
            return denial
        caps = side_capabilities(name, self.endpoints, self._tier_caps, self._catalog_records)
        strict = not self.endpoints.has_tier(name) and bool(
            getattr(self.endpoints.resolve(name), "strict", False)
        )
        return satisfies(caps, reqs, strict, name)

    def _route(
        self,
        prompt: str,
        kwargs: dict[str, Any],
        tier: str | None,
        router: str | None,
        threshold: float | None,
        reqs=None,
        decision=None,
        route_record=None,
    ) -> tuple[str, list[dict[str, Any]], ModelPair | None]:
        """Choose one endpoint for a prompt and record how it was chosen.

        A middleware may first name the tier to enter, which replaces
        the addressed one when that one accepts intent routing. A
        traffic rule or a middleware that returns a pair then bypasses
        the tree entirely: that pair is routed flat with the root
        level's resolved router and threshold, and the path carries a
        single entry naming where the pair came from.

        Parameters
        ----------
        prompt : str
            The request's prompt.
        kwargs : dict
            The request kwargs, which traffic rules read.
        tier : str, optional
            Tier addressed by the model name, if any.
        router : str, optional
            Request-level router from the model name, if any.
        threshold : float, optional
            Request-level threshold from the model name, if any.
        reqs : Requirements, optional
            What the request needs. None, or an empty one, leaves every
            level's router call where it was.

        Returns
        -------
        tuple[str, list[dict], ModelPair or None]
            The endpoint name to call, the decision path, and the pair
            the final level chose between when that level is a flat one.
            None when the walk ended inside the tier tree, whose sides
            the path already names.

        Raises
        ------
        RoutingError
            If no tier applies and the controller carries no flat pair.
        """
        data_class = getattr(reqs, "data_class", None)
        route_policy = getattr(reqs, "route_policy", None)
        inherited = {
            "parent_router": None,
            "parent_threshold": None,
            "request_router": router,
            "request_threshold": threshold,
            "default_router": self.default_router,
            "default_threshold": self.default_threshold,
        }

        # A traffic rule bypasses the tree outright, so it is consulted
        # first: classifying a prompt whose tier is about to be thrown
        # away would be paid for and never used.
        overridden = self.traffic_manager.get_model_pair(prompt, kwargs)
        pair_from = "traffic_rule"

        tier_from = ignored_tier = None
        if overridden is None:
            tier, tier_from, ignored_tier = self._apply_intent_tier(
                prompt, tier, data_class, route_policy
            )
            overridden = self._get_model_pair_for_prompt(prompt, data_class, route_policy)
            pair_from = "middleware"

        if overridden is None and tier is not None:
            picked, path = resolve_tier(
                tier,
                prompt,
                inherited,
                self.endpoints,
                functools.partial(
                    self._run_router, data_class=data_class, route_policy=route_policy
                ),
                requirements=reqs,
                check=self._can_serve,
                error_cls=route_error(RoutingError),
            )
            if decision is not None:
                path[0]["decision"] = decision.as_dict()
            if route_record is not None:
                path[0]["policy_routing"] = route_record.as_dict()
            if ignored_tier is not None:
                path[0]["intent_ignored"] = ignored_tier
            elif tier_from is not None:
                path[0]["tier_from"] = tier_from
            return picked, path, None

        # A bypassed pair, or a controller with no tiers at all, routes
        # flat at the root level's resolved values.
        root_tier = self.endpoints.get_tier(tier) if tier is not None else None
        level = resolve_level(
            root_tier.router if root_tier else None,
            root_tier.threshold if root_tier else None,
            inherited,
        )

        # Traffic rules were checked at startup; a middleware answers per
        # request, so its pair is checked here.
        if overridden is not None and self.endpoints.strict_registry:
            try:
                check_pair(self.endpoints, overridden, pair_from)
            except ValueError as exc:
                raise RoutingError(str(exc)) from exc

        pair = overridden or self.default_model_pair
        if pair is None:
            raise RoutingError(
                "No tier applies to this request and the controller carries "
                "no strong_model/weak_model pair."
            )

        forced = forced_side(
            pair_from if overridden is not None else "the configured pair",
            str(pair.strong),
            str(pair.weak),
            reqs,
            self._can_serve,
            route_error(RoutingError),
        )

        gate: list[dict[str, str]] = []
        if forced is None:
            picked, win_rate, *gate = self._run_router(
                level["router"], level["threshold"], prompt, pair, data_class, route_policy
            )
        else:
            picked, win_rate = forced[0], None

        entry = {
            "tier": None,
            "router": level["router"],
            "router_from": level["router_from"],
            "threshold": level["threshold"],
            "threshold_from": level["threshold_from"],
            "win_rate": win_rate,
            "picked": picked,
        }
        if forced is not None:
            entry["capability_forced"] = forced[1]
            entry["capability_requirement"] = forced[2]
        if forced is None and gate:
            entry.update(gate[0])
        if overridden is not None:
            entry["pair_from"] = pair_from
        if decision is not None:
            entry["decision"] = decision.as_dict()
        if route_record is not None:
            entry["policy_routing"] = route_record.as_dict()

        return picked, [entry], pair

    def _models_to_try(
        self,
        picked: str,
        path: list[dict[str, Any]],
        pair: ModelPair | None,
        reqs=None,
        denied: list[str] | None = None,
    ) -> tuple[str, list[str], str | None, str | None]:
        """Build the fallback chain for one request.

        Order: the canary when one fires, the picked leaf, then the
        sibling of the final pick. A tier-valued sibling is descended by
        its `weak` side without running any router.

        Parameters
        ----------
        picked : str
            Endpoint the routing decision landed on.
        path : list[dict]
            The decision path, whose last tier entry names the sibling.
        pair : ModelPair or None
            The pair the final level chose between when that level is a
            flat one, so the fallback stays inside the pair the request
            was routed against rather than the controller's default.
        reqs : Requirements, optional
            What the request needs; a sibling that cannot serve it is
            dropped. See `capabilities.usable_sibling`.
        denied : list[str], optional
            Receives the `policy:` or `gate:` reason of every candidate
            dropped for its class or by a gate, so a walk that ends with none left can
            refuse 403 instead of reporting an upstream error.

        Returns
        -------
        tuple[str, list[str], str or None, str or None]
            The endpoint the request is attributed to, the ordered names
            to try, the sibling endpoint itself, and the name it was
            written under.
        """
        if denied is None:
            denied = []
        canary = self.quality_manager.canary_config.canary_model
        if self.quality_manager.should_canary() and self._walkable(canary, reqs, denied):
            model_to_use = canary
        else:
            model_to_use = picked

        models_to_try = [model_to_use]
        if picked not in models_to_try:
            models_to_try.append(picked)

        # A side backed by a candidate pool is walked to its end before
        # the other side is tried.
        for name in self._pool_tail(path, picked, reqs, denied):
            if name not in models_to_try:
                models_to_try.append(name)

        found = sibling_of(path, self.endpoints, pair)
        sibling, reference = found if found is not None else (None, None)

        if not self._walkable(sibling, reqs, denied):
            sibling, reference = None, None

        if sibling is not None and sibling not in models_to_try:
            models_to_try.append(sibling)
            for name in self._pool_tail(path, sibling, reqs, denied):
                if name not in models_to_try:
                    models_to_try.append(name)

        gated = list(dict.fromkeys(r for r in denied if r.startswith("gate:")))
        if gated and path:
            path[-1]["egress_gate_denied"] = gated

        return model_to_use, models_to_try, sibling, reference

    def _walkable(self, name: str | None, reqs, denied: list[str] | None = None) -> bool:
        """Return whether `name` may be tried after the pick failed.

        The one check every fallback candidate, a sibling or a pool
        member, passes before it is added to the walk: a candidate that
        cannot serve the request is dropped. The class check and the
        egress gates sit in `_can_serve`, so a pool cannot route around
        what the sibling is held to. A candidate refused for its class
        or by a gate is noted in `denied`.
        """
        if usable_sibling(name, reqs, self._can_serve):
            return True
        reason = self._policy_reason(name, reqs) if name is not None else None
        if reason is not None and denied is not None:
            denied.append(reason)
        return False

    def _pool_tail(
        self, path: list[dict[str, Any]], head: str, reqs, denied: list[str] | None = None
    ) -> list[str]:
        """Return the pool members after `head` on its tier side, serveable ones.

        Empty when the final level is a flat pair, `head` is not a tier
        side's resolved endpoint, or the side has no pool.
        """
        last = path[-1] if path else None
        if not last or last.get("tier") is None:
            return []
        tier = self.endpoints.get_tier(last["tier"])
        for side in ("strong", "weak"):
            if getattr(tier, side) != head:
                continue
            pool = self.endpoints.pool_for(tier.name, side)
            return [name for name in pool[1:] if self._walkable(name, reqs, denied)]
        return []

    def _no_inner_retries(self, call_kwargs: dict[str, Any]) -> None:
        """Stop the client library retrying a call itself under cooldown.

        The provider SDK underneath litellm retries a 429 on its own,
        sleeping for the upstream's `Retry-After`: a call the resilience
        layer counts once would reach the endpoint several times and
        hold the request for the wait. With rate-limit cooldown on the
        resilience layer owns every retry, so the SDK's are switched
        off unless the request sets its own.
        """
        if self.resilience.config.rate_limit_cooldown:
            call_kwargs.setdefault("num_retries", 0)

    def _cooling_skip(self, name: str, walk: list[dict[str, Any]]) -> bool:
        """Note `name` in `walk` and return True when it is rate-limit cooling."""
        until = self.resilience.cooling(name)
        if until is None:
            return False
        walk.append({"endpoint": name, "outcome": "cooling", "until": until})
        logger.info("endpoint %s is cooling until %.0f; skipped without a call", name, until)
        return True

    def _raise_exhausted(
        self,
        names: list[str],
        last_err: Exception | None,
        denied: list[str] | None = None,
    ) -> None:
        """Raise what ends a walk that reached no answer.

        Every candidate cooling is `EndpointsCooling`, naming them. A
        candidate dropped for the request's class means the walk could
        not go on where it would have: `PolicyDenied`, never a move to
        another class. Any other mix raises the last real error.
        """
        cooling = {n: u for n in names if (u := self.resilience.cooling(n)) is not None}
        if cooling and len(cooling) == len(set(names)):
            raise EndpointsCooling(cooling, self.resilience.clock())
        if denied:
            raise PolicyDenied(f"{denied[0]}: no endpoint this request may reach could answer")
        if last_err:
            raise last_err

    def _attach_path(
        self,
        res,
        path: list[dict[str, Any]],
        used: str,
        sibling: str | None,
        sibling_reference: str | None,
        walk: list[dict[str, Any]] | None = None,
    ):
        """Log the decision path and attach it to a response.

        Recorded in `_hidden_params` only: an extra attribute would leak
        into the cached `model_dump()`.

        `fallback_from` is written only when the sibling is what
        answered. A canary also differs from the pick, but it is the
        intended target of its request rather than a fallback, so it
        leaves the path unlabelled.

        Parameters
        ----------
        res : ModelResponse
            The response to annotate.
        path : list[dict]
            The decision path.
        used : str
            Endpoint that answered.
        sibling : str or None
            The fallback endpoint for this request, if it has one.
        sibling_reference : str or None
            Name that sibling was written under on its tier, recorded as
            `fallback_from` when the sibling is what answered.
        walk : list[dict], optional
            What the fallback walk did before `used` answered: each
            endpoint skipped while cooling or failed. Recorded on the
            last entry as `walk`, ending with the one that served, and
            left off when the pick answered at once.

        Returns
        -------
        ModelResponse
            The same response.
        """
        final = [dict(entry) for entry in path]
        if final and self.egress_gates is not None:
            passed = self.egress_gates.passed(used)
            if passed:
                final[-1]["egress_gate"] = {"passed": passed}
        if final and walk:
            final[-1]["walk"] = [*walk, {"endpoint": used, "outcome": "served"}]
        if final and sibling is not None and used == sibling:
            final[-1]["fallback_from"] = sibling_reference or sibling

        logger.info("routing path: %s -> %s", final, used)

        try:
            res._hidden_params["routellm_path"] = final
            res._hidden_params["routellm_endpoint"] = used
        except (AttributeError, TypeError):
            logger.warning("response carries no _hidden_params; path not attached")

        return res

    def _area_of(self, path: list[dict]) -> str | None:
        """Return the area of the deepest tier on a decision path.

        A flat pair has no tier at all, so the lookup must tolerate a
        null one; a registry with no `areas:` answers None for every
        tier. This is what closes the loop: the recorded trace carries
        its area, so the next aggregation has areas without being
        handed the config.

        Parameters
        ----------
        path : list[dict]
            The decision path `_route` produced.

        Returns
        -------
        str or None
            The area name, or None when there is none.
        """
        if not path:
            return None
        return self.endpoints.area_of(path[-1].get("tier"))

    def _endpoint_call_params(
        self, model_name: str, reqs=None
    ) -> tuple[str, str | None, str | None, dict[str, Any]]:
        """Turn a model name into the parameters of one litellm call.

        The name is resolved through the endpoint registry, its
        credentials fall back to the controller defaults, and the load
        balancer overrides either when it returns one. Precedence:
        balancer, then endpoint, then controller default.

        Parameters
        ----------
        model_name : str
            Endpoint name, or a raw litellm model name.
        reqs : Requirements, optional
            The request's. With a data class, a load balancer selects
            only among targets the class may use.

        Returns
        -------
        tuple[str, str or None, str or None, dict]
            The `(model, api_base, api_key, extra)` to pass to litellm.
            `extra` is the endpoint's, and request kwargs override it.
        """
        endpoint = self.endpoints.resolve(model_name)
        api_base, api_key = endpoint.credentials(self.api_base, self.api_key)

        # Balancers stay keyed on the routed name, as the cache and
        # trace keys are, so a named endpoint balances under its name.
        # Membership decides whether one applies: balance() echoes its
        # argument when none is registered, which is indistinguishable
        # from a balancer whose target model equals the routed name.
        if model_name in self.traffic_manager.load_balancers:
            try:
                model, balanced_key, balanced_base = self.traffic_manager.balance(
                    model_name, self._balancer_filter(reqs)
                )
            except NoAllowedTarget:
                targets = self.traffic_manager.load_balancers[model_name].config.endpoints
                reason = self._policy_reason(targets[0].model, reqs) or "policy:?:reach"
                raise PolicyDenied(
                    f"{reason}: no target of load balancer {model_name!r} is allowed"
                ) from None
            if self.endpoints.strict_registry:
                # A target names a registry endpoint, so the call carries
                # that endpoint's model, reach and params, never a raw
                # string. The balancer's own base and key still win.
                endpoint = self.endpoints.resolve(model)
                model = endpoint.model
                api_base, api_key = endpoint.credentials(self.api_base, self.api_key)
        else:
            model, balanced_key, balanced_base = endpoint.model, None, None

        api_base = balanced_base or api_base
        extra = {
            **endpoint.extra,
            **drop_kwargs(model, api_base, endpoint.drop_params, endpoint.extra),
        }
        return model, api_base, balanced_key or api_key, extra

    def _balancer_filter(self, reqs):
        """Return the target check a balancer applies, None when the class gates nothing."""
        if self.policy is None and self.egress_gates is None:
            return None
        return lambda target: self._policy_reason(target, reqs) is None

    @staticmethod
    def _cache_params(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Everything that shapes the answer, for the cache key.

        The sampling params, tools and the like, plus a digest of the
        whole conversation: the cache's own `prompt` is only the last
        message, which would let two different conversations ending in
        the same words share one answer. `_context` digests all but the
        last turn's text, which scopes the semantic match to the same
        context.
        """
        params = {k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
        params["_conversation"] = conversation_key(kwargs.get("messages"))
        params["_context"] = context_key(kwargs.get("messages"))
        return params

    def _cache_scope(self, effective_class: str | None) -> tuple[str, str]:
        """Return `(namespace, mode)` for this request's cache use.

        The namespace is the digest of the caller and the effective data
        class, so an entry is never served across either. The mode is the
        class's `cache:` under a loaded policy, `none`, `exact` or `both`;
        a class the policy does not list, or one that is blocked, vaulted
        or redacted, gets `none`. Without a policy it is `both`, which the
        cache narrows to whatever its config enables.
        """
        principal = current_principal.get()
        namespace = cache_namespace(
            principal.principal if principal is not None else None, effective_class
        )
        if self.policy is None:
            return namespace, "both"
        spec = self.policy.classes.get(effective_class) if effective_class else None
        if spec is None or spec.treatment != "raw":
            return namespace, "none"
        return namespace, spec.cache or "none"

    def batch_calculate_win_rate(self, prompts, router: str):
        """Score every prompt's strong win rate with one router.

        Used by evaluations and threshold calibration, which need the raw
        scores before any threshold is applied.

        Parameters
        ----------
        prompts : pandas.Series
            Prompts to score.
        router : str
            Name of an initialized router.

        Returns
        -------
        pandas.Series
            Strong win rate per prompt, same index as `prompts`.

        Raises
        ------
        RoutingError
            If `router` was not initialized.
        """
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]

        if not router_instance.NO_PARALLEL:
            # Needs `pandarallel.initialize()` by the caller.
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)
        if self.progress_bar:
            return prompts.__class__(
                [
                    router_instance.calculate_strong_win_rate(p)
                    for p in tqdm(prompts, desc=f"Scoring {router}")
                ],
                index=prompts.index,
            )
        return prompts.apply(router_instance.calculate_strong_win_rate)

    def _validate_router_threshold(self, router: str, threshold: float):
        if router not in self.routers:
            raise RoutingError(f"Router {router} not found.")

        if not (0 <= threshold <= 1):
            raise RoutingError(f"Threshold {threshold} must be between 0 and 1.")

    @staticmethod
    def _is_402(exc) -> bool:
        """Report whether `exc` carries an HTTP 402 Payment Required.

        The status code is authoritative wherever the exception exposes
        one, either directly or on an attached response. Matching the
        message text alone is unsound in both directions: it misses a
        real 402 whose message never spells the digits, and it fires on
        an unrelated error that merely happens to contain "402" -- a
        token count, a model name, a request id -- which would pay a
        blockchain charge for a failure that never asked for one.

        The substring remains the fallback, and only the fallback, for
        an exception that exposes no status code at all; litellm raises
        such errors with the code in the message.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the downstream call.

        Returns
        -------
        bool
            True when the error is a 402 challenge.
        """
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)

        if status is not None:
            try:
                return int(status) == 402
            except (TypeError, ValueError):
                pass

        return "402" in str(exc)

    @staticmethod
    def _extract_402_response(exc):
        """Recover the 402's headers, body and URL from an exception.

        The payment gateway can only sign what the server actually
        asked for, so the raw response has to survive the trip through
        the exception. litellm is inconsistent here: the errors it
        builds for an authenticated or rate-limited call take a
        `response`, but the generic `APIError` it raises for a status it
        has no class for -- 402 among them -- carries only a `request`.
        Where the response is absent this returns empty wire data, and
        the gateway decides whether an unreadable challenge is payable.
        Inventing an amount and a currency here would sign a blank
        cheque against a challenge nobody read.

        Parameters
        ----------
        exc : BaseException
            The exception raised by the downstream call.

        Returns
        -------
        tuple[dict[str, str], bytes, dict, str]
            Lowercased response headers, raw body bytes, the decoded
            JSON body (empty dict when it does not decode), and the URL
            of the refused request. Any component absent comes back
            empty.
        """
        response = getattr(exc, "response", None)

        headers: dict[str, str] = {}
        raw_headers = getattr(response, "headers", None)
        if raw_headers is not None:
            try:
                headers = {str(k).lower(): str(v) for k, v in dict(raw_headers).items()}
            except (TypeError, ValueError):
                headers = {}

        body = getattr(response, "content", None)
        if not isinstance(body, (bytes, bytearray)):
            body = b""
        body = bytes(body)

        # The descriptive fields are read from the body the server sent,
        # never assumed. A provider that states its price in JSON is
        # quoted back accurately; one that states nothing yields empty
        # strings rather than a fabricated amount.
        decoded: dict = {}
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                candidate = json_fn()
            except Exception:
                candidate = None
            if isinstance(candidate, dict):
                decoded = candidate

        url = getattr(getattr(response, "request", None), "url", None)
        if url is None:
            url = getattr(getattr(exc, "request", None), "url", None)

        return headers, body, decoded, "" if url is None else str(url)

    def _may_pay(self, endpoint: str | None) -> bool:
        """Whether `endpoint` is authorised to charge the configured wallet.

        This is the second of the two seams a 402 can be paid at. The
        transport under litellm settles the ones it sees; this one
        catches the errors litellm raises for everything else, which is
        exactly what happens when the transport declined. Scoping only
        the transport would therefore close the hole and leave it open.

        Here the endpoint is known by name, so the question is its own
        `pay:` flag rather than a URL match. A name the registry does
        not know resolves to a passthrough endpoint, whose `pay` is
        False because no config line ever wrote one: an unconfigured
        model is never payable.

        Parameters
        ----------
        endpoint : str or None
            The routed endpoint name. None means the caller did not say,
            which only the internal helper's direct callers do.

        Returns
        -------
        bool
            True when a payment may be signed for this call.
        """
        if endpoint is None:
            # No name to check. The request path always passes one, so
            # this is reached only by a caller that already narrowed
            # the call itself, and refusing here would break it.
            return True
        return bool(self.endpoints.resolve(endpoint).pay)

    def _payment_cap(self, endpoint: str | None):
        """Return the cap binding a payment to `endpoint`, and whose it is.

        Authorisation says who may charge; this says how much.

        The endpoint is known here by name, so its own `max_payment:`
        is read off it directly rather than looked up by URL. That
        matters for an endpoint carrying no `api_base`: it has no base
        URL of its own to key a cap on, and resolving it through the
        default one would apply a sibling's cap to it.

        The ceiling still clamps: an endpoint cap above
        `--max-payment` is refused the same way it would be in the
        transport seam, so the two seams cannot disagree about what
        the operator allowed.

        Parameters
        ----------
        endpoint : str or None
            The routed endpoint name.

        Returns
        -------
        tuple
            `(cap, source)`, or `(None, None)` when no limit is
            configured, which leaves the gateway's own default
            standing rather than removing it.
        """
        if self.payment_limits is None or endpoint is None:
            return None, None

        from routellm.payment.limits import PaymentLimits

        own = self.endpoints.resolve(endpoint).max_payment
        if own is None:
            base = self.endpoints.resolve(endpoint).api_base or self.api_base
            if not base:
                return (
                    self.payment_limits.global_cap,
                    None if self.payment_limits.global_cap is None else "global",
                )
            return self.payment_limits.effective(base)

        # One synthetic base carrying this endpoint's own cap, so the
        # min() and the naming are the limits' own rule rather than a
        # second copy of it here.
        probe = "https://endpoint.invalid/"
        return PaymentLimits(
            global_cap=self.payment_limits.global_cap,
            per_base={probe: own},
        ).effective(probe)

    async def _request_with_payment(self, call_fn, endpoint=None):
        """Retry a refused call once, carrying proof of payment.

        A 402 is parsed, paid and retried only when a payment gateway is
        configured AND the endpoint was authorised to charge; otherwise
        the error propagates untouched, exactly as it would with no
        wallet at all.

        Parameters
        ----------
        call_fn : callable
            Coroutine function taking a dict of extra headers.
        endpoint : str, optional
            The routed endpoint name, whose `pay:` flag decides whether
            a payment may be signed. None skips the check.

        Returns
        -------
        Any
            The downstream response, from the first call or the retry.
        """
        try:
            return await call_fn({})
        except Exception as e:
            # Check if it's a 402 challenge; the status code decides
            # where the error carries one, the message text otherwise.
            if self._is_402(e) and self.payment_gateway and self._may_pay(endpoint):
                import logging

                logging.getLogger(__name__).info("Received 402 challenge, attempting to pay...")

                headers, body, stated, resource_url = self._extract_402_response(e)
                # The cap belongs to this payment, not to the gateway:
                # one wallet serves every endpoint and each may be
                # capped differently, so the seam that knows which
                # endpoint it called is the one that states the figure.
                cap, cap_source = self._payment_cap(endpoint)

                # The budget is the sum of what every payment
                # authorised, and this seam fires exactly when the
                # transport declined -- so a budget enforced only
                # there is a budget with a hole in it. Reserved before
                # signing, because refusing has to happen before a
                # wallet is authorised.
                if self.payment_budget:
                    refused = self.payment_budget.debit(cap)
                    if refused is not None:
                        raise RuntimeError(refused) from e

                challenge = PaymentChallenge(
                    scheme=stated.get("scheme") or self.payment_gateway.name,
                    network=stated.get("network") or self.payment_gateway.networks[0],
                    amount=stated.get("amount", ""),
                    currency=stated.get("currency", ""),
                    payload=stated,
                    headers=headers,
                    body=body,
                    resource_url=stated.get("resource") or resource_url,
                    max_amount=cap,
                    cap_source=cap_source,
                )

                try:
                    receipt = await self.payment_gateway.pay(challenge)
                except BaseException:
                    # Nothing was signed, so the reservation was never
                    # spent and goes back. Keeping it would let a
                    # repeatedly failing payment drain the budget
                    # without ever paying anyone.
                    if self.payment_budget:
                        self.payment_budget.refund(cap)
                    raise

                # The header name belongs to the protocol version the
                # server chose, so it travels on the receipt rather than
                # being assumed here.
                return await call_fn({receipt.header_name: receipt.proof})

            if self._is_402(e) and self.payment_gateway:
                import logging

                logging.getLogger(__name__).info(
                    "Received 402 from %r, which is not authorised to "
                    "charge this wallet; set `pay: true` on it to allow "
                    "payment. Re-raising unpaid.",
                    endpoint,
                )
            raise

    def completion(
        self,
        *,
        router: str | None = None,
        threshold: float | None = None,
        transcribe_language: str | None = None,
        request_id: str | None = None,
        declared_class: str | None = None,
        conversation_id: str | None = None,
        principal: Principal | None = None,
        attributes: RequestAttributes | None = None,
        **kwargs,
    ):
        """Synchronous completion with intelligent routing.

        Matches OpenAI Chat Completions API with extensions for router-based
        model selection. Routes prompts to strong/weak models based on
        difficulty score and threshold.

        Router and threshold can be specified explicitly or parsed from
        the model name, which also addresses a tier. A tiered request
        walks the tree, one router call per level, and the decision path
        comes back on `response._hidden_params["routellm_path"]`.

        Cache, resilience, and trace keys are the routed name: the
        endpoint name for a configured endpoint, the raw model name
        otherwise. Adopting endpoint names therefore changes existing
        SQLite cache keys, and prior entries for the same model go
        unread.

        Streaming: with `stream=True` this returns the provider's stream
        as it is. Unlike `acompletion`, it reads no first chunk, so a
        leaf that fails on the first read is not replaced by its sibling,
        and it records no trace. Streaming fallback and tracing hold only
        on the async path (`acompletion`, and so the server).

        Parameters
        ----------
        router : str, optional
            Router name. Parsed from model name if not provided.
        threshold : float, optional
            Routing threshold in [0, 1]. Parsed from model name if not
            provided.
        transcribe_language : str, optional
            Language hint for audio transcription, used only when the
            addressed tier transcribes audio parts. Overrides the
            configured `language`.
        request_id : str, optional
            Correlates the transcription log line with the caller's own.
        declared_class : str, optional
            Data class the caller declares; the decision step never
            lowers it. Read only when `decision:` is configured.
        conversation_id : str, optional
            Key for the decision step's sticky floor; derived from the
            conversation's opening message when absent.
        principal : Principal, optional
            The authenticated caller. Never forwarded upstream; it is
            carried on `routellm.auth.current_principal` for policy,
            audit and cache code on this request's call path.
        attributes : RequestAttributes, optional
            The request's routing attributes (project, client, ...).
            Read only when `policy_routing` is configured; never forwarded.
        **kwargs
            Standard OpenAI completion parameters (messages, temperature,
            top_p, etc.). model field can encode router and threshold.

        Returns
        -------
        ModelResponse
            Completion response matching OpenAI format.

        Raises
        ------
        RoutingError
            If router/threshold invalid or model name malformed.
        TranscriptionError
            If the addressed tier transcribes audio and that fails. The
            raw audio is never routed instead.
        """
        if principal is not None:
            current_principal.set(principal)
        tier = None
        request_name = kwargs.get("model")
        if request_name is not None:
            tier, router, threshold = self._parse_model_name(request_name)
        else:
            request_name = router

        # Audio becomes text before anything reads the messages, so the
        # router, the requirements and the cache key all see the transcript.
        transcriber = self._transcriber_for(kwargs, tier)
        if transcriber is not None:
            kwargs = {
                **kwargs,
                "messages": transcription.transcribe_sync(
                    kwargs["messages"], transcriber, transcribe_language, request_id
                ),
            }

        # A vision request's content is a list; routers want a string.
        prompt = requirements_module._prompt_text(kwargs["messages"])
        reqs = requirements_module.derive(
            kwargs["messages"],
            kwargs,
            request_name,
            self.endpoints.audio_tokens_per_second,
        )
        session_id = kwargs.get("user") or str(uuid.uuid4())

        # 0. Decide: label the request before anything routes it.
        decision = self._decide_sync(kwargs, declared_class, conversation_id)
        tier, effective, reqs, route_record = self._policy_route(
            tier, declared_class, decision, reqs, attributes, principal
        )
        reqs = self._check_entry_tier(tier, declared_class, decision, reqs, effective)

        # 1. Route: traffic rules and middleware bypass the tree, a tier
        #    walks it, and a bare pair routes flat. Each side is checked
        #    for fitness and class before its router runs.
        routed_model, path, effective_pair = self._route(
            prompt,
            kwargs,
            tier,
            router,
            threshold,
            reqs,
            decision=decision,
            route_record=route_record,
        )
        area = self._area_of(path)

        # 2. Apply canary testing and build the fallback chain, which
        #    stays inside the pair this request was routed against.
        denied: list[str] = []
        model_to_use, models_to_try, sibling, sibling_reference = self._models_to_try(
            routed_model, path, effective_pair, reqs, denied
        )
        is_canary = model_to_use != routed_model

        self.model_counts[request_name][model_to_use] += 1

        # A stream is neither served from the cache nor stored in it, and
        # `model_dump` on its wrapper would raise: the iterator goes
        # straight back to the caller, with the path attached.
        streaming = bool(kwargs.get("stream"))

        # Try cache
        cache_params = self._cache_params(kwargs)
        # The cache reads the one effective class (policy floor, declared
        # and decision), never the declared or decided one alone.
        request_class = self._request_class(declared_class, decision, effective)
        cache_ns, cache_mode = self._cache_scope(request_class)
        # A request a transform rewrites is never cached: a stored reply
        # could not be restored without that request's own state.
        use_cache = (
            not streaming
            and cache_mode != "none"
            and not (
                self.transforms.active and self.transforms.selects(context_for(self, reqs, None))
            )
        )
        fuzzy = cache_mode == "both" and self._semantic_cache_allowed(
            request_class,
            getattr(reqs, "route_policy", None),
        )
        cached_res = (
            self.cache.get(prompt, model_to_use, cache_params, cache_ns, fuzzy)
            if use_cache
            else None
        )
        if cached_res:
            from litellm.utils import ModelResponse

            res = ModelResponse(**cached_res)
            # Record trace for cached response too
            self.quality_manager.record_trace(
                prompt,
                model_to_use,
                res.model_dump(),
                {"cached": True, "is_canary": is_canary},
                path=path,
                request_model=request_name,
                endpoint=model_to_use,
                session_id=session_id,
                latency_ms=0,
                area=area,
                request_ctx=reqs,
            )
            return self._attach_path(res, path, model_to_use, sibling, sibling_reference)

        last_err = None
        walk: list[dict[str, Any]] = []
        for model_name in models_to_try:
            if self._cooling_skip(model_name, walk):
                continue

            # 3. Resolve the endpoint, then apply load balancing
            model, curr_api_base, curr_api_key, extra = self._endpoint_call_params(model_name, reqs)

            # Rewrite the messages for this candidate. A transform that
            # raises ends the request here: no call, no fallback.
            call_src, applied = forward_call(self, kwargs, model_name, reqs, request_id)
            rec_prompt = (
                requirements_module._prompt_text(call_src["messages"]) if applied.active else prompt
            )

            def _call(call_src=call_src):
                kwargs_copy = {**extra, **call_src}
                kwargs_copy["model"] = model
                log_dropped(model_name, model, curr_api_base, kwargs_copy)
                kwargs_copy.pop("extra_headers", None)
                self._no_inner_retries(kwargs_copy)
                return completion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    extra_headers=outbound_headers(
                        extra.get("extra_headers"), kwargs.get("extra_headers")
                    ),
                    **kwargs_copy,
                )

            try:
                # Wrap with resilience
                started = time.monotonic()
                res = self.resilience.wrap_completion(model_name, _call)
                latency_ms = int((time.monotonic() - started) * 1000)

                if streaming:
                    if applied.active:
                        res = RestoringIterator(res, applied.stream_restore())
                    return self._attach_path(
                        res, path, model_name, sibling, sibling_reference, walk
                    )

                # Cache the response
                try:
                    if use_cache:
                        self.cache.put(
                            prompt, model_name, cache_params, res.model_dump(), cache_ns, fuzzy
                        )
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).error(f"Failed to cache response: {e!s}")

                # Record trace: the transformed prompt and reply only
                self.quality_manager.record_trace(
                    rec_prompt,
                    model_name,
                    res.model_dump(),
                    {"is_canary": is_canary and model_name == model_to_use},
                    path=path,
                    request_model=request_name,
                    endpoint=model_name,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    area=area,
                    request_ctx=reqs,
                )

                res = applied.restore_response(res)
                return self._attach_path(res, path, model_name, sibling, sibling_reference, walk)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {e!s}"
                )
                walk.append(
                    {
                        "endpoint": model_name,
                        "outcome": "failed",
                        "error": type(e).__name__,
                    }
                )
                last_err = e
                continue

        self._raise_exhausted(models_to_try, last_err, denied)

    def _stream_ended(
        self,
        chunks,
        error,
        *,
        prompt,
        messages,
        model_name,
        is_canary,
        path,
        request_name,
        session_id,
        started,
        area,
        comparison=None,
        reqs=None,
    ):
        """Record a streamed completion's trace once the stream is over.

        Runs when the stream is exhausted or fails, never before: the
        trace needs the whole answer. A failed stream records nothing,
        since a partial answer is not one the quality data should learn
        from; the failure itself is logged.

        Assembling the answer and writing the trace are blocking work,
        so they run on the default executor: the stream's last chunk and
        `[DONE]` do not wait on them. The write is tracked, never
        dropped, and a failure is logged.
        """
        # Only now is the stream's outcome known: the breaker and the
        # rate-limit cooldown hear it here, not at the first chunk.
        self.resilience.record_stream_end(model_name, error)

        if error is not None:
            logger.warning(f"Model {model_name} failed mid-stream. Error: {error!s}")
            return

        # Taken now: how long the stream took, not how long the write waited.
        latency_ms = int((time.monotonic() - started) * 1000)
        chunks = list(chunks)

        def record():
            res = stream_chunk_builder(chunks, messages=messages)
            if res is None:
                return
            self.quality_manager.record_trace(
                prompt,
                model_name,
                res.model_dump(),
                {"is_canary": is_canary},
                path=path,
                request_model=request_name,
                endpoint=model_name,
                session_id=session_id,
                latency_ms=latency_ms,
                area=area,
                comparison=comparison,
                request_ctx=reqs,
            )
            logger.info("routing path at stream end: %s -> %s", path, model_name)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            record()
            return
        write = loop.run_in_executor(None, record)
        self._trace_writes.add(write)
        write.add_done_callback(self._trace_written)

    def _trace_written(self, write: asyncio.Future) -> None:
        """Forget a finished stream-trace write, logging a failure."""
        self._trace_writes.discard(write)
        if not write.cancelled() and write.exception() is not None:
            logger.error(f"Recording a stream trace failed: {write.exception()!s}")

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: str | None = None,
        threshold: float | None = None,
        transcribe_language: str | None = None,
        request_id: str | None = None,
        compare: str | None = None,
        declared_class: str | None = None,
        conversation_id: str | None = None,
        principal: Principal | None = None,
        attributes: RequestAttributes | None = None,
        **kwargs,
    ):
        if principal is not None:
            current_principal.set(principal)
        tier = None
        request_name = kwargs.get("model")
        if request_name is not None:
            tier, router, threshold = self._parse_model_name(request_name)
        else:
            request_name = router
        audit_note(entry_tier=tier)

        # Audio becomes text before anything reads the messages, so the
        # router, the requirements and the cache key all see the
        # transcript. Streaming is unaffected: this finishes before routing.
        transcriber = self._transcriber_for(kwargs, tier)
        if transcriber is not None:
            kwargs = {
                **kwargs,
                "messages": await transcription.transcribe(
                    kwargs["messages"], transcriber, transcribe_language, request_id
                ),
            }

        # A vision request's content is a list; routers want a string.
        prompt = requirements_module._prompt_text(kwargs["messages"])
        reqs = requirements_module.derive(
            kwargs["messages"],
            kwargs,
            request_name,
            self.endpoints.audio_tokens_per_second,
        )
        session_id = kwargs.get("user") or str(uuid.uuid4())

        # 0. Decide: label the request before anything routes it. Awaited,
        #    so a slow decision model never holds the event loop.
        decision = await self._decide(kwargs, declared_class, conversation_id)
        tier, effective, reqs, route_record = self._policy_route(
            tier, declared_class, decision, reqs, attributes, principal
        )
        audit_note(
            effective_class=self._request_class(declared_class, decision, effective),
            policy=route_record.policy if route_record is not None else None,
        )
        reqs = self._check_entry_tier(tier, declared_class, decision, reqs, effective)

        # 1. Route: traffic rules and middleware bypass the tree, a tier
        #    walks it, and a bare pair routes flat. Each side is checked
        #    for fitness and class before its router runs.
        routed_model, path, effective_pair = self._route(
            prompt,
            kwargs,
            tier,
            router,
            threshold,
            reqs,
            decision=decision,
            route_record=route_record,
        )
        area = self._area_of(path)

        # 2. Apply canary testing and build the fallback chain, which
        #    stays inside the pair this request was routed against.
        denied: list[str] = []
        model_to_use, models_to_try, sibling, sibling_reference = self._models_to_try(
            routed_model, path, effective_pair, reqs, denied
        )
        is_canary = model_to_use != routed_model

        self.model_counts[request_name][model_to_use] += 1

        # A stream is neither served from the cache nor stored in it: the
        # cache holds whole responses, and a stream has none to hold.
        streaming = bool(kwargs.get("stream"))

        # Try cache
        cache_params = self._cache_params(kwargs)
        # The cache reads the one effective class (policy floor, declared
        # and decision), never the declared or decided one alone.
        request_class = self._request_class(declared_class, decision, effective)
        cache_ns, cache_mode = self._cache_scope(request_class)
        # A request a transform rewrites is never cached: a stored reply
        # could not be restored without that request's own state.
        use_cache = (
            not streaming
            and cache_mode != "none"
            and not (
                self.transforms.active and self.transforms.selects(context_for(self, reqs, None))
            )
        )
        fuzzy = cache_mode == "both" and self._semantic_cache_allowed(
            request_class,
            getattr(reqs, "route_policy", None),
        )
        audit_note(cache="miss" if use_cache and self.cache.config.enabled else "bypass")
        cached_res = (
            await self.cache.aget(prompt, model_to_use, cache_params, cache_ns, fuzzy)
            if use_cache
            else None
        )
        if cached_res:
            from litellm.utils import ModelResponse

            res = ModelResponse(**cached_res)
            audit_note(cache="hit")
            # Record trace for cached response too
            self.quality_manager.record_trace(
                prompt,
                model_to_use,
                res.model_dump(),
                {"cached": True, "is_canary": is_canary},
                path=path,
                request_model=request_name,
                endpoint=model_to_use,
                session_id=session_id,
                latency_ms=0,
                area=area,
                request_ctx=reqs,
            )
            return self._attach_path(res, path, model_to_use, sibling, sibling_reference)

        # Shadow comparison: background tasks that record and nothing
        # else. They start only on a cache miss, run beside the primary
        # and never see the cache or the client.
        primary_mark = None
        if self.comparison is not None:
            plan = self.comparison.plan(
                header=compare,
                path=path,
                pair=effective_pair,
                picked=routed_model,
                reqs=reqs,
            )
            if plan is not None:
                primary_mark = {"id": plan.id, "role": "primary"}
                self.comparison.launch(
                    plan,
                    kwargs=kwargs,
                    prompt=prompt,
                    request_model=request_name,
                    path=path,
                    session_id=session_id,
                    reqs=reqs,
                )

        last_err = None
        walk: list[dict[str, Any]] = []
        for model_name in models_to_try:
            if self._cooling_skip(model_name, walk):
                continue

            # 3. Resolve the endpoint, then apply load balancing
            model, curr_api_base, curr_api_key, extra = self._endpoint_call_params(model_name, reqs)

            # Rewrite the messages for this candidate. A transform that
            # raises ends the request here: no call, no fallback.
            call_src, applied = forward_call(self, kwargs, model_name, reqs, request_id)
            rec_prompt = (
                requirements_module._prompt_text(call_src["messages"]) if applied.active else prompt
            )

            async def _call(extra_headers=None, call_src=call_src):
                kwargs_copy = {**extra, **call_src}
                kwargs_copy["model"] = model
                log_dropped(model_name, model, curr_api_base, kwargs_copy)
                # The endpoint's headers, the request's, then a payment
                # proof, merged into the one `extra_headers` litellm takes.
                kwargs_copy.pop("extra_headers", None)
                self._no_inner_retries(kwargs_copy)
                return await acompletion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    extra_headers=outbound_headers(
                        extra.get("extra_headers"), kwargs.get("extra_headers"), extra_headers
                    ),
                    **kwargs_copy,
                )

            try:
                started = time.monotonic()

                async def _open(_call=_call, model_name=model_name, applied=applied):
                    res = await self._request_with_payment(_call, endpoint=model_name)
                    if not streaming:
                        return res
                    # The sibling can only take over before the first
                    # chunk, so the first chunk is read inside the same
                    # guarded call that opened the stream.
                    return await RoutedStream.open(
                        res,
                        on_end=lambda chunks, err, model_name=model_name: self._stream_ended(
                            chunks,
                            err,
                            prompt=rec_prompt,
                            messages=call_src["messages"],
                            model_name=model_name,
                            is_canary=is_canary and model_name == model_to_use,
                            path=path,
                            request_name=request_name,
                            session_id=session_id,
                            started=started,
                            area=area,
                            comparison=primary_mark,
                            reqs=reqs,
                        ),
                        restore=applied.stream_restore() if applied.active else None,
                    )

                res = await self.resilience.wrap_acompletion(
                    model_name,
                    _open,
                    timeout_ms=self.endpoints.resolve(model_name).timeout_ms,
                    defer_success=streaming,
                )
                latency_ms = int((time.monotonic() - started) * 1000)

                if streaming:
                    return self._attach_path(
                        res, path, model_name, sibling, sibling_reference, walk
                    )

                # 4. Validate canary if needed
                if is_canary and model_name == model_to_use:
                    passed = await self.quality_manager.validate_canary(
                        res.choices[0].message.content, rec_prompt
                    )
                    if not passed:
                        import logging

                        logging.getLogger(__name__).warning(
                            f"Canary model {model_name} failed validation, trying fallback."
                        )
                        raise Exception(f"Canary validation failed for {model_name}")

                # Cache the response
                try:
                    if use_cache:
                        await self.cache.aput(
                            prompt, model_name, cache_params, res.model_dump(), cache_ns, fuzzy
                        )
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).error(f"Failed to cache response: {e!s}")

                # Record trace: the transformed prompt and reply only
                self.quality_manager.record_trace(
                    rec_prompt,
                    model_name,
                    res.model_dump(),
                    {"is_canary": is_canary and model_name == model_to_use},
                    path=path,
                    request_model=request_name,
                    endpoint=model_name,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    area=area,
                    comparison=primary_mark,
                    request_ctx=reqs,
                )

                res = applied.restore_response(res)
                return self._attach_path(res, path, model_name, sibling, sibling_reference, walk)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {e!s}"
                )
                walk.append(
                    {
                        "endpoint": model_name,
                        "outcome": "failed",
                        "error": type(e).__name__,
                    }
                )
                last_err = e
                continue

        audit_note(walk=walk)
        self._raise_exhausted(models_to_try, last_err, denied)
