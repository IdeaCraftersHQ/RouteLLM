"""Main controller for intelligent model routing.

Implements a router-based system that selects between strong and weak
models based on prompt difficulty. Supports caching, resilience,
quality management, and payment gateway integration.
"""

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Protocol

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm import requirements as requirements_module
from routellm.caching import Cache, CacheConfig
from routellm.capabilities import (
    build_tier_index,
    catalog_records,
    satisfies,
    side_capabilities,
    usable_sibling,
)
from routellm.endpoints import EndpointRegistry, Selector
from routellm.hints import TYPESAFE_INSTALL_HINT
from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentChallenge
from routellm.quality import QualityManager
from routellm.resilience import Resilience, ResilienceConfig
from routellm.routers.embeddings import configure_embeddings
from routellm.routers.routers import ROUTER_CLS
from routellm.routing import (
    forced_side,
    parse_model_name,
    resolve_level,
    resolve_tier,
    sibling_of,
)
from routellm.traffic import TrafficManager
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


class Controller:
    """Main router controller for intelligent model routing.

    Orchestrates routing decisions, caching, resilience, quality management,
    and payment handling. Matches OpenAI API while supporting advanced
    routing features.
    """
    def __init__(
        self,
        routers: list[str],
        strong_model: Optional[str] = None,
        weak_model: Optional[str] = None,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
        middleware: Optional[List[Middleware]] = None,
        payment_gateway: Optional[PaymentGateway] = None,
        resilience_config: Optional[ResilienceConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        traffic_manager: Optional[TrafficManager] = None,
        quality_manager: Optional[QualityManager] = None,
        endpoints: Optional[EndpointRegistry] = None,
        default_router: Optional[str] = None,
        default_threshold: float = 0.5,
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
        payment_gateway : PaymentGateway, optional
            Gateway for 402 payment challenges.
        resilience_config : ResilienceConfig, optional
            Configuration for circuit breaker and retry logic.
        cache_config : CacheConfig, optional
            Configuration for completion caching.
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

        Raises
        ------
        ValueError
            If both models are None and no `default` tier is configured,
            a tier names a router that is not registered, a tier's
            policy resolves to no endpoint or to the same endpoint on
            both sides, or `strong_model`/`weak_model` names a tier.
        """
        self.endpoints = endpoints or EndpointRegistry()

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
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        self.middleware = middleware or []
        self.payment_gateway = payment_gateway
        self.resilience = Resilience(resilience_config)
        self.cache = Cache(cache_config)
        self.traffic_manager = traffic_manager or TrafficManager()
        self.quality_manager = quality_manager or QualityManager()

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        # Routers that embed prompts build their client on first use;
        # point it at the `embedding` endpoint before any is constructed.
        configure_embeddings(self.endpoints)

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

        router_pbar = None
        if self.progress_bar:
            router_pbar = tqdm(total=len(to_build), desc="Initializing routers")

        for router in to_build:
            router_config = config.get(router, {})
            self.routers[router] = ROUTER_CLS[router](**router_config)
            if router_pbar:
                router_pbar.update(1)

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

    def _get_model_pair_for_prompt(self, prompt: str) -> Optional[ModelPair]:
        """Get the model pair a middleware asks for, or None."""
        for m in self.middleware:
            pair = m.get_model_pair(prompt)
            if pair:
                return pair
        return None

    def _get_tier_for_prompt(self, prompt: str) -> Optional[str]:
        """Get the tier a middleware asks the request to enter, or None.

        `get_tier` is optional on the middleware protocol, so it is
        looked up with `getattr`: middleware written before the hook
        carries only `get_model_pair` and is skipped here.

        Parameters
        ----------
        prompt : str
            The request's prompt.

        Returns
        -------
        str or None
            The first tier name a middleware names, or None when none
            does.
        """
        for m in self.middleware:
            hook = getattr(m, "get_tier", None)
            if hook is None:
                continue
            tier = hook(prompt)
            if tier:
                return tier
        return None

    def _apply_intent_tier(
        self, prompt: str, tier: Optional[str]
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
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

        chosen = self._get_tier_for_prompt(prompt)
        if chosen is None:
            return tier, None, None

        if not self.endpoints.has_tier(chosen):
            known = ", ".join(self.endpoints.tier_names()) or "<none>"
            raise RoutingError(
                f"Middleware selected unknown tier: {chosen}. "
                f"Configured tiers: {known}"
            )

        if not self.endpoints.get_tier(tier).accepts_intent_routing():
            logger.debug(
                "tier %s does not accept intent routing; ignoring %s",
                tier,
                chosen,
            )
            return tier, None, chosen

        return chosen, "intent", None

    def _parse_model_name(
        self, model_name: str
    ) -> tuple[Optional[str], Optional[str], Optional[float]]:
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

    def _run_router(
        self, router: str, threshold: float, prompt: str, pair: ModelPair
    ) -> tuple[str, Optional[float]]:
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

        Returns
        -------
        tuple[str, float or None]
            The side picked, and the win rate behind it, None when the
            router reports none.

        Raises
        ------
        RoutingError
            If the router is unknown or the threshold is out of range.
        """
        self._validate_router_threshold(router, threshold)
        instance = self.routers[router]

        # Routers predating the hook, and test doubles, may carry only
        # `route`; they decide for themselves and report no score.
        scored = getattr(instance, "route_with_score", None)
        if scored is None:
            return instance.route(prompt, threshold, pair), None

        return scored(prompt, threshold, pair)

    def _can_serve(self, name: str, reqs) -> Optional[str]:
        """Return the first requirement `name` cannot serve, or None.

        A tier reads its cached union and is never strict: a tier is
        not an endpoint, and its union already assumes the best of
        every reachable leaf. See `capabilities.satisfies`.
        """
        caps = side_capabilities(
            name, self.endpoints, self._tier_caps, self._catalog_records
        )
        strict = not self.endpoints.has_tier(name) and bool(
            getattr(self.endpoints.resolve(name), "strict", False)
        )
        return satisfies(caps, reqs, strict, name)

    def _route(
        self,
        prompt: str,
        kwargs: dict[str, Any],
        tier: Optional[str],
        router: Optional[str],
        threshold: Optional[float],
        reqs=None,
    ) -> tuple[str, list[dict[str, Any]], Optional[ModelPair]]:
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
            tier, tier_from, ignored_tier = self._apply_intent_tier(prompt, tier)
            overridden = self._get_model_pair_for_prompt(prompt)
            pair_from = "middleware"

        if overridden is None and tier is not None:
            picked, path = resolve_tier(
                tier,
                prompt,
                inherited,
                self.endpoints,
                self._run_router,
                requirements=reqs,
                check=self._can_serve,
                error_cls=RoutingError,
            )
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
            RoutingError,
        )

        if forced is None:
            picked, win_rate = self._run_router(
                level["router"], level["threshold"], prompt, pair
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
        if overridden is not None:
            entry["pair_from"] = pair_from

        return picked, [entry], pair

    def _models_to_try(
        self,
        picked: str,
        path: list[dict[str, Any]],
        pair: Optional[ModelPair],
        reqs=None,
    ) -> tuple[str, list[str], Optional[str], Optional[str]]:
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

        Returns
        -------
        tuple[str, list[str], str or None, str or None]
            The endpoint the request is attributed to, the ordered names
            to try, the sibling endpoint itself, and the name it was
            written under.
        """
        if self.quality_manager.should_canary():
            model_to_use = self.quality_manager.canary_config.canary_model
        else:
            model_to_use = picked

        models_to_try = [model_to_use]
        if picked not in models_to_try:
            models_to_try.append(picked)

        found = sibling_of(path, self.endpoints, pair)
        sibling, reference = found if found is not None else (None, None)

        if not usable_sibling(sibling, reqs, self._can_serve):
            sibling, reference = None, None

        if sibling is not None and sibling not in models_to_try:
            models_to_try.append(sibling)

        return model_to_use, models_to_try, sibling, reference

    def _attach_path(
        self,
        res,
        path: list[dict[str, Any]],
        used: str,
        sibling: Optional[str],
        sibling_reference: Optional[str],
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

        Returns
        -------
        ModelResponse
            The same response.
        """
        final = [dict(entry) for entry in path]
        if final and sibling is not None and used == sibling:
            final[-1]["fallback_from"] = sibling_reference or sibling

        logger.info("routing path: %s -> %s", final, used)

        try:
            res._hidden_params["routellm_path"] = final
        except (AttributeError, TypeError):
            logger.warning("response carries no _hidden_params; path not attached")

        return res

    def _area_of(self, path: List[dict]) -> Optional[str]:
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
        self, model_name: str
    ) -> tuple[str, Optional[str], Optional[str], dict[str, Any]]:
        """Turn a model name into the parameters of one litellm call.

        The name is resolved through the endpoint registry, its
        credentials fall back to the controller defaults, and the load
        balancer overrides either when it returns one. Precedence:
        balancer, then endpoint, then controller default.

        Parameters
        ----------
        model_name : str
            Endpoint name, or a raw litellm model name.

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
            model, balanced_key, balanced_base = self.traffic_manager.balance(
                model_name
            )
        else:
            model, balanced_key, balanced_base = endpoint.model, None, None

        return model, balanced_base or api_base, balanced_key or api_key, endpoint.extra

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

    def _may_pay(self, endpoint: Optional[str]) -> bool:
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

                logging.getLogger(__name__).info(
                    "Received 402 challenge, attempting to pay..."
                )

                headers, body, stated, resource_url = self._extract_402_response(e)
                challenge = PaymentChallenge(
                    scheme=stated.get("scheme") or self.payment_gateway.name,
                    network=stated.get("network")
                    or self.payment_gateway.networks[0],
                    amount=stated.get("amount", ""),
                    currency=stated.get("currency", ""),
                    payload=stated,
                    headers=headers,
                    body=body,
                    resource_url=stated.get("resource") or resource_url,
                )

                receipt = await self.payment_gateway.pay(challenge)
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
        router: Optional[str] = None,
        threshold: Optional[float] = None,
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

        Parameters
        ----------
        router : str, optional
            Router name. Parsed from model name if not provided.
        threshold : float, optional
            Routing threshold in [0, 1]. Parsed from model name if not
            provided.
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
        """
        tier = None
        request_name = kwargs.get("model")
        if request_name is not None:
            tier, router, threshold = self._parse_model_name(request_name)
        else:
            request_name = router

        # A vision request's content is a list; routers want a string.
        prompt = requirements_module._prompt_text(kwargs["messages"])
        reqs = requirements_module.derive(kwargs["messages"], kwargs, request_name)
        session_id = kwargs.get("user") or str(uuid.uuid4())

        # 1. Route: traffic rules and middleware bypass the tree, a tier
        #    walks it, and a bare pair routes flat. Each side is checked
        #    for fitness before its router runs.
        routed_model, path, effective_pair = self._route(
            prompt, kwargs, tier, router, threshold, reqs
        )
        area = self._area_of(path)

        # 2. Apply canary testing and build the fallback chain, which
        #    stays inside the pair this request was routed against.
        model_to_use, models_to_try, sibling, sibling_reference = self._models_to_try(
            routed_model, path, effective_pair, reqs
        )
        is_canary = model_to_use != routed_model

        self.model_counts[request_name][model_to_use] += 1

        # Try cache
        cache_params = {k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
        cached_res = self.cache.get(prompt, model_to_use, cache_params)
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
            )
            return self._attach_path(
                res, path, model_to_use, sibling, sibling_reference
            )

        last_err = None
        for model_name in models_to_try:
            # 3. Resolve the endpoint, then apply load balancing
            model, curr_api_base, curr_api_key, extra = self._endpoint_call_params(
                model_name
            )

            def _call():
                kwargs_copy = {**extra, **kwargs}
                kwargs_copy["model"] = model
                return completion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    **kwargs_copy,
                )

            try:
                # Wrap with resilience
                started = time.monotonic()
                res = self.resilience.wrap_completion(
                    model_name,
                    _call
                )
                latency_ms = int((time.monotonic() - started) * 1000)

                # Cache the response
                try:
                    self.cache.put(prompt, model_name, cache_params, res.model_dump())
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to cache response: {str(e)}")
                
                # Record trace
                self.quality_manager.record_trace(
                    prompt,
                    model_name,
                    res.model_dump(),
                    {"is_canary": is_canary and model_name == model_to_use},
                    path=path,
                    request_model=request_name,
                    endpoint=model_name,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    area=area,
                )

                return self._attach_path(
                    res, path, model_name, sibling, sibling_reference
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {str(e)}"
                )
                last_err = e
                continue

        if last_err:
            raise last_err

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        tier = None
        request_name = kwargs.get("model")
        if request_name is not None:
            tier, router, threshold = self._parse_model_name(request_name)
        else:
            request_name = router

        # A vision request's content is a list; routers want a string.
        prompt = requirements_module._prompt_text(kwargs["messages"])
        reqs = requirements_module.derive(kwargs["messages"], kwargs, request_name)
        session_id = kwargs.get("user") or str(uuid.uuid4())

        # 1. Route: traffic rules and middleware bypass the tree, a tier
        #    walks it, and a bare pair routes flat. Each side is checked
        #    for fitness before its router runs.
        routed_model, path, effective_pair = self._route(
            prompt, kwargs, tier, router, threshold, reqs
        )
        area = self._area_of(path)

        # 2. Apply canary testing and build the fallback chain, which
        #    stays inside the pair this request was routed against.
        model_to_use, models_to_try, sibling, sibling_reference = self._models_to_try(
            routed_model, path, effective_pair, reqs
        )
        is_canary = model_to_use != routed_model

        self.model_counts[request_name][model_to_use] += 1

        # Try cache
        cache_params = {k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
        cached_res = await self.cache.aget(prompt, model_to_use, cache_params)
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
            )
            return self._attach_path(
                res, path, model_to_use, sibling, sibling_reference
            )

        last_err = None
        for model_name in models_to_try:
            # 3. Resolve the endpoint, then apply load balancing
            model, curr_api_base, curr_api_key, extra = self._endpoint_call_params(
                model_name
            )

            async def _call(extra_headers={}):
                kwargs_copy = {**extra, **kwargs}
                kwargs_copy["model"] = model
                return await acompletion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    extra_headers=extra_headers,
                    **kwargs_copy,
                )

            try:
                started = time.monotonic()
                res = await self.resilience.wrap_acompletion(
                    model_name,
                    lambda: self._request_with_payment(_call, endpoint=model_name)
                )
                latency_ms = int((time.monotonic() - started) * 1000)
                
                # 4. Validate canary if needed
                if is_canary and model_name == model_to_use:
                    passed = await self.quality_manager.validate_canary(
                        res.choices[0].message.content, 
                        prompt
                    )
                    if not passed:
                        import logging
                        logging.getLogger(__name__).warning(f"Canary model {model_name} failed validation, trying fallback.")
                        raise Exception(f"Canary validation failed for {model_name}")

                # Cache the response
                try:
                    await self.cache.aput(prompt, model_name, cache_params, res.model_dump())
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to cache response: {str(e)}")
                
                # Record trace
                self.quality_manager.record_trace(
                    prompt,
                    model_name,
                    res.model_dump(),
                    {"is_canary": is_canary and model_name == model_to_use},
                    path=path,
                    request_model=request_name,
                    endpoint=model_name,
                    session_id=session_id,
                    latency_ms=latency_ms,
                    area=area,
                )

                return self._attach_path(
                    res, path, model_name, sibling, sibling_reference
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {str(e)}"
                )
                last_err = e
                continue
        
        if last_err:
            raise last_err
