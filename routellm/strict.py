"""Strict registry mode: no raw model name reaches litellm.

With `strict_registry` on, every model the controller can call must be
a registry endpoint. `EndpointRegistry.resolve` refuses anything else
at call time; this module checks, once at controller construction,
every place a name enters from outside the registry: the flat pair,
traffic rules, the canary target, load balancer keys and targets, and
the pairs and detection model a middleware carries. A middleware can
also return a pair per request, so `check_pair` runs on that too.
"""

from collections.abc import Iterable
from typing import Any


def check_name(registry, name: str | None, where: str) -> None:
    """Raise unless `name` is a registry endpoint.

    Parameters
    ----------
    registry : EndpointRegistry
        The registry the name must be declared in.
    name : str or None
        The name to check.
    where : str
        What names it, for the message.

    Raises
    ------
    ValueError
        If `name` is empty, a tier, or not a configured endpoint.
    """
    known = ", ".join(registry.names()) or "<none>"
    if not name:
        raise ValueError(f"strict_registry: {where} names no registry endpoint.")
    if name in registry.names():
        return
    kind = "a tier, not an endpoint" if registry.has_tier(name) else "not a registry endpoint"
    raise ValueError(
        f"strict_registry: {where} {name!r} is {kind}; raw model names are "
        f"refused. Configured endpoints: {known}"
    )


def check_pair(registry, pair: Any, where: str) -> None:
    """Raise unless both sides of `pair` are registry endpoints."""
    # A `ModelPair` says `strong`; a traffic rule says `strong_model`.
    strong = getattr(pair, "strong", None) or getattr(pair, "strong_model", None)
    weak = getattr(pair, "weak", None) or getattr(pair, "weak_model", None)
    check_name(registry, strong, f"{where} strong model")
    check_name(registry, weak, f"{where} weak model")


def _check_middleware(registry, middleware: Iterable[Any]) -> None:
    for m in middleware:
        label = type(m).__name__
        default = getattr(m, "default_model_pair", None)
        if default is not None:
            check_pair(registry, default, f"{label} default pair")
        for mapping in getattr(m, "intent_mappings", None) or []:
            if mapping.model_pair is not None:
                check_pair(registry, mapping.model_pair, f"{label} intent {mapping.intent!r} pair")
        # A built-in classifier given a bare model string calls litellm
        # with it; one given an endpoint or a detector does not.
        raw = getattr(m, "intent_detection_model", None)
        if (
            raw is not None
            and getattr(m, "intent_endpoint", None) is None
            and getattr(m, "intent_detector", None) is None
        ):
            check_name(registry, raw, f"{label} intent detection model")


def check_strict_registry(
    registry,
    *,
    strong_model: str | None,
    weak_model: str | None,
    traffic_manager,
    quality_manager,
    middleware: Iterable[Any],
) -> None:
    """Refuse, at startup, every raw model name the controller was given.

    Raises
    ------
    ValueError
        On the first name that is not a registry endpoint, naming where
        it came from.
    """
    if strong_model is not None or weak_model is not None:
        check_name(registry, strong_model, "strong_model")
        check_name(registry, weak_model, "weak_model")

    for i, rule in enumerate(traffic_manager.rules):
        check_pair(registry, rule, f"traffic rule {i}")

    for key, balancer in traffic_manager.load_balancers.items():
        check_name(registry, key, "load balancer key")
        for target in balancer.config.endpoints:
            check_name(registry, target.model, f"load balancer {key!r} target")

    canary = quality_manager.canary_config
    if canary.canary_model:
        check_name(registry, canary.canary_model, "canary_model")

    _check_middleware(registry, middleware)
