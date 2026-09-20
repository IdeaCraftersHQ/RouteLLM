"""Model-name parsing and the recursive walk over tiers.

Kept out of `controller.py` so that module stays under its size budget:
these are plain functions over an `EndpointRegistry` plus a callable
that runs one router, which also makes them testable without a
controller.

Four model-name forms address a tier or a flat pair::

    premium:router-jev-0.7   tier, with a request-level router/threshold
    router-jev-0.7           legacy flat form, or the `default` tier
    router-premium           tier, router-prefixed
    premium                  tier, bare

Walking a tier runs one router per level on the original prompt and
returns the endpoint it lands on together with a path, one entry per
level, recording where that level's router and threshold came from.
"""

import logging
from typing import Any, Callable, Optional

from routellm.endpoints import EndpointRegistry
from routellm.types import ModelPair

logger = logging.getLogger(__name__)

# Where a level's router or threshold came from, most specific first.
SOURCE_TIER = "tier"
SOURCE_PARENT = "parent"
SOURCE_REQUEST = "request"
SOURCE_DEFAULT = "default"


def parse_model_name(
    model_name: str,
    registry: EndpointRegistry,
    error_cls: type,
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    """Split a model name into a tier, a router, and a threshold.

    Parse order: a ':' qualifies a tier with a request-level
    `router-<r>-<thr>`; otherwise a `router-` prefix is the legacy flat
    form when its last '-' segment parses as a threshold, and a tier
    name otherwise; anything else is a bare tier name. The legacy form
    addresses the `default` tier when one is configured, and the flat
    pair when none is.

    Parameters
    ----------
    model_name : str
        The request's model string.
    registry : EndpointRegistry
        Registry whose tiers the name is checked against.
    error_cls : type
        Exception class raised for a malformed or unknown name.

    Returns
    -------
    tuple[str or None, str or None, float or None]
        The `(tier, router, threshold)` the name asks for. Each is None
        when the name does not carry it.

    Raises
    ------
    error_cls
        If the name is malformed, or names a tier that is not
        configured.
    """
    if not model_name or not model_name.strip():
        raise error_cls("Invalid model name: model name must be a non-empty string")

    if ":" in model_name:
        tier, _, remainder = model_name.partition(":")
        router, threshold = _parse_router_form(remainder, model_name, error_cls)
        return _known_tier(tier, registry, error_cls), router, threshold

    if model_name.startswith("router-"):
        remainder = model_name[len("router-"):]
        threshold = _as_threshold(remainder.rsplit("-", 1)[-1])
        if threshold is not None:
            router, threshold = _parse_router_form(model_name, model_name, error_cls)
            tier = "default" if registry.has_tier("default") else None
            return tier, router, threshold
        return _known_tier(remainder, registry, error_cls), None, None

    return _known_tier(model_name, registry, error_cls), None, None


def _parse_router_form(
    text: str, model_name: str, error_cls: type
) -> tuple[str, float]:
    """Parse the `router-<name>-<threshold>` form out of `text`."""
    parts = text.split("-")
    if len(parts) != 3 or parts[0] != "router":
        raise error_cls(
            f"Invalid model name: {model_name}. The router part must be in "
            "the format 'router-[router name]-[threshold]'"
        )

    threshold = _as_threshold(parts[2])
    if threshold is None:
        raise error_cls(
            f"Invalid threshold: {parts[2]}. Threshold must be a float in [0, 1]."
        )

    return parts[1], threshold


def _as_threshold(text: str) -> Optional[float]:
    """Return `text` as a threshold in [0, 1], or None when it is not one."""
    try:
        value = float(text)
    except ValueError:
        return None

    return value if 0 <= value <= 1 else None


def _known_tier(name: str, registry: EndpointRegistry, error_cls: type) -> str:
    """Return `name` when it is a configured tier, else raise."""
    if registry.has_tier(name):
        return name

    known = ", ".join(registry.tier_names()) or "<none>"
    raise error_cls(f"Unknown tier: {name}. Configured tiers: {known}")


def resolve_level(
    tier_router: Optional[str],
    tier_threshold: Optional[float],
    inherited: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one level's router and threshold, first hit winning.

    Order: the tier's own value, the parent level's resolved value, the
    request-level value from the model name, the controller defaults.

    Parameters
    ----------
    tier_router : str, optional
        Router named by this tier, if any.
    tier_threshold : float, optional
        Threshold named by this tier, if any.
    inherited : dict
        Carries `parent_router` / `parent_threshold` (None at the root),
        `request_router` / `request_threshold`, and `default_router` /
        `default_threshold`.

    Returns
    -------
    dict
        `{"router", "router_from", "threshold", "threshold_from"}`.
    """
    router, router_from = _first_hit(
        tier_router,
        inherited.get("parent_router"),
        inherited.get("request_router"),
        inherited.get("default_router"),
    )
    threshold, threshold_from = _first_hit(
        tier_threshold,
        inherited.get("parent_threshold"),
        inherited.get("request_threshold"),
        inherited.get("default_threshold"),
    )

    return {
        "router": router,
        "router_from": router_from,
        "threshold": threshold,
        "threshold_from": threshold_from,
    }


def _first_hit(own, parent, request, default) -> tuple[Any, str]:
    """Return the first non-None candidate and the source it came from."""
    for value, source in (
        (own, SOURCE_TIER),
        (parent, SOURCE_PARENT),
        (request, SOURCE_REQUEST),
        (default, SOURCE_DEFAULT),
    ):
        if value is not None:
            return value, source

    return None, SOURCE_DEFAULT


def forced_side(
    label: str,
    strong: str,
    weak: str,
    requirements: Optional[Any],
    check: Optional[Callable[[str, Any], Optional[str]]],
    error_cls: type,
) -> Optional[tuple[str, str, str]]:
    """Return the side capability checking forces, or None for "unchanged".

    None means the router decides as it always has: either no
    requirements were derived, or both sides can serve the request.

    Parameters
    ----------
    label : str
        Tier name, or a description of the flat pair, for the error.
    strong, weak : str
        The two sides.
    requirements : Requirements, optional
        What the request needs.
    check : callable, optional
        `(side, requirements) -> failed requirement name or None`.
    error_cls : type
        Exception raised when neither side can serve the request.

    Returns
    -------
    tuple[str, str, str] or None
        `(picked, "strong" | "weak", requirement the OTHER side failed)`,
        or None when the level is unchanged.

    Raises
    ------
    error_cls
        If neither side can serve the request.
    """
    if requirements is None or check is None or requirements.is_empty():
        return None

    strong_failed = check(strong, requirements)
    weak_failed = check(weak, requirements)

    if strong_failed is None and weak_failed is None:
        return None

    if strong_failed is None:
        return strong, "strong", weak_failed
    if weak_failed is None:
        return weak, "weak", strong_failed

    raise error_cls(
        f"Tier {label!r} has no side that can serve this request: "
        f"{strong_failed}. Configured sides: {strong} ({strong_failed}), "
        f"{weak} ({weak_failed})."
    )


def resolve_tier(
    tier_name: str,
    prompt: str,
    inherited: dict[str, Any],
    registry: EndpointRegistry,
    run_router: Callable[[str, float, str], float],
    *,
    requirements: Optional[Any] = None,
    check: Optional[Callable[[str, Any], Optional[str]]] = None,
    error_cls: type = ValueError,
) -> tuple[str, list[dict[str, Any]]]:
    """Walk the tier tree from `tier_name` down to one endpoint.

    Each level resolves its own router and threshold, runs the router on
    the ORIGINAL prompt, and takes the strong side when the win rate
    clears the threshold. A side that is itself a tier is recursed into
    with this level's values as the inherited ones.

    With `requirements` and `check`, each level's two sides are checked
    for fitness BEFORE its router runs, so a request that needs vision
    never pays for a classifier call that was going to pick a blind
    model. Three outcomes: both sides pass and the level is unchanged;
    exactly one passes and it is taken with no router call, the entry
    recording `capability_forced` and `capability_requirement`; neither
    passes and `error_cls` is raised naming the tier and the failures.

    Parameters
    ----------
    tier_name : str
        Tier to enter.
    prompt : str
        The request's prompt, passed unchanged to every level.
    inherited : dict
        Request-level and controller-default values, as
        `resolve_level` reads them.
    registry : EndpointRegistry
        Registry holding the tiers and endpoints.
    run_router : callable
        `(router, threshold, prompt, pair) -> (picked, win_rate)`, where
        `win_rate` is None for a router that reports no score. It is
        also where the router name is validated.
    requirements : Requirements, optional
        What the request needs. None, or an empty one, leaves the
        existing code path untouched, byte for byte.
    check : callable, optional
        `(side, requirements) -> failed requirement name or None`.
    error_cls : type
        Exception raised when no side can serve the request. Passed in
        so this module keeps importing nothing from `controller.py`.

    Returns
    -------
    tuple[str, list[dict]]
        The endpoint name landed on, and one path entry per level.

    Raises
    ------
    error_cls
        If neither side of some level can serve the request.
    """
    tier = registry.get_tier(tier_name)
    level = resolve_level(tier.router, tier.threshold, inherited)

    forced = forced_side(
        tier_name,
        str(tier.strong),
        str(tier.weak),
        requirements,
        check,
        error_cls,
    )

    if forced is None:
        picked, win_rate = run_router(
            level["router"],
            level["threshold"],
            prompt,
            ModelPair(strong=tier.strong, weak=tier.weak),
        )
    else:
        picked, win_rate = forced[0], None

    entry = {
        "tier": tier_name,
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

    if not registry.has_tier(picked):
        return picked, [entry]

    child_inherited = {
        **inherited,
        "parent_router": level["router"],
        "parent_threshold": level["threshold"],
    }
    leaf, child_path = resolve_tier(
        picked,
        prompt,
        child_inherited,
        registry,
        run_router,
        requirements=requirements,
        check=check,
        error_cls=error_cls,
    )

    return leaf, [entry] + child_path


def sibling_of(
    path: list[dict[str, Any]],
    registry: EndpointRegistry,
    pair: Optional[ModelPair] = None,
) -> Optional[tuple[str, str]]:
    """Return the fallback endpoint on the other side of the final pick.

    The sibling is the side not taken at the level of the final pick. A
    tier-valued sibling is descended by its `weak` side WITHOUT running
    any router, so the failure path stays deterministic.

    The final level may be a flat one, either because a traffic rule or
    middleware bypassed the tree or because the controller has no tiers.
    Its two sides are not on the path, so they arrive as `pair`, and the
    fallback stays inside the pair the request was actually routed
    against rather than the controller's default one.

    Parameters
    ----------
    path : list[dict]
        The decision path, as `resolve_tier` returns it.
    registry : EndpointRegistry
        Registry holding the tiers and endpoints.
    pair : ModelPair, optional
        The two sides of the final level when it is a flat one.

    Returns
    -------
    tuple[str, str] or None
        The `(endpoint, reference)` pair, where `reference` is the name
        written on the tier, a tier name when the sibling was descended.
        None when there is no other side to fall back to.
    """
    last = path[-1] if path else None
    if last is None:
        return None

    if last.get("tier") is None:
        if pair is None:
            return None
        sibling = pair.weak if last["picked"] == pair.strong else pair.strong
        return sibling, sibling

    tier = registry.get_tier(last["tier"])
    reference = tier.weak if last["picked"] == tier.strong else tier.strong

    # Validation caps the depth, so this descent always terminates.
    sibling = reference
    while registry.has_tier(sibling):
        sibling = registry.get_tier(sibling).weak

    return sibling, reference
