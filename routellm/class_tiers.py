"""Class to allowed tiers: the entry check and the startup reachability walk.

A policy class names the entry tiers a request of that class may use
(`classes.<name>.tiers`). Two checks hold it:

* at request time, the tier a request addresses must be one of its
  effective class's tiers, else `PolicyDenied` (403);
* at startup, every leaf reachable from each allowed tier must sit
  within the class's reach and jurisdiction, else the server refuses to
  start, listing class, tier, leaf and the offending tag.

"Reachable" is everything a request entering the tier can be sent to,
on the way in or after a failure: both sides of every level, nested
tiers, the sibling the failure path falls back to, and every member of
a candidate pool. Names that bypass the tree for any request are held
to the same rule for every class: traffic-rule pairs, the canary model,
middleware pairs, and the targets of a load balancer keyed on a
reachable leaf.

Nothing here reads a request body; the request path passes the class
and the tier in.
"""

from collections.abc import Iterable
from typing import Any

from routellm.policy import ClassPolicy, Policy, PolicyDenied, PolicyError
from routellm.routing import sibling_of

__all__ = ["check_class_tiers", "check_entry_tier", "reachable_leaves"]


def check_entry_tier(policy: Policy, tier: str | None, data_class: str | None) -> None:
    """Refuse a request whose tier is not one of its class's entry tiers.

    Parameters
    ----------
    policy : Policy
        The loaded policy.
    tier : str, optional
        The tier the request addresses; None for a flat pair.
    data_class : str, optional
        The effective class: the declared one, or the decision step's
        when that is stricter. None (nothing known) is refused.

    Raises
    ------
    PolicyDenied
        Naming the class and the tier.
    """
    spec = policy.classes.get(data_class) if data_class is not None else None
    if spec is None:
        raise PolicyDenied(f"data class {data_class!r} is not in the policy: request refused")
    if tier is None or tier not in spec.tiers:
        shown = repr(tier) if tier is not None else "<none: flat pair>"
        raise PolicyDenied(
            f"data class {data_class!r} may not use tier {shown} "
            f"(allowed: {', '.join(spec.tiers) or '<none>'})"
        )


def reachable_leaves(registry, entry: str) -> dict[str, str]:
    """Return every endpoint a request entering tier `entry` can reach.

    Parameters
    ----------
    registry : EndpointRegistry
        The registry, selectors already resolved to endpoint names.
    entry : str
        The entry tier.

    Returns
    -------
    dict[str, str]
        Endpoint name to how it is reached, first way found.
    """
    found: dict[str, str] = {}
    _walk(registry, entry, found)
    return found


def _walk(registry, tier_name: str, found: dict[str, str]) -> None:
    for side in ("strong", "weak"):
        members = registry.pool_for(tier_name, side)
        head = members[0]
        if registry.has_tier(head):
            _walk(registry, head, found)
            continue
        for i, name in enumerate(members):
            how = f"{side} side of tier {tier_name!r}" + (" pool" if i else "")
            found.setdefault(name, how)
        _fallback(registry, tier_name, head, found)


def _fallback(registry, tier_name: str, picked: str, found: dict[str, str]) -> None:
    """Add the sibling the failure path tries after `picked`.

    Today it is always a leaf the walk reaches as a side of some level;
    asking `sibling_of` keeps the check tied to the real fallback rule
    should that ever reach further. The sibling's pool members are walked
    with its own side.
    """
    sibling = sibling_of([{"tier": tier_name, "picked": picked}], registry)
    if sibling is None:
        return
    name, _ = sibling
    found.setdefault(name, f"fallback sibling of {picked!r} in tier {tier_name!r}")


def _bypass_targets(traffic_manager, quality_manager, middleware: Iterable[Any]) -> dict[str, str]:
    """Endpoints a request can reach whatever tier it entered."""
    found: dict[str, str] = {}
    for i, rule in enumerate(traffic_manager.rules):
        for name in (rule.strong_model, rule.weak_model):
            found.setdefault(name, f"traffic rule {i}")
    canary = quality_manager.canary_config.canary_model
    if canary:
        found.setdefault(canary, "canary_model")
    for m in middleware:
        label = type(m).__name__
        pairs = [(f"{label} default pair", getattr(m, "default_model_pair", None))]
        for mapping in getattr(m, "intent_mappings", None) or []:
            pairs.append((f"{label} intent {mapping.intent!r} pair", mapping.model_pair))
        for where, pair in pairs:
            if pair is not None:
                for name in (pair.strong, pair.weak):
                    if name:
                        found.setdefault(name, where)
    return found


def _balanced(traffic_manager, found: dict[str, str]) -> dict[str, str]:
    """Targets of a load balancer keyed on a reached endpoint."""
    extra: dict[str, str] = {}
    for key in found:
        balancer = traffic_manager.load_balancers.get(key)
        if balancer is None:
            continue
        for target in balancer.config.endpoints:
            extra.setdefault(target.model, f"load balancer {key!r} target")
    return extra


def _offending(endpoint, spec: ClassPolicy) -> list[tuple[str, str]]:
    """Return `(tag, what the class allows)` for each way `endpoint` exceeds `spec`."""
    out: list[tuple[str, str]] = []
    allowed_reach = ", ".join(spec.reach)
    if not endpoint.reach:
        out.append(("<no reach tag>", f"reach {allowed_reach}"))
    out.extend((tag, f"reach {allowed_reach}") for tag in sorted(endpoint.reach - set(spec.reach)))
    if spec.jur and endpoint.jur not in spec.jur:
        tag = f"jur:{endpoint.jur}" if endpoint.jur else "<no jur tag>"
        out.append((tag, f"jur {', '.join(spec.jur)}"))
    return out


def check_class_tiers(
    policy: Policy,
    registry,
    *,
    traffic_manager,
    quality_manager,
    middleware: Iterable[Any] = (),
) -> None:
    """Refuse, at startup, a class whose tiers can reach a leaf it forbids.

    Parameters
    ----------
    policy : Policy
        The loaded policy.
    registry : EndpointRegistry
        The registry with selectors resolved (pairing already ran).
    traffic_manager, quality_manager : object
        Where traffic rules, load balancers and the canary model live.
    middleware : iterable
        Middleware that can return a pair of its own.

    Raises
    ------
    PolicyError
        Listing every violation as class, tier, leaf and offending tag;
        an allowed tier the registry does not have is listed too.
    """
    bypass = _bypass_targets(traffic_manager, quality_manager, middleware)
    bypass.update({k: v for k, v in _balanced(traffic_manager, bypass).items() if k not in bypass})
    problems: list[str] = []

    for cls, spec in policy.classes.items():
        if spec.refuses_all:
            continue
        reached: list[tuple[str, str, str]] = []
        for tier in spec.tiers:
            if not registry.has_tier(tier):
                known = ", ".join(registry.tier_names()) or "<none>"
                problems.append(f"class {cls!r}: tier {tier!r} is not a tier (known: {known})")
                continue
            leaves = reachable_leaves(registry, tier)
            leaves.update({k: v for k, v in _balanced(traffic_manager, leaves).items()})
            reached += [(f"tier {tier!r}", leaf, how) for leaf, how in leaves.items()]
        reached += [("any tier", leaf, how) for leaf, how in bypass.items()]

        for where, leaf, how in reached:
            try:
                endpoint = registry.get(leaf)
            except KeyError:
                problems.append(f"class {cls!r}, {where}: leaf {leaf!r} ({how}) is not an endpoint")
                continue
            for tag, allowed in _offending(endpoint, spec):
                problems.append(
                    f"class {cls!r}, {where}: leaf {leaf!r} ({how}) carries {tag!r}; "
                    f"class allows {allowed}"
                )

    if problems:
        raise PolicyError(f"policy {policy.source}: " + "; ".join(problems))
