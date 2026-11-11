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

At request time the same rule runs per candidate: `class_denial` answers
for one name (a leaf, or a tier, all of whose leaves must pass), and
`Controller._can_serve` asks it for every tier side, forced side,
sibling, pool member, load-balancer target, canary, traffic-rule or
middleware pair and comparison target. A denied candidate is never
called and never replaced by another class or tier; when none is left
the request is refused with `policy:<class>:<reach|jur>`.

Nothing here reads a request body; the request path passes the class
and the tier in.
"""

import re
from collections.abc import Iterable
from typing import Any

from routellm.policy import ClassPolicy, Policy, PolicyDenied, PolicyError
from routellm.routing import sibling_of

#: A refusal reason: `policy:<class>:...` or `gate:<name>:...`.
_REASON = re.compile(r"(?<!\w)(?:policy|gate):")

__all__ = [
    "check_class_tiers",
    "check_entry_tier",
    "class_denial",
    "reachable_leaves",
    "redact_aux_denial",
    "route_error",
]


def class_denial(policy: Policy, registry, name: str, data_class: str | None) -> str | None:
    """Return `policy:<class>:<reach|jur>` when `name` may not serve `data_class`.

    Parameters
    ----------
    policy : Policy
        The loaded policy.
    registry : EndpointRegistry
        The registry; a tier name is judged by every leaf it can reach.
    name : str
        An endpoint or tier name.
    data_class : str, optional
        The effective class. None, or a class the policy does not know
        or vaults, serves nothing.

    Returns
    -------
    str or None
        The reason, naming the class and what it was refused on (reach
        before jurisdiction); None when every leaf passes.
    """
    spec = policy.classes.get(data_class) if data_class is not None else None
    if spec is None or spec.refuses_all:
        return f"policy:{data_class}:reach"
    if registry.has_tier(name):
        leaves = list(reachable_leaves(registry, name))
    else:
        leaves = [name]
    for leaf in leaves:
        try:
            endpoint = registry.resolve(leaf)
        except Exception:
            return f"policy:{data_class}:reach"
        for _tag, allowed in _offending(endpoint, spec):
            return f"policy:{data_class}:{allowed.split()[0]}"
    return None


def redact_aux_denial(policy: Policy, registry, name: str, data_class: str | None) -> str | None:
    """Return `redact:<class>:aux_not_local` when a redact class's content may not go to `name`.

    A redact class reaches the cloud only through the redact transform
    on the main call. Every other call that carries its content (hosted
    and embedding routers, the semantic cache's embedding, the intent
    detector, the comparison judge, the decision classifier) is not
    transformed, so it is local-only: every leaf `name` can reach must
    carry the `local` reach label and no other. None for any other class.
    """
    spec = policy.classes.get(data_class) if data_class is not None else None
    if spec is None or spec.treatment != "redact":
        return None
    leaves = list(reachable_leaves(registry, name)) if registry.has_tier(name) else [name]
    for leaf in leaves:
        try:
            reach = registry.resolve(leaf).reach
        except Exception:
            return f"redact:{data_class}:aux_not_local"
        if not reach or not set(reach) <= {"local"}:
            return f"redact:{data_class}:aux_not_local"
    return None


def check_decision_endpoint(policy: Policy, registry, endpoint: str) -> None:
    """Refuse a decision endpoint some class may not reach.

    The decision step reads a request before any class exists, so its
    prompt goes to the endpoint whatever the request turns out to be.
    The endpoint must therefore pass every class that can result; a
    `vault` class allows nothing and is left out, as the step runs
    before a request is known to be vaulted.

    Raises
    ------
    PolicyError
        Naming the endpoint and each class that refuses it.
    """
    refused = []
    for name, spec in policy.classes.items():
        if spec.refuses_all:
            continue
        denial = class_denial(policy, registry, endpoint, name) or redact_aux_denial(
            policy, registry, endpoint, name
        )
        if denial is not None:
            refused.append(f"{name} ({denial.rsplit(':', 1)[-1]})")
    if refused:
        raise PolicyError(
            f"decision endpoint {endpoint!r} reads every request before its class is known, "
            f"but class {', '.join(refused)} may not reach it; use an endpoint every class allows "
            "(a local one)"
        )


#: Treatments whose content must stay on the host (redact) or never leave it (block).
_HOST_BOUND = ("redact", "block")


def transcriber_denial(policy: Policy, config, data_class: str | None) -> str | None:
    """Return `transcriber:<class>:not_local` when `data_class` may not be transcribed by `config`.

    The audio of a redact or block class cannot be redacted before it
    is transcribed, so only a transcriber tagged `local` may hear it.
    None for any other class, a local transcriber or one that is off.
    """
    spec = policy.classes.get(data_class) if data_class is not None else None
    if spec is None or spec.treatment not in _HOST_BOUND:
        return None
    if config is None or not config.active or config.is_local:
        return None
    return f"transcriber:{data_class}:not_local"


def check_transcriber(policy: Policy, registry) -> None:
    """Refuse a transcriber a redact or block class may not use.

    Audio is transcribed before the request's class is known (the
    transcript is what the decision step and the routers read), so
    every transcriber that can run, the global one and each tier's,
    must be local when any class redacts or blocks.

    Raises
    ------
    PolicyError
        Naming the transcriber and each class that refuses it.
    """
    classes = [n for n, c in policy.classes.items() if c.treatment in _HOST_BOUND]
    if not classes:
        return
    for scope, config in registry.transcribers().items():
        if config.active and not config.is_local:
            where = "global" if scope is None else scope
            raise PolicyError(
                f"transcriber {where!r} hears audio before its class is known, but class "
                f"{', '.join(classes)} (redact or block) may not reach a transcriber that is "
                "not tagged `local`; point it at a local service and add `tags: [local]`, "
                "or set `mode: passthrough`"
            )


def route_error(plain: type[Exception]):
    """Return a factory for the error a failed candidate check raises.

    A message carrying a `policy:` or `gate:` reason becomes
    `PolicyDenied` (403), anything else is `plain` (a capability the
    side cannot serve).
    """

    def make(message: str) -> Exception:
        if _REASON.search(message):
            return PolicyDenied(message)
        return plain(message)

    return make


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
