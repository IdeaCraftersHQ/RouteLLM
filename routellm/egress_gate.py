"""Egress gates: a hard check in front of a tagged group of endpoints.

A gate is keyed on an endpoint tag expression, so it names no provider:
the deployment says which tag marks the endpoints it wants held back.

```yaml
egress_gates:
  - name: gated
    select: "tag:gated"             # tag terms only: tag:x  !tag:x  tag:a|b
    deny_classes: [private]         # classes that never reach a selected endpoint
    require_policy_allow: false     # see below
    models:
      allow:                        # glob on the endpoint's model id
        - "vendor/*"
        - model: "vendor/free-*"    # an entry may also demand request params
          require: {extra_body.provider.data_collection: deny}
      deny: ["*:legacy"]            # deny wins over allow
    also_check: [extra_body.models] # dotted paths in `extra` holding more model ids
```

Two moments, one rule set:

* at load, every endpoint the selector matches passes `models`: its own
  model id and every id found under `also_check` (ids a provider tries
  on its own side, invisible at call time) must match `allow` and no
  `deny` entry, and every `require` of each matching `allow` entry must
  equal the value at that path in the endpoint's `extra`. An omitted
  `models.allow` leaves ids unlimited; a present, empty one allows none.
  A violation stops startup naming gate, endpoint and id.
* at request time `denial` answers for one candidate and joins the class
  check at `Controller._policy_reason`, so a tier side, forced side,
  sibling, pool member, balancer target, shadow and comparison target
  all pass it. A refused candidate is never called and never replaced by
  another class; `gate:<name>:<why>` is the reason, and with none left
  the request is refused 403.

A class not yet decided is the most restrictive: a request with no class
is refused by a gate that lists `deny_classes`. `require_policy_allow`
needs the request's routing policy (`policy_routing:`): a leaf it selects
is admitted only when that policy lists one of its tags in `allow_tags`
(and none in `deny_tags`). Without policy routing it refuses every
request to its endpoints.

Nothing here reads a request body.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from routellm.class_tiers import reachable_leaves
from routellm.governance import TagTerm, check_tag_term, parse_tag_term

__all__ = ["UNDECIDED", "EgressGateError", "EgressGates", "GateSpec", "build_egress_gates"]

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")

#: The class reason for a request whose class is not known.
UNDECIDED = "undecided"


class EgressGateError(ValueError):
    """An `egress_gates:` section is malformed, or an endpoint violates a gate."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AllowEntry(_Strict):
    """One `models.allow` entry: a glob, and request params its endpoints must carry."""

    model: str = Field(min_length=1)
    require: dict[str, Any] = Field(default_factory=dict)


class Models(_Strict):
    """Allow and deny lists on model ids; `allow: None` leaves ids unlimited."""

    allow: list[AllowEntry] | None = None
    deny: list[str] = Field(default_factory=list)

    @field_validator("allow", mode="before")
    @classmethod
    def _entries(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("must be a list")
        return [{"model": v} if isinstance(v, str) else v for v in value]


class GateSpec(_Strict):
    """One gate as written in the config."""

    name: str
    select: str
    deny_classes: list[str] = Field(default_factory=list)
    require_policy_allow: bool = False
    models: Models = Field(default_factory=Models)
    also_check: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _LABEL.match(value):
            raise ValueError(f"must match {_LABEL.pattern}")
        return value

    @field_validator("also_check")
    @classmethod
    def _paths(cls, value: list[str]) -> list[str]:
        for path in value:
            if not path or any(not part for part in path.split(".")):
                raise ValueError(f"{path!r} is not a dotted path")
        return value

    @property
    def request_rules(self) -> bool:
        """Whether the gate judges anything at request time."""
        return bool(self.deny_classes) or self.require_policy_allow


@dataclass
class _Gate:
    spec: GateSpec
    terms: tuple[TagTerm, ...] = ()
    matched: frozenset[str] = frozenset()


def _terms(spec: GateSpec, known: frozenset[str]) -> tuple[TagTerm, ...]:
    out = []
    for term in spec.select.split():
        if not (term.startswith("tag:") or term.startswith("!tag:")):
            raise ValueError(f"select: {term!r} is not a tag term (tag:x, !tag:x, tag:a|b)")
        parsed = parse_tag_term(term, spec.select)
        check_tag_term(parsed, known, spec.select)
        out.append(parsed)
    if not out:
        raise ValueError("select: write at least one tag term")
    return tuple(out)


def _lookup(extra: dict[str, Any], path: str) -> Any:
    node: Any = extra
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _ids_at(extra: dict[str, Any], path: str) -> list[str]:
    value = _lookup(extra, path)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"{path}: expected a model id or a list of ids, got {value!r}")


class EgressGates:
    """The loaded gates; `check` binds them to a registry, `denial` answers per request."""

    def __init__(self, specs: Iterable[GateSpec]):
        self._gates = [_Gate(spec) for spec in specs]
        self._leaves: dict[str, tuple[str, ...]] = {}
        self.checked = False

    @property
    def names(self) -> list[str]:
        return [g.spec.name for g in self._gates]

    @property
    def needs_class(self) -> bool:
        """Whether any gate judges the request's class or policy."""
        return any(g.spec.request_rules for g in self._gates)

    def check(self, registry, policy=None, *, decision_endpoint: str | None = None) -> None:
        """Validate every gate against the registry and bind each to its endpoints.

        Parameters
        ----------
        registry : EndpointRegistry
            The registry, selectors already resolved to endpoint names.
        policy : Policy, optional
            The loaded policy; `deny_classes` and `require_policy_allow`
            need one, and every class named must be in it.
        decision_endpoint : str, optional
            The decision step's endpoint. It reads a request before any
            class exists, so a gate that judges classes may not hold it.

        Raises
        ------
        EgressGateError
            Listing every problem as gate, endpoint and model id.
        """
        problems: list[str] = []
        seen: set[str] = set()
        for gate in self._gates:
            spec = gate.spec
            where = f"egress gate {spec.name!r}"
            if spec.name in seen:
                problems.append(f"{where}: name used twice")
            seen.add(spec.name)
            try:
                gate.terms = _terms(spec, registry.known_namespaces)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
                continue
            if spec.request_rules and policy is None:
                problems.append(
                    f"{where}: deny_classes and require_policy_allow need a `policy:`; "
                    "a request has no class without one"
                )
            elif policy is not None:
                unknown = [c for c in spec.deny_classes if c not in policy.classes]
                if unknown:
                    problems.append(
                        f"{where}: deny_classes {', '.join(map(repr, unknown))} not in policy "
                        f"classes (known: {', '.join(policy.classes)})"
                    )
            gate.matched = frozenset(
                name
                for name in registry.names()
                if all(term.matches(registry.get(name).tags) for term in gate.terms)
            )
            if not gate.matched:
                problems.append(f"{where}: select {spec.select!r} matches no endpoint")
            if spec.request_rules and decision_endpoint in gate.matched:
                problems.append(
                    f"{where}: decision endpoint {decision_endpoint!r} is gated, but the "
                    "decision step reads the request before its class is known"
                )
            for name in sorted(gate.matched):
                problems.extend(self._check_endpoint(gate, name, registry.get(name)))

        if problems:
            raise EgressGateError("; ".join(problems))
        self.checked = True

    @staticmethod
    def _check_endpoint(gate: _Gate, name: str, endpoint) -> list[str]:
        spec = gate.spec
        where = f"egress gate {spec.name!r}: endpoint {name!r}"
        found: list[tuple[str, str]] = [("model", endpoint.model)]
        out: list[str] = []
        for path in spec.also_check:
            try:
                found += [(path, i) for i in _ids_at(endpoint.extra, path)]
            except ValueError as exc:
                out.append(f"{where}: also_check {exc}")
        allow = spec.models.allow
        for source, model_id in found:
            at = f"{where} model id {model_id!r} (from {source})"
            denied = next((p for p in spec.models.deny if fnmatchcase(model_id, p)), None)
            if denied is not None:
                out.append(f"{at} is on models.deny ({denied!r})")
                continue
            if allow is None:
                continue
            entries = [e for e in allow if fnmatchcase(model_id, e.model)]
            if not entries:
                listed = ", ".join(repr(e.model) for e in allow) or "<empty: nothing is allowed>"
                out.append(f"{at} is not on models.allow ({listed})")
            for entry in entries:
                for path, want in entry.require.items():
                    have = _lookup(endpoint.extra, path)
                    if have != want:
                        out.append(
                            f"{at} matches allow entry {entry.model!r}, which requires "
                            f"{path} = {want!r}; the endpoint has {have!r}"
                        )
        return out

    def _leaves_of(self, registry, name: str) -> tuple[str, ...]:
        if not registry.has_tier(name):
            return (name,)
        if name not in self._leaves:
            self._leaves[name] = tuple(reachable_leaves(registry, name))
        return self._leaves[name]

    def denial(self, registry, name: str, data_class: str | None, route_policy=None) -> str | None:
        """Return `gate:<name>:<why>` when `name` may not receive this request.

        Parameters
        ----------
        registry : EndpointRegistry
            Judges a tier by every leaf it can reach.
        name : str
            An endpoint or tier name.
        data_class : str, optional
            The request's effective class. None is a class not yet
            decided and counts as the most restrictive.
        route_policy : RoutePolicy, optional
            The request's selected routing policy. A gate with
            `require_policy_allow` admits a leaf only when the policy's
            `allow_tags` lists one of its tags and `deny_tags` none; with
            no policy (policy routing off, or a caller with no request) it
            refuses every leaf it selects.

        Returns
        -------
        str or None
            The first reason; None when no gate refuses any leaf.
        """
        for leaf in self._leaves_of(registry, name):
            for gate in self._gates:
                spec = gate.spec
                if leaf not in gate.matched or not spec.request_rules:
                    continue
                if spec.require_policy_allow and (
                    route_policy is None or not route_policy.admits(registry.get(leaf).tags)
                ):
                    return f"gate:{spec.name}:policy"
                if spec.deny_classes and data_class is None:
                    return f"gate:{spec.name}:class:{UNDECIDED}"
                if data_class in spec.deny_classes:
                    return f"gate:{spec.name}:class:{data_class}"
        return None

    def passed(self, name: str) -> list[str]:
        """Return the gates that judged endpoint `name` (and so passed it)."""
        return [g.spec.name for g in self._gates if name in g.matched and g.spec.request_rules]


def build_egress_gates(file_config: dict | None, registry=None) -> EgressGates | None:
    """Return the gates the `egress_gates:` key names, else None.

    Only parses; `EgressGates.check` validates against the registry and
    policy, and the controller runs it at construction.

    Raises
    ------
    EgressGateError
        If the key is not a list of mappings or a gate is malformed;
        the message names the gate and the field.
    """
    raw = (file_config or {}).get("egress_gates")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise EgressGateError("`egress_gates:` must be a non-empty list of gates")
    specs = []
    for i, item in enumerate(raw):
        label = item.get("name", i) if isinstance(item, dict) else i
        if not isinstance(item, dict):
            raise EgressGateError(f"egress gate {label!r}: must be a mapping")
        try:
            specs.append(GateSpec.model_validate(item))
        except ValidationError as exc:
            problems = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                problems.append(f"{loc}: {err['msg'].removeprefix('Value error, ')}")
            raise EgressGateError(f"egress gate {label!r}: " + "; ".join(problems)) from exc
    return EgressGates(specs)
