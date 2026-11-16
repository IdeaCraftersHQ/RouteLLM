"""Policy routing: request attributes and a routing policy pick the tier.

Replaces a threshold at the entry with a rule list. Everything here is
config: the header each attribute is read from, the attribute names, the
classes, the intent labels and the tiers.

```yaml
policy_routing:
  attributes:                    # attribute -> header it is read from
    project: {header: X-Route-Project}      # header defaults to X-Route-<Name>
    client: {pattern: '^[a-z]{1,16}$'}      # pattern defaults to a safe slug
  key: project                   # the attribute whose value names the policy
  entry: [auto]                  # tiers addressed by clients; rules pick the real tier
  rules:                         # first match wins; every matcher must hold
    - {when: {class: private}, tier: private}
    - {when: {client: codex}, tier: codex}
    - {when: {intent: [hard]}, tier: hard}
    - {when: {modality: image}, tier: vision}
  else: code                     # taken when no rule matches
  default_policy: default        # used for a missing, invalid or unknown key
  policies:
    default:
      data_class: general        # floor; nothing lowers the effective class below it
      tiers: [code, private, vision]   # tiers this policy may use
      fallback: code             # taken for a rule target outside `tiers`; else 403
      allow_tags: []             # tags a gate with `require_policy_allow` admits
      deny_tags: []              # an endpoint with one of these is never a candidate
```

Matchers: `class` (the effective class), `intent` (the decision step's
label; never matches when none ran), `modality` (`text`, `image`,
`audio`) and any configured attribute name. A list value is any-of.

A policy never widens a class: the tier it picks still passes the
class's entry-tier check and every per-leaf check, and a rule whose tier
the class does not list is a startup error. Nothing here reads a body.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from routellm.class_tiers import _offending, reachable_leaves
from routellm.policy import Policy, PolicyDenied, PolicyError

__all__ = [
    "RESERVED_MATCHERS",
    "PolicyRouter",
    "PolicyRoutingConfig",
    "PolicyRoutingError",
    "RequestAttributes",
    "Selection",
    "build_policy_routing",
    "current_attributes",
]

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_HEADER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")

#: Value shape an attribute accepts unless it sets `pattern`.
DEFAULT_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,62}$"

#: Matchers with a meaning of their own; no attribute may take one of these names.
RESERVED_MATCHERS = frozenset({"class", "intent", "modality"})
MODALITIES = ("text", "image", "audio", "video")


class PolicyRoutingError(PolicyError):
    """The `policy_routing:` section is invalid or contradicts the policy file."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttributeSpec(_Strict):
    """Where one request attribute is read from, and what a valid value looks like."""

    header: str | None = None
    pattern: str = DEFAULT_PATTERN

    @field_validator("header")
    @classmethod
    def _header_name(cls, value):
        if value is not None and not _HEADER.match(value):
            raise ValueError(f"{value!r} is not a header name")
        return value

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"not a regular expression: {exc}") from exc
        return value


class Rule(_Strict):
    """One routing rule: every matcher in `when` holds -> `tier`."""

    when: dict[str, str | list[str]]
    tier: str

    @field_validator("when")
    @classmethod
    def _matchers(cls, value):
        if not value:
            raise ValueError("when: write at least one matcher (use `else` for the rest)")
        for key, want in value.items():
            labels = [want] if isinstance(want, str) else want
            if not labels or not all(isinstance(w, str) and w for w in labels):
                raise ValueError(f"when.{key}: a label or a non-empty list of labels")
        return value

    def wants(self, key: str) -> list[str] | None:
        want = self.when.get(key)
        return None if want is None else ([want] if isinstance(want, str) else list(want))


class RoutePolicy(_Strict):
    """What one project may use: a class floor, tiers, a fallback and tag lists."""

    data_class: str | None = None
    tiers: list[str] = Field(min_length=1)
    fallback: str | None = None
    allow_tags: list[str] = Field(default_factory=list)
    deny_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fallback_allowed(self) -> RoutePolicy:
        if self.fallback is not None and self.fallback not in self.tiers:
            raise ValueError(f"fallback: {self.fallback!r} must be one of its tiers")
        return self

    def admits(self, tags) -> bool:
        """Whether a gate that requires the policy's allow admits an endpoint with `tags`."""
        tags = set(tags)
        return bool(tags & set(self.allow_tags)) and not tags & set(self.deny_tags)


class PolicyRoutingConfig(_Strict):
    """The `policy_routing:` section as written."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    attributes: dict[str, AttributeSpec] = Field(default_factory=dict)
    key: str
    entry: list[str] = Field(min_length=1)
    rules: list[Rule] = Field(default_factory=list)
    otherwise: str = Field(alias="else")
    default_policy: str = "default"
    policies: dict[str, RoutePolicy]

    @model_validator(mode="after")
    def _consistent(self) -> PolicyRoutingConfig:
        for name in self.attributes:
            if not _LABEL.match(str(name)):
                raise ValueError(f"attributes.{name}: name must match {_LABEL.pattern}")
            if name in RESERVED_MATCHERS:
                raise ValueError(
                    f"attributes.{name}: reserved matcher name "
                    f"({', '.join(sorted(RESERVED_MATCHERS))})"
                )
        if self.key not in self.attributes:
            raise ValueError(f"key: {self.key!r} is not an attribute")
        for i, rule in enumerate(self.rules):
            for matcher in rule.when:
                if matcher not in RESERVED_MATCHERS and matcher not in self.attributes:
                    raise ValueError(
                        f"rules[{i}].when: {matcher!r} is neither an attribute "
                        f"nor one of {', '.join(sorted(RESERVED_MATCHERS))}"
                    )
            bad = [m for m in rule.wants("modality") or [] if m not in MODALITIES]
            if bad:
                raise ValueError(
                    f"rules[{i}].when.modality: {', '.join(map(repr, bad))} "
                    f"(known: {', '.join(MODALITIES)})"
                )
        if self.default_policy not in self.policies:
            raise ValueError(
                f"policies: a policy named {self.default_policy!r} (default_policy) is required"
            )
        return self


@dataclass(frozen=True)
class RequestAttributes:
    """The request attributes read off its headers.

    `values` holds each valid one; `invalid` names the attributes whose
    header was present but did not match the pattern (treated as unknown).
    """

    values: Mapping[str, str] = field(default_factory=dict)
    invalid: tuple[str, ...] = ()


#: The attributes of the request being served, for code with no `Request`
#: in hand. None outside a request.
current_attributes: ContextVar[RequestAttributes | None] = ContextVar(
    "current_attributes", default=None
)


@dataclass(frozen=True)
class Selection:
    """What policy routing decided for one request, recorded on the path."""

    policy: str
    policy_from: str  # attribute | absent | invalid | unknown
    attributes: Mapping[str, str]
    classes: Mapping[str, str | None]  # source -> class it proposed
    effective: str | None
    intent: str | None
    addressed: str | None
    tier: str | None
    rule: int | str | None  # index, `else`, or None when no rule ran
    refused: Mapping[str, str] | None = None  # tier the policy refused -> fallback taken

    def as_dict(self) -> dict[str, Any]:
        """Return the record written into the routing path."""
        record: dict[str, Any] = {
            "policy": self.policy,
            "policy_from": self.policy_from,
            "attributes": dict(self.attributes),
            "classes": {k: v for k, v in self.classes.items() if v is not None},
            "effective_class": self.effective,
            "intent": self.intent,
            "addressed": self.addressed,
            "tier": self.tier,
            "rule": self.rule,
        }
        if self.refused:
            record["refused"] = dict(self.refused)
        return record


class PolicyRouter:
    """The loaded section; binds to the class policy and answers per request."""

    def __init__(self, config: PolicyRoutingConfig, class_policy: Policy | None):
        self.config = config
        self.class_policy = class_policy
        self.policies = config.policies

    # -- attributes -------------------------------------------------------

    def attributes(self, headers: Mapping[str, str]) -> RequestAttributes:
        """Read and validate each configured attribute off `headers`.

        A missing header, or one whose value fails its pattern, is an
        unknown attribute (never an error): the policy falls to the default.
        """
        lowered = {k.lower(): v for k, v in headers.items()}
        values: dict[str, str] = {}
        invalid: list[str] = []
        for name, spec in self.config.attributes.items():
            header = spec.header or f"X-Route-{name.title()}"
            raw = lowered.get(header.lower())
            if raw is None or raw == "":
                continue
            if re.fullmatch(spec.pattern, raw):
                values[name] = raw
            else:
                invalid.append(name)
        return RequestAttributes(values, tuple(invalid))

    def policy_for(self, attrs: RequestAttributes) -> tuple[str, str]:
        """Return `(policy name, why)`; why is attribute, absent, invalid or unknown."""
        key = self.config.key
        default = self.config.default_policy
        if key in attrs.invalid:
            return default, "invalid"
        value = attrs.values.get(key)
        if value is None:
            return default, "absent"
        if value not in self.policies:
            return default, "unknown"
        return value, "attribute"

    def selected(self, attrs: RequestAttributes | None, principal=None) -> tuple[str, str]:
        """Return `(policy name, why)` for a request, enforcing who may select what.

        The one place the principal's `projects` list is applied, shared
        by chat (`select`), `/v1/compare` and the class declaration.

        Raises
        ------
        PolicyDenied
            A non-default policy the caller may not use (`policy_from: forbidden`).
        """
        name, why = self.policy_for(attrs or RequestAttributes())
        if why == "attribute" and name != self.config.default_policy:
            allowed = (
                self.class_policy.principal_projects(principal.principal if principal else None)
                if self.class_policy is not None
                else []
            )
            if name not in allowed:
                raise PolicyDenied(
                    f"routing policy {name!r} is not allowed for this caller "
                    "(policy_from: forbidden)"
                )
        return name, why

    def floor(self, attrs: RequestAttributes | None, principal=None) -> str | None:
        """Return the class floor of the policy this request selects, or None.

        Raises
        ------
        PolicyDenied
            As `selected`.
        """
        name, _ = self.selected(attrs, principal)
        return self.policies[name].data_class

    # -- class ------------------------------------------------------------

    def _rank(self, name: str) -> int:
        names = self.class_policy.names if self.class_policy else []
        if name not in names:
            raise PolicyDenied(f"data class {name!r} is not in the policy: request refused")
        return names.index(name)

    def stricter(self, *classes: str | None) -> str | None:
        """Return the most restrictive of `classes`, in the policy file's order."""
        known = [c for c in classes if c is not None]
        return max(known, key=self._rank) if known else None

    # -- selection --------------------------------------------------------

    @staticmethod
    def _modalities(reqs) -> set[str]:
        found = set()
        if getattr(reqs, "vision", False):
            found.add("image")
        if getattr(reqs, "audio", False):
            found.add("audio")
        if getattr(reqs, "video", False):
            found.add("video")
        return found or {"text"}

    def _matches(self, rule: Rule, facts: dict[str, Any]) -> bool:
        for key, want in rule.when.items():
            wanted = [want] if isinstance(want, str) else want
            have = facts.get(key)
            if have is None:
                return False
            if isinstance(have, set):
                if not have & set(wanted):
                    return False
            elif have not in wanted:
                return False
        return True

    def select(
        self,
        tier: str | None,
        declared_class: str | None,
        decision,
        reqs,
        attrs: RequestAttributes | None,
        principal=None,
    ) -> Selection:
        """Pick the policy, the effective class and the tier for one request.

        Parameters
        ----------
        tier : str, optional
            The tier the model name addresses. One of `entry` is replaced
            by the first matching rule's tier; any other stands as
            addressed.
        declared_class : str, optional
            The validated declared class (the policy file's default when
            none was sent).
        decision : Decision, optional
            The decision step's result; None when none is configured.
        reqs : Requirements, optional
            Read for the request's modality.
        attrs : RequestAttributes, optional
            The request's attributes; None is no attribute at all.
        principal : Principal, optional
            The authenticated caller. A header may select a non-default
            policy only if the policy file's `principals.<name>.projects`
            lists it; with no principal only the default applies.

        Raises
        ------
        PolicyDenied
            A policy the caller may not use (`policy_from: forbidden`), a
            tier the policy does not allow with no fallback (naming
            policy and tier), or a class the policy file does not list.
        """
        attrs = attrs or RequestAttributes()
        name, why = self.selected(attrs, principal)
        route_policy = self.policies[name]
        classes = {
            "policy": route_policy.data_class,
            "declared": declared_class,
            "decision": decision.data_class if decision is not None else None,
        }
        effective = self.stricter(*classes.values())
        intent = decision.intent if decision is not None else None

        refused = None
        rule: int | str | None = None
        target = tier
        if tier is not None and tier in self.config.entry:
            facts = {
                **attrs.values,
                "class": effective,
                "intent": intent,
                "modality": self._modalities(reqs),
            }
            target, rule = self.config.otherwise, "else"
            for i, candidate in enumerate(self.config.rules):
                if self._matches(candidate, facts):
                    target, rule = candidate.tier, i
                    break
            if target not in route_policy.tiers:
                if route_policy.fallback is None:
                    raise PolicyDenied(
                        f"routing policy {name!r} does not allow tier {target!r} "
                        f"(allowed: {', '.join(route_policy.tiers)}) and has no fallback"
                    )
                refused = {target: route_policy.fallback}
                target = route_policy.fallback
        elif tier is not None and tier not in route_policy.tiers:
            # An explicit address is never moved: the client chose this tier.
            raise PolicyDenied(
                f"routing policy {name!r} does not allow tier {tier!r} "
                f"(allowed: {', '.join(route_policy.tiers)})"
            )
        return Selection(
            policy=name,
            policy_from=why,
            attributes=dict(attrs.values),
            classes=classes,
            effective=effective,
            intent=intent,
            addressed=tier,
            tier=target,
            rule=rule,
            refused=refused,
        )

    # -- candidates -------------------------------------------------------

    def tag_denial(self, registry, name: str, policy_name: str | None) -> str | None:
        """Return `policy:<name>:tag:<tag>` when a leaf under `name` carries a denied tag."""
        route_policy = self.policies.get(policy_name) if policy_name else None
        if route_policy is None or not route_policy.deny_tags:
            return None
        leaves = list(reachable_leaves(registry, name)) if registry.has_tier(name) else [name]
        for leaf in leaves:
            for tag in sorted(set(registry.get(leaf).tags) & set(route_policy.deny_tags)):
                return f"policy:{policy_name}:tag:{tag}"
        return None

    # -- startup ----------------------------------------------------------

    def check(self, registry, class_policy: Policy | None, decision_config=None) -> None:
        """Validate the section against the registry, the policy file and the classifier.

        Raises
        ------
        PolicyRoutingError
            Listing every problem: unknown tiers, classes or intents, a
            rule whose tier its class does not list, a class a policy can
            serve through no tier, a selectable tier whose leaves exceed
            the class's reach.
        """
        cfg = self.config
        problems: list[str] = []
        self.class_policy = class_policy
        if class_policy is None:
            raise PolicyRoutingError(
                "policy_routing: needs a `policy:` file; a request has no class without one"
            )
        classes = class_policy.classes

        def tier_ok(tier: str, where: str) -> bool:
            if registry.has_tier(tier):
                return True
            known = ", ".join(registry.tier_names()) or "<none>"
            problems.append(f"{where}: {tier!r} is not a tier (known: {known})")
            return False

        for tier in cfg.entry:
            tier_ok(tier, "entry")
        for i, rule in enumerate(cfg.rules):
            tier_ok(rule.tier, f"rules[{i}].tier")
            for cls in rule.wants("class") or []:
                if cls not in classes:
                    problems.append(f"rules[{i}].when.class: {cls!r} is not a policy class")
                elif rule.tier not in classes[cls].tiers:
                    problems.append(
                        f"rules[{i}]: tier {rule.tier!r} is not an entry tier of class "
                        f"{cls!r}; a policy cannot widen a class"
                    )
            if decision_config is not None:
                for label in rule.wants("intent") or []:
                    if label not in decision_config.intents:
                        problems.append(
                            f"rules[{i}].when.intent: {label!r} is not a decision intent"
                        )
        tier_ok(cfg.otherwise, "else")
        if decision_config is not None:
            ranks = [list(classes).index(c) for c in decision_config.data_classes if c in classes]
            if ranks != sorted(ranks):
                problems.append(
                    "decision.data_classes order disagrees with the policy file's `classes:` "
                    f"order ({', '.join(classes)}); both run least to most restrictive"
                )
        for pname, rp in cfg.policies.items():
            for tier in rp.tiers:
                tier_ok(tier, f"policies.{pname}.tiers")
            if rp.data_class is not None and rp.data_class not in classes:
                problems.append(
                    f"policies.{pname}.data_class: {rp.data_class!r} is not a policy class"
                )
            if (
                rp.data_class is not None
                and decision_config is not None
                and rp.data_class in getattr(decision_config, "declarable", ())
            ):
                problems.append(
                    f"policies.{pname}.data_class: {rp.data_class!r} is a decision.declarable "
                    "class; a floor would give it to a request that did not declare it"
                )
        for who, spec in class_policy.principals.items():
            for project in spec.projects:
                if project not in cfg.policies:
                    problems.append(
                        f"principals.{who}.projects: {project!r} is not a routing policy "
                        f"(known: {', '.join(cfg.policies)})"
                    )
        if not problems:
            problems += self._walk(registry, classes)
        if problems:
            raise PolicyRoutingError("policy_routing: " + "; ".join(problems))

    def _walk(self, registry, classes) -> list[str]:
        """Walk every tier a policy can select for each class it can apply to."""
        cfg = self.config
        names = list(classes)
        problems: list[str] = []
        for pname, rp in cfg.policies.items():
            floor = names.index(rp.data_class) if rp.data_class in names else 0
            for cls in names[floor:]:
                spec = classes[cls]
                if spec.refuses_all:
                    continue
                selectable = {cfg.otherwise}
                if rp.fallback:
                    selectable.add(rp.fallback)
                for rule in cfg.rules:
                    if rule.wants("class") is None or cls in rule.wants("class"):
                        selectable.add(rule.tier)
                usable = sorted(selectable & set(rp.tiers) & set(spec.tiers))
                if not usable:
                    problems.append(
                        f"policy {pname!r}, class {cls!r}: no tier the rules can select is an "
                        f"entry tier of the class (class tiers: {', '.join(spec.tiers)})"
                    )
                for tier in usable:
                    for leaf, how in reachable_leaves(registry, tier).items():
                        for tag, allowed in _offending(registry.get(leaf), spec):
                            problems.append(
                                f"policy {pname!r}, class {cls!r}, tier {tier!r}: leaf {leaf!r} "
                                f"({how}) carries {tag!r}; class allows {allowed}"
                            )
        return problems


def build_policy_routing(file_config: dict | None, class_policy: Policy | None = None):
    """Return the router the `policy_routing:` key names, else None.

    Only parses; `PolicyRouter.check` validates against the registry,
    and the controller runs it at construction.

    Raises
    ------
    PolicyRoutingError
        If the section is malformed; the message names the field.
    """
    raw = (file_config or {}).get("policy_routing")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PolicyRoutingError("`policy_routing:` must be a mapping")
    try:
        config = PolicyRoutingConfig.model_validate(raw)
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            problems.append(f"{loc}: {err['msg'].removeprefix('Value error, ')}")
        raise PolicyRoutingError("policy_routing: " + "; ".join(problems)) from exc
    return PolicyRouter(config, class_policy)
