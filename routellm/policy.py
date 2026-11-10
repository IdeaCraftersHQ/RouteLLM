"""Policy file: data classes mapped to reach, jurisdiction and treatment.

A policy names the data classes a request may declare and, for each,
where it may travel and what happens to it. It denies by default: a
class the file does not list is refused, a request without a class is
refused unless `default_class` names one, and a principal reaches only
the classes listed for it.

```yaml
default_class: null            # or a class name; null refuses a request with none
classes:
  public:
    treatment: raw             # raw | redact | block | vault
    reach: [local, vpn, cloud] # reach labels an endpoint may carry
    jur: []                    # allowed jurisdictions; empty = any
    tiers: [default]           # entry tiers a request of this class may use
    cache: exact               # none | exact | both
  secret:
    treatment: block
    reach: [local]
    jur: [ca]
    tiers: [private]
  sealed:
    treatment: vault           # every request refused, no model call
principals:
  analyst: {classes: [public, secret], projects: [research]}
```

Nothing here reads a request or an endpoint. `Policy` is the loaded,
validated object the request path and the startup checks consult.
Loading a policy turns on `governance` and `strict_registry`.
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from routellm.governance import CLOUD_TAG, REACH_TAGS

__all__ = [
    "ClassPolicy",
    "Policy",
    "PolicyDenied",
    "PolicyError",
    "PolicyRefusal",
    "PrincipalPolicy",
    "build_policy",
    "check_decision_classes",
    "load_policy",
    "parse_policy",
]

#: Whether a `redact` class may reach the cloud: it may, because the
#: redact transform (`routellm.redact.transform`) runs first. A policy
#: that lets a redact class off the host without a `redact:` config is
#: refused when the controller starts, not here (this file reads no config).
REDACTION_AVAILABLE = True

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_JUR = re.compile(r"^[a-z]{2}$")

Treatment = Literal["raw", "redact", "block", "vault"]
CacheMode = Literal["none", "exact", "both"]


class PolicyError(ValueError):
    """The policy file is unreadable or invalid; the message names file, class and field."""


class PolicyDenied(Exception):
    """A request is refused by the policy once it is being routed (403).

    The message names the class and the tier or leaf it was refused on.
    """

    status = 403


class PolicyRefusal(Exception):
    """A request's class is refused by the policy.

    `code` is `unknown_class`, `missing_class` or `vault`, so the caller
    maps it to a status without parsing the message.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassPolicy(_Strict):
    """One data class: where it may go and what happens to it."""

    treatment: Treatment
    reach: list[str] = Field(default_factory=list)
    jur: list[str] = Field(default_factory=list)
    tiers: list[str] = Field(default_factory=list)
    cache: CacheMode | None = None

    @model_validator(mode="after")
    def _rules(self) -> "ClassPolicy":
        for value in self.reach:
            if value not in REACH_TAGS:
                raise ValueError(
                    f"reach: {value!r} is not a reach label (known: {', '.join(sorted(REACH_TAGS))})"
                )
        for value in self.jur:
            if not _JUR.match(value):
                raise ValueError(
                    f"jur: {value!r} must be a lowercase ISO 3166-1 alpha-2 country code"
                )
        for tier in self.tiers:
            if not tier:
                raise ValueError("tiers: empty tier name")
        if self.cache is None:
            self.cache = "exact" if self.treatment == "raw" else "none"

        if self.treatment == "vault":
            for field in ("reach", "jur", "tiers"):
                if getattr(self, field):
                    raise ValueError(
                        f"{field}: a vault class refuses every request and allows nothing; "
                        f"remove {field}"
                    )
            if self.cache != "none":
                raise ValueError("cache: a vault class makes no model call; use none")
            return self

        if not self.reach:
            raise ValueError("reach: list at least one reach label; an empty list allows nothing")
        if not self.tiers:
            raise ValueError("tiers: list at least one entry tier; an empty list allows nothing")
        if self.treatment == "block" and CLOUD_TAG in self.reach:
            raise ValueError("reach: a block class never reaches the cloud; remove 'cloud'")
        if self.treatment == "redact" and CLOUD_TAG in self.reach and not REDACTION_AVAILABLE:
            raise ValueError(
                "reach: 'cloud' on a redact class needs the redaction treatment, which is "
                "not available; remove 'cloud'"
            )
        return self

    @property
    def refuses_all(self) -> bool:
        """True when every request of this class is refused with no model call."""
        return self.treatment == "vault"


class PrincipalPolicy(_Strict):
    """The classes one principal may send, and the routing policies it may select.

    `projects` names `policy_routing` policies a request header may ask
    for; empty means the default policy only.
    """

    classes: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


class Policy(_Strict):
    """The loaded policy; `source` is the file it came from, for messages."""

    classes: dict[str, ClassPolicy]
    default_class: str | None = None
    principals: dict[str, PrincipalPolicy] = Field(default_factory=dict)
    source: str = "<policy>"

    @model_validator(mode="after")
    def _consistent(self) -> "Policy":
        if not self.classes:
            raise ValueError("classes: list at least one class")
        for name in self.classes:
            if not _LABEL.match(str(name)):
                raise ValueError(f"classes.{name}: name must match {_LABEL.pattern}")
        if self.default_class is not None and self.default_class not in self.classes:
            raise ValueError(
                f"default_class: {self.default_class!r} is not a class (known: "
                f"{', '.join(self.classes)}); use null to refuse a request with none"
            )
        for principal, spec in self.principals.items():
            if not principal:
                raise ValueError("principals: empty principal name")
            for cls in spec.classes:
                if cls not in self.classes:
                    raise ValueError(
                        f"principals.{principal}.classes: {cls!r} is not a class "
                        f"(known: {', '.join(self.classes)})"
                    )
        return self

    @property
    def names(self) -> list[str]:
        """The class names, in file order."""
        return list(self.classes)

    def resolve(self, declared: str | None) -> tuple[str, ClassPolicy]:
        """Return `(name, policy)` for a request's declared class.

        Raises
        ------
        PolicyRefusal
            `missing_class` when nothing is declared and `default_class`
            is null; `unknown_class` when the class is not listed;
            `vault` when the class refuses every request.
        """
        name = declared if declared else self.default_class
        if name is None:
            raise PolicyRefusal(
                "missing_class",
                "request declares no data class and the policy has no default_class",
            )
        spec = self.classes.get(name)
        if spec is None:
            raise PolicyRefusal(
                "unknown_class",
                f"data class {name!r} is not in the policy (known: {', '.join(self.classes)})",
            )
        if spec.refuses_all:
            raise PolicyRefusal("vault", f"data class {name!r} is vaulted: requests are refused")
        return name, spec

    def principal_projects(self, principal: str | None) -> list[str]:
        """The routing policies `principal` may select; none for an unknown or absent one."""
        spec = self.principals.get(principal) if principal is not None else None
        return list(spec.projects) if spec is not None else []

    def principal_allows(self, principal: str | None, name: str) -> bool:
        """Whether `principal` may send class `name`; unknown principals may send none."""
        spec = self.principals.get(principal) if principal is not None else None
        return spec is not None and name in spec.classes


def _format(source: str, exc: ValidationError) -> str:
    problems = []
    for err in exc.errors():
        loc = [str(p) for p in err["loc"]]
        msg = err["msg"].removeprefix("Value error, ")
        if len(loc) >= 2 and loc[0] == "classes":
            if loc[2:]:
                problems.append(f"class {loc[1]!r} {'.'.join(loc[2:])}: {msg}")
            else:
                problems.append(f"class {loc[1]!r} {msg}")
        elif loc:
            problems.append(f"{'.'.join(loc)}: {msg}")
        else:
            problems.append(msg)
    return f"policy {source}: " + "; ".join(problems)


def parse_policy(raw: Any, source: str = "<policy>") -> Policy:
    """Validate a parsed policy mapping.

    Raises
    ------
    PolicyError
        On any violation; the message names `source`, the class and the field.
    """
    if not isinstance(raw, dict):
        raise PolicyError(f"policy {source}: must be a mapping, got {type(raw).__name__}")
    if "source" in raw:
        raise PolicyError(f"policy {source}: source: unknown key")
    try:
        return Policy.model_validate({**raw, "source": source})
    except ValidationError as exc:
        raise PolicyError(_format(source, exc)) from exc


def load_policy(path: str | Path) -> Policy:
    """Read and validate the policy file at `path`.

    Raises
    ------
    PolicyError
        If the file is missing, not YAML, or invalid.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise PolicyError(f"policy {path}: cannot read: {exc.strerror or exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(f"policy {path}: not valid YAML: {exc}") from exc
    return parse_policy(raw, str(path))


def build_policy(file_config: dict | None, resolve_path=None) -> Policy | None:
    """Return the policy the `policy:` key names, else None.

    A policy implies `governance: true` and `strict_registry: true`;
    both are set on `file_config` here so the registry is built under
    them whatever the config said.

    Parameters
    ----------
    file_config : dict, optional
        The merged config; mutated to switch the two modes on.
    resolve_path : callable, optional
        `(key, value) -> Path`, resolving a relative path against the
        file that set it (`LoadedConfig.resolve_path`).
    """
    if file_config is None or file_config.get("policy") is None:
        return None
    value = file_config["policy"]
    if not isinstance(value, str) or not value:
        raise PolicyError("`policy:` must be a path to a policy file")
    path = resolve_path("policy", value) if resolve_path else Path(value)
    policy = load_policy(path)
    file_config["governance"] = True
    file_config["strict_registry"] = True
    return policy


def check_decision_classes(policy: Policy, decision_classes) -> None:
    """Refuse a classifier that can emit a class the policy does not list.

    The policy is the source of truth for class names. A label the
    classifier could return that the policy lacks would be refused as an
    unknown class on every request it labels, so it fails at startup.
    The policy may list more classes than the classifier emits: callers
    can declare their own.

    Raises
    ------
    PolicyError
        Naming each classifier class missing from the policy.
    """
    missing = [c for c in decision_classes if c not in policy.classes]
    if missing:
        raise PolicyError(
            f"policy {policy.source}: decision.data_classes {', '.join(map(repr, missing))} "
            f"not in policy classes (known: {', '.join(policy.classes)})"
        )
