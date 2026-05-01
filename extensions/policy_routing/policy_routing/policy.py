"""classify_to_policy — c12n classification → routellm CachePolicy.

Pure mapping logic; no I/O, no state. Consumers wire this into
routellm's cache hook (see US-0102) via integrations.build_cache_hook.

The c12n ClassificationResult is duck-typed: any object exposing the
following attrs (any subset; missing = no signal) is accepted:

    pii_labels       : Iterable[str]   e.g. {"EMAIL", "PHONE", "SSN"}
    jailbreak_score  : float in [0, 1]
    toxicity_score   : float in [0, 1]
    code_content     : Optional[str]   e.g. "python", "go", None
    domain           : Optional[str]   e.g. "legal", "math", "cs", "general"
    complexity       : Optional[str]   one of "easy", "medium", "hard"
    cost_estimate    : Optional[str]   one of "low", "medium", "high"
    output_format    : Optional[str]   e.g. "code", "json", "yaml", "prose"

Policy table (showcase scenario 3, locked spec):

    PII (any of EMAIL/PHONE/SSN)              skip cache; tenant-scoped namespace
    Jailbreak score >= JAILBREAK_HIGH         skip cache (audit handled upstream)
    Toxicity  score >= TOXICITY_HIGH          skip cache
    CodeContent: <lang>                       exact-match only; long TTL
    Domain: legal                             workspace-scoped; short TTL
    Domain: math|cs                           workspace-scoped; long TTL
    Complexity: easy + cost_estimate: low     skip cache (cheap to recompute)
    Complexity: hard + cost_estimate: high    semantic; long TTL
    OutputFormat: code|json|yaml              exact-match; high TTL
    (else)                                    DEFAULT policy

Precedence: safety (PII / jailbreak / toxicity) wins over all
others. Then code_content. Then domain. Then complexity+cost. Then
output_format. Else default.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants — single source of truth for thresholds + TTLs.
# ---------------------------------------------------------------------------

# Sentinel namespace value: consumers (routellm cache hook) must
# resolve this to the request's workspace_id at lookup time. See
# US-0102 acceptance criteria.
DEFAULT_NAMESPACE_KEY = "__workspace__"

# Tenant-scoped namespace sentinel: PII case requires tenant-level
# isolation rather than workspace-level (privacy spec calls for it
# explicitly because tenant covers the legal/billing boundary).
TENANT_NAMESPACE_KEY = "__tenant__"

# TTL buckets (seconds). Tuned per spec rationale columns:
TTL_SHORT = 60 * 15  # 15 min — legal: confidentiality fades fast
TTL_DEFAULT = 60 * 60 * 24  # 24 h  — matches CacheConfig default
TTL_LONG = 60 * 60 * 24 * 7  # 7 d   — code/factual/structured: stable

# Score thresholds.
JAILBREAK_HIGH = 0.7
TOXICITY_HIGH = 0.7

# PII labels that trigger no-cache. Mirrors c12n SIGNALS.md PII set.
PII_BLOCKING_LABELS = frozenset({"EMAIL", "PHONE", "SSN"})

# Output formats that require exact-match (semantic match would
# corrupt structure-sensitive results).
STRUCTURED_OUTPUT_FORMATS = frozenset({"code", "json", "yaml"})

# Code-content languages — any non-empty value triggers exact-match.
# Stored as a marker; actual language string is opaque to mapping.

# ---------------------------------------------------------------------------
# Public types — mirror routellm/types.py US-0102 spec.
# ---------------------------------------------------------------------------


class CacheAction(str, enum.Enum):
    """Legal action values per US-0102."""

    SKIP = "skip"
    LOOKUP_ONLY = "lookup_only"
    LOOKUP_AND_STORE = "lookup_and_store"
    STORE_ONLY = "store_only"


class MatchStrategy(str, enum.Enum):
    """Legal match values per US-0102."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    BOTH = "both"


@dataclass(frozen=True)
class CachePolicy:
    """Cache-hook decision returned to routellm core.

    Mirrors routellm/types.py CachePolicy (planned by US-0102).
    Frozen so it's hashable + safe to pass between threads.

    Attributes
    ----------
    action: SKIP / LOOKUP_ONLY / LOOKUP_AND_STORE / STORE_ONLY.
    ttl: store TTL in seconds; overrides CacheConfig.ttl_seconds.
    namespace: cache key prefix. DEFAULT_NAMESPACE_KEY means
        "resolve workspace_id from request"; TENANT_NAMESPACE_KEY
        means "resolve tenant_id"; any other string is used verbatim
        (e.g. "global:domain:math").
    match: exact / semantic / both.
    reason: human-readable trace label for audit log + debugging.
    metadata: free-form dict for downstream hooks (audit, routing).
    """

    action: CacheAction = CacheAction.LOOKUP_AND_STORE
    ttl: int = TTL_DEFAULT
    namespace: str = DEFAULT_NAMESPACE_KEY
    match: MatchStrategy = MatchStrategy.BOTH
    reason: str = "default"
    metadata: tuple = field(default_factory=tuple)


# Module-level default — reused everywhere the table doesn't fire.
DEFAULT_POLICY = CachePolicy()


# ---------------------------------------------------------------------------
# Mapping function.
# ---------------------------------------------------------------------------


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute getter — works for dataclasses, namedtuples,
    plain classes, dict-like, TypedDict instances."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_pii(classification: Any) -> bool:
    labels = _attr(classification, "pii_labels") or ()
    return any(label in PII_BLOCKING_LABELS for label in labels)


def classify_to_policy(request: Any, classification: Any) -> CachePolicy:
    """Map a (request, ClassificationResult) pair to a CachePolicy.

    `request` is unused by the current mapping rules but kept in the
    signature to match the US-0102 hook contract; downstream rule
    extensions may need request-local context (e.g. workspace_id
    overrides keyed off auth claims).
    """
    del request  # reserved for future rules; matches hook signature.

    # ---- safety class — never cache ------------------------------------
    if _has_pii(classification):
        return CachePolicy(
            action=CacheAction.SKIP,
            namespace=TENANT_NAMESPACE_KEY,
            ttl=0,
            match=MatchStrategy.EXACT,
            reason="pii_detected",
        )

    if (_attr(classification, "jailbreak_score") or 0.0) >= JAILBREAK_HIGH:
        return CachePolicy(
            action=CacheAction.SKIP,
            ttl=0,
            match=MatchStrategy.EXACT,
            reason="jailbreak_high",
        )

    if (_attr(classification, "toxicity_score") or 0.0) >= TOXICITY_HIGH:
        return CachePolicy(
            action=CacheAction.SKIP,
            ttl=0,
            match=MatchStrategy.EXACT,
            reason="toxicity_high",
        )

    # ---- code content — determinism ------------------------------------
    code_content = _attr(classification, "code_content")
    if code_content:
        return CachePolicy(
            action=CacheAction.LOOKUP_AND_STORE,
            ttl=TTL_LONG,
            match=MatchStrategy.EXACT,
            reason=f"code_content:{code_content}",
        )

    # ---- domain class --------------------------------------------------
    domain = _attr(classification, "domain")
    if domain == "legal":
        return CachePolicy(
            action=CacheAction.LOOKUP_AND_STORE,
            ttl=TTL_SHORT,
            namespace=DEFAULT_NAMESPACE_KEY,  # workspace-scoped per T-0193
            match=MatchStrategy.BOTH,
            reason="domain:legal",
        )
    if domain in ("math", "cs"):
        return CachePolicy(
            action=CacheAction.LOOKUP_AND_STORE,
            ttl=TTL_LONG,
            namespace=DEFAULT_NAMESPACE_KEY,  # workspace-scoped per T-0193
            match=MatchStrategy.BOTH,
            reason=f"domain:{domain}",
        )

    # ---- complexity + cost interaction ---------------------------------
    complexity = _attr(classification, "complexity")
    cost_estimate = _attr(classification, "cost_estimate")
    if complexity == "easy" and cost_estimate == "low":
        return CachePolicy(
            action=CacheAction.SKIP,
            ttl=0,
            reason="cheap_to_recompute",
        )
    if complexity == "hard" and cost_estimate == "high":
        return CachePolicy(
            action=CacheAction.LOOKUP_AND_STORE,
            ttl=TTL_LONG,
            match=MatchStrategy.SEMANTIC,
            reason="high_value_reuse",
        )

    # ---- output format -------------------------------------------------
    output_format = _attr(classification, "output_format")
    if output_format in STRUCTURED_OUTPUT_FORMATS:
        return CachePolicy(
            action=CacheAction.LOOKUP_AND_STORE,
            ttl=TTL_LONG,
            match=MatchStrategy.EXACT,
            reason=f"output_format:{output_format}",
        )

    return DEFAULT_POLICY


__all__ = [
    "CacheAction",
    "CachePolicy",
    "MatchStrategy",
    "DEFAULT_NAMESPACE_KEY",
    "DEFAULT_POLICY",
    "JAILBREAK_HIGH",
    "PII_BLOCKING_LABELS",
    "STRUCTURED_OUTPUT_FORMATS",
    "TENANT_NAMESPACE_KEY",
    "TOXICITY_HIGH",
    "TTL_DEFAULT",
    "TTL_LONG",
    "TTL_SHORT",
    "classify_to_policy",
]
