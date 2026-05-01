"""Tests covering each row of the cache policy table (scenario 3 spec).

Uses a fake ClassificationResult dataclass so we don't pull c12n
bindings into the test loop. The mapping function only inspects
documented attributes; any duck-typed shape works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pytest
from policy_routing import (
    DEFAULT_NAMESPACE_KEY,
    CacheAction,
    CachePolicy,
    MatchStrategy,
    classify_to_policy,
)
from policy_routing.policy import (
    DEFAULT_POLICY,
    JAILBREAK_HIGH,
    TENANT_NAMESPACE_KEY,
    TOXICITY_HIGH,
    TTL_DEFAULT,
    TTL_LONG,
    TTL_SHORT,
)


@dataclass
class FakeClassification:
    """Stand-in for c12n.ClassificationResult shape."""

    pii_labels: Iterable[str] = field(default_factory=tuple)
    jailbreak_score: float = 0.0
    toxicity_score: float = 0.0
    code_content: Optional[str] = None
    domain: Optional[str] = None
    complexity: Optional[str] = None
    cost_estimate: Optional[str] = None
    output_format: Optional[str] = None


# Request is unused by current rules — sentinel keeps the signature
# real-world-ish. Tests should not rely on its structure.
REQUEST = object()


# ---------------------------------------------------------------------------
# Row 1 — PII labels (EMAIL / PHONE / SSN) → SKIP cache, tenant ns.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["EMAIL", "PHONE", "SSN"])
def test_pii_label_skips_cache_with_tenant_namespace(label):
    classification = FakeClassification(pii_labels=(label,))
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.ttl == 0
    assert policy.namespace == TENANT_NAMESPACE_KEY
    assert policy.reason == "pii_detected"


def test_pii_label_unknown_does_not_trigger():
    # Non-blocking PII labels (e.g. "NAME") should NOT skip cache;
    # only the explicit blocking set fires.
    classification = FakeClassification(pii_labels=("NAME",))
    policy = classify_to_policy(REQUEST, classification)
    assert policy == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Row 2 — Jailbreak: high → SKIP cache.
# ---------------------------------------------------------------------------


def test_jailbreak_high_skips_cache():
    classification = FakeClassification(jailbreak_score=JAILBREAK_HIGH)
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.reason == "jailbreak_high"


def test_jailbreak_below_threshold_uses_default():
    classification = FakeClassification(jailbreak_score=JAILBREAK_HIGH - 0.01)
    policy = classify_to_policy(REQUEST, classification)
    assert policy == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Row 3 — Toxicity: high → SKIP cache.
# ---------------------------------------------------------------------------


def test_toxicity_high_skips_cache():
    classification = FakeClassification(toxicity_score=TOXICITY_HIGH)
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.reason == "toxicity_high"


# ---------------------------------------------------------------------------
# Row 4 — CodeContent: <lang> → exact-match only, long TTL.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["python", "go", "rust", "typescript"])
def test_code_content_uses_exact_match_long_ttl(lang):
    classification = FakeClassification(code_content=lang)
    policy = classify_to_policy(REQUEST, classification)
    assert policy.match is MatchStrategy.EXACT
    assert policy.ttl == TTL_LONG
    assert policy.action is CacheAction.LOOKUP_AND_STORE
    assert policy.reason == f"code_content:{lang}"


# ---------------------------------------------------------------------------
# Row 5 — Domain: legal → workspace-scoped, short TTL.
# ---------------------------------------------------------------------------


def test_domain_legal_workspace_scoped_short_ttl():
    classification = FakeClassification(domain="legal")
    policy = classify_to_policy(REQUEST, classification)
    assert policy.namespace == DEFAULT_NAMESPACE_KEY  # workspace_id
    assert policy.ttl == TTL_SHORT
    assert policy.reason == "domain:legal"


# ---------------------------------------------------------------------------
# Row 6 — Domain: math/cs → workspace-scoped (per T-0193), long TTL.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["math", "cs"])
def test_domain_math_or_cs_workspace_scoped_long_ttl(domain):
    classification = FakeClassification(domain=domain)
    policy = classify_to_policy(REQUEST, classification)
    assert policy.namespace == DEFAULT_NAMESPACE_KEY
    assert policy.ttl == TTL_LONG
    assert policy.action is CacheAction.LOOKUP_AND_STORE
    assert policy.reason == f"domain:{domain}"


# ---------------------------------------------------------------------------
# Row 7 — Complexity: easy + cost: low → SKIP (not worth storing).
# ---------------------------------------------------------------------------


def test_easy_low_cost_skips_cache():
    classification = FakeClassification(complexity="easy", cost_estimate="low")
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.reason == "cheap_to_recompute"


def test_easy_alone_does_not_skip():
    # Without explicit low cost_estimate, fall through to default —
    # the table requires BOTH conditions to fire.
    classification = FakeClassification(complexity="easy")
    policy = classify_to_policy(REQUEST, classification)
    assert policy == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Row 8 — Complexity: hard + cost: high → semantic cache, long TTL.
# ---------------------------------------------------------------------------


def test_hard_high_cost_uses_semantic_long_ttl():
    classification = FakeClassification(complexity="hard", cost_estimate="high")
    policy = classify_to_policy(REQUEST, classification)
    assert policy.match is MatchStrategy.SEMANTIC
    assert policy.ttl == TTL_LONG
    assert policy.action is CacheAction.LOOKUP_AND_STORE
    assert policy.reason == "high_value_reuse"


# ---------------------------------------------------------------------------
# Row 9 — OutputFormat: code/json/yaml → exact-match, high TTL.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["code", "json", "yaml"])
def test_structured_output_format_exact_match_long_ttl(fmt):
    classification = FakeClassification(output_format=fmt)
    policy = classify_to_policy(REQUEST, classification)
    assert policy.match is MatchStrategy.EXACT
    assert policy.ttl == TTL_LONG
    assert policy.reason == f"output_format:{fmt}"


def test_prose_output_format_uses_default():
    classification = FakeClassification(output_format="prose")
    policy = classify_to_policy(REQUEST, classification)
    assert policy == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Default + precedence behavior.
# ---------------------------------------------------------------------------


def test_empty_classification_uses_default():
    policy = classify_to_policy(REQUEST, FakeClassification())
    assert policy == DEFAULT_POLICY
    assert policy.action is CacheAction.LOOKUP_AND_STORE
    assert policy.ttl == TTL_DEFAULT
    assert policy.namespace == DEFAULT_NAMESPACE_KEY
    assert policy.match is MatchStrategy.BOTH


def test_pii_takes_precedence_over_code_content():
    # Safety wins over determinism: a code-content prompt that also
    # carries PII must never be cached.
    classification = FakeClassification(
        pii_labels=("EMAIL",),
        code_content="python",
    )
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.reason == "pii_detected"


def test_jailbreak_takes_precedence_over_domain():
    classification = FakeClassification(
        jailbreak_score=0.95,
        domain="math",
    )
    policy = classify_to_policy(REQUEST, classification)
    assert policy.action is CacheAction.SKIP
    assert policy.reason == "jailbreak_high"


def test_dict_shape_classification_works():
    # The mapper accepts dict-like classification objects (TypedDict
    # consumers) — duck-typing path.
    classification = {"domain": "legal"}
    policy = classify_to_policy(REQUEST, classification)
    assert policy.reason == "domain:legal"
    assert policy.ttl == TTL_SHORT


# ---------------------------------------------------------------------------
# CachePolicy dataclass — shape contract.
# ---------------------------------------------------------------------------


def test_cache_policy_is_frozen_and_hashable():
    p1 = CachePolicy()
    p2 = CachePolicy()
    assert p1 == p2
    assert hash(p1) == hash(p2)
    with pytest.raises(Exception):
        p1.action = CacheAction.SKIP  # type: ignore[misc]


def test_cache_action_string_round_trip():
    # Action values must round-trip as strings — needed for JSON
    # serialisation in the audit log (US-0101).
    assert CacheAction.SKIP.value == "skip"
    assert CacheAction.LOOKUP_AND_STORE.value == "lookup_and_store"
    assert MatchStrategy.EXACT.value == "exact"
    assert MatchStrategy.BOTH.value == "both"


# ---------------------------------------------------------------------------
# Integrations — build_cache_hook smoke test.
# ---------------------------------------------------------------------------


def test_build_cache_hook_returns_callable_matching_contract():
    from policy_routing.integrations import build_cache_hook

    hook = build_cache_hook()
    classification = FakeClassification(domain="legal")
    policy = hook(REQUEST, classification)
    assert policy.reason == "domain:legal"


def test_build_cache_hook_invokes_classifier_when_missing():
    from policy_routing.integrations import build_cache_hook

    calls = []

    def classifier(req):
        calls.append(req)
        return FakeClassification(code_content="python")

    hook = build_cache_hook(classifier=classifier)
    policy = hook(REQUEST, classification=None)
    assert calls == [REQUEST]
    assert policy.match is MatchStrategy.EXACT
    assert policy.reason == "code_content:python"


def test_build_cache_hook_no_classifier_no_classification_returns_default():
    from policy_routing.integrations import build_cache_hook

    hook = build_cache_hook()
    policy = hook(REQUEST, classification=None)
    assert policy == DEFAULT_POLICY
