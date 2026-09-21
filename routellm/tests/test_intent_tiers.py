"""Tests for intents choosing which tier a request enters.

Covers the `intent_routing` field on `Tier`, the middleware `get_tier`
hook the controller consults before the pair bypass, the
`intent_tiers` mapping on `IntentModelSelector`, the `intents:` config
section the server builds a middleware from, and the explain line the
pairing CLI prints per intent.

Routers are stubbed the same way as in `test_tiers.py`: a fixed win
rate registered into `routellm.controller.ROUTER_CLS`, which the root
conftest has already replaced with a stub dict.
"""

import builtins
import sys
from unittest.mock import MagicMock

import pytest

import routellm.controller
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry, Tier
from routellm.middleware.intent_model_selector import (
    IntentModelSelector,
    IntentModelMapping,
)
from routellm.routers.base import Router
from routellm.traffic import TrafficManager, TrafficRule
from routellm.types import ModelPair

CONFIG = {
    "endpoints": {
        "cloud_strong": {"model": "gpt-4o"},
        "frontier_local": {"model": "ollama_chat/qwen3:32b"},
        "local_fast": {"model": "ollama_chat/qwen3:8b"},
    },
    "tiers": {
        "premium": {
            "router": "hi",
            "threshold": 0.33,
            "strong": "cloud_strong",
            "weak": "frontier_local",
        },
        "legal": {
            "router": "hi",
            "threshold": 0.33,
            "intent_routing": True,
            "strong": "cloud_strong",
            "weak": "local_fast",
        },
        "default": {
            "router": "hi",
            "threshold": 0.12,
            "strong": "premium",
            "weak": "local_fast",
        },
    },
}


class _StubRouter(Router):
    """Router returning a fixed win rate."""

    win_rate = 0.5

    def calculate_strong_win_rate(self, prompt):
        return type(self).win_rate


def _make_router(win_rate):
    """Build a fresh stub router class with its own win rate."""

    class Stub(_StubRouter):
        pass

    Stub.win_rate = win_rate
    return Stub


@pytest.fixture
def hi_router(monkeypatch):
    """Stub router scoring 0.9, registered under the name 'hi'."""
    cls = _make_router(0.9)
    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "hi", lambda **kw: cls())
    return cls


class _TierMiddleware:
    """Middleware naming a tier to enter and no pair."""

    def __init__(self, tier):
        self.tier = tier

    def get_tier(self, prompt):
        return self.tier

    def get_model_pair(self, prompt):
        return None


class _CountingTierMiddleware(_TierMiddleware):
    """Tier middleware counting how often it was asked."""

    def __init__(self, tier):
        super().__init__(tier)
        self.calls = 0

    def get_tier(self, prompt):
        self.calls += 1
        return self.tier


def _controller(middleware, hi_router):
    """Build a tiered controller carrying `middleware`."""
    return Controller(
        routers=["hi"],
        endpoints=EndpointRegistry.from_config(CONFIG),
        middleware=middleware,
        default_router="hi",
        default_threshold=0.5,
        progress_bar=False,
    )


# ---------------------------------------------------------------------------
# Tier.intent_routing
# ---------------------------------------------------------------------------


def test_default_tier_accepts_intent_routing_without_declaring_it():
    tier = Tier(name="default", strong="a", weak="b")

    assert tier.accepts_intent_routing() is True


def test_named_tier_refuses_intent_routing_by_default():
    tier = Tier(name="premium", strong="a", weak="b")

    assert tier.accepts_intent_routing() is False


def test_explicit_intent_routing_wins_over_the_name():
    opted_in = Tier(name="legal", intent_routing=True, strong="a", weak="b")
    opted_out = Tier(name="default", intent_routing=False, strong="a", weak="b")

    assert opted_in.accepts_intent_routing() is True
    assert opted_out.accepts_intent_routing() is False


# ---------------------------------------------------------------------------
# The controller's get_tier hook
# ---------------------------------------------------------------------------


def test_middleware_tier_is_entered_from_the_default_tier(hi_router):
    controller = _controller([_TierMiddleware("premium")], hi_router)

    picked, path, pair = controller._route("hi", {}, "default", None, None)

    assert picked == "cloud_strong"
    assert [entry["tier"] for entry in path] == ["premium"]
    assert path[0]["tier_from"] == "intent"
    assert pair is None


def test_an_explicit_tier_ignores_the_middleware_tier(hi_router):
    controller = _controller([_TierMiddleware("legal")], hi_router)

    picked, path, _ = controller._route("hi", {}, "premium", None, None)

    assert [entry["tier"] for entry in path] == ["premium"]
    assert path[0].get("tier_from") is None
    assert path[0]["intent_ignored"] == "legal"


def test_a_tier_declaring_intent_routing_honours_the_middleware(hi_router):
    controller = _controller([_TierMiddleware("legal")], hi_router)

    picked, path, _ = controller._route("hi", {}, "legal", None, None)

    assert [entry["tier"] for entry in path] == ["legal"]
    assert path[0]["tier_from"] == "intent"


def test_no_tier_from_the_middleware_leaves_the_request_where_it_was(hi_router):
    controller = _controller([_TierMiddleware(None)], hi_router)

    picked, path, _ = controller._route("hi", {}, "default", None, None)

    assert [entry["tier"] for entry in path] == ["default", "premium"]
    assert path[0].get("tier_from") is None


def test_an_unknown_tier_from_the_middleware_is_an_error(hi_router):
    controller = _controller([_TierMiddleware("nope")], hi_router)

    with pytest.raises(routellm.controller.RoutingError, match="nope"):
        controller._route("hi", {}, "default", None, None)


def test_a_traffic_rule_skips_the_classifier_entirely(hi_router):
    """A fully bypassed request must not pay for a classification."""
    middleware = _CountingTierMiddleware("premium")
    controller = Controller(
        routers=["hi"],
        endpoints=EndpointRegistry.from_config(CONFIG),
        middleware=[middleware],
        traffic_manager=TrafficManager(
            rules=[
                TrafficRule(
                    pattern="special",
                    strong_model="cloud_strong",
                    weak_model="local_fast",
                )
            ]
        ),
        default_router="hi",
        default_threshold=0.5,
        progress_bar=False,
    )

    picked, path, pair = controller._route("this is special", {}, "default", None, None)

    assert path[0]["pair_from"] == "traffic_rule"
    assert middleware.calls == 0


def test_a_middleware_pair_still_bypasses_the_tree(hi_router):
    class _PairMiddleware:
        def get_model_pair(self, prompt):
            return ModelPair(strong="cloud_strong", weak="local_fast")

    controller = _controller([_PairMiddleware()], hi_router)

    picked, path, pair = controller._route("hi", {}, "default", None, None)

    assert path[0]["pair_from"] == "middleware"
    assert path[0]["tier"] is None
    assert pair is not None


# ---------------------------------------------------------------------------
# IntentModelSelector.intent_tiers
# ---------------------------------------------------------------------------


class _FixedDetector:
    """Detector returning a fixed label and counting its calls."""

    def __init__(self, intent):
        self.intent = intent
        self.calls = 0

    def detect_intent(self, prompt):
        self.calls += 1
        return self.intent


def test_selector_maps_a_detected_intent_to_a_tier():
    detector = _FixedDetector("legal")
    selector = IntentModelSelector(
        intent_mappings=[],
        default_model_pair=ModelPair(strong="s", weak="w"),
        intent_detector=detector,
        intent_tiers={"legal": "premium"},
    )

    assert selector.get_tier("draft a contract") == "premium"


def test_selector_returns_none_for_an_unmapped_intent():
    detector = _FixedDetector("general")
    selector = IntentModelSelector(
        intent_mappings=[],
        default_model_pair=ModelPair(strong="s", weak="w"),
        intent_detector=detector,
        intent_tiers={"legal": "premium"},
    )

    assert selector.get_tier("hello") is None


def test_selector_get_tier_reuses_the_intent_cache():
    detector = _FixedDetector("legal")
    selector = IntentModelSelector(
        intent_mappings=[],
        default_model_pair=ModelPair(strong="s", weak="w"),
        intent_detector=detector,
        intent_tiers={"legal": "premium"},
    )

    selector.get_tier("same prompt")
    selector.get_tier("same prompt")

    assert detector.calls == 1


def test_a_tier_only_selector_pairs_with_the_default():
    detector = _FixedDetector("legal")
    selector = IntentModelSelector(
        intent_mappings=[],
        default_model_pair=ModelPair(strong="s", weak="w"),
        intent_detector=detector,
        intent_tiers={"legal": "premium"},
    )

    assert selector.get_model_pair("draft a contract") == ModelPair(strong="s", weak="w")


def test_intent_tiers_leave_get_model_pair_alone_for_mapping_callers():
    detector = _FixedDetector("legal")
    selector = IntentModelSelector(
        intent_mappings=[
            IntentModelMapping(
                intent="legal",
                model_pair=ModelPair(strong="big", weak="small"),
                description="legal work",
            )
        ],
        default_model_pair=ModelPair(strong="s", weak="w"),
        intent_detector=detector,
        intent_tiers={"legal": "premium"},
    )

    assert selector.get_model_pair("draft a contract") == ModelPair(strong="big", weak="small")


# ---------------------------------------------------------------------------
# The intents: config section
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Registry holding the tiers the intents map to."""
    return EndpointRegistry.from_config(CONFIG)


@pytest.fixture(scope="module")
def server_config():
    """Import the server module, which parses argv at import time."""
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def test_no_intents_section_builds_no_middleware(registry, server_config):
    assert server_config.build_intents({}, registry) is None


def test_intents_section_builds_a_litellm_backed_selector(registry, server_config):
    config = {
        "intents": {
            "detector": "litellm",
            "model": "gpt-4o-mini",
            "descriptions": {"legal": "contracts and statutes"},
            "tiers": {"legal": "legal"},
        }
    }

    selector = server_config.build_intents(config, registry)

    assert isinstance(selector, IntentModelSelector)
    assert selector.intent_tiers == {"legal": "legal"}
    assert selector.intent_detector is None
    assert selector.intent_detection_model == "gpt-4o-mini"
    assert [m.intent for m in selector.intent_mappings] == ["legal"]
    assert selector.intent_mappings[0].description == "contracts and statutes"


def test_a_mapping_to_a_missing_tier_is_rejected(registry, server_config):
    config = {"intents": {"detector": "litellm", "tiers": {"legal": "ghost"}}}

    with pytest.raises(ValueError, match="legal.*ghost"):
        server_config.build_intents(config, registry)


def test_an_intents_section_without_tiers_is_rejected(registry, server_config):
    config = {"intents": {"detector": "litellm", "descriptions": {"legal": "x"}}}

    with pytest.raises(ValueError, match="non-empty `tiers` mapping"):
        server_config.build_intents(config, registry)


def test_an_unknown_detector_is_rejected(registry, server_config):
    config = {"intents": {"detector": "magic", "tiers": {"legal": "legal"}}}

    with pytest.raises(ValueError, match="magic"):
        server_config.build_intents(config, registry)


def test_the_jev_detector_is_imported_lazily(registry, server_config, monkeypatch):
    fake = MagicMock()
    detector = MagicMock()
    fake.JevIntentDetector.return_value = detector
    monkeypatch.setitem(sys.modules, "routellm_typesafe.intent_detector", fake)

    config = {
        "intents": {
            "detector": "jev",
            "model": "jev-1",
            "confidence_floor": 0.7,
            "descriptions": {"legal": "contracts"},
            "tiers": {"legal": "legal"},
        }
    }
    selector = server_config.build_intents(config, registry)

    assert selector.intent_detector is detector
    kwargs = fake.JevIntentDetector.call_args.kwargs
    assert kwargs["model"] == "jev-1"
    assert kwargs["confidence_floor"] == 0.7
    assert kwargs["descriptions"] == {"legal": "contracts"}


def test_a_missing_typesafe_extension_names_the_install(registry, server_config, monkeypatch):
    real_import = builtins.__import__

    def _refuse(name, *rest, **kwargs):
        if name.startswith("routellm_typesafe"):
            raise ImportError("No module named 'routellm_typesafe'")
        return real_import(name, *rest, **kwargs)

    monkeypatch.delitem(sys.modules, "routellm_typesafe.intent_detector", raising=False)
    monkeypatch.delitem(sys.modules, "routellm_typesafe", raising=False)
    monkeypatch.setattr(builtins, "__import__", _refuse)

    config = {"intents": {"detector": "jev", "tiers": {"legal": "legal"}}}

    with pytest.raises(ImportError, match="pip install -e extensions/typesafe"):
        server_config.build_intents(config, registry)


# ---------------------------------------------------------------------------
# Explain surface
# ---------------------------------------------------------------------------


def test_explain_lists_each_intent_and_its_tier(tmp_path):
    from routellm.pairing import _explain

    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  legal: {strong: big, weak: small, intent_routing: true}\n"
        "  default: {strong: big, weak: small}\n"
        "intents:\n"
        "  detector: litellm\n"
        "  tiers:\n"
        "    legal: legal\n"
    )

    report = _explain(str(config))

    assert "intent legal -> tier legal" in report
