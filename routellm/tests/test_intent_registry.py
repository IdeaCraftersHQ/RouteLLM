"""Intent detection resolves through the endpoint registry.

No default cloud model: an unset detector model disables detection, a
raw name is refused at startup, and every detector exposes the registry
endpoint it calls so an egress gate can veto it.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import routellm.controller
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector
from routellm.routers.base import Router

CONFIG = {
    "endpoints": {
        "classifier": {
            "model": "ollama_chat/qwen3:1b",
            "api_base": "http://127.0.0.1:11500",
            "tags": ["local"],
        },
        "jev_upstream": {
            "model": "jev-1",
            "api_base": "https://jev.example.com",
            "api_key_env": "JEV_KEY",
            "tags": ["cloud"],
        },
        "big": {"model": "gpt-4o"},
        "small": {"model": "ollama_chat/qwen3:8b"},
    },
    "tiers": {"legal": {"strong": "big", "weak": "small"}},
}

MAPPINGS = [IntentModelMapping(intent="legal", description="contracts")]


@pytest.fixture
def registry():
    return EndpointRegistry.from_config(CONFIG)


@pytest.fixture(scope="module")
def server_config():
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def _intents(**spec):
    return {"intents": {"detector": "litellm", "tiers": {"legal": "legal"}, **spec}}


# --- selector: no default model ------------------------------------------


@patch("litellm.completion")
def test_selector_without_model_makes_no_completion_call(mock_completion):
    selector = IntentModelSelector(MAPPINGS)

    assert selector.detect_intent("hello") == "general"
    mock_completion.assert_not_called()


@patch("litellm.completion")
def test_confidence_analysis_without_model_makes_no_completion_call(mock_completion):
    selector = IntentModelSelector(MAPPINGS)

    assert selector.analyze_intent_confidence("hello")["best_match"] == "general"
    mock_completion.assert_not_called()


# --- build_intents: litellm detector --------------------------------------


@patch("litellm.completion")
def test_unset_model_disables_detection_with_one_info_line(
    mock_completion, registry, server_config, caplog
):
    with caplog.at_level(logging.INFO):
        selector = server_config.build_intents(_intents(), registry)

    assert selector is None
    assert len([r for r in caplog.records if r.levelno == logging.INFO]) == 1
    mock_completion.assert_not_called()


def test_raw_model_name_is_a_startup_error(registry, server_config):
    with pytest.raises(ValueError, match="gpt-4o-mini.*registry endpoint"):
        server_config.build_intents(_intents(model="gpt-4o-mini"), registry)


def test_tier_name_is_not_an_endpoint(registry, server_config):
    with pytest.raises(ValueError, match="legal.*tier"):
        server_config.build_intents(_intents(model="legal"), registry)


@patch("litellm.completion")
def test_registry_endpoint_is_called_with_its_own_params(mock_completion, registry, server_config):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="legal"))]
    )
    selector = server_config.build_intents(_intents(model="classifier"), registry)

    assert selector.egress_endpoint == "classifier"
    assert selector.detect_intent("sue them") == "legal"
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "ollama_chat/qwen3:1b"
    assert kwargs["api_base"] == "http://127.0.0.1:11500"


# --- build_intents: jev detector -------------------------------------------


def _jev(**spec):
    return {"intents": {"detector": "jev", "tiers": {"legal": "legal"}, **spec}}


def test_jev_without_upstream_is_a_startup_error(registry, server_config):
    with pytest.raises(ValueError, match="upstream"):
        server_config.build_intents(_jev(), registry)


def test_jev_with_unknown_upstream_is_a_startup_error(registry, server_config):
    with pytest.raises(ValueError, match="ghost.*registry endpoint"):
        server_config.build_intents(_jev(upstream="ghost"), registry)


def test_jev_upstream_is_exposed_and_sets_the_base_url(registry, server_config, monkeypatch):
    fake = MagicMock()
    fake.JevIntentDetector.return_value = MagicMock(upstream="jev_upstream")
    monkeypatch.setitem(sys.modules, "routellm_typesafe.intent_detector", fake)

    selector = server_config.build_intents(_jev(upstream="jev_upstream"), registry)

    assert selector.egress_endpoint == "jev_upstream"
    kwargs = fake.JevIntentDetector.call_args.kwargs
    assert kwargs["base_url"] == "https://jev.example.com"
    assert kwargs["upstream"] == "jev_upstream"


# --- routers that call a hosted upstream -----------------------------------


class _Hosted(Router):
    requires_upstream = True

    def __init__(self, upstream=None, base_url=None):
        self.upstream = upstream
        self.base_url = base_url

    def calculate_strong_win_rate(self, prompt):
        return 0.5


def _controller(monkeypatch, router_config):
    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "hosted", _Hosted)
    return Controller(
        routers=["hosted"],
        strong_model="big",
        weak_model="small",
        endpoints=EndpointRegistry.from_config(CONFIG),
        config={"hosted": router_config},
        progress_bar=False,
    )


def test_hosted_router_without_upstream_refuses_to_load(monkeypatch):
    with pytest.raises(ValueError, match="hosted.*upstream"):
        _controller(monkeypatch, {})


def test_hosted_router_with_unknown_upstream_refuses_to_load(monkeypatch):
    with pytest.raises(ValueError, match="ghost.*registry endpoint"):
        _controller(monkeypatch, {"upstream": "ghost"})


def test_hosted_router_gets_base_url_and_exposes_its_endpoint(monkeypatch):
    controller = _controller(monkeypatch, {"upstream": "jev_upstream"})

    router = controller.routers["hosted"]
    assert router.upstream == "jev_upstream"
    assert router.base_url == "https://jev.example.com"
