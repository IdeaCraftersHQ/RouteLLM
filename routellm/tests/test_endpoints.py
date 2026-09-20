"""Tests for the endpoint registry and the controller's use of it.

Covers `EndpointRegistry` construction and lookup, per-endpoint
credential resolution, and the controller path that turns an endpoint
into litellm call parameters.
"""
import logging

import pytest
from pydantic import ValidationError

from routellm.endpoints import Endpoint, EndpointRegistry

CONFIG = {
    "endpoints": {
        "cloud_strong": {
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "tags": ["tools", "vision"],
            "quality": 90,
            "extra": {"timeout": 60},
        },
        "local_fast": {
            "model": "ollama_chat/qwen3:8b",
            "api_base": "http://127.0.0.1:11500",
            "tags": ["local"],
        },
    }
}


@pytest.fixture
def registry():
    """Registry built from the two-endpoint sample config."""
    return EndpointRegistry.from_config(CONFIG)


# ---------------------------------------------------------------------------
# Registry construction and lookup
# ---------------------------------------------------------------------------


def test_from_config_builds_named_endpoints(registry):
    assert registry.names() == ["cloud_strong", "local_fast"]

    strong = registry.get("cloud_strong")
    assert strong.name == "cloud_strong"
    assert strong.model == "gpt-4o"
    assert strong.api_key_env == "OPENAI_API_KEY"
    assert strong.tags == ["tools", "vision"]
    assert strong.quality == 90
    assert strong.extra == {"timeout": 60}


def test_from_config_defaults_fill_omitted_fields(registry):
    local = registry.get("local_fast")

    assert local.api_base == "http://127.0.0.1:11500"
    assert local.api_key_env is None
    assert local.quality is None
    assert local.extra == {}


def test_from_config_without_endpoints_key_is_empty():
    assert EndpointRegistry.from_config({}).names() == []


def test_from_config_ignores_unknown_top_level_keys():
    config = {"tiers": {"fast": {}}, "mf": {"checkpoint_path": "x"}, **CONFIG}

    assert EndpointRegistry.from_config(config).names() == [
        "cloud_strong",
        "local_fast",
    ]


def test_endpoint_name_charset_rejected():
    with pytest.raises(ValidationError):
        Endpoint(name="cloud-strong", model="gpt-4o")


def test_endpoint_empty_model_rejected():
    with pytest.raises(ValidationError):
        Endpoint(name="cloud_strong", model="")


def test_get_unknown_name_lists_known_names(registry):
    with pytest.raises(KeyError) as excinfo:
        registry.get("missing")

    message = str(excinfo.value)
    assert "cloud_strong" in message
    assert "local_fast" in message


# ---------------------------------------------------------------------------
# resolve(): named endpoints and raw model passthrough
# ---------------------------------------------------------------------------


def test_resolve_known_name_returns_endpoint(registry):
    assert registry.resolve("cloud_strong") is registry.get("cloud_strong")


def test_resolve_raw_model_warns_once(registry, caplog):
    caplog.set_level(logging.WARNING, logger="routellm.endpoints")

    first = registry.resolve("gpt-4")

    assert first.name == "gpt-4"
    assert first.model == "gpt-4"
    assert first.api_base is None
    assert first.api_key_env is None
    assert len(caplog.records) == 1
    assert "gpt-4" in caplog.records[0].message

    caplog.clear()
    second = registry.resolve("gpt-4")

    assert second.model == "gpt-4"
    assert caplog.records == []


def test_resolve_warns_once_per_distinct_name(registry, caplog):
    caplog.set_level(logging.WARNING, logger="routellm.endpoints")

    registry.resolve("gpt-4")
    registry.resolve("mistral-7b")

    assert len(caplog.records) == 2


# ---------------------------------------------------------------------------
# credentials()
# ---------------------------------------------------------------------------


def test_credentials_reads_env_at_call_time(registry, monkeypatch):
    endpoint = registry.get("cloud_strong")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        endpoint.credentials(None, None)

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    assert endpoint.credentials(None, None) == (None, "test")


def test_credentials_missing_env_names_endpoint_and_variable(registry, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        registry.get("cloud_strong").credentials(None, "default-key")

    message = str(excinfo.value)
    assert "cloud_strong" in message
    assert "OPENAI_API_KEY" in message


def test_credentials_fall_back_to_defaults(registry):
    api_base, api_key = registry.get("local_fast").credentials(
        "https://default.base", "default-key"
    )

    assert api_base == "http://127.0.0.1:11500"
    assert api_key == "default-key"


def test_credentials_of_anonymous_endpoint_are_the_defaults(registry):
    endpoint = registry.resolve("gpt-4")

    assert endpoint.credentials("https://default.base", "default-key") == (
        "https://default.base",
        "default-key",
    )


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------


@pytest.fixture
def controller(registry):
    """Controller wired to the sample registry, caching disabled."""
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    return Controller(
        routers=["random"],
        strong_model="cloud_strong",
        weak_model="local_fast",
        endpoints=registry,
        api_base="https://default.base",
        api_key="default-key",
        cache_config=CacheConfig(enabled=False),
    )


@pytest.fixture
def mock_completion(monkeypatch):
    """Patch `routellm.controller.completion`, return the mock."""
    from unittest.mock import MagicMock

    res = MagicMock()
    res.model_dump.return_value = {"id": "test", "choices": []}
    mock = MagicMock(return_value=res)
    monkeypatch.setattr("routellm.controller.completion", mock)
    return mock


def test_controller_defaults_to_empty_registry():
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        cache_config=CacheConfig(enabled=False),
    )

    assert controller.endpoints.names() == []


def test_controller_passes_endpoint_call_params(controller, mock_completion):
    # The fake router routes to the weak model, i.e. local_fast.
    controller.completion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}],
    )

    kwargs = mock_completion.call_args[1]
    assert kwargs["model"] == "ollama_chat/qwen3:8b"
    assert kwargs["api_base"] == "http://127.0.0.1:11500"
    assert kwargs["api_key"] == "default-key"


def test_controller_request_kwargs_win_over_extra(
    controller, mock_completion, monkeypatch
):
    # Only cloud_strong carries extra, so route to the strong model.
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        controller.routers["random"],
        "route",
        lambda prompt, threshold, model_pair: model_pair.strong,
    )

    controller.completion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}],
        timeout=5,
    )

    kwargs = mock_completion.call_args[1]
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["api_base"] == "https://default.base"
    assert kwargs["api_key"] == "test"
    assert kwargs["timeout"] == 5


def test_controller_extra_applied_when_request_is_silent(
    controller, mock_completion, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        controller.routers["random"],
        "route",
        lambda prompt, threshold, model_pair: model_pair.strong,
    )

    controller.completion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert mock_completion.call_args[1]["timeout"] == 60


def test_controller_balancer_overrides_endpoint(controller, mock_completion):
    from routellm.traffic import (
        LoadBalancer,
        LoadBalancerConfig,
        LoadBalancerEndpoint,
    )

    controller.traffic_manager.load_balancers["local_fast"] = LoadBalancer(
        LoadBalancerConfig(
            strategy="round-robin",
            endpoints=[
                LoadBalancerEndpoint(
                    model="balanced-model",
                    api_base="https://balanced.base",
                    api_key="balanced-key",
                )
            ],
        )
    )

    controller.completion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}],
    )

    kwargs = mock_completion.call_args[1]
    assert kwargs["model"] == "balanced-model"
    assert kwargs["api_base"] == "https://balanced.base"
    assert kwargs["api_key"] == "balanced-key"


def test_controller_raw_model_still_works(mock_completion):
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        api_base="https://default.base",
        api_key="default-key",
        cache_config=CacheConfig(enabled=False),
    )

    controller.completion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}],
    )

    kwargs = mock_completion.call_args[1]
    assert kwargs["model"] == "gpt-3.5-turbo"
    assert kwargs["api_base"] == "https://default.base"
    assert kwargs["api_key"] == "default-key"
