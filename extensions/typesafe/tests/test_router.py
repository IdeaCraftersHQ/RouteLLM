"""Tests for JevRouter against a mock TypeSafe transport.

All requests go through httpx2.MockTransport, never the network. Every
fixture sets TYPESAFE_API_KEY: TypeSafeClient requires it even with a
mock transport, since the key check happens before any request is sent.
"""
import json
import logging

import httpx2
import pytest
import typesafe_sdk

import routellm.routers.registry as registry

from routellm.types import ModelPair
from routellm_typesafe.router import DEFAULT_CRITERIA, JevRouter

RESPONSE_BODY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 1, "output_tokens": 0},
    "answers": {"strong": {"type": "noul", "noul": 0.83}},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")


@pytest.fixture
def captured():
    """Mutable box the mock handler stashes the decoded request body into."""
    return {}


@pytest.fixture
def router(captured):
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=RESPONSE_BODY)

    router = JevRouter(model="jev-1.13.0", transport=httpx2.MockTransport(handler))
    yield router
    router.close()


def test_win_rate_is_noul_probability(router):
    assert router.calculate_strong_win_rate("hello world") == 0.83


def test_route_threshold(router):
    pair = ModelPair(strong="gpt-4", weak="gpt-3.5-turbo")

    assert router.route("hello world", 0.5, pair) == "gpt-4"
    assert router.route("hello world", 0.9, pair) == "gpt-3.5-turbo"


def test_prompt_truncated(captured):
    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=RESPONSE_BODY)

    router = JevRouter(
        max_prompt_chars=10, transport=httpx2.MockTransport(handler)
    )
    router.calculate_strong_win_rate("x" * 100)
    router.close()

    assert len(captured["body"]["state"]["prompt"]) == 10


def test_model_from_config(router, captured):
    router.calculate_strong_win_rate("hello world")

    assert captured["body"]["model"] == "jev-1.13.0"


def test_custom_criteria_in_request(captured):
    criteria = {"true": "custom yes case", "false": "custom no case"}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=RESPONSE_BODY)

    router = JevRouter(
        criteria=criteria, transport=httpx2.MockTransport(handler)
    )
    router.calculate_strong_win_rate("hello world")
    router.close()

    assert captured["body"]["questions"]["strong"]["criteria"] == criteria


def test_default_criteria_not_shared_between_routers():
    def handler(request):
        return httpx2.Response(200, json=RESPONSE_BODY)

    router_a = JevRouter(transport=httpx2.MockTransport(handler))
    router_b = JevRouter(transport=httpx2.MockTransport(handler))
    try:
        assert router_a.criteria is not router_b.criteria

        router_a.criteria["true"] = "mutated"

        assert DEFAULT_CRITERIA["true"] != "mutated"
        assert router_b.criteria["true"] != "mutated"
    finally:
        router_a.close()
        router_b.close()


def test_debug_log_records_response_model_id(router, caplog):
    caplog.set_level(logging.DEBUG, logger="routellm_typesafe.router")

    router.calculate_strong_win_rate("hello world")

    assert any(
        RESPONSE_BODY["model"] in record.message
        and str(RESPONSE_BODY["usage"]["input_tokens"]) in record.message
        for record in caplog.records
    )


def test_api_error_propagates():
    def handler(request):
        return httpx2.Response(429, json={"error": "rate limited"})

    router = JevRouter(transport=httpx2.MockTransport(handler), max_retries=0)
    try:
        with pytest.raises(typesafe_sdk.TypeSafeRateLimitError):
            router.calculate_strong_win_rate("hello world")
    finally:
        router.close()


class _FakeEntryPoint:
    """Stand-in for the `jev` entry point this package declares."""

    name = "jev"

    def load(self):
        return JevRouter


# The real entry point only resolves once the package is pip-installed,
# which the standalone test run does not require. Patching
# `entry_points` in the registry namespace exercises the same code path
# with the same name and target class the pyproject declares.
def test_registered_through_entry_point(monkeypatch):
    def fake_entry_points(group=None):
        assert group == registry.ENTRY_POINT_GROUP
        return [_FakeEntryPoint()]

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)

    saved = dict(registry.ROUTER_CLS)
    saved_failures = dict(registry.discovery_failures)
    registry.reset_registry()
    try:
        assert registry.discover_routers() == ["jev"]
        assert registry.get_router_class("jev") is JevRouter
    finally:
        registry.reset_registry()
        registry.ROUTER_CLS.update(saved)
        registry.discovery_failures.update(saved_failures)


_ABSENT = object()


def test_str_is_jev(router):
    # Router.__str__ resolves through the registry, which is real and
    # torch-free even though routellm.routers.routers is stubbed here.
    saved = registry.ROUTER_CLS.get("jev", _ABSENT)
    registry.register_router("jev", JevRouter, replace=True)
    try:
        assert str(router) == "jev"
    finally:
        if saved is _ABSENT:
            registry.ROUTER_CLS.pop("jev", None)
        else:
            registry.ROUTER_CLS["jev"] = saved
