"""Tests for JevRouter against a mock TypeSafe transport.

All requests go through httpx2.MockTransport, never the network. Every
fixture sets TYPESAFE_API_KEY: TypeSafeClient requires it even with a
mock transport, since the key check happens before any request is sent.
"""
import importlib.util
import json
import logging
import sys

import httpx2
import pytest
import typesafe_sdk

import routellm.routers.registry as registry

from routellm.routers.typesafe.router import DEFAULT_CRITERIA, JevRouter
from routellm.types import ModelPair

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
    caplog.set_level(logging.DEBUG, logger="routellm.routers.typesafe.router")

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


def test_missing_sdk_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ImportError, match=r"routellm\[typesafe\]"):
        JevRouter()


# `routellm.routers.routers` imports torch at module level; the root
# conftest.py stubs it out under pytest so ROUTER_CLS there is fake. To
# check the real registration, load routers.py directly from its file path
# under a private module name, bypassing the stub in sys.modules. Its
# register_router calls populate the shared registry dict.
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
# routers.py transitively imports causal_llm/prompt_format.py, a
# pre-existing module with Pydantic v1-style @validator decorators; that
# deprecation warning is unrelated to JevRouter and only surfaces because
# this test loads the real module.
@pytest.mark.filterwarnings("ignore::pydantic.PydanticDeprecatedSince20")
def test_registered():
    saved = dict(registry.ROUTER_CLS)
    registry.reset_registry()
    try:
        spec = importlib.util.spec_from_file_location(
            "_real_routellm_routers_for_test",
            importlib.util.find_spec("routellm.routers.typesafe.router").origin.replace(
                "typesafe/router.py", "routers.py"
            ),
        )
        real_routers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_routers)

        assert real_routers.ROUTER_CLS["jev"] is JevRouter
        assert registry.get_router_class("jev") is JevRouter
    finally:
        registry.reset_registry()
        registry.ROUTER_CLS.update(saved)


def test_str_is_jev(router):
    # Router.__str__ resolves through the registry, which is real and
    # torch-free even though routellm.routers.routers is stubbed here.
    registry.register_router("jev", JevRouter, replace=True)
    try:
        assert str(router) == "jev"
    finally:
        registry.ROUTER_CLS.pop("jev", None)
