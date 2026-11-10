"""Streaming, cache and per-endpoint timeout together, against fake upstreams.

Each feature has its own tests; these cover what only shows up when they
meet: a stream with the cache switched on, a slow leaf opened as a stream,
and a tool-call conversation under the conversation-key cache. Upstreams
are threads on 127.0.0.1; nothing leaves the machine.
"""

import asyncio
import os
import time

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.resilience import ResilienceConfig
from routellm.tests.fake_upstream import FakeUpstream

pytestmark = pytest.mark.slow

_USER = [{"role": "user", "content": "hi"}]


def _controller(tmp_path, a, b, a_timeout_ms=None, cache=True):
    os.environ["FAKE_UPSTREAM_KEY"] = "sk-fake"
    endpoint = {"api_key_env": "FAKE_UPSTREAM_KEY", "extra": {"max_retries": 0}}
    a_cfg = {"model": "openai/model-a", "api_base": a.base, **endpoint}
    if a_timeout_ms:
        a_cfg["timeout_ms"] = a_timeout_ms
    # The suite stubs the random router to always pick the weak side, so
    # `a` is the weak leaf: the picked one, with `b` its sibling.
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "a": a_cfg,
                "b": {"model": "openai/model-b", "api_base": b.base, **endpoint},
            },
            "tiers": {
                "default": {"router": "random", "threshold": 0.0, "strong": "b", "weak": "a"}
            },
        }
    )
    return Controller(
        routers=["random"],
        strong_model="x",
        weak_model="x",
        endpoints=registry,
        resilience_config=ResilienceConfig(timeout_ms=30000, max_retries=0),
        cache_config=CacheConfig(enabled=cache, db_path=str(tmp_path / "cache.db")),
    )


def _ask(controller, messages, **kwargs):
    return asyncio.run(controller.acompletion(model="default", messages=messages, **kwargs))


def _drain(stream):
    async def go():
        return [chunk async for chunk in stream]

    return asyncio.run(go())


@pytest.fixture
def pair():
    a, b = FakeUpstream("A"), FakeUpstream("B")
    yield a, b
    for fake in (a, b):
        fake.server.shutdown()
        fake.server.server_close()


def _spy_cache(controller):
    calls = []
    for name in ("get", "put", "aget", "aput"):
        real = getattr(controller.cache, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        setattr(controller.cache, name, spy)
    return calls


def test_streamed_request_with_the_cache_on_never_touches_the_cache(pair, tmp_path):
    a, b = pair
    controller = _controller(tmp_path, a, b)
    calls = _spy_cache(controller)

    for _ in range(2):
        stream = _ask(controller, _USER, stream=True)
        assert any(chunk.choices[0].delta.content for chunk in _drain(stream) if chunk.choices)

    assert len(a.hits) == 2
    assert calls == []

    # Nothing was stored either: the same request, not streamed, goes upstream.
    _ask(controller, _USER)
    assert len(a.hits) == 3
    assert "aget" in calls and "aput" in calls


def test_endpoint_timeout_before_the_first_chunk_falls_back_to_the_sibling(pair, tmp_path):
    a, b = pair
    a.delay = 2.0
    controller = _controller(tmp_path, a, b, a_timeout_ms=300, cache=False)

    started = time.monotonic()
    stream = _ask(controller, _USER, stream=True)
    elapsed = time.monotonic() - started
    text = "".join(c.choices[0].delta.content or "" for c in _drain(stream) if c.choices)

    assert text == "hello from B"
    assert len(a.hits) == 1 and len(b.hits) == 1
    assert elapsed < 1.5


def test_tool_call_conversation_is_cached_under_the_whole_conversation(pair, tmp_path):
    a, b = pair
    a.mode = "tools"
    controller = _controller(tmp_path, a, b)
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    ]

    def history(result):
        call = {
            "id": "call_0",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        return [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_0", "content": result},
        ]

    first = _ask(controller, history("one"), tools=tools)
    again = _ask(controller, history("one"), tools=tools)
    assert len(a.hits) == 1
    assert again.choices[0].message.tool_calls[0].function.name == "read_file"
    assert again.choices[0].message.tool_calls[0].id == first.choices[0].message.tool_calls[0].id

    # Same last message, different history: its own entry.
    _ask(controller, history("two"), tools=tools)
    assert len(a.hits) == 2

    # And the entry sits under the conversation key, not the last message alone.
    kwargs = {"messages": history("one"), "tools": tools}
    params = controller._cache_params(kwargs)
    assert params["_conversation"]
    assert controller.cache.get("one", "a", params) is not None
