"""Server cache and resilience, driven from config against a local fake.

The upstream is a thread on 127.0.0.1 that counts what reaches it, so
every assertion is about calls that actually crossed the wire. Nothing
leaves the machine.
"""

import json
import os
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from routellm.caching import CacheConfig
from routellm.config import ConfigError
from routellm.controller import Controller
from routellm.resilience import ResilienceConfig
from routellm.server_config import build_cache_config, build_resilience_config


class Upstream:
    """OpenAI-shaped fake; `mode` is ok, fail or slow."""

    def __init__(self, mode="ok", delay=0.0):
        self.mode, self.delay, self.hits = mode, delay, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                fake.hits.append(json.loads(raw))
                if fake.delay:
                    time.sleep(fake.delay)
                if fake.mode == "fail":
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"boom"}}')
                    return
                out = {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"answer {len(fake.hits)}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(out).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    fake = Upstream()
    yield fake
    fake.close()


def _controller(fake, cache=None, resilience=None):
    cache = cache or CacheConfig(enabled=False)
    return Controller(
        routers=["random"],
        strong_model="openai/m",
        weak_model="openai/m",
        api_base=fake.base,
        api_key="sk-fake",
        cache_config=cache,
        resilience_config=resilience,
    )


def _ask(controller, messages, **kwargs):
    import asyncio

    route = kwargs.pop("route", {"router": "random", "threshold": 0.5})
    return asyncio.run(controller.acompletion(messages=messages, **route, **kwargs))


TIER = {"model": "default"}


def _cache(tmp_path, **overrides):
    return build_cache_config({"cache": {"enabled": True, **overrides}}, str(tmp_path / "c.yaml"))


def _conv(first, last="continue"):
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": last},
    ]


def _image(data):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            ],
        }
    ]


# ---------------------------------------------------------------- cache


def test_cache_is_off_when_the_config_says_nothing(upstream, tmp_path):
    config = build_cache_config({}, str(tmp_path / "c.yaml"))
    assert config.enabled is False
    assert build_cache_config(None, None).enabled is False

    controller = _controller(upstream, cache=config)
    _ask(controller, _conv("ctx ONE"))
    _ask(controller, _conv("ctx TWO"))
    assert len(upstream.hits) == 2


def test_default_server_config_creates_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_cache_config({}, str(tmp_path / "c.yaml"))
    from routellm.caching import Cache

    Cache(config)
    assert not [p for p in os.listdir(tmp_path) if "cache" in p]


def test_identical_conversation_hits(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))
    first = _ask(controller, _conv("ctx ONE"))
    second = _ask(controller, _conv("ctx ONE"))
    assert len(upstream.hits) == 1
    assert second.choices[0].message.content == first.choices[0].message.content


def test_different_history_same_last_message_misses(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))
    _ask(controller, _conv("ctx ONE"))
    _ask(controller, _conv("ctx TWO totally different"))
    assert len(upstream.hits) == 2


def test_different_system_prompt_misses(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))
    base = [{"role": "user", "content": "hi"}]
    _ask(controller, [{"role": "system", "content": "be terse"}, *base])
    _ask(controller, [{"role": "system", "content": "be verbose"}, *base])
    assert len(upstream.hits) == 2


def test_different_image_misses(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))
    _ask(controller, _image("AAAA"))
    _ask(controller, _image("BBBB"))
    assert len(upstream.hits) == 2
    _ask(controller, _image("AAAA"))
    assert len(upstream.hits) == 2


def test_different_tools_or_params_miss(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))
    msgs = [{"role": "user", "content": "hi"}]
    tool = lambda name: [  # noqa: E731
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    ]
    _ask(controller, msgs, tools=tool("a"))
    _ask(controller, msgs, tools=tool("b"))
    _ask(controller, msgs, temperature=0.9)
    assert len(upstream.hits) == 3


def test_tool_call_history_is_part_of_the_key(upstream, tmp_path):
    controller = _controller(upstream, cache=_cache(tmp_path))

    def history(result):
        return [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": result},
        ]

    _ask(controller, history("one"))
    _ask(controller, history("two"))
    assert len(upstream.hits) == 2


def test_stream_requests_are_never_cached(upstream, tmp_path, monkeypatch):
    """A stream request neither reads nor writes the cache."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    reply = MagicMock()
    reply.choices = [MagicMock()]
    reply.model_dump.return_value = {"id": "c", "choices": []}
    calls = AsyncMock(return_value=reply)
    monkeypatch.setattr("routellm.controller.acompletion", calls)

    controller = _controller(upstream, cache=_cache(tmp_path))
    gets, puts = [], []
    real_get, real_put = controller.cache.get, controller.cache.put
    controller.cache.get = lambda *a, **k: gets.append(1) or real_get(*a, **k)
    controller.cache.put = lambda *a, **k: puts.append(1) or real_put(*a, **k)

    msgs = [{"role": "user", "content": "hi"}]
    for _ in range(2):
        asyncio.run(
            controller.acompletion(router="random", threshold=0.5, messages=msgs, stream=True)
        )
    assert calls.call_count == 2
    assert gets == [] and puts == []

    # the same request without stream does use the cache
    asyncio.run(controller.acompletion(router="random", threshold=0.5, messages=msgs))
    asyncio.run(controller.acompletion(router="random", threshold=0.5, messages=msgs))
    assert calls.call_count == 3


def test_cache_db_is_created_0600_and_resolved_against_the_config(tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = build_cache_config(
        {"cache": {"enabled": True, "db_path": "state/c.db", "ttl_seconds": 7}},
        str(tmp_path / "conf" / "config.yaml"),
    )
    assert config.db_path == str(tmp_path / "conf" / "state" / "c.db")
    assert config.ttl_seconds == 7

    from routellm.caching import Cache

    Cache(config)
    assert stat.S_IMODE(os.stat(config.db_path).st_mode) == 0o600
    assert not list(elsewhere.iterdir())


def test_unknown_cache_key_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="cache"):
        build_cache_config({"cache": {"enabled": True, "ttl": 5}}, None)


# ----------------------------------------------------------- resilience


def test_resilience_defaults_when_unset():
    assert build_resilience_config({}) == ResilienceConfig()
    assert build_resilience_config(None) == ResilienceConfig()


def test_resilience_section_maps_onto_the_config():
    config = build_resilience_config(
        {
            "resilience": {
                "timeout_ms": 90000,
                "max_retries": 0,
                "retry_on_codes": [503],
                "backoff_factor": 3.0,
                "initial_backoff_ms": 100,
                "cb_fail_max": 9,
                "cb_fail_wait_ms": 1234,
            }
        }
    )
    assert config.timeout_ms == 90000
    assert config.max_retries == 0
    assert config.retry_on_codes == [503]
    assert config.backoff_factor == 3.0
    assert config.initial_backoff_ms == 100
    assert config.cb_fail_max == 9
    assert config.cb_fail_wait_ms == 1234


def test_unknown_resilience_key_is_refused():
    with pytest.raises(ConfigError, match="resilience"):
        build_resilience_config({"resilience": {"timeout": 5}})


def test_timeout_from_config_is_observed_against_a_slow_upstream():
    slow = Upstream(delay=1.5)
    try:
        config = build_resilience_config({"resilience": {"timeout_ms": 300, "max_retries": 0}})
        controller = _controller(slow, resilience=config)
        started = time.monotonic()
        with pytest.raises(Exception):
            _ask(controller, [{"role": "user", "content": "hi"}])
        assert time.monotonic() - started < 1.2
    finally:
        slow.close()


def test_slow_upstream_succeeds_under_a_longer_configured_timeout():
    slow = Upstream(delay=0.6)
    try:
        config = build_resilience_config({"resilience": {"timeout_ms": 5000, "max_retries": 0}})
        controller = _controller(slow, resilience=config)
        res = _ask(controller, [{"role": "user", "content": "hi"}])
        assert res.choices[0].message.content == "answer 1"
    finally:
        slow.close()


def test_max_retries_from_config_is_observed_against_a_failing_upstream():
    for retries in (0, 2):
        failing = Upstream(mode="fail")
        try:
            config = build_resilience_config(
                {"resilience": {"max_retries": retries, "initial_backoff_ms": 10}}
            )
            controller = Controller(
                routers=["random"],
                strong_model="x",
                weak_model="x",
                endpoints=_registry(failing),
                resilience_config=config,
                cache_config=CacheConfig(enabled=False),
            )
            with pytest.raises(Exception):
                _ask(controller, [{"role": "user", "content": "hi"}], route=TIER)
            assert len(failing.hits) == retries + 1
        finally:
            failing.close()


def _registry(fake, **endpoint):
    from routellm.endpoints import EndpointRegistry

    os.environ["FAKE_UPSTREAM_KEY"] = "sk-fake"

    return EndpointRegistry.from_config(
        {
            "endpoints": {
                "e": {
                    "model": "openai/m",
                    "api_base": fake.base,
                    "api_key_env": "FAKE_UPSTREAM_KEY",
                    "extra": {"max_retries": 0},
                    **endpoint,
                }
            },
            "tiers": {
                "default": {"router": "random", "threshold": 0.5, "strong": "e", "weak": "e"}
            },
        }
    )


def test_per_endpoint_timeout_overrides_the_global_one():
    slow = Upstream(delay=1.5)
    try:
        controller = Controller(
            routers=["random"],
            strong_model="x",
            weak_model="x",
            endpoints=_registry(slow, timeout_ms=300),
            resilience_config=ResilienceConfig(timeout_ms=30000, max_retries=0),
            cache_config=CacheConfig(enabled=False),
        )
        started = time.monotonic()
        with pytest.raises(Exception):
            _ask(controller, [{"role": "user", "content": "hi"}], route=TIER)
        assert time.monotonic() - started < 1.2
        assert len(slow.hits) >= 1
    finally:
        slow.close()


def test_server_keys_are_not_handed_to_the_routers(tmp_path):
    from routellm.tests.test_openai_server_models import _run

    body = (
        "print(json.dumps(server.build_router_config("
        "{'cache': {'enabled': True}, 'resilience': {'max_retries': 1}, 'mf': {'a': 1}})))\n"
    )
    assert _run(tmp_path, ["openai_server", "--routers", "random"], body) == {"mf": {"a": 1}}


# ----------------------------------------------------- the whole server


def test_server_default_config_makes_two_upstream_calls(tmp_path):
    from routellm.tests.test_openai_server_models import _run

    fake = Upstream()
    try:
        config = tmp_path / "config.yaml"
        config.write_text(
            "endpoints:\n"
            f"  e: {{model: openai/m, api_base: '{fake.base}', api_key_env: FAKE_UPSTREAM_KEY, extra: {{max_retries: 0}}}}\n"
            "tiers:\n"
            "  default: {strong: e, weak: e, router: random, threshold: 0.5}\n"
        )
        body = (
            "msgs = lambda a: [{'role': 'user', 'content': a}, {'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'continue'}]\n"
            "import litellm\n"
            "controller.acompletion = litellm.acompletion\n"
            "with TestClient(server.app) as c:\n"
            "    for a in ('ONE', 'TWO'):\n"
            "        r = c.post('/v1/chat/completions', json={'model': 'default', 'messages': msgs(a)})\n"
            "        assert r.status_code == 200, r.text\n"
            "    print(json.dumps({'enabled': server.CONTROLLER.cache.config.enabled}))\n"
        )
        out = _run(
            tmp_path,
            ["openai_server", "--config", str(config), "--routers", "random"],
            body,
            env={"FAKE_UPSTREAM_KEY": "sk-fake"},
            cwd=tmp_path,
        )
        assert out == {"enabled": False}
        assert len(fake.hits) == 2
    finally:
        fake.close()
