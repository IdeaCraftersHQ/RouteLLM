"""Rate-limit-aware fallback and candidate pools, against local fakes.

Every upstream is a thread on 127.0.0.1 that answers 429 (with or
without `Retry-After`), a daily-limit 429, or an in-stream rate-limit
event, and counts what reaches it. Time is a fake clock set on the
resilience layer, so "10 s later" and "after the reset" cost nothing.
Nothing leaves the loopback interface.
"""

import asyncio
import email.utils

import pytest

from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.resilience import (
    EndpointsCooling,
    Resilience,
    ResilienceConfig,
    rate_limit_info,
)
from routellm.tests.fake_upstream import FakeUpstream

NOW = 1_800_000_000.0


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now


def _config(fakes, tier, **tiers):
    endpoints = {
        name: {
            "model": "openai/m",
            "api_base": fake.base,
            "api_key_env": "FAKE_KEY",
            **extra,
        }
        for name, (fake, extra) in fakes.items()
    }
    return {
        "endpoints": endpoints,
        "tiers": {"default": {"router": "random", "threshold": 0.0, **tier}, **tiers},
    }


def _controller(config, clock=None, **resilience):
    resilience.setdefault("rate_limit_cooldown", True)
    resilience.setdefault("max_retries", 2)
    resilience.setdefault("initial_backoff_ms", 1)
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(config),
        default_router="random",
        default_threshold=0.0,
        resilience_config=ResilienceConfig(**resilience),
        progress_bar=False,
    )
    controller.cache.config.enabled = False
    if clock is not None:
        controller.resilience.clock = clock
    return controller


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")


MESSAGES = [{"role": "user", "content": "hi"}]


def _ask(controller, **kwargs):
    return asyncio.run(controller.acompletion(model="default", messages=MESSAGES, **kwargs))


def _path(res):
    return res._hidden_params["routellm_path"]


# The suite's stand-in random router always takes the weak side, so the
# endpoint under test sits on `weak` and its sibling on `strong`.


def _walk(res):
    return _path(res)[-1].get("walk", [])


# --------------------------------------------------------- header parsing


class _Err(Exception):
    def __init__(self, headers=None, body=None, message="", status_code=429):
        super().__init__(message)
        self.status_code = status_code
        self.litellm_response_headers = headers
        self.body = body
        self.message = message


def _info(error, **overrides):
    return rate_limit_info(error, ResilienceConfig(rate_limit_cooldown=True, **overrides), NOW)


def test_retry_after_seconds_is_read_from_the_exception_headers():
    info = _info(_Err(headers={"retry-after": "30"}))
    assert info.until == NOW + 30 and info.daily is False


def test_retry_after_http_date():
    date = email.utils.formatdate(NOW + 90, usegmt=True)
    assert _info(_Err(headers={"Retry-After": date})).until == pytest.approx(NOW + 90, abs=1)


def test_reset_header_epoch_seconds_and_milliseconds():
    assert _info(_Err(headers={"x-ratelimit-reset": str(int(NOW + 120))})).until == NOW + 120
    assert _info(_Err(headers={"x-ratelimit-reset": str(int((NOW + 120) * 1000))})).until == (
        NOW + 120
    )


def test_reset_stated_in_the_body_metadata_is_read():
    body = {"message": "slow down", "metadata": {"headers": {}}}
    body["metadata"]["headers"]["X-RateLimit-Reset"] = str(int((NOW + 500) * 1000))
    assert _info(_Err(body=body)).until == NOW + 500


def test_a_reset_in_the_past_counts_as_unstated():
    assert _info(_Err(headers={"x-ratelimit-reset": str(int(NOW - 5))})).until == NOW + 60


def test_no_stated_reset_uses_the_configured_default():
    assert _info(_Err(), rate_limit_default_cooldown_s=45).until == NOW + 45


def test_daily_marker_in_the_message_uses_the_daily_default_and_a_stated_reset():
    err = _Err(message="Rate limit exceeded: free-models-per-day")
    info = _info(err, rate_limit_daily_cooldown_s=1800)
    assert info.daily is True and info.until == NOW + 1800
    err.litellm_response_headers = {"x-ratelimit-reset": str(int((NOW + 4000) * 1000))}
    assert _info(err).until == NOW + 4000


def test_a_stated_reset_is_capped():
    info = _info(_Err(headers={"retry-after": "999999"}), rate_limit_max_cooldown_s=600)
    assert info.until == NOW + 600


def test_original_exception_is_searched():
    inner = _Err(headers={"retry-after": "12"})
    outer = Exception("wrapped")
    outer.status_code = 429
    outer.original_exception = inner
    assert _info(outer).until == NOW + 12


# ------------------------------------------------------------- cooldown


def _pair(a_mode="429", **a_attrs):
    a, b = FakeUpstream("A", a_mode), FakeUpstream("B")
    for key, value in a_attrs.items():
        setattr(a, key, value)
    return a, b, _config({"a": (a, {}), "b": (b, {})}, {"strong": "b", "weak": "a"})


def test_429_with_retry_after_is_one_call_then_skipped_then_retried():
    a, b, config = _pair(retry_after="30")
    clock = Clock()
    controller = _controller(config, clock)

    res = _ask(controller)
    assert res.choices[0].message.content == "hello from B"
    assert len(a.hits) == 1  # no same-endpoint retry on a 429
    assert _walk(res)[0]["endpoint"] == "a"

    clock.now += 10
    res = _ask(controller)
    assert len(a.hits) == 1  # cooling: never called
    assert res.choices[0].message.content == "hello from B"
    skipped = _walk(res)[0]
    assert skipped["endpoint"] == "a" and skipped["outcome"] == "cooling"
    assert skipped["until"] == NOW + 30

    clock.now += 21  # past the reset
    _ask(controller)
    assert len(a.hits) == 2


def test_daily_limit_429_cools_until_the_stated_reset():
    a, b, config = _pair("429_daily", reset_ms=int((NOW + 3600) * 1000))
    clock = Clock()
    controller = _controller(config, clock)

    _ask(controller)
    assert len(a.hits) == 1

    clock.now += 3000
    _ask(controller)
    assert len(a.hits) == 1

    clock.now += 601
    _ask(controller)
    assert len(a.hits) == 2


def test_429_without_a_stated_reset_cools_for_the_default():
    a, b, config = _pair()
    clock = Clock()
    controller = _controller(config, clock, rate_limit_default_cooldown_s=45)

    _ask(controller)
    clock.now += 44
    _ask(controller)
    assert len(a.hits) == 1
    clock.now += 2
    _ask(controller)
    assert len(a.hits) == 2


def test_429_does_not_count_toward_the_breaker_when_cooling_is_on():
    a, b, config = _pair(retry_after="5")
    controller = _controller(config, Clock())
    _ask(controller)
    assert controller.resilience.get_cb("a").failures == 0


def test_cooldown_is_off_by_default_and_429_retries_in_place():
    a, b, config = _pair()
    controller = _controller(config, rate_limit_cooldown=False)
    _ask(controller)
    first = len(a.hits)
    assert first >= 3  # retried in place: one call plus max_retries, at least
    assert controller.resilience.cooling("a") is None
    _ask(controller)
    assert len(a.hits) >= first + 3  # nothing is remembered between requests


def test_sync_completion_skips_a_cooling_endpoint_too():
    a, b, config = _pair(retry_after="30")
    clock = Clock()
    controller = _controller(config, clock)
    controller.completion(model="default", messages=MESSAGES)
    controller.completion(model="default", messages=MESSAGES)
    assert len(a.hits) == 1


# ---------------------------------------------------------------- pools


def _pool(modes, weak_mode="ok", pool=3):
    fakes = {f"p{i}": FakeUpstream(f"P{i}", mode) for i, mode in enumerate(modes, 1)}
    weak = FakeUpstream("W", weak_mode)
    config = _config(
        {
            **{
                name: (fake, {"tags": ["pool"], "quality": 100 - i})
                for i, (name, fake) in enumerate(fakes.items())
            },
            "w": (weak, {"tags": ["spare"]}),
        },
        {
            "weak": {"select": "tag:pool", "order": "quality_desc", "pool": pool},
            "strong": "w",
        },
    )
    return fakes, weak, config


def test_pool_walks_candidates_before_leaving_the_side():
    fakes, weak, config = _pool(["429", "429", "ok"])
    controller = _controller(config, Clock())

    res = _ask(controller)
    assert res.choices[0].message.content == "hello from P3"
    assert [len(f.hits) for f in fakes.values()] == [1, 1, 1]
    assert len(weak.hits) == 0
    assert [(w["endpoint"], w["outcome"]) for w in _walk(res)] == [
        ("p1", "failed"),
        ("p2", "failed"),
        ("p3", "served"),
    ]

    res = _ask(controller)
    assert [len(f.hits) for f in fakes.values()] == [1, 1, 2]
    assert [(w["endpoint"], w["outcome"]) for w in _walk(res)] == [
        ("p1", "cooling"),
        ("p2", "cooling"),
        ("p3", "served"),
    ]


def test_pool_of_one_is_todays_behaviour():
    fakes, weak, config = _pool(["500", "ok", "ok"], pool=1)
    controller = _controller(config, Clock(), max_retries=0)
    res = _ask(controller)
    assert res.choices[0].message.content == "hello from W"  # sibling, not p2
    assert len(fakes["p2"].hits) == 0


def test_pool_then_sibling_when_the_whole_pool_fails():
    fakes, weak, config = _pool(["429", "429", "429"])
    controller = _controller(config, Clock())
    res = _ask(controller)
    assert res.choices[0].message.content == "hello from W"
    assert _path(res)[-1]["fallback_from"] == "w"


def test_all_cooling_error_names_every_endpoint():
    fakes, weak, config = _pool(["429", "429", "429"], weak_mode="429")
    controller = _controller(config, Clock())

    with pytest.raises(EndpointsCooling) as first:
        _ask(controller)
    assert {"p1", "p2", "p3", "w"} <= set(first.value.endpoints)
    before = [len(f.hits) for f in fakes.values()] + [len(weak.hits)]

    with pytest.raises(EndpointsCooling) as second:
        _ask(controller)
    assert {"p1", "p2", "p3", "w"} <= set(second.value.endpoints)
    for name in ("p1", "p2", "p3", "w"):
        assert name in str(second.value)
    assert [len(f.hits) for f in fakes.values()] + [len(weak.hits)] == before


def test_pool_member_that_cannot_serve_the_request_is_skipped():
    """Every pool candidate passes the same serve check as a sibling."""
    fakes, weak, config = _pool(["429", "ok", "ok"])
    config["endpoints"]["p2"]["capabilities"] = {"vision": False}
    image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    for fake_name in ("p1", "p3"):
        config["endpoints"][fake_name]["capabilities"] = {"vision": True}
    controller = _controller(config, Clock())
    res = asyncio.run(controller.acompletion(model="default", messages=image))
    assert len(fakes["p2"].hits) == 0
    assert res.choices[0].message.content == "hello from P3"


def test_local_tier_never_lists_an_openrouter_endpoint():
    local = FakeUpstream("L")
    cloud = FakeUpstream("C")
    config = _config(
        {
            "l1": (local, {"tags": ["local"], "quality": 50}),
            "l2": (local, {"tags": ["local"], "quality": 40}),
            "l3": (local, {"tags": ["local"], "quality": 30}),
            "or1": (cloud, {"tags": ["cloud", "openrouter", "free"], "quality": 99}),
        },
        {
            "strong": {"select": "tag:local", "pool": 4},
            "weak": {"select": "tag:local", "order": "quality_asc", "pool": 4},
        },
    )
    controller = _controller(config)
    tier = controller.endpoints.get_tier("default")
    listed = {tier.strong, tier.weak, *controller.endpoints.pool_for("default", "strong")}
    listed |= set(controller.endpoints.pool_for("default", "weak"))
    assert listed == {"l1", "l2", "l3"}


# ----------------------------------------------------- stream and breaker


def _consume(controller):
    async def go():
        stream = await controller.acompletion(model="default", messages=MESSAGES, stream=True)
        out = []
        try:
            async for chunk in stream:
                out.append(chunk)
        except Exception as exc:
            return out, exc
        return out, None

    return asyncio.run(go())


@pytest.mark.parametrize(("mode", "cooldown"), [("die_after", True), ("stream_error_after", False)])
def test_failure_after_the_first_chunk_counts_against_the_breaker(mode, cooldown):
    a, b, config = _pair(mode)
    controller = _controller(config, Clock(), rate_limit_cooldown=cooldown)
    cb = controller.resilience.get_cb("a")
    cb.failures = 2
    chunks, err = _consume(controller)
    assert err is not None and len(chunks) >= 1
    assert cb.failures == 3


def test_clean_stream_still_counts_as_success():
    a, b, config = _pair("ok")
    controller = _controller(config, Clock())
    cb = controller.resilience.get_cb("a")
    cb.failures = 2
    chunks, err = _consume(controller)
    assert err is None
    assert cb.failures == 0


def test_in_stream_rate_limit_event_cools_the_endpoint():
    a, b, config = _pair("stream_error_after")
    clock = Clock()
    controller = _controller(config, clock)
    _consume(controller)
    assert len(a.hits) == 1
    a.mode = "ok"
    chunks, err = _consume(controller)
    assert err is None
    assert len(a.hits) == 1  # still cooling: the next request skipped it
    assert "hello from B" in chunks[0].choices[0].delta.content


def test_in_stream_rate_limit_before_the_first_chunk_falls_to_the_sibling():
    a, b, config = _pair("stream_error_first")
    controller = _controller(config, Clock())
    chunks, err = _consume(controller)
    assert err is None
    assert "hello from B" in chunks[0].choices[0].delta.content
    assert controller.resilience.cooling("a") is not None


# --------------------------------------------------------- timeout retry


def test_timeout_retries_are_configurable():
    attempts = []

    async def count(retry_on_timeout):
        resilience = Resilience(
            ResilienceConfig(
                timeout_ms=50,
                max_retries=2,
                initial_backoff_ms=1,
                retry_on_timeout=retry_on_timeout,
            )
        )

        async def call():
            attempts.append(1)
            await asyncio.sleep(1)

        with pytest.raises(TimeoutError):
            await resilience.wrap_acompletion("slow", call)

    asyncio.run(count(True))
    assert len(attempts) == 3
    attempts.clear()
    asyncio.run(count(False))
    assert len(attempts) == 1
