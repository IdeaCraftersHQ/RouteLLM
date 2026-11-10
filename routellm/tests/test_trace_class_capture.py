"""`traces.capture_content` never stores text for a class that is not `raw`.

With a policy loaded, a trace keeps digests only when the request's
effective class treats data as `redact`, `block` or `vault`, or is not in
the policy. A `raw` class keeps the opt-in; no policy keeps it as it was.
"""

import asyncio
import json
from pathlib import Path

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.policy import parse_policy
from routellm.quality import FineTuneConfig, QualityManager
from routellm.requirements import Requirements
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import build_comparison_config
from routellm.tests.fake_upstream import FakeUpstream

CANARY = "CANARY-PROMPT-5d21"
MESSAGES = [{"role": "user", "content": CANARY}]


class _PinStrong:
    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()

POLICY = parse_policy(
    {
        "default_class": "open",
        "classes": {
            "open": {
                "treatment": "raw",
                "reach": ["local"],
                "tiers": ["default"],
                "cache": "exact",
            },
            "masked": {"treatment": "redact", "reach": ["local"], "tiers": ["default"]},
            "sealed": {"treatment": "block", "reach": ["local"], "tiers": ["default"]},
            "gone": {"treatment": "vault"},
        },
    },
    "p.yaml",
)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")


def _response(text="reply"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": {}}


def _manager(tmp_path, policy=POLICY, capture=True):
    manager = QualityManager(
        fine_tune_config=FineTuneConfig(
            enabled=True, trace_dir=str(tmp_path / "traces"), capture_content=capture
        )
    )
    manager.policy = policy
    return manager


def _traces(tmp_path):
    return [json.loads(p.read_text()) for p in sorted(Path(tmp_path / "traces").glob("*.json"))]


def _text(tmp_path):
    return "".join(p.read_text() for p in Path(tmp_path / "traces").glob("*.json"))


# -- the gate -----------------------------------------------------------------


@pytest.mark.parametrize(
    "ctx, captured",
    [
        ("open", True),
        ("masked", False),
        ("sealed", False),
        ("gone", False),
        ("not-a-class", False),
        (None, False),
        (Requirements(data_class="open"), True),
        (Requirements(data_class="masked"), False),
        (Requirements(), False),
    ],
)
def test_gate_by_effective_class(tmp_path, ctx, captured):
    manager = _manager(tmp_path)
    manager.record_trace(CANARY, "m", _response(), request_ctx=ctx)
    (trace,) = _traces(tmp_path)
    assert trace["metadata"]["content_captured"] is captured
    assert (CANARY in _text(tmp_path)) is captured
    if captured:
        assert "content_withheld_reason" not in trace["metadata"]
    else:
        assert trace["metadata"]["content_withheld_reason"]
        assert trace["input"]["prompt_sha256"]
        assert "output" not in trace


def test_reason_names_the_treatment_or_the_unknown_class(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace(CANARY, "m", _response(), request_ctx="masked")
    manager.record_trace(CANARY, "m", _response(), request_ctx=None)
    reasons = [t["metadata"]["content_withheld_reason"] for t in _traces(tmp_path)]
    assert any("redact" in r for r in reasons)
    assert any("unknown" in r for r in reasons)


def test_no_policy_is_unchanged(tmp_path):
    manager = _manager(tmp_path, policy=None)
    manager.record_trace(CANARY, "m", _response(), request_ctx=None)
    manager.record_trace(CANARY, "m", _response(), request_ctx="anything")
    assert all(t["metadata"]["content_captured"] is True for t in _traces(tmp_path))


def test_capture_off_gives_no_reason(tmp_path):
    manager = _manager(tmp_path, capture=False)
    manager.record_trace(CANARY, "m", _response(), request_ctx="masked")
    (trace,) = _traces(tmp_path)
    assert trace["metadata"]["content_captured"] is False
    assert "content_withheld_reason" not in trace["metadata"]


# -- the request path ----------------------------------------------------------


@pytest.fixture
def ups():
    made = {name: FakeUpstream(name) for name in "abc"}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def _controller(ups, tmp_path, *, policy=POLICY, cache=False):
    config = {
        "endpoints": {
            name: {
                "model": f"openai/model-{name}",
                "api_base": ups[name].base,
                "api_key_env": "FAKE_KEY",
                "tags": ["local"],
            }
            for name in ups
        },
        "tiers": {
            "default": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "b"}
        },
    }
    kwargs = {"policy": policy, "strict_registry": True} if policy else {}
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(config),
        cache_config=CacheConfig(enabled=cache, db_path=str(tmp_path / "cache.db")),
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(
                enabled=True, trace_dir=str(tmp_path / "traces"), capture_content=True
            )
        ),
        comparison_config=build_comparison_config({"comparison": {"enabled": True}}),
        **kwargs,
    )
    controller.resilience.config.max_retries = 0
    return controller


def _ask(controller, declared, **kwargs):
    async def go():
        res = await controller.acompletion(
            model="default", messages=MESSAGES, declared_class=declared, **kwargs
        )
        if kwargs.get("stream"):
            [chunk async for chunk in res]
        await controller.comparison.drain()
        return res

    return asyncio.run(go())


@pytest.mark.parametrize(
    "declared, captured", [("open", True), ("masked", False), ("sealed", False)]
)
def test_primary_and_shadow_traces_follow_the_class(ups, tmp_path, declared, captured):
    controller = _controller(ups, tmp_path)
    _ask(controller, declared, compare="b")

    rows = _traces(tmp_path)
    roles = {(t["routellm"].get("comparison") or {}).get("role") for t in rows}
    assert {None, "shadow"} <= roles or {"primary", "shadow"} <= roles
    assert len(rows) >= 2
    for trace in rows:
        assert trace["metadata"]["content_captured"] is captured
    assert (CANARY in _text(tmp_path)) is captured


def test_streamed_trace_follows_the_class(ups, tmp_path):
    controller = _controller(ups, tmp_path)
    _ask(controller, "masked", stream=True)
    rows = _traces(tmp_path)
    assert rows
    assert all(t["metadata"]["content_captured"] is False for t in rows)
    assert CANARY not in _text(tmp_path)


def test_streamed_trace_of_a_raw_class_keeps_text(ups, tmp_path):
    controller = _controller(ups, tmp_path)
    _ask(controller, "open", stream=True)
    rows = _traces(tmp_path)
    assert rows
    assert all(t["metadata"]["content_captured"] is True for t in rows)
    assert CANARY in _text(tmp_path)


def test_cached_hit_of_a_raw_class_keeps_text(ups, tmp_path):
    # A non-raw class never reads or writes the cache, so only `raw` hits.
    controller = _controller(ups, tmp_path, cache=True)
    _ask(controller, "open")
    _ask(controller, "open")
    rows = _traces(tmp_path)
    assert any(t["metadata"].get("cached") for t in rows)
    assert all(t["metadata"]["content_captured"] is True for t in rows)


def test_sync_path_follows_the_class(ups, tmp_path):
    controller = _controller(ups, tmp_path)
    controller.completion(model="default", messages=MESSAGES, declared_class="sealed")
    (trace,) = _traces(tmp_path)
    assert trace["metadata"]["content_captured"] is False
    assert CANARY not in _text(tmp_path)


def test_sync_path_of_a_raw_class_keeps_text_cached_or_not(ups, tmp_path):
    controller = _controller(ups, tmp_path, cache=True)
    for _ in range(2):
        controller.completion(model="default", messages=MESSAGES, declared_class="open")
    rows = _traces(tmp_path)
    assert any(t["metadata"].get("cached") for t in rows) and len(rows) == 2
    assert all(t["metadata"]["content_captured"] is True for t in rows)


def test_no_policy_keeps_capturing(ups, tmp_path):
    controller = _controller(ups, tmp_path, policy=None)
    _ask(controller, None)
    assert all(t["metadata"]["content_captured"] is True for t in _traces(tmp_path))
    assert CANARY in _text(tmp_path)
