"""Decision step wired into config load and the Controller."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from routellm.controller import Controller, RoutingError
from routellm.decision import build_decision, current_decision
from routellm.endpoints import EndpointRegistry
from routellm.resilience import ResilienceConfig
from routellm.tests.decision_fakes import CLASSES, INTENTS, answer

ENDPOINTS = {
    "decider": {
        "model": "ollama_chat/tiny",
        "api_base": "http://127.0.0.1:11500",
        "tags": ["local"],
    },
    "cloud_decider": {"model": "gpt-4o", "tags": ["cloud"]},
    "big": {"model": "gpt-4o"},
    "small": {
        "model": "ollama_chat/small",
        "api_base": "http://127.0.0.1:11500",
        "tags": ["local"],
    },
}


def section(**over):
    return {
        "endpoint": "decider",
        "require_tags": ["local"],
        "data_classes": CLASSES,
        "intents": INTENTS,
        "timeout_ms": 500,
        "min_confidence": 0.5,
        **over,
    }


def registry():
    return EndpointRegistry.from_config(
        {
            "endpoints": ENDPOINTS,
            "tiers": {
                "default": {"router": "random", "threshold": 0.0, "strong": "big", "weak": "small"}
            },
        }
    )


def build(**over):
    return build_decision({"decision": section(**over)}, registry())


# -- load -------------------------------------------------------------------


def test_missing_required_tag_is_a_load_error_naming_endpoint_and_tag():
    with pytest.raises(ValueError) as info:
        build(endpoint="cloud_decider")
    msg = str(info.value)
    assert "cloud_decider" in msg and "'local'" in msg


def test_required_tag_present_loads():
    assert build().endpoint == "decider"


def test_no_decision_section_builds_nothing():
    assert build_decision({}, registry()) is None
    assert build_decision(None, registry()) is None


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"endpoint": "nope"}, "not a registry endpoint"),
        ({"endpoint": "default"}, "is a tier"),
        ({"provider": "mystery"}, "unknown 'mystery'"),
        ({"on_error": "shrug"}, "on_error"),
        ({"data_classes": {}}, "at least one"),
        (
            {"recognizers": [{"name": "r", "type": "regex", "class": "zzz", "patterns": ["a"]}]},
            "zzz",
        ),
        (
            {
                "recognizers": [
                    {"name": "r", "type": "regex", "class": "private", "patterns": ["("]}
                ]
            },
            "bad pattern",
        ),
        ({"recognizers": [{"name": "r", "type": "wat", "class": "private"}]}, "unknown type"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_bad_config_is_a_load_error(over, needle):
    with pytest.raises(ValueError, match=needle):
        build(**over)


def test_decision_key_is_not_router_config(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["routellm.openai_server"])
    from routellm.openai_server import build_router_config

    assert build_router_config({"decision": {"x": 1}, "other": 2}) == {"other": 2}


# -- controller ---------------------------------------------------------------


def controller(decision, reply=None, delay=0.0):
    """A Controller whose every litellm call is a scripted fake."""
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        if delay and kwargs["model"] == "ollama_chat/tiny":
            await asyncio.sleep(delay)
        if kwargs["model"] == "ollama_chat/tiny":
            content = reply(kwargs) if callable(reply) else (reply or answer())
            if isinstance(content, Exception):
                raise content
        else:
            content = "served"
        res = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            _hidden_params={},
            model_dump=lambda: {"choices": [{"message": {"content": content}}]},
        )
        return res

    ctrl = Controller(
        routers=["random"],
        default_router="random",
        endpoints=registry(),
        decision=decision,
        progress_bar=False,
        cache_config=None,
        resilience_config=ResilienceConfig(max_retries=0),
    )
    return ctrl, calls, fake


def decider_calls(calls):
    return [c for c in calls if c["model"] == "ollama_chat/tiny"]


MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def patched(monkeypatch):
    def apply(fake):
        monkeypatch.setattr("routellm.controller.acompletion", fake)

    return apply


def test_decision_lands_on_path_and_context(patched):
    ctrl, calls, fake = controller(build(), answer("internal", "code", 0.9))
    patched(fake)
    res = asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    path = res._hidden_params["routellm_path"]
    assert path[0]["decision"] == {
        "data_class": "internal",
        "source": "model",
        "intent": "code",
        "confidence": 0.9,
    }
    sent = decider_calls(calls)[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["api_base"] == "http://127.0.0.1:11500"
    assert current_decision() is None or current_decision().data_class in CLASSES


def test_context_var_visible_to_routing(patched, monkeypatch):
    seen = []
    ctrl, _calls, fake = controller(build(), answer("private", "code"))
    orig = ctrl._route

    def spy(*a, **k):
        seen.append(current_decision())
        return orig(*a, **k)

    ctrl._route = spy
    patched(fake)
    asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    assert seen[0].data_class == "private"


def test_no_decision_section_makes_no_decision_call(patched):
    ctrl, calls, fake = controller(None)
    patched(fake)
    res = asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    assert decider_calls(calls) == []
    assert "decision" not in res._hidden_params["routellm_path"][0]


def test_timeout_resolves_most_restrictive_in_path(patched):
    ctrl, _calls, fake = controller(build(timeout_ms=50), delay=1.0)
    patched(fake)
    res = asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    rec = res._hidden_params["routellm_path"][0]["decision"]
    assert (rec["data_class"], rec["source"], rec["error"]) == ("private", "error", "timeout")


def test_transport_error_recorded(patched):
    ctrl, _calls, fake = controller(build(), ConnectionError("down"))
    patched(fake)
    res = asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    assert res._hidden_params["routellm_path"][0]["decision"]["error"] == "transport"


def test_refuse_becomes_routing_error(patched):
    ctrl, calls, fake = controller(build(on_error="refuse"), ConnectionError("down"))
    patched(fake)
    with pytest.raises(RoutingError, match="on_error is refuse"):
        asyncio.run(ctrl.acompletion(model="router-default", messages=MSG))
    assert [c for c in calls if c["model"] != "ollama_chat/tiny"] == []


def test_recognizer_hit_makes_no_decision_call(patched):
    rec = [{"name": "k", "type": "markers", "class": "private", "markers": ["BEGIN PRIVATE KEY"]}]
    ctrl, calls, fake = controller(build(recognizers=rec))
    patched(fake)
    res = asyncio.run(
        ctrl.acompletion(
            model="router-default",
            messages=[{"role": "user", "content": "-----BEGIN PRIVATE KEY-----"}],
        )
    )
    assert decider_calls(calls) == []
    assert res._hidden_params["routellm_path"][0]["decision"]["source"] == "recognizer"


def test_declared_class_skips_model_and_is_not_sent_upstream(patched):
    ctrl, calls, fake = controller(build())
    patched(fake)
    asyncio.run(ctrl.acompletion(model="router-default", messages=MSG, declared_class="private"))
    assert decider_calls(calls) == []
    assert all("declared_class" not in c for c in calls)


def test_slow_decision_does_not_delay_unrelated_request(patched):
    clf = build(timeout_ms=3000)
    ctrl, _calls, fake = controller(clf, delay=0.4)
    plain, _plain_calls, _ = controller(None)
    patched(fake)

    async def scenario():
        started = time.perf_counter()
        slow = asyncio.create_task(ctrl.acompletion(model="router-default", messages=MSG))
        await asyncio.sleep(0.02)
        await plain.acompletion(model="router-default", messages=MSG)
        unrelated = time.perf_counter() - started
        await slow
        return unrelated

    assert asyncio.run(scenario()) < 0.3


def test_sync_completion_runs_the_decision_step(monkeypatch):
    ctrl, _calls, fake = controller(build(), answer("internal", "chat"))
    results = []

    def sync_fake(**kwargs):
        results.append(kwargs)
        content = answer("internal", "chat") if kwargs["model"] == "ollama_chat/tiny" else "ok"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            _hidden_params={},
            model_dump=lambda: {},
        )

    monkeypatch.setattr("routellm.controller.acompletion", fake)
    monkeypatch.setattr("routellm.controller.completion", sync_fake)
    res = ctrl.completion(model="router-default", messages=MSG)
    assert res._hidden_params["routellm_path"][0]["decision"]["data_class"] == "internal"
    assert json.loads(answer())  # sanity
