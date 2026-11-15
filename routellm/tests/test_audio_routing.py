"""Tests for audio input: detection, capability, forcing and refusal.

An `input_audio` (or `audio_url`) content part is a requirement like an
image is. Only an endpoint that DECLARES audio serves it: unknown is a
refusal here, even when `strict` is off, because a text-only leaf would
drop the part and answer a question nobody asked. Native Ollama
transports (`ollama_chat/`, `ollama/`) cannot carry the part through
litellm and never count as audio capable.

Routers are stubs and every upstream is a mock; no network is touched.
"""

from typing import ClassVar

import pytest

import routellm.controller
from routellm import requirements as requirements_module
from routellm.caching import CacheConfig
from routellm.capabilities import Capabilities, build_tier_index, merge, satisfies
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.pairing import ModelRecord
from routellm.requirements import Requirements, derive
from routellm.routers.base import Router

CONFIG = {
    "endpoints": {
        "hearing": {
            "model": "openai/audio-model",
            "api_base": "http://127.0.0.1:1/v1",
            "capabilities": {"audio": True, "vision": True},
        },
        "hearing_blind": {
            "model": "openai/audio-model-2",
            "api_base": "http://127.0.0.1:1/v1",
            "capabilities": {"modalities_in": ["text", "audio"], "vision": False},
        },
        "deaf": {
            "model": "openai/text-model",
            "api_base": "http://127.0.0.1:1/v1",
            "capabilities": {"modalities_in": ["text"], "vision": True},
        },
        "silent": {
            "model": "openai/other-model",
            "api_base": "http://127.0.0.1:1/v1",
        },
    },
}


def _tiers(strong, weak):
    return {
        **CONFIG,
        "tiers": {"default": {"router": "hi", "threshold": 0.12, "strong": strong, "weak": weak}},
    }


def _audio(part_type="input_audio", extra=None):
    part = {"type": part_type, part_type: {"data": "QUJD", "format": "wav"}}
    content = [{"type": "text", "text": "transcribe this"}, part]
    if extra:
        content.append(extra)
    return [{"role": "user", "content": content}]


def _image():
    return {"type": "image_url", "image_url": {"url": "https://x/y.png"}}


class _Stub(Router):
    calls: ClassVar[list] = []

    def calculate_strong_win_rate(self, prompt):
        type(self).calls.append(prompt)
        return 0.9


@pytest.fixture
def hi_router(monkeypatch):
    class Stub(_Stub):
        calls: ClassVar[list] = []

    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "hi", lambda **kw: Stub())
    return Stub


@pytest.fixture(autouse=True)
def _no_catalog(monkeypatch, tmp_path):
    from routellm import pairing

    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])


def _route(config, messages, tmp_path):
    controller = Controller(
        routers=["hi"],
        strong_model=None,
        weak_model=None,
        endpoints=EndpointRegistry.from_config(config),
        cache_config=CacheConfig(enabled=False),
    )
    request = {"model": "default", "messages": messages}
    tier, router, threshold = controller._parse_model_name("default")
    prompt = requirements_module._prompt_text(messages)
    reqs = derive(messages, request, "default")
    return controller._route(prompt, request, tier, router, threshold, reqs)


# -- detection ---------------------------------------------------------------


@pytest.mark.parametrize("part_type", ["input_audio", "audio_url"])
def test_audio_part_sets_audio(part_type):
    reqs = derive(_audio(part_type), {}, "default")

    assert reqs.audio is True
    assert reqs.vision is False
    assert reqs.is_empty() is False


def test_image_and_audio_set_both():
    reqs = derive(_audio(extra=_image()), {}, "default")

    assert reqs.audio is True
    assert reqs.vision is True


def test_text_only_request_is_unchanged():
    reqs = derive([{"role": "user", "content": "say input_audio"}], {}, "default")

    assert reqs.audio is False
    assert Requirements(context_needed=None).is_empty()


def test_audio_part_does_not_break_the_token_estimate():
    """litellm's counter rejects `input_audio`; the estimate must survive."""
    big = "A" * 400_000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "input_audio", "input_audio": {"data": big, "format": "wav"}},
            ],
        }
    ]

    reqs = derive(messages, {"max_tokens": 10}, "default")

    assert reqs.context_needed is not None
    # The base64 payload is skipped, not counted as text.
    assert reqs.context_needed < 1000


# -- capability --------------------------------------------------------------


def test_modalities_in_audio_means_audio():
    assert Capabilities(modalities_in=["text", "audio"]).audio is True
    assert Capabilities(modalities_in=["text"]).audio is False
    assert Capabilities().audio is None


def test_catalog_record_audio_modality():
    record = ModelRecord(provider="x", id="y", modalities_in=["text", "audio"])

    assert merge(None, record, None).audio is True


def test_unknown_audio_refuses_even_when_not_strict():
    reqs = Requirements(audio=True)

    assert satisfies(Capabilities(), reqs, False, "a") == "audio"
    assert satisfies(Capabilities(audio=False), reqs, False, "a") == "audio"
    assert satisfies(Capabilities(audio=True), reqs, False, "a") is None


def test_unknown_vision_still_serves_when_not_strict():
    assert satisfies(Capabilities(), Requirements(vision=True), False, "a") is None


def test_tier_union_carries_audio():
    index = build_tier_index(EndpointRegistry.from_config(_tiers("hearing", "deaf")), {})

    assert index["default"].audio is True


@pytest.mark.parametrize("model", ["ollama_chat/m", "ollama/m"])
def test_native_ollama_transport_is_never_audio_capable(model, caplog):
    config = {"endpoints": {"x": {"model": model, "capabilities": {"audio": True}}}}
    registry = EndpointRegistry.from_config(config)
    index = build_tier_index(
        EndpointRegistry.from_config(
            {
                **config,
                "tiers": {"t": {"router": "hi", "threshold": 0.5, "strong": "x", "weak": "x"}},
            }
        ),
        {},
    )

    assert registry is not None
    assert index["t"].audio is False


# -- routing -----------------------------------------------------------------


def test_audio_forces_the_one_audio_side_without_a_router_call(hi_router, tmp_path):
    picked, path, _ = _route(_tiers("deaf", "hearing"), _audio(), tmp_path)

    assert picked == "hearing"
    assert hi_router.calls == []
    assert path[0]["capability_forced"] == "weak"
    assert path[0]["capability_requirement"] == "audio"


def test_no_audio_side_raises_naming_audio(hi_router, tmp_path):
    with pytest.raises(RoutingError) as excinfo:
        _route(_tiers("deaf", "silent"), _audio(), tmp_path)

    assert "audio" in str(excinfo.value)
    assert hi_router.calls == []


def test_image_plus_audio_needs_a_side_with_both(hi_router, tmp_path):
    messages = _audio(extra=_image())

    picked, _, _ = _route(_tiers("hearing_blind", "hearing"), messages, tmp_path)
    assert picked == "hearing"

    with pytest.raises(RoutingError):
        _route(_tiers("hearing_blind", "deaf"), messages, tmp_path)


def test_text_request_routes_as_before(hi_router, tmp_path):
    picked, path, _ = _route(
        _tiers("deaf", "silent"), [{"role": "user", "content": "hi"}], tmp_path
    )

    assert picked == "deaf"
    assert len(hi_router.calls) == 1
    assert "capability_forced" not in path[0]


# -- server ------------------------------------------------------------------

_SERVER_BODY = """
import json, sys
from unittest.mock import AsyncMock, MagicMock
sys.argv = {argv!r}
import routellm.openai_server as server
from fastapi.testclient import TestClient

res = MagicMock()
res._hidden_params = {{}}
res.model_dump.return_value = {{"id": "c1", "choices": []}}
upstream = AsyncMock(return_value=res)
server.acompletion = upstream
import routellm.controller as controller
controller.acompletion = upstream

with TestClient(server.app) as client:
    reply = client.post("/v1/chat/completions", json={{
        "model": "default",
        "messages": [{{"role": "user", "content": [
            {{"type": "text", "text": "transcribe"}},
            {{"type": "input_audio", "input_audio": {{"data": "QUJD", "format": "wav"}}}},
        ]}}],
    }})
    print(json.dumps({{"status": reply.status_code, "body": reply.json(),
                      "upstream_calls": upstream.call_count}}))
"""


def _serve(tmp_path, endpoints):
    import json
    import subprocess
    import sys
    from pathlib import Path

    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n" + endpoints + "tiers:\n  default:\n    router: random\n"
        "    threshold: 0.5\n    strong: a\n    weak: b\n"
    )
    from routellm.pairing import write_snapshot

    # An empty fresh snapshot: the server must not fetch the catalog.
    write_snapshot(tmp_path / "catalog.json", [])
    argv = ["openai_server", "--config", str(config), "--routers", "random"]
    result = subprocess.run(
        [sys.executable, "-c", _SERVER_BODY.format(argv=argv)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ROUTELLM_CATALOG_CACHE": str(tmp_path / "catalog.json"),
        },
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_server_refuses_audio_with_400_and_no_upstream_call(tmp_path):
    payload = _serve(
        tmp_path,
        "  a:\n    model: openai/m1\n    capabilities: {modalities_in: [text]}\n"
        "  b:\n    model: ollama_chat/m2\n    capabilities: {audio: true}\n",
    )

    assert payload["status"] == 400
    assert payload["body"]["object"] == "error"
    assert "audio" in payload["body"]["message"]
    assert payload["upstream_calls"] == 0


def test_server_forwards_audio_to_the_audio_side(tmp_path):
    payload = _serve(
        tmp_path,
        "  a:\n    model: openai/m1\n    capabilities: {modalities_in: [text]}\n"
        "  b:\n    model: openai/m2\n    api_base: http://127.0.0.1:1/v1\n"
        "    capabilities: {audio: true}\n",
    )

    assert payload["status"] == 200
    assert payload["upstream_calls"] == 1


def test_token_counter_never_sees_an_audio_part(monkeypatch):
    """The real counter raises on `input_audio`; it must not be handed one."""

    def counter(model=None, messages=None):
        for message in messages:
            for part in message["content"]:
                if part.get("type") == "input_audio":
                    raise ValueError("Invalid content item type: input_audio")
        return 7

    monkeypatch.setattr(requirements_module.litellm, "token_counter", counter)

    reqs = derive(_audio(), {"max_tokens": 3}, "default")

    assert reqs.context_needed == 10
