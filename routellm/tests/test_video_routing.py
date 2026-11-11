"""Video through the controller and the server: preprocess-then-route.

The preprocessor and the LLM upstream are loopback fakes.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from routellm import requirements as requirements_module
from routellm.caching import CacheConfig
from routellm.capabilities import Capabilities, satisfies
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.requirements import Requirements, derive
from routellm.resilience import ResilienceConfig
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.fake_video import LEAK, FakeVideoService
from routellm.tests.subproc import SUBPROCESS_TIMEOUT
from routellm.video import VideoError

REPO_ROOT = Path(__file__).resolve().parents[2]


def vpart(raw=b"clip"):
    data = base64.b64encode(raw).decode()
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}}


def user(*parts):
    return [{"role": "user", "content": list(parts)}]


def text(value):
    return {"type": "text", "text": value}


@pytest.fixture
def svc():
    fake = FakeVideoService()
    yield fake
    fake.close()


@pytest.fixture
def llm():
    return FakeUpstream("llm")


# -- capability and requirements ---------------------------------------------


def test_video_is_a_requirement_only_a_leaf_listing_it_serves():
    reqs = Requirements(video=True)

    assert satisfies(Capabilities(modalities_in=["text", "video"]), reqs, False) is None
    assert satisfies(Capabilities(modalities_in=["text", "image"]), reqs, False) == "video"
    assert satisfies(Capabilities(), reqs, False) == "video"
    assert satisfies(Capabilities(vision=True), Requirements(vision=True), False) is None
    assert not Requirements(video=True).is_empty()


def test_the_token_estimate_skips_video_and_counts_frames_as_images(monkeypatch):
    seen = []

    def counter(model, messages):
        seen.append(messages)
        return 7

    monkeypatch.setattr(requirements_module.litellm, "token_counter", counter)
    reqs = derive(user(text("hi"), vpart(b"x" * 5000)), {}, "m")

    assert reqs.video and not reqs.vision
    assert "video" not in str(seen[0])
    frame = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}
    derived = derive(user(text("hi"), frame), {}, "m")
    assert derived.vision and "image_url" in str(seen[1])


# -- through the controller --------------------------------------------------


def _controller(
    tmp_path, llm, video, *, modalities=("text", "image"), cache=False, tier=None, vision=None
):
    os.environ["FAKE_UPSTREAM_KEY"] = "sk-fake"
    leaf = {
        "api_base": llm.base,
        "api_key_env": "FAKE_UPSTREAM_KEY",
        "extra": {"max_retries": 0},
        "capabilities": {"modalities_in": list(modalities)},
    }
    if vision is not None:
        leaf["capabilities"]["vision"] = vision
    spec = {
        "endpoints": {
            "a": {"model": "openai/model-a", **leaf},
            "b": {"model": "openai/model-b", **leaf},
        },
        "tiers": {
            "default": {
                "router": "random",
                "threshold": 0.0,
                "strong": "b",
                "weak": "a",
                **(tier or {}),
            }
        },
    }
    if video is not None:
        spec["video"] = video
    return Controller(
        routers=["random"],
        strong_model="x",
        weak_model="x",
        endpoints=EndpointRegistry.from_config(spec),
        resilience_config=ResilienceConfig(timeout_ms=30000, max_retries=0),
        cache_config=CacheConfig(enabled=cache, db_path=str(tmp_path / "cache.db")),
    )


def _ask(controller, messages, **kw):
    return asyncio.run(controller.acompletion(model="default", messages=messages, **kw))


@pytest.mark.slow
def test_a_vision_leaf_serves_the_preprocessed_video(tmp_path, llm, svc):
    controller = _controller(tmp_path, llm, {"api_base": svc.base})

    res = _ask(controller, user(text("what happens?"), vpart(b"clip")))

    assert res.choices[0].message.content == "hello from llm"
    (_, _, body) = llm.hits[0]
    assert "video_url" not in str(body) and base64.b64encode(b"clip").decode() not in str(body)
    parts = body["messages"][0]["content"]
    assert parts[0]["text"] == "what happens?"
    assert sum(p["type"] == "image_url" for p in parts) == 3
    assert any("said: clip" in p.get("text", "") for p in parts)


@pytest.mark.slow
def test_frames_need_a_vision_capable_leaf(tmp_path, llm, svc):
    controller = _controller(
        tmp_path, llm, {"api_base": svc.base}, modalities=("text",), vision=False
    )

    with pytest.raises(RoutingError, match="vision"):
        _ask(controller, user(vpart()))


@pytest.mark.slow
def test_streaming_requests_are_preprocessed_first(tmp_path, llm, svc):
    controller = _controller(tmp_path, llm, {"api_base": svc.base})

    async def go():
        stream = await controller.acompletion(model="default", messages=user(vpart()), stream=True)
        return [chunk async for chunk in stream]

    assert asyncio.run(go())
    assert len(svc.hits) == 1 and "image_url" in str(llm.hits[-1][2])


@pytest.mark.slow
def test_passthrough_forwards_the_part_only_to_a_video_leaf(tmp_path, llm, svc):
    blind = _controller(tmp_path, llm, {"mode": "passthrough"})
    with pytest.raises(RoutingError, match="video"):
        _ask(blind, user(vpart()))

    seeing = _controller(tmp_path, llm, {"mode": "passthrough"}, modalities=("text", "video"))
    _ask(seeing, user(vpart(b"raw")))

    assert svc.hits == []
    assert base64.b64encode(b"raw").decode() in str(llm.hits[-1][2])


@pytest.mark.slow
def test_refuse_and_a_missing_section_are_a_400_naming_the_section(tmp_path, llm):
    refusing = _controller(tmp_path, llm, {"mode": "refuse"})
    with pytest.raises(VideoError) as err:
        _ask(refusing, user(vpart()))
    assert err.value.status_code == 400 and err.value.code == "video_refused"

    absent = _controller(tmp_path, llm, None)
    with pytest.raises(VideoError, match="`video:`") as err:
        _ask(absent, user(vpart()))
    assert err.value.status_code == 400 and err.value.code == "video_not_configured"
    assert llm.hits == []


@pytest.mark.slow
def test_a_tier_override_picks_the_options(tmp_path, llm, svc):
    controller = _controller(
        tmp_path,
        llm,
        {"api_base": svc.base},
        tier={"video": {"layout": "contact_sheet", "max_frames": 3}},
    )

    _ask(controller, user(vpart()))

    assert svc.hits[0]["fields"]["layout"] == "contact_sheet"
    assert svc.hits[0]["fields"]["max_frames"] == "3"
    parts = llm.hits[0][2]["messages"][0]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 1


@pytest.mark.slow
def test_a_failed_preprocess_never_reaches_the_llm(tmp_path, llm):
    controller = _controller(
        tmp_path, llm, {"api_base": "http://127.0.0.1:1"}, modalities=("text", "image", "video")
    )

    with pytest.raises(VideoError):
        _ask(controller, user(vpart()))

    assert llm.hits == []


@pytest.mark.slow
def test_a_non_local_preprocessor_is_denied_before_upload(tmp_path, llm, svc):
    from routellm.policy import parse_policy

    policy = parse_policy(
        {
            "classes": {
                "c": {"treatment": "redact", "reach": ["local", "cloud"], "tiers": ["default"]}
            },
            "default_class": "c",
        },
        "p",
    )
    controller = _controller(tmp_path, llm, {"api_base": svc.base})
    controller.policy = policy

    from routellm.controller import PolicyDenied

    with pytest.raises(PolicyDenied, match="video:c:not_local"):
        _ask(controller, user(vpart()), declared_class="c")
    assert svc.hits == [] and llm.hits == []


@pytest.mark.slow
def test_the_cache_key_follows_the_frames(tmp_path, llm, svc):
    controller = _controller(tmp_path, llm, {"api_base": svc.base}, cache=True)

    _ask(controller, user(vpart(b"alpha")))
    _ask(controller, user(vpart(b"alpha")))
    assert len(llm.hits) == 1

    _ask(controller, user(vpart(b"bravo")))
    assert len(llm.hits) == 2 and len(svc.hits) == 3


# -- through the server ------------------------------------------------------

_SERVER = """
import base64, json, sys
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.fake_video import FakeVideoService
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

LLM = FakeUpstream("llm")
SVC = FakeVideoService()
leaf = {{"api_base": LLM.base, "api_key_env": "FAKE_KEY",
         "capabilities": {{"modalities_in": ["text", "image"]}}}}
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{"a": {{"model": "openai/model-a", **leaf}}, "b": {{"model": "openai/model-b", **leaf}}}},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "b", "weak": "a"}}}},
    "video": {{"api_base": SVC.base, "timeout_ms": 400, "max_video_bytes": 64}},
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient


def video(raw):
    return {{"type": "video_url", "video_url": {{"url": "data:video/mp4;base64," + base64.b64encode(raw).decode()}}}}


def ask(part, **extra):
    headers = extra.pop("headers", {{}})
    return client.post("/v1/chat/completions", headers=headers, json={{
        "model": "default", "messages": [{{"role": "user", "content": [part]}}], **extra}})


out = {{}}
with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000
    ok = ask(video(b"hello"), headers={{"X-RouteLLM-Transcribe-Language": "fr"}})
    out["ok"] = [ok.status_code, SVC.hits[-1]["fields"].get("language"), len(LLM.hits)]
    out["remote"] = ask({{"type": "video_url", "video_url": {{"url": "http://x.example/a.mp4"}}}}).status_code
    out["shape"] = ask({{"type": "video_url", "video_url": 3}}).status_code
    out["big"] = ask(video(b"x" * 200)).status_code
    SVC.mode = "400"
    out["refused"] = ask(video(b"hello")).json()
    SVC.mode = "500"
    out["err"] = ask(video(b"hello")).json() | {{"_status": 0}}
    out["err_status"] = ask(video(b"hello")).status_code
    SVC.mode = "ok"
    SVC.delay = 1.5
    slow = ask(video(b"hello"))
    out["slow"] = [slow.status_code, slow.json()]
    out["llm_hits"] = len(LLM.hits)
print("@@RESULT@@" + json.dumps(out))
"""


@pytest.mark.slow
def test_server_status_codes_codes_and_no_leaks(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _SERVER.format(config=str(tmp_path / "config.json"))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "FAKE_KEY": "sk-fake",
        },
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    out = json.loads(line.split("@@RESULT@@", 1)[1])

    assert out["ok"] == [200, "fr", 1]
    assert out["remote"] == 400 and out["shape"] == 400 and out["big"] == 413

    refused = out["refused"]
    assert refused["code"] == "ocr_unavailable" and "ocr_unavailable" in refused["message"]
    assert LEAK not in str(refused) and refused["request_id"] in refused["message"]

    assert out["err_status"] == 502 and LEAK not in str(out["err"])
    assert out["err"]["request_id"] in out["err"]["message"]

    status, body = out["slow"]
    assert status == 504 and body["code"] == "video_timeout"
    assert out["llm_hits"] == 1
