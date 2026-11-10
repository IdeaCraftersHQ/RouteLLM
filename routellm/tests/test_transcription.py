"""Transcribe-then-route: audio parts become transcript text before routing.

The transcriber and the LLM upstream are fakes on 127.0.0.1; nothing
leaves the machine. Unit tests drive `routellm.transcription` and the
registry lint directly; the `slow` ones run a real `Controller` (and, in
a subprocess, the server) against both fakes.
"""

import asyncio
import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from routellm import transcription
from routellm.caching import CacheConfig
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.resilience import ResilienceConfig
from routellm.tests.fake_transcriber import LEAK, FakeTranscriber
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.subproc import SUBPROCESS_TIMEOUT
from routellm.transcription import TranscriberConfig, TranscriptionError, format_transcript

REPO_ROOT = Path(__file__).resolve().parents[2]


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def audio_part(raw=b"hello", fmt="wav"):
    return {"type": "input_audio", "input_audio": {"data": b64(raw), "format": fmt}}


def user(*parts):
    return [{"role": "user", "content": list(parts)}]


def text(value):
    return {"type": "text", "text": value}


def config(base, **kw):
    return TranscriberConfig(api_base=base, **kw)


@pytest.fixture
def stt():
    fake = FakeTranscriber()
    yield fake
    fake.close()


# -- formatting --------------------------------------------------------------

RESULT = {
    "text": "ignored when segments exist",
    "segments": [
        {"start": 1.2, "end": 4.8, "text": " Hi there ", "speaker": "SPEAKER_07"},
        {"start": 5.0, "end": 6.0, "text": "Hello", "speaker": "SPEAKER_02"},
        {"start": 6.0, "end": 7.04, "text": "Again", "speaker": "SPEAKER_07"},
    ],
}


def test_default_format_labels_speakers_by_first_appearance():
    out = format_transcript(RESULT, TranscriberConfig(api_base="http://x/v1"))

    assert out.splitlines() == [
        "[00:01.2-00:04.8] Speaker 1: Hi there",
        "[00:05.0-00:06.0] Speaker 2: Hello",
        "[00:06.0-00:07.0] Speaker 1: Again",
    ]


def test_segments_without_speaker_carry_no_label():
    result = {"segments": [{"start": 0, "end": 2.5, "text": "alone"}]}

    out = format_transcript(result, TranscriberConfig(api_base="http://x/v1"))

    assert out == "[00:00.0-00:02.5] alone"


def test_mixed_speaker_and_unlabelled_segments():
    result = {
        "segments": [
            {"start": 0, "end": 1, "text": "a", "speaker": 0},
            {"start": 1, "end": 2, "text": "b", "speaker": None},
        ]
    }

    out = format_transcript(result, TranscriberConfig(api_base="http://x/v1"))

    assert out.splitlines() == ["[00:00.0-00:01.0] Speaker 1: a", "[00:01.0-00:02.0] b"]


def test_custom_templates_and_long_timestamps():
    cfg = TranscriberConfig(
        api_base="http://x/v1", format="{speaker} @{start}: {text}", format_no_speaker="{text}"
    )
    result = {"segments": [{"start": 3725.4, "end": 3730, "text": "late", "speaker": "a"}]}

    assert format_transcript(result, cfg) == "Speaker 1 @1:02:05.4: late"


def test_no_segments_falls_back_to_text_then_to_a_marker():
    cfg = TranscriberConfig(api_base="http://x/v1")

    assert format_transcript({"text": " bare "}, cfg) == "bare"
    assert format_transcript({"text": "", "segments": []}, cfg) == transcription.NO_SPEECH


def test_unknown_template_placeholder_is_rejected():
    with pytest.raises(ValueError, match="unknown placeholder"):
        TranscriberConfig(api_base="http://x/v1", format="{who}: {text}")


# -- config ------------------------------------------------------------------


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="extra|Extra"):
        EndpointRegistry.from_config(
            {"transcriber": {"api_base": "http://x/v1", "api_basee": "typo"}}
        )


def test_transcribe_mode_needs_a_base_but_passthrough_does_not():
    with pytest.raises(ValueError, match="api_base"):
        EndpointRegistry.from_config({"transcriber": {"mode": "transcribe"}})

    registry = EndpointRegistry.from_config({"transcriber": {"mode": "passthrough"}})
    assert registry.transcriber_for(None).active is False


def test_tier_overrides_global_key_by_key():
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m"}, "b": {"model": "n"}},
            "tiers": {
                "t": {"strong": "a", "weak": "b", "transcriber": {"language": "fr"}},
                "u": {"strong": "a", "weak": "b", "transcriber": {"mode": "passthrough"}},
            },
            "transcriber": {"api_base": "http://x/v1", "language": "en", "model": "w"},
        }
    )

    assert registry.transcriber_for("t").language == "fr"
    assert registry.transcriber_for("t").model == "w"
    assert registry.transcriber_for(None).language == "en"
    assert registry.transcriber_for("u").active is False


def _local_tiers(transcriber, tier_transcriber=None, tags=("local",), other_tags=("local",)):
    tier = {"strong": "a", "weak": "b"}
    if tier_transcriber is not None:
        tier["transcriber"] = tier_transcriber
    return {
        "endpoints": {
            "a": {"model": "m", "api_base": "http://127.0.0.1:11434/v1", "tags": list(tags)},
            "b": {"model": "n", "api_base": "http://127.0.0.1:11434/v1", "tags": list(other_tags)},
        },
        "tiers": {"private": tier},
        "transcriber": transcriber,
    }


def test_local_only_tier_rejects_a_non_local_transcriber():
    with pytest.raises(ValueError, match="local-only"):
        EndpointRegistry.from_config(_local_tiers({"api_base": "http://stt.example.com/v1"}))


def test_local_only_tier_accepts_a_local_transcriber():
    registry = EndpointRegistry.from_config(
        _local_tiers({"api_base": "http://127.0.0.1:9/v1", "tags": ["local"]})
    )

    assert registry.transcriber_for("private").is_local


def test_a_tier_override_cannot_smuggle_in_a_cloud_transcriber():
    spec = _local_tiers(
        {"api_base": "http://127.0.0.1:9/v1", "tags": ["local"]},
        {"api_base": "http://stt.example.com/v1", "tags": []},
    )

    with pytest.raises(ValueError, match="local-only"):
        EndpointRegistry.from_config(spec)


def test_the_rule_reaches_through_nested_tiers():
    spec = {
        "endpoints": {
            "a": {"model": "m", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
            "b": {"model": "n", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
        },
        "tiers": {
            "inner": {"strong": "a", "weak": "b"},
            "outer": {"strong": "inner", "weak": "b"},
        },
        "transcriber": {"api_base": "http://stt.example.com/v1"},
    }

    with pytest.raises(ValueError, match="local-only"):
        EndpointRegistry.from_config(spec)


def test_a_tier_with_a_cloud_endpoint_may_use_a_cloud_transcriber():
    spec = _local_tiers({"api_base": "http://stt.example.com/v1"}, other_tags=("cloud",))

    assert EndpointRegistry.from_config(spec).transcriber_for("private").active


def test_passthrough_is_fine_on_a_local_only_tier():
    spec = _local_tiers({"api_base": "http://stt.example.com/v1", "mode": "passthrough"})

    assert EndpointRegistry.from_config(spec).transcriber_for("private").active is False


# -- transcribe() against the fake ------------------------------------------


def run(messages, cfg, **kw):
    return asyncio.run(transcription.transcribe(messages, cfg, **kw))


def test_audio_is_replaced_by_the_formatted_transcript(stt):
    out = run(user(text("what was said?"), audio_part(b"hello")), config(stt.base))

    assert out == user(
        text("what was said?"),
        text("[00:01.2-00:04.8] Speaker 1: hello one\n[00:05.0-00:06.5] Speaker 2: hello two"),
    )
    assert not transcription.has_audio(out)


def test_the_request_to_the_transcriber(stt, monkeypatch):
    monkeypatch.setenv("STT_KEY", "k-1")

    run(user(audio_part(b"hello", "mp3")), config(stt.base, api_key_env="STT_KEY", model="m-1"))

    (hit,) = stt.hits
    assert hit["path"] == "/v1/audio/transcriptions"
    assert hit["headers"]["Authorization"] == "Bearer k-1"
    assert hit["fields"]["model"] == "m-1"
    assert hit["fields"]["response_format"] == "verbose_json"
    assert hit["fields"]["timestamp_granularities[]"] == "segment"
    assert hit["filename"] == "audio.mp3"
    assert hit["audio"] == b"hello"
    assert "language" not in hit["fields"]


def test_language_from_the_request_wins_over_the_config_default(stt):
    run(user(audio_part()), config(stt.base, language="en"), language="fr")
    run(user(audio_part()), config(stt.base, language="en"))
    run(user(audio_part()), config(stt.base))

    assert [h["fields"].get("language") for h in stt.hits] == ["fr", "en", None]


def test_a_bad_language_is_refused_before_any_call(stt):
    with pytest.raises(TranscriptionError) as err:
        run(user(audio_part()), config(stt.base), language="en; rm -rf")

    assert err.value.status_code == 400
    assert stt.hits == []


def test_multiple_audio_parts_each_become_a_part_in_place(stt):
    messages = [
        {"role": "user", "content": [audio_part(b"one"), text("and"), audio_part(b"two")]},
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [audio_part(b"three")]},
    ]

    out = run(messages, config(stt.base))

    assert [h["audio"] for h in stt.hits] == [b"one", b"two", b"three"]
    first = out[0]["content"]
    assert first[0]["text"].startswith("[00:01.2-00:04.8] Speaker 1: one")
    assert first[1] == text("and")
    assert first[2]["text"].startswith("[00:01.2-00:04.8] Speaker 1: two")
    assert out[1] == messages[1]
    assert out[2]["content"][0]["text"].startswith("[00:01.2-00:04.8] Speaker 1: three")
    # The caller's own messages are not touched.
    assert messages[0]["content"][0]["type"] == "input_audio"


def test_text_and_image_parts_survive_untouched(stt):
    image = {"type": "image_url", "image_url": {"url": "https://x/y.png"}}

    out = run(user(text("look"), image, audio_part()), config(stt.base))

    assert out[0]["content"][:2] == [text("look"), image]
    assert out[0]["content"][2]["type"] == "text"


def test_a_transcriber_without_speakers_gives_unlabelled_lines():
    fake = FakeTranscriber(
        answer=lambda f, n, a: {"segments": [{"start": 0.0, "end": 1.0, "text": "plain"}]}
    )
    try:
        out = run(user(audio_part()), config(fake.base))
    finally:
        fake.close()

    assert out[0]["content"][0]["text"] == "[00:00.0-00:01.0] plain"


def test_sync_form_matches_async(stt):
    messages = user(audio_part(b"x"))

    assert transcription.transcribe_sync(messages, config(stt.base)) == run(
        messages, config(stt.base)
    )


# -- fail closed -------------------------------------------------------------


def test_transcriber_down_is_a_502():
    with pytest.raises(TranscriptionError) as err:
        run(user(audio_part()), config("http://127.0.0.1:1/v1"))

    assert err.value.status_code == 502


def test_transcriber_500_is_a_502_that_leaks_nothing():
    fake = FakeTranscriber(mode="500")
    try:
        with pytest.raises(TranscriptionError) as err:
            run(user(audio_part()), config(fake.base))
    finally:
        fake.close()

    assert err.value.status_code == 502
    assert LEAK not in err.value.public
    assert fake.base not in err.value.public
    assert LEAK in err.value.detail  # kept for the server log


def test_unreadable_answer_is_a_502():
    fake = FakeTranscriber(mode="garbage")
    try:
        with pytest.raises(TranscriptionError) as err:
            run(user(audio_part()), config(fake.base))
    finally:
        fake.close()

    assert err.value.status_code == 502


def test_timeout_is_a_504(stt):
    stt.delay = 1.0

    with pytest.raises(TranscriptionError) as err:
        run(user(audio_part()), config(stt.base, timeout_ms=100))

    assert err.value.status_code == 504


def test_oversize_audio_is_a_413_and_never_sent(stt):
    with pytest.raises(TranscriptionError) as err:
        run(user(audio_part(b"x" * 100)), config(stt.base, max_audio_bytes=50))

    assert err.value.status_code == 413
    assert stt.hits == []


def test_audio_url_is_refused_by_default(stt):
    part = {"type": "audio_url", "audio_url": {"url": f"{stt.base}/media/a.wav"}}

    with pytest.raises(TranscriptionError) as err:
        run(user(part), config(stt.base))

    assert err.value.status_code == 400
    assert stt.hits == [] and stt.gets == []


def test_audio_url_is_fetched_only_when_allowed(stt):
    local = stt.base.replace("127.0.0.1", "localhost")
    part = {"type": "audio_url", "audio_url": {"url": f"{local}/media/a.wav"}}

    out = run(
        user(part),
        config(
            stt.base,
            allow_urls=True,
            allowed_url_hosts=["localhost"],
            allow_private_url_hosts=True,
        ),
    )

    assert stt.gets == ["/v1/media/a.wav"]
    assert stt.hits[0]["audio"] == b"fetched"
    assert stt.hits[0]["filename"] == "audio.wav"
    assert out[0]["content"][0]["type"] == "text"


def test_an_oversize_fetched_url_is_a_413(stt):
    stt.media = b"x" * 100
    local = stt.base.replace("127.0.0.1", "localhost")
    part = {"type": "audio_url", "audio_url": {"url": f"{local}/media/a.wav"}}

    with pytest.raises(TranscriptionError) as err:
        run(
            user(part),
            config(
                stt.base,
                allow_urls=True,
                max_audio_bytes=50,
                allowed_url_hosts=["localhost"],
                allow_private_url_hosts=True,
            ),
        )

    assert err.value.status_code == 413
    assert stt.hits == []


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_audio", "input_audio": {"data": "!!notb64", "format": "wav"}},
        {"type": "input_audio", "input_audio": {"data": "QUJD"}},
        {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "../x"}},
        {"type": "input_audio"},
    ],
)
def test_malformed_audio_parts_are_a_400(stt, part):
    with pytest.raises(TranscriptionError) as err:
        run(user(part), config(stt.base))

    assert err.value.status_code == 400
    assert stt.hits == []


# -- through the controller --------------------------------------------------


def _controller(tmp_path, llm, stt_base, *, transcriber=None, cache=False, audio_leaf=False):
    os.environ["FAKE_UPSTREAM_KEY"] = "sk-fake"
    caps = {"modalities_in": ["text", "audio"]} if audio_leaf else {"modalities_in": ["text"]}
    leaf = {
        "api_base": llm.base,
        "api_key_env": "FAKE_UPSTREAM_KEY",
        "extra": {"max_retries": 0},
        "capabilities": caps,
    }
    section = {"api_base": stt_base} if transcriber is None else transcriber
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "a": {"model": "openai/model-a", **leaf},
                "b": {"model": "openai/model-b", **leaf},
            },
            "tiers": {
                "default": {"router": "random", "threshold": 0.0, "strong": "b", "weak": "a"}
            },
            "transcriber": section,
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


def _ask(controller, messages, **kw):
    return asyncio.run(controller.acompletion(model="default", messages=messages, **kw))


@pytest.fixture
def llm():
    fake = FakeUpstream("llm")
    yield fake


pytestmark_slow = pytest.mark.slow


@pytest.mark.slow
def test_a_text_only_tier_serves_a_transcribed_audio_request(tmp_path, llm, stt):
    controller = _controller(tmp_path, llm, stt.base)

    res = _ask(controller, user(text("summarise"), audio_part(b"hello")))

    assert res.choices[0].message.content == "hello from llm"
    (_, _, body) = llm.hits[0]
    parts = body["messages"][0]["content"]
    assert "input_audio" not in str(body)
    assert b64(b"hello") not in str(body)
    assert any("Speaker 1: hello one" in p.get("text", "") for p in parts)


@pytest.mark.slow
def test_passthrough_mode_still_needs_an_audio_side(tmp_path, llm, stt):
    deaf = _controller(tmp_path, llm, stt.base, transcriber={"mode": "passthrough"})
    with pytest.raises(RoutingError, match="audio"):
        _ask(deaf, user(audio_part()))

    hearing = _controller(
        tmp_path, llm, stt.base, transcriber={"mode": "passthrough"}, audio_leaf=True
    )
    _ask(hearing, user(audio_part(b"raw")))

    assert stt.hits == []
    assert b64(b"raw") in str(llm.hits[-1][2])


@pytest.mark.slow
def test_streaming_requests_are_transcribed_first(tmp_path, llm, stt):
    controller = _controller(tmp_path, llm, stt.base)

    async def go():
        stream = await controller.acompletion(
            model="default", messages=user(audio_part(b"hello")), stream=True
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(go())

    assert chunks
    assert len(stt.hits) == 1
    assert "Speaker 1: hello one" in str(llm.hits[-1][2])


@pytest.mark.slow
def test_a_failed_transcription_never_reaches_the_llm(tmp_path, llm):
    controller = _controller(tmp_path, llm, "http://127.0.0.1:1/v1")

    with pytest.raises(TranscriptionError):
        _ask(controller, user(audio_part()))

    assert llm.hits == []


@pytest.mark.slow
def test_a_failed_transcription_never_falls_back_to_an_audio_side(tmp_path, llm):
    controller = _controller(tmp_path, llm, "http://127.0.0.1:1/v1", audio_leaf=True)

    with pytest.raises(TranscriptionError):
        _ask(controller, user(audio_part(b"raw")))

    assert llm.hits == []


@pytest.mark.slow
def test_the_cache_key_follows_the_transcript(tmp_path, llm, stt):
    controller = _controller(tmp_path, llm, stt.base, cache=True)

    _ask(controller, user(audio_part(b"alpha")))
    _ask(controller, user(audio_part(b"alpha")))
    assert len(llm.hits) == 1

    _ask(controller, user(audio_part(b"bravo")))
    assert len(llm.hits) == 2
    # Transcription runs before the cache is consulted, so a hit still
    # costs one transcription.
    assert len(stt.hits) == 3


@pytest.mark.slow
def test_info_logs_carry_a_summary_and_no_transcript(tmp_path, llm, stt, caplog):
    controller = _controller(tmp_path, llm, stt.base)

    with caplog.at_level(logging.INFO):
        _ask(controller, user(text("SECRET-QUESTION"), audio_part(b"SECRETWORDS")))

    log = caplog.text
    assert "SECRETWORDS" not in log
    assert b64(b"SECRETWORDS") not in log
    summary = next(r.getMessage() for r in caplog.records if "transcribed audio" in r.getMessage())
    assert "parts=1" in summary
    assert "seconds=6.5" in summary
    assert "segments=2" in summary
    assert "speakers=2" in summary
    assert "latency_ms=" in summary


# -- through the server ------------------------------------------------------

_SERVER = """
import json, logging, io, sys
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.fake_transcriber import FakeTranscriber
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

LLM = FakeUpstream("llm")
STT = FakeTranscriber()
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{
        "a": {{"model": "openai/model-a", "api_base": LLM.base, "api_key_env": "FAKE_KEY",
               "capabilities": {{"modalities_in": ["text"]}}}},
        "b": {{"model": "openai/model-b", "api_base": LLM.base, "api_key_env": "FAKE_KEY",
               "capabilities": {{"modalities_in": ["text"]}}}},
    }},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "b", "weak": "a"}}}},
    "transcriber": {{"api_base": STT.base, "timeout_ms": 400, "max_audio_bytes": 64}},
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient

buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
handler.addFilter(lambda r: r.name == "root" or r.name.startswith("routellm"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


def audio(raw):
    import base64
    return {{"type": "input_audio",
            "input_audio": {{"data": base64.b64encode(raw).decode(), "format": "wav"}}}}


def ask(raw, **extra):
    headers = extra.pop("headers", {{}})
    return client.post("/v1/chat/completions", headers=headers, json={{
        "model": "default", "messages": [{{"role": "user", "content": [audio(raw)]}}], **extra}})


out = {{}}
with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000  # starved CPU is not an upstream timeout
    ok = ask(b"hello", headers={{"X-RouteLLM-Transcribe-Language": "fr"}})
    out["ok"] = [ok.status_code, STT.hits[-1]["fields"].get("language"), len(LLM.hits)]

    stream = ask(b"hello", stream=True)
    out["stream"] = [stream.status_code, "[DONE]" in stream.text, len(LLM.hits)]

    big = ask(b"x" * 200)
    out["big"] = [big.status_code, big.json()]

    STT.mode = "500"
    err = ask(b"hello")
    out["err"] = [err.status_code, err.json()]

    STT.mode = "ok"
    STT.delay = 1.5
    slow = ask(b"hello")
    out["slow"] = [slow.status_code, slow.json()]

    out["llm_hits"] = len(LLM.hits)
    out["log"] = buffer.getvalue()
print("@@RESULT@@" + json.dumps(out))
"""


@pytest.mark.slow
def test_server_status_codes_request_ids_and_no_leaks(tmp_path):
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
    import json

    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    out = json.loads(line.split("@@RESULT@@", 1)[1])

    assert out["ok"] == [200, "fr", 1]
    assert out["stream"] == [200, True, 2]

    status, body = out["big"]
    assert status == 413 and "request id" in body["message"] and body["request_id"]

    status, body = out["err"]
    assert status == 502
    assert body["request_id"] in body["message"]
    assert LEAK not in str(body)

    status, body = out["slow"]
    assert status == 504 and body["request_id"]

    # Only the two good requests ever reached the LLM.
    assert out["llm_hits"] == 2
    assert LEAK in out["log"]  # detail stays server-side, under the same id
    assert "hello one" not in out["log"]


# -- intent re-routing ---------------------------------------------------------


def _intent_spec(transcriber, intents, intent_routing=None, tags_open=("cloud",)):
    open_tier = {"strong": "c", "weak": "d"}
    if intent_routing is not None:
        open_tier["intent_routing"] = intent_routing
    return {
        "endpoints": {
            "a": {"model": "m", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
            "b": {"model": "n", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
            "c": {"model": "o", "tags": list(tags_open)},
            "d": {"model": "p", "tags": list(tags_open)},
        },
        "tiers": {
            "private": {
                "strong": "a",
                "weak": "b",
                "transcriber": {"api_base": "http://127.0.0.1:9/v1", "tags": ["local"]},
            },
            "default": open_tier,
        },
        "transcriber": transcriber,
        "intents": {"tiers": intents},
    }


CLOUD_STT = {"api_base": "http://stt.example.com/v1"}


def test_a_tier_that_intent_routing_can_send_to_a_local_only_tier_needs_a_local_transcriber():
    # `default` accepts intent routing and an intent maps to the local-only
    # `private`; audio would be transcribed in the cloud before that move.
    spec = _intent_spec(CLOUD_STT, {"confidential": "private"})

    with pytest.raises(ValueError, match="intent"):
        EndpointRegistry.from_config(spec)


def test_the_same_config_is_fine_with_a_local_transcriber():
    spec = _intent_spec({**CLOUD_STT, "tags": ["local"]}, {"confidential": "private"})

    assert EndpointRegistry.from_config(spec).transcriber_for("default").is_local


def test_a_tier_that_does_not_accept_intent_routing_is_not_affected():
    spec = _intent_spec(CLOUD_STT, {"confidential": "private"}, intent_routing=False)

    assert EndpointRegistry.from_config(spec).transcriber_for("default").active


def test_intents_that_reach_no_local_only_tier_are_fine():
    spec = _intent_spec(CLOUD_STT, {"general": "default"})

    assert EndpointRegistry.from_config(spec).transcriber_for("default").active
