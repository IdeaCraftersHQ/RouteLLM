"""Advisory token estimate for audio parts.

`litellm.token_counter` cannot count audio, so the context estimate adds
a flat cost per second of audio: read off a WAV header, or guessed from
the byte length with conservative bitrates. A long clip then exceeds a
small context window, and a transcribed one is counted as its transcript
only.

Everything runs on in-memory fakes and 127.0.0.1; no network.
"""

import asyncio
import base64
import io
import os
import wave

import pytest

from routellm import requirements
from routellm.caching import CacheConfig
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.requirements import derive
from routellm.resilience import ResilienceConfig
from routellm.tests.fake_transcriber import FakeTranscriber
from routellm.tests.fake_upstream import FakeUpstream


def _wav(seconds, rate=8000, width=1, channels=1):
    """Return the bytes of a silent WAV `seconds` long."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(b"\x00" * int(seconds * rate * width * channels))
    return buf.getvalue()


def _part(raw, fmt="wav"):
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(raw).decode(), "format": fmt},
    }


def _msgs(*parts):
    return [{"role": "user", "content": list(parts)}]


def _audio_tokens(raw, fmt="wav", **kw):
    """Tokens the audio alone adds on top of the empty prompt."""
    base = derive(_msgs({"type": "text", "text": "x"}), {}, "default").context_needed
    got = derive(_msgs({"type": "text", "text": "x"}, _part(raw, fmt)), {}, "default", **kw)
    return got.context_needed - base


# -- duration ----------------------------------------------------------------


def test_wav_duration_comes_from_the_header_not_the_byte_length():
    # 8 kHz 8-bit mono is 8000 B/s; 16 kHz 16-bit stereo is 64000 B/s.
    slow = requirements.audio_seconds(_part(_wav(10, 8000, 1, 1)))
    fast = requirements.audio_seconds(_part(_wav(10, 16000, 2, 2)))

    assert slow == pytest.approx(10, abs=0.01)
    assert fast == pytest.approx(10, abs=0.01)


def test_wav_with_a_streaming_data_size_uses_the_bytes_that_follow():
    raw = bytearray(_wav(5, 8000, 1, 1))
    idx = raw.index(b"data")
    raw[idx + 4 : idx + 8] = b"\xff\xff\xff\xff"

    assert requirements.audio_seconds(_part(bytes(raw))) == pytest.approx(5, abs=0.01)


def test_compressed_audio_is_estimated_from_bytes_at_a_conservative_bitrate():
    # 40000 bytes of mp3 at the assumed 32 kbit/s (4000 B/s) is 10 s.
    assert requirements.audio_seconds(_part(b"\x00" * 40000, "mp3")) == pytest.approx(10)


def test_a_malformed_wav_falls_back_to_a_pcm_rate_over_the_bytes():
    seconds = requirements.audio_seconds(_part(b"not a riff header" * 1000, "wav"))

    assert seconds is not None and seconds > 0


def test_undecodable_base64_still_gives_a_length_based_estimate():
    part = {"type": "input_audio", "input_audio": {"data": "!!!" * 4000, "format": "mp3"}}

    assert requirements.audio_seconds(part) > 0


def test_a_part_with_no_payload_has_no_duration():
    assert requirements.audio_seconds({"type": "audio_url", "audio_url": {"url": "http://x"}}) == 0
    assert requirements.audio_seconds({"type": "input_audio", "input_audio": {}}) == 0


# -- tokens ------------------------------------------------------------------


def test_audio_adds_the_default_rate_per_second():
    got = _audio_tokens(_wav(10))

    assert got == round(10 * requirements.DEFAULT_AUDIO_TOKENS_PER_SECOND)


def test_the_rate_is_configurable_per_call():
    assert _audio_tokens(_wav(10), audio_tokens_per_second=100) == 1000


def test_every_audio_part_counts():
    base = derive(_msgs({"type": "text", "text": "x"}), {}, "d", audio_tokens_per_second=10)
    two = derive(
        _msgs({"type": "text", "text": "x"}, _part(_wav(10)), _part(_wav(5))),
        {},
        "d",
        audio_tokens_per_second=10,
    )

    assert two.context_needed - base.context_needed == 150


def test_the_estimate_survives_the_char_fallback_counter(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(requirements.litellm, "token_counter", boom)

    assert _audio_tokens(_wav(10), audio_tokens_per_second=100) == 1000


def test_the_request_itself_is_not_altered():
    messages = _msgs(_part(_wav(2)))
    before = repr(messages)

    derive(messages, {}, "d")

    assert repr(messages) == before


# -- config ------------------------------------------------------------------


def test_config_sets_the_rate_on_the_registry():
    registry = EndpointRegistry.from_config({"audio_tokens_per_second": 50})

    assert registry.audio_tokens_per_second == 50


def test_the_rate_defaults_when_the_config_omits_it():
    registry = EndpointRegistry.from_config({})

    assert registry.audio_tokens_per_second == requirements.DEFAULT_AUDIO_TOKENS_PER_SECOND


@pytest.mark.parametrize("bad", [0, -3, "fast", True])
def test_a_bad_rate_is_a_load_error(bad):
    with pytest.raises(ValueError, match="audio_tokens_per_second"):
        EndpointRegistry.from_config({"audio_tokens_per_second": bad})


# -- through the controller --------------------------------------------------


def _config(base, *, context, transcriber, rate=100, audio=True):
    caps = {"modalities_in": ["text", "audio"] if audio else ["text"], "context": context}
    leaf = {
        "api_base": base,
        "api_key_env": "FAKE_UPSTREAM_KEY",
        "extra": {"max_retries": 0},
        "capabilities": caps,
    }
    cfg = {
        "audio_tokens_per_second": rate,
        "endpoints": {
            "a": {"model": "openai/model-a", **leaf},
            "b": {"model": "openai/model-b", **leaf},
        },
        "tiers": {"default": {"router": "random", "threshold": 0.0, "strong": "b", "weak": "a"}},
    }
    if transcriber is not None:
        cfg["transcriber"] = transcriber
    return cfg


def _controller(tmp_path, config):
    os.environ["FAKE_UPSTREAM_KEY"] = "sk-fake"
    return Controller(
        routers=["random"],
        strong_model="x",
        weak_model="x",
        endpoints=EndpointRegistry.from_config(config),
        resilience_config=ResilienceConfig(timeout_ms=30000, max_retries=0),
        cache_config=CacheConfig(enabled=False),
    )


def _ask(controller, messages):
    return asyncio.run(controller.acompletion(model="default", messages=messages))


@pytest.mark.slow
def test_a_long_clip_exceeds_the_leaf_context_and_is_refused_in_passthrough(tmp_path):
    fake = FakeUpstream("llm")
    controller = _controller(
        tmp_path, _config(fake.base, context=1000, transcriber={"mode": "passthrough"})
    )

    # 60 s at 100 tokens/s is 6000 tokens against a 1000-token window.
    with pytest.raises(RoutingError, match="context"):
        _ask(controller, _msgs(_part(_wav(60))))

    assert fake.hits == []


@pytest.mark.slow
def test_a_short_clip_still_fits_in_passthrough(tmp_path):
    fake = FakeUpstream("llm")
    controller = _controller(
        tmp_path, _config(fake.base, context=1000, transcriber={"mode": "passthrough"})
    )

    # 2 s at 100 tokens/s is 200 tokens.
    _ask(controller, _msgs(_part(_wav(2))))

    assert len(fake.hits) == 1


@pytest.mark.slow
def test_the_estimate_is_not_double_counted_after_transcription(tmp_path):
    fake = FakeUpstream("llm")
    stt = FakeTranscriber(answer=lambda fields, name, audio: {"text": "a short transcript"})
    try:
        controller = _controller(
            tmp_path,
            _config(fake.base, context=1000, transcriber={"api_base": stt.base}, audio=False),
        )

        # Counted as audio this would be 6000 tokens and refused; the
        # transcript is a handful of words.
        _ask(controller, _msgs(_part(_wav(60))))
    finally:
        stt.close()

    assert len(fake.hits) == 1
