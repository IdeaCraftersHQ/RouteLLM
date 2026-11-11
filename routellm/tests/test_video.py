"""Video preprocessing: part shapes, config, rendering, errors, privacy checks.

The preprocessor is a loopback fake of the HTTP contract; nothing leaves
the machine. Controller and server tests live in `test_video_routing`.
"""

import asyncio
import base64
import logging

import pytest

from routellm import video
from routellm.class_tiers import check_video, video_denial
from routellm.endpoints import EndpointRegistry
from routellm.policy import PolicyError, parse_policy
from routellm.tests.fake_video import FRAME, LEAK, SHEET, FakeVideoService
from routellm.video import VideoConfig, VideoError


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def vpart(raw=b"clip", sub="mp4"):
    return {"type": "video_url", "video_url": {"url": f"data:video/{sub};base64,{b64(raw)}"}}


def fpart(raw=b"clip", name="clip.mp4"):
    return {
        "type": "file",
        "file": {"file_data": f"data:video/mp4;base64,{b64(raw)}", "filename": name},
    }


def text(value):
    return {"type": "text", "text": value}


def user(*parts):
    return [{"role": "user", "content": list(parts)}]


def cfg(base, **kw):
    return VideoConfig(api_base=base, **kw)


@pytest.fixture
def svc():
    fake = FakeVideoService()
    yield fake
    fake.close()


def run(messages, config, **kw):
    return asyncio.run(video.preprocess(messages, config, **kw))


# -- detection and shapes ----------------------------------------------------


def test_both_accepted_shapes_are_video_and_other_parts_are_not():
    assert video.has_video(user(text("hi"), vpart()))
    assert video.has_video(user(fpart()))
    assert not video.has_video(user(text("hi")))
    pdf = {
        "type": "file",
        "file": {"file_data": "data:application/pdf;base64,AA==", "filename": "a.pdf"},
    }
    assert not video.has_video(user(pdf))
    assert not video.has_video([{"role": "user", "content": "video_url"}])


@pytest.mark.parametrize(
    "part, code",
    [
        ({"type": "video_url", "video_url": "data:video/mp4;base64,AA=="}, "video_bad_part"),
        ({"type": "video_url", "video_url": {}}, "video_bad_part"),
        (
            {"type": "video_url", "video_url": {"url": "data:image/png;base64,AA=="}},
            "video_bad_part",
        ),
        ({"type": "file", "file": {"file_data": "AAAA", "filename": "x.mp4"}}, "video_bad_part"),
        ({"type": "file", "file": {"file_data": "data:video/mp4,plain"}}, "video_bad_part"),
        (
            {"type": "video_url", "video_url": {"url": "https://x.example/a.mp4"}},
            "video_url_refused",
        ),
    ],
)
def test_unknown_or_remote_shapes_are_a_400_before_any_call(part, code, svc):
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(part), cfg(svc.base))

    assert err.value.status_code == 400 and err.value.code == code
    assert svc.hits == []


def test_a_remote_url_is_still_refused_with_allow_urls_and_says_so(svc):
    part = {"type": "video_url", "video_url": {"url": "http://x.example/a.mp4"}}

    with pytest.raises(VideoError, match="not fetched") as err:
        video.preprocess_sync(user(part), cfg(svc.base, allow_urls=True))

    assert err.value.status_code == 400 and svc.hits == []


def test_bad_base64_and_oversize_are_refused_before_any_call(svc):
    bad = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,@@@@"}}
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(bad), cfg(svc.base))
    assert err.value.status_code == 400

    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart(b"x" * 200)), cfg(svc.base, max_video_bytes=64))
    assert err.value.status_code == 413 and err.value.code == "video_too_large"
    assert svc.hits == []


# -- config ------------------------------------------------------------------


def test_unknown_key_and_bad_ranges_are_rejected():
    with pytest.raises(ValueError, match="extra|Extra"):
        EndpointRegistry.from_config({"video": {"api_base": "http://x", "api_basee": "t"}})
    with pytest.raises(ValueError, match="max_frames"):
        EndpointRegistry.from_config({"video": {"api_base": "http://x", "max_frames": 65}})
    with pytest.raises(ValueError, match="layout"):
        EndpointRegistry.from_config({"video": {"api_base": "http://x", "layout": "grid"}})
    with pytest.raises(ValueError, match="api_base"):
        EndpointRegistry.from_config({"video": {"mode": "preprocess"}})
    # passthrough and refuse need no service
    assert (
        EndpointRegistry.from_config({"video": {"mode": "refuse"}}).video_for(None).active is False
    )


def test_tier_overrides_global_key_by_key():
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m"}, "b": {"model": "n"}},
            "tiers": {
                "t": {
                    "strong": "a",
                    "weak": "b",
                    "video": {"max_frames": 4, "layout": "contact_sheet"},
                },
                "u": {"strong": "a", "weak": "b", "video": {"mode": "passthrough"}},
            },
            "video": {"api_base": "http://x", "max_frames": 8},
        }
    )

    assert registry.video_for("t").max_frames == 4
    assert registry.video_for("t").layout == "contact_sheet"
    assert registry.video_for(None).max_frames == 8
    assert registry.video_for("u").active is False
    assert registry.video_for("missing") is None


def _local_spec(section, tier_section=None, other_tags=("local",)):
    tier = {"strong": "a", "weak": "b"}
    if tier_section is not None:
        tier["video"] = tier_section
    return {
        "endpoints": {
            "a": {"model": "m", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
            "b": {"model": "n", "api_base": "http://127.0.0.1:11434/v1", "tags": list(other_tags)},
        },
        "tiers": {"private": tier},
        "video": section,
    }


def test_local_only_tier_rejects_a_non_local_preprocessor():
    with pytest.raises(ValueError, match="local-only"):
        EndpointRegistry.from_config(_local_spec({"api_base": "http://video.example.com"}))
    with pytest.raises(ValueError, match="local-only"):
        EndpointRegistry.from_config(
            _local_spec(
                {"api_base": "http://127.0.0.1:9", "tags": ["local"]},
                {"api_base": "http://video.example.com", "tags": []},
            )
        )
    ok = EndpointRegistry.from_config(
        _local_spec({"api_base": "http://127.0.0.1:9", "tags": ["local"]})
    )
    assert ok.video_for("private").is_local
    mixed = _local_spec({"api_base": "http://video.example.com"}, other_tags=("cloud",))
    assert EndpointRegistry.from_config(mixed).video_for("private").active
    refuse = _local_spec({"api_base": "http://video.example.com", "mode": "refuse"})
    assert EndpointRegistry.from_config(refuse).video_for("private").active is False


def test_a_local_tag_needs_a_non_public_api_base():
    with pytest.raises(ValueError, match="not a loopback"):
        EndpointRegistry.from_config(
            {"video": {"api_base": "https://video.example.com", "tags": ["local"]}}
        )
    open_ = {"api_base": "https://video.example.com", "tags": ["local"], "allow_public_base": True}
    assert EndpointRegistry.from_config({"video": open_}).video_for(None).is_local
    for base in ("http://localhost:9", "http://10.1.2.3:9", "http://video.internal"):
        EndpointRegistry.from_config({"video": {"api_base": base, "tags": ["local"]}})


def test_intent_routing_to_a_local_only_tier_needs_a_local_preprocessor():
    spec = {
        "endpoints": {
            "a": {"model": "m", "api_base": "http://127.0.0.1:11434/v1", "tags": ["local"]},
            "c": {"model": "o", "tags": ["cloud"]},
        },
        "tiers": {
            "private": {"strong": "a", "weak": "a", "video": {"mode": "refuse"}},
            "default": {"strong": "c", "weak": "c"},
        },
        "video": {"api_base": "http://video.example.com"},
        "intents": {"tiers": {"secret": "private"}},
    }
    with pytest.raises(ValueError, match="intent routing"):
        EndpointRegistry.from_config(spec)
    spec["tiers"]["default"]["intent_routing"] = False
    EndpointRegistry.from_config(spec)


def _policy(treatment):
    spec = {"treatment": treatment, "reach": ["local", "cloud"], "tiers": ["default"]}
    if treatment == "block":
        spec["reach"] = ["local"]
    return parse_policy({"classes": {"c": spec}, "default_class": "c"}, "p")


def _registry(section, tier_section=None):
    tier = {"strong": "a", "weak": "a"}
    if tier_section is not None:
        tier["video"] = tier_section
    return EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m", "tags": ["cloud"]}},
            "tiers": {"default": tier},
            "video": section,
        }
    )


CLOUD = {"api_base": "https://video.example.com"}
LOCAL = {"api_base": "http://127.0.0.1:9", "tags": ["local"]}


@pytest.mark.parametrize("treatment", ["redact", "block"])
def test_startup_refuses_a_non_local_preprocessor_under_redact_or_block(treatment):
    with pytest.raises(PolicyError, match="'global'"):
        check_video(_policy(treatment), _registry(CLOUD))
    with pytest.raises(PolicyError, match="'default'"):
        check_video(_policy(treatment), _registry(LOCAL, CLOUD | {"tags": []}))
    check_video(_policy(treatment), _registry(LOCAL))
    check_video(_policy(treatment), _registry({"mode": "refuse"}))
    check_video(_policy(treatment), _registry(None))


def test_raw_classes_do_not_care_and_denial_is_per_class():
    check_video(_policy("raw"), _registry(CLOUD))
    config = VideoConfig(**CLOUD)
    assert video_denial(_policy("redact"), config, "c") == "video:c:not_local"
    assert video_denial(_policy("redact"), config, None) is None
    assert video_denial(_policy("redact"), VideoConfig(**LOCAL), "c") is None
    assert video_denial(_policy("raw"), config, "c") is None


# -- the call and the rendering ----------------------------------------------


def test_the_request_to_the_service(svc, monkeypatch):
    monkeypatch.setenv("VIDEO_KEY", "k-123")
    config = cfg(
        svc.base,
        api_key_env="VIDEO_KEY",
        max_frames=6,
        layout="frames",
        max_side=512,
        transcribe=True,
        ocr=True,
        scene_detection=False,
        require_transcript=True,
    )

    video.preprocess_sync(user(fpart(b"abc", name="SECRET-NAME.mp4")), config, language="fr")

    (hit,) = svc.hits
    assert hit["path"] == "/v1/video/preprocess"
    assert hit["headers"]["Authorization"] == "Bearer k-123"
    assert hit["video"] == b"abc"
    assert "SECRET-NAME" not in str(hit["filename"])
    assert hit["fields"] == {
        "max_frames": "6",
        "layout": "frames",
        "max_side": "512",
        "transcribe": "true",
        "ocr": "true",
        "scene_detection": "false",
        "require_transcript": "true",
        "language": "fr",
    }


def test_a_missing_key_variable_is_a_502_before_any_call(svc, monkeypatch):
    monkeypatch.delenv("VIDEO_KEY", raising=False)

    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base, api_key_env="VIDEO_KEY"))

    assert err.value.status_code == 502 and svc.hits == []


def test_a_bad_language_is_refused_before_any_call(svc):
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base), language="en; drop")
    assert err.value.status_code == 400 and svc.hits == []


def test_layout_frames_replaces_the_part_in_place_in_order(svc):
    svc.transcript = "hello"
    out = video.preprocess_sync(
        user(text("before"), vpart(b"v1"), text("between"), fpart(b"v2"), text("after")),
        cfg(svc.base, ocr=True),
    )

    parts = out[0]["content"]
    kinds = [p["type"] for p in parts]
    assert kinds[0] == "text" and parts[0]["text"] == "before"
    texts = [p["text"] for p in parts if p["type"] == "text"]
    assert "Video 1: 93.4s, 2 scenes" in texts
    assert "Video 2: 93.4s, 2 scenes" in texts
    assert texts.index("between") < texts.index("Video 2: 93.4s, 2 scenes")
    assert texts[-1] == "after" or "after" in texts
    assert parts[-1] == text("after")
    assert "video_url" not in str(out) and "file_data" not in str(out)
    first = parts[1:8]
    assert first[0]["text"].startswith("Video 1")
    assert first[1]["text"] == "[scene 1, 0.0-12.3 s]"
    assert [p["type"] for p in first[2:4]] == ["image_url", "image_url"]
    assert first[4]["text"] == "[scene 2, 12.3-93.4 s]"
    assert first[5]["type"] == "image_url"
    assert first[6]["text"].startswith("Transcript:\n[00:00 SPEAKER_00] hello v1")
    assert "[01:05] no speaker here" in first[6]["text"]
    assert "On-screen text:\n[scene 1] OCR-ONE" in texts
    assert len(svc.hits) == 2


def test_contact_sheet_layout_sends_one_image(svc):
    out = video.preprocess_sync(user(vpart()), cfg(svc.base, layout="contact_sheet"))

    parts = out[0]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert [i["image_url"]["url"] for i in images] == [SHEET]
    assert "frames at 1.2, 6.0 s" in str(parts)


def test_no_transcript_is_stated_not_silent(svc):
    out = video.preprocess_sync(user(vpart()), cfg(svc.base, transcribe=True))
    assert "Transcript:" in str(out)

    quiet = video.preprocess_sync(user(vpart()), cfg(svc.base, transcribe=False))
    assert "Transcript" not in str(quiet)

    answer = {
        "duration_s": 1,
        "has_audio": False,
        "transcript": None,
        "scenes": [{"index": 0, "start_s": 0, "end_s": 1, "frames": []}],
    }
    rendered = video.render(answer, cfg(svc.base))
    assert rendered[-1]["text"] == "Transcript: [no audio track]"
    assert rendered[0]["text"] == "Video 1: 1.0s, 1 scene"


def test_a_long_transcript_is_bounded_and_says_so():
    segments = [{"start_s": i, "end_s": i + 1, "text": f"line {i} " + "w" * 20} for i in range(100)]
    answer = {"duration_s": 100, "scenes": [], "transcript": {"segments": segments}}

    out = video.render(answer, VideoConfig(api_base="http://x", max_transcript_chars=200))

    body = out[-1]["text"]
    assert body.endswith("[transcript truncated]")
    assert len(body) < 200 + 60
    assert "line 99" not in body
    empty = video.render(
        {"scenes": [], "transcript": {"text": "", "segments": []}}, VideoConfig(api_base="http://x")
    )
    assert video.NO_SPEECH in empty[-1]["text"]


def test_a_frame_that_is_not_an_inline_image_is_never_forwarded(svc):
    svc.mode = "remote"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base))
    assert err.value.status_code == 502 and err.value.code == "video_bad_answer"


def test_sync_and_async_agree(svc):
    messages = user(text("q"), vpart(b"same"))
    assert run(messages, cfg(svc.base)) == video.preprocess_sync(messages, cfg(svc.base))


def test_the_input_messages_are_not_mutated(svc):
    messages = user(vpart())
    before = repr(messages)
    video.preprocess_sync(messages, cfg(svc.base))
    assert repr(messages) == before


# -- failures ----------------------------------------------------------------


def test_service_down_is_a_502():
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg("http://127.0.0.1:1"))
    assert (err.value.status_code, err.value.code) == (502, "video_unavailable")


def test_service_5xx_is_a_502_that_leaks_nothing(svc):
    svc.mode = "500"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base))
    assert err.value.status_code == 502 and err.value.code == "internal"
    assert LEAK not in err.value.public and LEAK in err.value.detail


def test_service_4xx_is_a_400_with_its_code_and_none_of_its_message(svc):
    svc.mode = "400"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base, ocr=True))
    assert err.value.status_code == 400 and err.value.code == "ocr_unavailable"
    assert "ocr_unavailable" in err.value.public and LEAK not in err.value.public


def test_a_service_auth_failure_is_our_502_not_the_clients_400(svc):
    svc.mode = "401"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base))
    assert err.value.status_code == 502 and LEAK not in err.value.public


def test_stt_failed_when_required_is_a_502_with_its_code(svc):
    svc.mode = "stt_failed"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base, require_transcript=True))
    assert (err.value.status_code, err.value.code) == (502, "stt_failed")


def test_unreadable_answer_is_a_502(svc):
    svc.mode = "garbage"
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base))
    assert err.value.status_code == 502


def test_timeout_is_a_504(svc):
    svc.delay = 1.0
    with pytest.raises(VideoError) as err:
        video.preprocess_sync(user(vpart()), cfg(svc.base, timeout_ms=100))
    assert (err.value.status_code, err.value.code) == (504, "video_timeout")


def test_info_log_has_counts_and_no_content(svc, caplog):
    svc.transcript = "SECRETWORDS"
    with caplog.at_level(logging.INFO):
        video.preprocess_sync(user(vpart(b"SECRETBYTES")), cfg(svc.base), request_id="r1")

    assert "SECRETWORDS" not in caplog.text and "SECRETBYTES" not in caplog.text
    assert b64(b"SECRETBYTES") not in caplog.text and FRAME not in caplog.text
    line = next(r.getMessage() for r in caplog.records if "preprocessed video" in r.getMessage())
    assert "parts=1" in line and "scenes=2" in line and "frames=3" in line
