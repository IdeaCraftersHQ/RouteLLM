"""Redact-treatment privacy cases: class `pii` may reach the cloud only as placeholders.

These run in the redact world (see `privacy_fixture.redact_files`): the
`pii` class has treatment `redact`, the cloud and gated fakes echo what
they are sent (`echo` mode), and the aux world's detector, embedding
endpoint and embedding router all point at the cloud fake. The claim:
no canary reaches an off-host fake by any path (main call, stream,
tool-call arguments, multimodal text, compare targets, auxiliary calls),
and the client still reads the original text.
"""

import json

from routellm.tests.privacy_fixture import AUDIO as AUDIO_MESSAGES
from routellm.tests.privacy_fixture import CANARIES, case

FR = "Bonjour, Jean Tremblay (jean.tremblay@example.com), NAS 046 454 286, dossier ACME-12345."
EN = "Please email Alice Smith and cc Marie Gagnon."
LAST = "Résume le dossier de Jean Tremblay et écris à Marie Gagnon."

HISTORY = [
    {"role": "system", "content": f"Contexte: {FR}"},
    {"role": "user", "content": [{"type": "text", "text": EN}]},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"client": "Jean Tremblay", "id": "ACME-12345"}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Alice Smith, 046 454 286"},
    {"role": "user", "content": LAST},
]
OFF_HOST = ("cloud_main", "gated_main")


def _sent_off_host(w) -> str:
    return json.dumps(
        [hit[2] for name in OFF_HOST for hit in w.fakes[name].hits], ensure_ascii=False
    )


def _leaks(w) -> list[str]:
    sent = _sent_off_host(w)
    return [c for c in CANARIES if c in sent]


def _echo(w, *names):
    for name in names:
        w.fakes[name].mode = "echo"


def _off_host_hits(w) -> int:
    return sum(len(w.fakes[n].hits) for n in OFF_HOST)


@case(
    "redact_main_call_sends_no_canary_and_the_client_reads_the_original",
    200,
    reaches_off_host=True,
    redact=True,
)
def main_call(w):
    _echo(w, "cloud_main")
    r = w.post("red_cloud", HISTORY, data_class="pii")
    content = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else None
    return {
        "status": r.status_code,
        "leaks": _leaks(w),
        "off_host_hits": _off_host_hits(w),
        "restored": content == f"echo: {LAST}",
        "placeholders_sent": "[[" in _sent_off_host(w),
        "mapping_free": "[[" not in (content or ""),
    }


@case("redact_stream_split_over_chunks_is_restored", 200, reaches_off_host=True, redact=True)
def stream(w):
    _echo(w, "cloud_main")
    status, events = w.stream("red_cloud", HISTORY, data_class="pii")
    chunks = [json.loads(e) for e in events if e != "[DONE]"]
    deltas = [c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]]
    return {
        "status": status,
        "leaks": _leaks(w),
        "off_host_hits": _off_host_hits(w),
        "restored": "".join(deltas) == f"echo: {LAST}",
        "pieces": len([d for d in deltas if d]),
        "placeholder_in_client_stream": any("[[" in d for d in deltas),
    }


@case("redact_sibling_fallback_is_redacted_too", 200, reaches_off_host=True, redact=True)
def fallback(w):
    _echo(w, "gated_main")
    w.fail("cloud_main")
    r = w.post("red_fallback", HISTORY, data_class="pii")
    return {
        "status": r.status_code,
        "leaks": _leaks(w),
        "gated_hits": len(w.fakes["gated_main"].hits),
        "cloud_hits": len(w.fakes["cloud_main"].hits),
    }


@case("redact_compare_targets_are_redacted", 200, reaches_off_host=True, redact=True)
def compare(w):
    _echo(w, "cloud_main", "gated_main")
    r = w.compare(["cloud_main", "gated_main"], messages=HISTORY, data_class="pii")
    return {
        "status": r.status_code,
        "leaks": _leaks(w),
        "hit_both": all(w.fakes[n].hits for n in OFF_HOST),
    }


@case("redact_redactor_failure_is_a_5xx_with_zero_provider_calls", 500, redact=True)
def redactor_fails(w):
    transform = w.server.CONTROLLER.transforms.transforms[0]
    real = transform.redactor._analyze

    def boom(text, language):
        raise RuntimeError("analyzer failed")

    transform.redactor._analyze = boom
    try:
        r = w.post("red_cloud", HISTORY, data_class="pii")
        s = w.stream("red_cloud", HISTORY, data_class="pii")
    finally:
        transform.redactor._analyze = real
    return {
        "status": r.status_code,
        "stream_status": s[0],
        "code": r.json().get("code"),
        "off_host_hits": _off_host_hits(w),
    }


@case("redact_local_leaf_stays_raw", 200, {"local_main": 1}, redact=True)
def local_raw(w):
    _echo(w, "local_main")
    r = w.post("red_local", HISTORY, data_class="pii")
    sent = json.dumps([h[2] for h in w.fakes["local_main"].hits], ensure_ascii=False)
    return {
        "status": r.status_code,
        "local_saw_raw": "Jean Tremblay" in sent and "[[" not in sent,
        "off_host_hits": _off_host_hits(w),
    }


@case("redact_unredactable_part_is_refused_not_sent", 500, redact=True)
def image_refused(w):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": EN},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }
    ]
    r = w.post("red_cloud", msgs, data_class="pii")
    return {"status": r.status_code, "off_host_hits": _off_host_hits(w)}


@case("redact_aux_detector_never_sees_the_class_off_host", 200, {"local_main": 1}, redact=True)
def aux_detector(w):
    for middleware in w.server.CONTROLLER.middleware:
        middleware.intent_cache.clear()
    r = w.post("default", HISTORY, data_class="pii")
    return {"status": r.status_code, "leaks": _leaks(w), "off_host_hits": _off_host_hits(w)}


@case("redact_aux_embedding_router_is_skipped", 200, {"local_small": 1}, redact=True)
def aux_router(w):
    w.admit("pii", "private_emb")
    r = w.post("private_emb", HISTORY, data_class="pii")
    return {"status": r.status_code, "leaks": _leaks(w), "off_host_hits": _off_host_hits(w)}


@case("redact_aux_semantic_cache_never_embeds", 200, {"local_main": 1}, redact=True)
def aux_semantic(w):
    import tempfile
    from pathlib import Path

    cache = w.server.CONTROLLER.cache
    cache.config.enabled = True
    cache.config.semantic_enabled = True
    cache.config.db_path = str(Path(tempfile.mkdtemp()) / "redact_cache.db")
    cache._init_db()
    _echo(w, "local_main")
    r = w.post("red_local", HISTORY, data_class="pii")
    return {"status": r.status_code, "leaks": _leaks(w), "off_host_hits": _off_host_hits(w)}


# -- fields outside `messages` ------------------------------------------------------------

CANARY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find the file of Jean Tremblay",
            "parameters": {
                "type": "object",
                "description": "Contact Marie Gagnon",
                "properties": {
                    "client": {
                        "type": "string",
                        "description": "e.g. Alice Smith",
                        "enum": ["ACME-12345", "other"],
                        "default": "jean.tremblay@example.com",
                        "examples": ["046 454 286"],
                    }
                },
                "required": ["client"],
            },
        },
    }
]
CANARY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "description": "Answer about Jean Tremblay",
        "schema": {
            "type": "object",
            "properties": {"who": {"type": "string", "enum": ["Alice Smith", "x"]}},
        },
    },
}
NAMED = [
    {"role": "user", "name": "Marie_Gagnon", "content": "bonjour"},
    {"role": "user", "name": "ACME-12345", "content": EN},
]


def _call_direct(w, **kw):
    """One call through the controller, as a library caller: fields the HTTP body drops."""
    controller = w.server.CONTROLLER
    try:
        controller.completion(
            model="red_cloud",
            messages=[{"role": "user", "content": "hello"}],
            declared_class="pii",
            **kw,
        )
        return 200, None
    except Exception as exc:
        return 500 if type(exc).__name__ == "TransformError" else 502, type(exc).__name__


@case("redact_request_fields_carry_no_canary", 200, reaches_off_host=True, redact=True)
def fields_http(w):
    _echo(w, "cloud_main")
    r = w.post(
        "red_cloud",
        [*NAMED, {"role": "user", "content": "ok"}],
        data_class="pii",
        user="jean.tremblay@example.com",
        tools=CANARY_TOOLS,
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        response_format=CANARY_FORMAT,
        stop=["END"],
        temperature=0.2,
    )
    sent = _sent_off_host(w)
    return {
        "status": r.status_code,
        "leaks": _leaks(w),
        "off_host_hits": _off_host_hits(w),
        "body": r.text[:300],
        "allowlisted_kept": '"temperature": 0.2' in sent and '"lookup"' in sent,
        "placeholders_sent": "[[" in sent,
    }


@case("redact_direct_fields_carry_no_canary", 200, reaches_off_host=True, redact=True)
def fields_direct(w):
    _echo(w, "cloud_main")
    status, kind = _call_direct(
        w,
        prediction={"type": "content", "content": "Jean Tremblay"},
        metadata={"owner": "Alice Smith"},
        functions=[{"name": "f", "description": "Marie Gagnon", "parameters": {"type": "object"}}],
        user="046 454 286",
    )
    return {
        "status": status,
        "kind": kind,
        "leaks": _leaks(w),
        "off_host_hits": _off_host_hits(w),
    }


@case("redact_unknown_field_is_refused_with_zero_calls", 500, redact=True)
def unknown_field(w):
    status, kind = _call_direct(w, frobnicate="Jean Tremblay")
    return {"status": status, "kind": kind, "off_host_hits": _off_host_hits(w)}


@case("redact_unredactable_name_is_refused_with_zero_calls", 500, redact=True)
def pii_in_a_name(w):
    tools = [{"type": "function", "function": {"name": "jean.tremblay@example.com"}}]
    status, kind = _call_direct(w, tools=tools)
    return {"status": status, "kind": kind, "off_host_hits": _off_host_hits(w)}


@case("redact_non_local_transcriber_is_refused_before_transcription", None, redact=True)
def transcriber_per_request(w):
    from routellm.transcription import TranscriberConfig

    endpoints = w.server.CONTROLLER.endpoints
    cloud = endpoints.resolve("cloud_main")
    config = TranscriberConfig(mode="transcribe", api_base=cloud.api_base, model="stt")
    endpoints._transcribers["red_cloud"] = config
    try:
        audio = w.post("red_cloud", AUDIO_MESSAGES, data_class="pii")
    finally:
        endpoints._transcribers.pop("red_cloud")
    return {"status": audio.status_code, "off_host_hits": _off_host_hits(w)}


# -- video: the transcript is user content like any other ------------------------------------

VIDEO_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What does the clip say?"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ],
    }
]


def _with_video(w, tier, base, tags, frames, body):
    """Run `body` with `tier` preprocessing video through the fake service at `base`."""
    from routellm.tests.fake_video import default_answer
    from routellm.video import VideoConfig

    def answer(fields, filename, video, transcript):
        out = default_answer(fields, filename, video, transcript)
        if not frames:
            out["scenes"] = [{**s, "frames": []} for s in out["scenes"]]
        return out

    endpoints = w.server.CONTROLLER.endpoints
    endpoints._videos[tier] = VideoConfig(api_base=base, tags=tags)
    try:
        return body(answer)
    finally:
        endpoints._videos.pop(tier)


@case(
    "redact_video_transcript_canary_never_reaches_a_cloud_leaf",
    200,
    reaches_off_host=True,
    redact=True,
)
def video_transcript(w):
    from routellm.tests.fake_video import FakeVideoService

    service = FakeVideoService()
    service.transcript = (
        "Jean Tremblay wrote to Marie Gagnon (ACME-12345) at jean.tremblay@example.com:"
    )
    _echo(w, "cloud_main")

    def body(answer):
        service.answer = answer
        return w.post("red_cloud", VIDEO_MESSAGES, data_class="pii")

    try:
        r = _with_video(w, "red_cloud", service.base, ["local"], False, body)
    finally:
        service.close()
    return {
        "status": r.status_code,
        "leaks": _leaks(w),
        "off_host_hits": _off_host_hits(w),
        "service_hits": len(service.hits),
        "placeholders_sent": "[[" in _sent_off_host(w),
        "transcript_sent": "Transcript:" in _sent_off_host(w),
    }


@case("redact_video_frames_are_refused_not_sent", 500, redact=True)
def video_frames(w):
    from routellm.tests.fake_video import FakeVideoService

    service = FakeVideoService()
    try:
        r = _with_video(
            w,
            "red_cloud",
            service.base,
            ["local"],
            True,
            lambda answer: w.post("red_cloud", VIDEO_MESSAGES, data_class="pii"),
        )
    finally:
        service.close()
    return {"status": r.status_code, "off_host_hits": _off_host_hits(w)}


@case("redact_non_local_video_preprocessor_is_refused_before_upload", 403, redact=True)
def video_per_request(w):
    from routellm.tests.fake_video import FakeVideoService

    service = FakeVideoService()
    try:
        r = _with_video(
            w,
            "red_cloud",
            service.base,
            [],
            True,
            lambda answer: w.post("red_cloud", VIDEO_MESSAGES, data_class="pii"),
        )
    finally:
        service.close()
    return {
        "status": r.status_code,
        "off_host_hits": _off_host_hits(w),
        "service_hits": len(service.hits),
    }
