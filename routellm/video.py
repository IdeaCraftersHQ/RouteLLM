"""Preprocess video parts into frames and a transcript before routing.

A chat request may carry a video content part. With a `video:` section
and its mode set to `preprocess`, each video is sent to a separate,
local preprocessing service and replaced by what it returns: image parts
(sampled frames, or one contact sheet) and text parts (a header, scene
labels, a transcript, on-screen text). The request then routes as an
ordinary vision request. RouteLLM is only the client of that service.

Accepted part shapes (anything else that claims to be video is a 400)::

    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
    {"type": "file", "file": {"file_data": "data:video/mp4;base64,...",
                              "filename": "clip.mp4"}}

A remote URL (http/https) is always refused: this server does not fetch
client-named URLs. `allow_urls` only changes the wording of the refusal;
fetching them is not implemented.

Configuration is a top-level `video:` section, optionally overridden key
by key under a tier's own `video:`::

    video:
      mode: preprocess          # preprocess | passthrough | refuse
      api_base: http://127.0.0.1:9100
      api_key_env: VIDEO_PREPROCESSOR_KEY
      timeout_ms: 600000
      max_video_bytes: 209715200
      tags: [local]
      max_frames: 12
      layout: frames            # frames | contact_sheet
      max_side: 768
      transcribe: true
      language: null
      ocr: false
      scene_detection: true
      require_transcript: false
      max_transcript_chars: 20000

Everything fails closed: an unreachable, slow, erroring or refusing
service ends the request with a `VideoError`; the raw video is never
forwarded instead. Neither the video, the frames nor the transcript is
logged above DEBUG; the INFO summary holds counts and latency only.
"""

import base64
import binascii
import logging
import os
import re
import time
from typing import Any

import httpx

from routellm.video_config import _LANGUAGE, LOCAL_TAG, VideoConfig, VideoError
from routellm.video_render import NO_SPEECH, num, render

logger = logging.getLogger(__name__)

__all__ = [
    "LOCAL_TAG",
    "NO_SPEECH",
    "VideoConfig",
    "VideoError",
    "check_parts",
    "has_video",
    "is_video_part",
    "preprocess",
    "preprocess_sync",
    "render",
]

_VIDEO_URL = re.compile(r"^data:video/([A-Za-z0-9.+-]{1,32});base64,", re.IGNORECASE)
_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VIDEO_NAME = re.compile(r"\.(mp4|m4v|mov|webm|mkv|avi|mpe?g|ogv|3gp|flv|wmv)$", re.IGNORECASE)


# -- detection ---------------------------------------------------------------


def is_video_part(part: Any) -> bool:
    """Return whether a content part is (or claims to be) a video."""
    if not isinstance(part, dict):
        return False
    kind = part.get("type")
    if kind == "video_url":
        return True
    if kind == "file":
        body = part.get("file")
        if not isinstance(body, dict):
            return False
        data = body.get("file_data")
        if not isinstance(data, str):
            return False
        if data[:11].lower() == "data:video/":
            return True
        name = body.get("filename")
        return isinstance(name, str) and bool(_VIDEO_NAME.search(name))
    return False


def _video_parts(messages: Any):
    """Yield `(message_index, part_index, part)` for every video part."""
    if not isinstance(messages, list):
        return
    for mi, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if is_video_part(part):
                yield mi, pi, part


def has_video(messages: Any) -> bool:
    """Return whether any message carries a video part."""
    return any(True for _ in _video_parts(messages))


def check_parts(messages: Any, config: VideoConfig) -> None:
    """Refuse a video part that is not an accepted shape (400, nothing sent)."""
    for _, _, part in _video_parts(messages):
        _data_url(part, config)


def _data_url(part: dict[str, Any], config: VideoConfig) -> str:
    """Return the part's `data:video/...` URL, or raise a 400."""
    if part.get("type") == "video_url":
        body = part.get("video_url")
        url = body.get("url") if isinstance(body, dict) else None
        shape = "video_url"
    else:
        body = part.get("file")
        url = body.get("file_data") if isinstance(body, dict) else None
        shape = "file"
    if not isinstance(url, str) or not url:
        raise VideoError(
            400,
            f"unrecognized video part: a {shape} part needs a data:video/<type>;base64 URL",
            code="video_bad_part",
        )
    if re.match(r"^https?://", url, re.IGNORECASE):
        if config.allow_urls:
            raise VideoError(
                400,
                "remote video URLs are not fetched by this server; send a data:video URL",
                code="video_url_unsupported",
            )
        raise VideoError(
            400,
            "remote video URLs are not accepted; send the video as a data:video URL",
            code="video_url_refused",
        )
    if not _VIDEO_URL.match(url):
        raise VideoError(
            400,
            f"unrecognized video part: a {shape} part needs a data:video/<type>;base64 URL",
            code="video_bad_part",
        )
    return url


def _decode(part: dict[str, Any], config: VideoConfig) -> tuple[bytes, str]:
    """Return `(video bytes, subtype)` for a part, or raise a `VideoError`."""
    url = _data_url(part, config)
    match = _VIDEO_URL.match(url)
    assert match is not None
    subtype = match.group(1).lower()
    data = url[match.end() :]
    if len(data) * 3 // 4 > config.max_video_bytes + 3:
        raise VideoError(
            413,
            "video part is larger than this server accepts",
            f"base64 length {len(data)}",
            "video_too_large",
        )
    try:
        video = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VideoError(400, "video part is not valid base64", code="video_bad_part") from exc
    if not video:
        raise VideoError(400, "video part is empty", code="video_bad_part")
    if len(video) > config.max_video_bytes:
        raise VideoError(
            413,
            "video part is larger than this server accepts",
            f"{len(video)} bytes",
            "video_too_large",
        )
    return video, subtype


# -- the service call --------------------------------------------------------


def _request(config: VideoConfig, language: str | None):
    """Return `(url, headers, form)` for one preprocess POST."""
    headers: dict[str, str] = {}
    if config.api_key_env:
        key = os.environ.get(config.api_key_env)
        if key is None:
            raise VideoError(
                502,
                "video preprocessor is not available",
                f"environment variable {config.api_key_env!r} is not set",
                "video_unavailable",
            )
        headers["Authorization"] = f"Bearer {key}"

    lang = language or config.language
    if lang and not _LANGUAGE.match(lang):
        raise VideoError(
            400, "language is not a language code such as en or pt-BR", code="video_bad_language"
        )

    def flag(value: bool) -> str:
        return "true" if value else "false"

    form = {
        "max_frames": str(config.max_frames),
        "layout": config.layout,
        "max_side": str(config.max_side),
        "transcribe": flag(config.transcribe),
        "ocr": flag(config.ocr),
        "scene_detection": flag(config.scene_detection),
        "require_transcript": flag(config.require_transcript),
    }
    if lang:
        form["language"] = lang
    return f"{(config.api_base or '').rstrip('/')}/v1/video/preprocess", headers, form


def _error_code(response: httpx.Response) -> str | None:
    """Return the service's own error code if it sent a well-formed one."""
    try:
        body = response.json()
        code = body["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None
    return code if isinstance(code, str) and _CODE.match(code) else None


def _parse(response: httpx.Response) -> dict[str, Any]:
    status = response.status_code
    if status != 200:
        code = _error_code(response)
        detail = f"HTTP {status}: {response.text[:300]}"
        if status == 504:
            raise VideoError(504, "video preprocessor timed out", detail, "video_timeout")
        if status in (401, 403) or status >= 500:
            # An auth failure is this server's misconfiguration, not the client's.
            tail = f" ({code})" if code and status >= 500 else ""
            raise VideoError(
                502,
                f"video preprocessor returned an error{tail}",
                detail,
                code if (code and status >= 500) else "video_unavailable",
            )
        raise VideoError(
            400,
            f"video preprocessor refused the video{f' ({code})' if code else ''}",
            detail,
            code or "video_refused",
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise VideoError(
            502, "video preprocessor returned an unreadable answer", code="video_bad_answer"
        ) from exc
    if not isinstance(result, dict):
        raise VideoError(
            502, "video preprocessor returned an unreadable answer", code="video_bad_answer"
        )
    return result


def _transport_error(exc: httpx.HTTPError) -> VideoError:
    if isinstance(exc, httpx.TimeoutException):
        return VideoError(504, "video preprocessor timed out", repr(exc), "video_timeout")
    return VideoError(502, "video preprocessor is not available", repr(exc), "video_unavailable")


class _Totals:
    """What one request's videos add up to, for the INFO line."""

    def __init__(self) -> None:
        self.parts = 0
        self.seconds = 0.0
        self.scenes = 0
        self.frames = 0
        self.latency_ms = 0

    def add(self, result: dict[str, Any], rendered: list[dict[str, Any]], latency_ms: int) -> None:
        scenes = result.get("scenes")
        self.parts += 1
        self.seconds += num(result.get("duration_s"))
        self.scenes += len(scenes) if isinstance(scenes, list) else 0
        self.frames += sum(1 for p in rendered if p.get("type") == "image_url")
        self.latency_ms += latency_ms

    def log(self, request_id: str | None) -> None:
        logger.info(
            "preprocessed video: parts=%d seconds=%.1f scenes=%d frames=%d "
            "latency_ms=%d request_id=%s",
            self.parts,
            self.seconds,
            self.scenes,
            self.frames,
            self.latency_ms,
            request_id,
        )

    def summary(self) -> dict[str, Any]:
        return {"parts": self.parts, "scenes": self.scenes, "frames": self.frames}


def _log_warnings(result: dict[str, Any]) -> None:
    warnings = result.get("warnings")
    for warning in warnings[:5] if isinstance(warnings, list) else []:
        if isinstance(warning, str):
            logger.info("video preprocessor warning: %s", warning.split(":", 1)[0][:64])


def _replace(messages: list, replacements: dict[tuple[int, int], list[dict[str, Any]]]) -> list:
    """Return `messages` with each video part swapped for its parts, in place."""
    out = []
    for mi, message in enumerate(messages):
        if not any(key[0] == mi for key in replacements):
            out.append(message)
            continue
        content: list[Any] = []
        for pi, part in enumerate(message["content"]):
            content.extend(replacements.get((mi, pi), [part]))
        out.append({**message, "content": content})
    return out


def _post_args(video: bytes, subtype: str, form: dict[str, str]) -> dict[str, Any]:
    # A fixed filename: the client's own is never sent to the service.
    return {"data": form, "files": {"file": (f"video.{subtype}", video, f"video/{subtype}")}}


def preprocess_sync(
    messages: Any,
    config: VideoConfig,
    language: str | None = None,
    request_id: str | None = None,
    summary: dict[str, Any] | None = None,
) -> Any:
    """Replace every video part in `messages` with frames and a transcript.

    Parameters
    ----------
    messages : Any
        The request's messages. Not mutated.
    config : VideoConfig
        An active (`preprocess`) config.
    language : str, optional
        Language hint from the request; wins over `config.language`.
    request_id : str, optional
        Only written to the log summary.
    summary : dict, optional
        Filled with counts (parts, scenes, frames), for the audit record.

    Raises
    ------
    VideoError
        On any failure; the caller must not route the raw video.
    """
    replacements: dict[tuple[int, int], list[dict[str, Any]]] = {}
    totals = _Totals()
    timeout = httpx.Timeout(config.timeout_ms / 1000)
    decoded = [(mi, pi, _decode(part, config)) for mi, pi, part in _video_parts(messages)]

    with httpx.Client(timeout=timeout) as client:
        for number, (mi, pi, (video, subtype)) in enumerate(decoded, 1):
            url, headers, form = _request(config, language)
            started = time.monotonic()
            try:
                result = _parse(
                    client.post(url, headers=headers, **_post_args(video, subtype, form))
                )
            except httpx.HTTPError as exc:
                raise _transport_error(exc) from exc
            rendered = render(result, config, number)
            _log_warnings(result)
            totals.add(result, rendered, int((time.monotonic() - started) * 1000))
            replacements[(mi, pi)] = rendered

    totals.log(request_id)
    if summary is not None:
        summary.update(totals.summary())
    return _replace(messages, replacements)


async def preprocess(
    messages: Any,
    config: VideoConfig,
    language: str | None = None,
    request_id: str | None = None,
    summary: dict[str, Any] | None = None,
) -> Any:
    """Async form of `preprocess_sync`; same contract."""
    replacements: dict[tuple[int, int], list[dict[str, Any]]] = {}
    totals = _Totals()
    timeout = httpx.Timeout(config.timeout_ms / 1000)
    decoded = [(mi, pi, _decode(part, config)) for mi, pi, part in _video_parts(messages)]

    async with httpx.AsyncClient(timeout=timeout) as client:
        for number, (mi, pi, (video, subtype)) in enumerate(decoded, 1):
            url, headers, form = _request(config, language)
            started = time.monotonic()
            try:
                result = _parse(
                    await client.post(url, headers=headers, **_post_args(video, subtype, form))
                )
            except httpx.HTTPError as exc:
                raise _transport_error(exc) from exc
            rendered = render(result, config, number)
            _log_warnings(result)
            totals.add(result, rendered, int((time.monotonic() - started) * 1000))
            replacements[(mi, pi)] = rendered

    totals.log(request_id)
    if summary is not None:
        summary.update(totals.summary())
    return _replace(messages, replacements)
