"""Transcribe audio parts to text before a request is routed.

A chat request may carry `input_audio` (base64) or `audio_url` content
parts. With a transcriber configured and its mode set to `transcribe`,
each audio part is sent to an OpenAI `/v1/audio/transcriptions`
compatible service and replaced by a text part holding the transcript,
one line per segment. The request then routes as an ordinary text
request: no audio requirement is left, so any side of the tier may serve
it. With mode `passthrough` (or no transcriber at all) nothing here
runs and audio routes to an audio-capable side as before.

Configuration is a top-level `transcriber:` section, optionally
overridden key by key under a tier's own `transcriber:`::

    transcriber:
      api_base: http://127.0.0.1:9000/v1
      api_key_env: TRANSCRIBER_API_KEY
      model: whisper-1
      timeout_ms: 60000
      max_audio_bytes: 26214400
      language: null
      format: "[{start}-{end}] {speaker}: {text}"
      format_no_speaker: "[{start}-{end}] {text}"
      allow_urls: false
      tags: [local]
      mode: transcribe

The request is POSTed as multipart with `response_format=verbose_json`
and `timestamp_granularities[]=segment`. Each entry of `segments` may
carry a `speaker`, a diarization extension some services add and the
OpenAI API does not; when it is absent the line is rendered with
`format_no_speaker` and carries no label.

Everything fails closed. A transcriber that is down, slow, erroring or
sent too much audio ends the request with a `TranscriptionError`; the
raw audio is never forwarded instead, and no other transcriber is tried.
Neither the audio nor any transcript is ever logged above DEBUG; the
INFO summary holds counts and latency only.
"""

import asyncio
import base64
import binascii
import ipaddress
import logging
import os
import re
import socket
import string
import time
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_FORMAT_NO_SPEAKER",
    "LOCAL_TAG",
    "TranscriberConfig",
    "TranscriptionError",
    "format_transcript",
    "has_audio",
    "transcribe",
    "transcribe_sync",
]

#: The tag that marks a transcriber (or an endpoint) as running locally.
#: The same label endpoints already use for `tag:local` selection.
LOCAL_TAG = "local"

DEFAULT_FORMAT = "[{start}-{end}] {speaker}: {text}"
DEFAULT_FORMAT_NO_SPEAKER = "[{start}-{end}] {text}"

#: Placeholders a format template may use.
_PLACEHOLDERS = frozenset({"start", "end", "speaker", "text"})

#: Written in place of a transcript that holds no speech at all.
NO_SPEECH = "[no speech detected]"

_AUDIO_PART_TYPES = ("input_audio", "audio_url")

_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
_FORMAT_NAME = re.compile(r"^[A-Za-z0-9]{1,8}$")

#: Content types sent for the audio formats `input_audio` names.
_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
}


class TranscriptionError(Exception):
    """The audio could not be transcribed; the request must not go on.

    Attributes
    ----------
    status_code : int
        The HTTP status a server should answer with.
    public : str
        A message safe to show a client: it names nothing about the
        transcriber and echoes nothing it said.
    detail : str
        What went wrong, for the server log only. It may name the
        transcriber and quote its answer.
    """

    def __init__(self, status_code: int, public: str, detail: str = ""):
        super().__init__(detail or public)
        self.status_code = status_code
        self.public = public
        self.detail = detail or public


class TranscriberConfig(BaseModel):
    """One transcriber's settings, global or merged with a tier's.

    Attributes
    ----------
    mode : str
        `transcribe` replaces audio with its transcript; `passthrough`
        leaves audio alone, to route to an audio-capable side.
    api_base : str, optional
        Base URL of the OpenAI-compatible service, up to and including
        `/v1`. Required for mode `transcribe`.
    api_key_env : str, optional
        Name of the environment variable holding the key, read at call
        time. Never the key itself.
    model : str
        The transcription model name sent to the service.
    timeout_ms : int
        Bound on one transcription call, in milliseconds.
    max_audio_bytes : int
        Largest single audio part accepted, in decoded bytes. Seconds
        are not limited: the duration is not known without decoding.
    language : str, optional
        Default language hint; a request may override it. None lets
        the service detect it.
    format : str
        Template for a line that has a speaker. Placeholders: `start`,
        `end`, `speaker`, `text`.
    format_no_speaker : str
        Template for a line whose segment carries no speaker.
    allowed_url_hosts : list[str]
        With `allow_urls`, the exact hostnames an `audio_url` may name
        (compared lower-cased, without a trailing dot). Empty refuses
        every URL.
    allow_private_url_hosts : bool
        DANGEROUS. Lets an allowlisted host resolve to loopback,
        private, link-local or other non-public addresses. Off by
        default; turning it on lets a client reach internal services
        through any allowlisted name that points inward.
    allow_urls : bool
        Whether `audio_url` parts may be fetched. Default False: they
        are refused, because fetching a client-named URL from the
        server is a request-forgery vector.
    allow_public_base : bool
        DANGEROUS. Lets a `local`-tagged transcriber name a public host.
    tags : list[str]
        Labels; `local` declares the service runs on this machine or
        network, which a tier whose endpoints are all `local` requires.
        A `local` transcriber's `api_base` must then name a loopback,
        private or internal host.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["transcribe", "passthrough"] = "transcribe"
    api_base: str | None = None
    api_key_env: str | None = None
    model: str = "whisper-1"
    timeout_ms: int = Field(default=60_000, gt=0)
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    language: str | None = None
    format: str = DEFAULT_FORMAT
    format_no_speaker: str = DEFAULT_FORMAT_NO_SPEAKER
    allow_urls: StrictBool = False
    allowed_url_hosts: list[str] = Field(default_factory=list)
    allow_private_url_hosts: StrictBool = False
    allow_public_base: StrictBool = False
    tags: list[str] = Field(default_factory=list)

    @field_validator("format", "format_no_speaker")
    @classmethod
    def _validate_template(cls, value: str) -> str:
        try:
            fields = [name for _, name, _, _ in string.Formatter().parse(value) if name is not None]
        except ValueError as exc:
            raise ValueError(f"bad template {value!r}: {exc}") from exc
        unknown = sorted(set(fields) - _PLACEHOLDERS)
        if unknown:
            raise ValueError(
                f"unknown placeholder(s) {', '.join(unknown)} in {value!r}; "
                f"use {', '.join(sorted(_PLACEHOLDERS))}"
            )
        return value

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        if value is not None and not _LANGUAGE.match(value):
            raise ValueError(f"language {value!r} is not a language code such as 'en' or 'pt-BR'")
        return value

    @field_validator("api_base")
    @classmethod
    def _validate_base(cls, value: str | None) -> str | None:
        if value is not None and not re.match(r"^https?://[^\s/]+", value):
            raise ValueError(f"api_base {value!r} must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _needs_base(self) -> "TranscriberConfig":
        if self.mode == "transcribe" and not self.api_base:
            raise ValueError("mode `transcribe` needs `api_base`")
        return self

    @property
    def is_local(self) -> bool:
        """Whether this transcriber declares itself local."""
        return LOCAL_TAG in self.tags

    @property
    def active(self) -> bool:
        """Whether audio parts are to be transcribed under this config."""
        return self.mode == "transcribe"


# -- detection ---------------------------------------------------------------


def has_audio(messages: Any) -> bool:
    """Return whether any message carries an audio part."""
    return any(True for _ in _audio_parts(messages))


def _audio_parts(messages: Any):
    """Yield `(message_index, part_index, part)` for every audio part."""
    if not isinstance(messages, list):
        return
    for mi, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") in _AUDIO_PART_TYPES:
                yield mi, pi, part


# -- formatting --------------------------------------------------------------


def _stamp(seconds: Any) -> str:
    """Render seconds as `MM:SS.t`, or `H:MM:SS.t` from one hour up."""
    try:
        tenths = round(float(seconds) * 10)
    except (TypeError, ValueError):
        tenths = 0
    tenths = max(tenths, 0)
    whole, tenth = divmod(tenths, 10)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{tenth}"
    return f"{minutes:02d}:{secs:02d}.{tenth}"


def format_transcript(result: dict[str, Any], config: TranscriberConfig) -> str:
    """Render a `verbose_json` transcription as one line per segment.

    Speakers are numbered by first appearance within this audio, so a
    service's own ids (`SPEAKER_00`) become `Speaker 1`, `Speaker 2`.
    A segment without a speaker uses `format_no_speaker`. A result with
    no segments falls back to its bare `text`; one with nothing at all
    renders as `[no speech detected]`.

    Parameters
    ----------
    result : dict
        The service's parsed JSON answer.
    config : TranscriberConfig
        Supplies the two templates.

    Returns
    -------
    str
        The transcript, newline-joined.
    """
    lines: list[str] = []
    numbers: dict[str, int] = {}

    segments = result.get("segments")
    for segment in segments if isinstance(segments, list) else []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker")
        values = {
            "start": _stamp(segment.get("start")),
            "end": _stamp(segment.get("end")),
            "text": text,
        }
        if speaker is None or str(speaker) == "":
            lines.append(config.format_no_speaker.format(**values, speaker=""))
            continue
        number = numbers.setdefault(str(speaker), len(numbers) + 1)
        lines.append(config.format.format(**values, speaker=f"Speaker {number}"))

    if lines:
        return "\n".join(lines)

    bare = str(result.get("text") or "").strip()
    return bare or NO_SPEECH


def _speakers(result: dict[str, Any]) -> int:
    segments = result.get("segments")
    found = {
        str(s["speaker"])
        for s in (segments if isinstance(segments, list) else [])
        if isinstance(s, dict) and s.get("speaker") not in (None, "")
    }
    return len(found)


def _seconds(result: dict[str, Any]) -> float:
    """Return the audio's length: the stated duration, else the last end."""
    try:
        return float(result["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    ends = []
    segments = result.get("segments")
    for segment in segments if isinstance(segments, list) else []:
        try:
            ends.append(float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(ends, default=0.0)


# -- one request -------------------------------------------------------------


class _Totals:
    """What one request's transcriptions add up to, for the INFO line."""

    def __init__(self) -> None:
        self.parts = 0
        self.seconds = 0.0
        self.segments = 0
        self.speakers = 0
        self.latency_ms = 0

    def add(self, result: dict[str, Any], latency_ms: int) -> None:
        segments = result.get("segments")
        self.parts += 1
        self.seconds += _seconds(result)
        self.segments += len(segments) if isinstance(segments, list) else 0
        self.speakers = max(self.speakers, _speakers(result))
        self.latency_ms += latency_ms

    def log(self, request_id: str | None) -> None:
        logger.info(
            "transcribed audio: parts=%d seconds=%.1f segments=%d speakers=%d "
            "latency_ms=%d request_id=%s",
            self.parts,
            self.seconds,
            self.segments,
            self.speakers,
            self.latency_ms,
            request_id,
        )


def _decode(part: dict[str, Any], config: TranscriberConfig) -> tuple[bytes, str] | str:
    """Return `(audio, format)` for an `input_audio` part, or an `audio_url`.

    A string result is the URL of an `audio_url` part still to fetch,
    and is only ever returned when the config allows fetching.

    Raises
    ------
    TranscriptionError
        On a URL part the config does not allow, a malformed part, or
        audio over `max_audio_bytes`.
    """
    if part.get("type") == "audio_url":
        if not config.allow_urls:
            raise TranscriptionError(
                400,
                "audio_url parts are not accepted; send the audio as input_audio",
            )
        raw = part.get("audio_url")
        url = raw.get("url") if isinstance(raw, dict) else raw
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            raise TranscriptionError(400, "audio_url must be an http(s) URL")
        return url

    body = part.get("input_audio")
    data = body.get("data") if isinstance(body, dict) else None
    fmt = body.get("format") if isinstance(body, dict) else None
    if not isinstance(data, str) or not data:
        raise TranscriptionError(400, "input_audio needs base64 `data`")
    if not isinstance(fmt, str) or not _FORMAT_NAME.match(fmt):
        raise TranscriptionError(400, "input_audio needs a `format` such as wav or mp3")

    # Refuse before decoding what is plainly too big.
    if len(data) * 3 // 4 > config.max_audio_bytes + 3:
        raise TranscriptionError(
            413, "audio part is larger than this server accepts", f"base64 length {len(data)}"
        )
    try:
        audio = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TranscriptionError(400, "input_audio `data` is not valid base64") from exc
    if len(audio) > config.max_audio_bytes:
        raise TranscriptionError(
            413, "audio part is larger than this server accepts", f"{len(audio)} bytes"
        )
    return audio, fmt.lower()


def _request(config: TranscriberConfig, audio: bytes, fmt: str, language: str | None):
    """Return `(url, headers, data, files)` for one transcription POST."""
    headers = {}
    if config.api_key_env:
        key = os.environ.get(config.api_key_env)
        if key is None:
            raise TranscriptionError(
                502,
                "transcriber is not available",
                f"environment variable {config.api_key_env!r} is not set",
            )
        headers["Authorization"] = f"Bearer {key}"

    data = {
        "model": config.model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    lang = language or config.language
    if lang:
        if not _LANGUAGE.match(lang):
            raise TranscriptionError(400, "language is not a language code such as en or pt-BR")
        data["language"] = lang

    files = {"file": (f"audio.{fmt}", audio, _MIME.get(fmt, "application/octet-stream"))}
    url = f"{(config.api_base or '').rstrip('/')}/audio/transcriptions"
    return url, headers, data, files


def _parse(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise TranscriptionError(
            502,
            "transcriber returned an error",
            f"HTTP {response.status_code}: {response.text[:300]}",
        )
    try:
        result = response.json()
    except ValueError as exc:
        raise TranscriptionError(502, "transcriber returned an unreadable answer") from exc
    if not isinstance(result, dict):
        raise TranscriptionError(502, "transcriber returned an unreadable answer")
    return result


def _transport_error(exc: httpx.HTTPError) -> TranscriptionError:
    if isinstance(exc, httpx.TimeoutException):
        return TranscriptionError(504, "transcriber timed out", repr(exc))
    return TranscriptionError(502, "transcriber is not available", repr(exc))


def _url_error(exc: Exception) -> TranscriptionError:
    return TranscriptionError(502, "audio_url could not be fetched", repr(exc))


def _fetched(url: str, content_type: str, body: bytes, config: TranscriberConfig):
    if len(body) > config.max_audio_bytes:
        raise TranscriptionError(
            413, "audio part is larger than this server accepts", f"{len(body)} bytes"
        )
    suffix = url.split("?", 1)[0].rsplit(".", 1)[-1].lower()
    if _FORMAT_NAME.match(suffix) and suffix in _MIME:
        return body, suffix
    for name, mime in _MIME.items():
        if content_type.split(";", maxsplit=1)[0].strip() == mime:
            return body, name
    return body, "wav"


def _replace(messages: list, replacements: dict[tuple[int, int], str]) -> list:
    """Return `messages` with each audio part swapped for a text part."""
    out = []
    for mi, message in enumerate(messages):
        if not any(key[0] == mi for key in replacements):
            out.append(message)
            continue
        content = [
            {"type": "text", "text": replacements[(mi, pi)]} if (mi, pi) in replacements else part
            for pi, part in enumerate(message["content"])
        ]
        out.append({**message, "content": content})
    return out


def transcribe_sync(
    messages: Any,
    config: TranscriberConfig,
    language: str | None = None,
    request_id: str | None = None,
) -> Any:
    """Replace every audio part in `messages` with its transcript.

    Parameters
    ----------
    messages : Any
        The request's messages. Not mutated.
    config : TranscriberConfig
        An active transcriber config.
    language : str, optional
        Language hint from the request; wins over `config.language`.
    request_id : str, optional
        Only written to the log summary.

    Returns
    -------
    Any
        A copy of `messages` with text parts where audio parts were.

    Raises
    ------
    TranscriptionError
        On any failure; the caller must not route the raw audio.
    """
    replacements: dict[tuple[int, int], str] = {}
    totals = _Totals()
    timeout = httpx.Timeout(config.timeout_ms / 1000)

    with httpx.Client(timeout=timeout) as client:
        for mi, pi, part in _audio_parts(messages):
            source = _decode(part, config)
            if isinstance(source, str):
                audio, fmt = _fetch_sync(client, source, config)
            else:
                audio, fmt = source
            url, headers, data, files = _request(config, audio, fmt, language)
            started = time.monotonic()
            try:
                result = _parse(client.post(url, headers=headers, data=data, files=files))
            except httpx.HTTPError as exc:
                raise _transport_error(exc) from exc
            totals.add(result, int((time.monotonic() - started) * 1000))
            replacements[(mi, pi)] = format_transcript(result, config)

    totals.log(request_id)
    return _replace(messages, replacements)


async def transcribe(
    messages: Any,
    config: TranscriberConfig,
    language: str | None = None,
    request_id: str | None = None,
) -> Any:
    """Async form of `transcribe_sync`; same contract."""
    replacements: dict[tuple[int, int], str] = {}
    totals = _Totals()
    timeout = httpx.Timeout(config.timeout_ms / 1000)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for mi, pi, part in _audio_parts(messages):
            source = _decode(part, config)
            if isinstance(source, str):
                audio, fmt = await _fetch_async(client, source, config)
            else:
                audio, fmt = source
            url, headers, data, files = _request(config, audio, fmt, language)
            started = time.monotonic()
            try:
                result = _parse(await client.post(url, headers=headers, data=data, files=files))
            except httpx.HTTPError as exc:
                raise _transport_error(exc) from exc
            totals.add(result, int((time.monotonic() - started) * 1000))
            replacements[(mi, pi)] = format_transcript(result, config)

    totals.log(request_id)
    return _replace(messages, replacements)


def _resolve(host: str, port: int) -> list[str]:
    """Return every address `host` resolves to, as strings."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise TranscriptionError(502, "audio_url could not be fetched", repr(exc)) from exc
    return [info[4][0] for info in infos]


def _public(address: str) -> bool:
    """Return whether `address` is a global unicast address."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _pin(url: str, config: TranscriberConfig) -> tuple[str, dict[str, str], str]:
    """Validate an `audio_url` and pin it to one checked address.

    Returns `(pinned_url, headers, hostname)`: the URL with its host
    replaced by the validated IP, the `Host` header to send, and the
    name for TLS SNI and certificate checks. The name is resolved once,
    here, so a second DNS answer cannot move the connection.

    Raises
    ------
    TranscriptionError
        On a host off the allowlist, credentials in the URL, or any
        resolved address that is not public (unless the operator opted
        in to private ones).
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").rstrip(".").lower()
    allowed = {h.rstrip(".").lower() for h in config.allowed_url_hosts}
    if not host or host not in allowed or parsed.username or parsed.password:
        raise TranscriptionError(400, "audio_url host is not allowed", f"refused {host!r}")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise TranscriptionError(400, "audio_url is malformed") from exc

    addresses = _resolve(host, port)
    if not addresses:
        raise TranscriptionError(502, "audio_url could not be fetched", f"{host} did not resolve")
    if not config.allow_private_url_hosts and not all(_public(a) for a in addresses):
        raise TranscriptionError(
            400, "audio_url host is not allowed", f"{host} resolves to {addresses}"
        )

    ip = sorted(addresses, key=lambda a: ":" in a)[0].split("%", 1)[0]
    netloc = f"[{ip}]" if ":" in ip else ip
    if parsed.port:
        netloc += f":{parsed.port}"
    pinned = parsed._replace(netloc=netloc).geturl()
    header_host = host + (f":{parsed.port}" if parsed.port else "")
    return pinned, {"Host": header_host}, host


def _fetch_sync(client: httpx.Client, url: str, config: TranscriberConfig):
    pinned, headers, host = _pin(url, config)
    try:
        with client.stream(
            "GET", pinned, headers=headers, extensions={"sni_hostname": host}
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > config.max_audio_bytes:
                    break
            content_type = response.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        raise _url_error(exc) from exc
    return _fetched(url, content_type, bytes(body), config)


async def _fetch_async(client: httpx.AsyncClient, url: str, config: TranscriberConfig):
    pinned, headers, host = await asyncio.to_thread(_pin, url, config)
    try:
        async with client.stream(
            "GET", pinned, headers=headers, extensions={"sni_hostname": host}
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > config.max_audio_bytes:
                    break
            content_type = response.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        raise _url_error(exc) from exc
    return _fetched(url, content_type, bytes(body), config)
