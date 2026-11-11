"""Turn a video preprocessor's answer into the message parts that replace the video."""

import re
from typing import Any

from routellm.video_config import VideoConfig, VideoError

__all__ = ["NO_SPEECH", "num", "render"]

_IMAGE_URL = re.compile(r"^data:image/(jpeg|png|webp);base64,", re.IGNORECASE)

#: Written in place of a transcript that holds no speech at all.
NO_SPEECH = "[no speech detected]"


def _bad(why: str) -> VideoError:
    return VideoError(
        502, "video preprocessor returned an unreadable answer", why, "video_bad_answer"
    )


def num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(float(value), 0.0)


def _clock(seconds: Any) -> str:
    whole = int(num(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image(url: Any) -> dict[str, Any]:
    # Only an inline image is ever forwarded: a URL from the service would
    # make the leaf fetch whatever it names.
    if not isinstance(url, str) or not _IMAGE_URL.match(url):
        raise _bad("an image is not a data:image URL")
    return {"type": "image_url", "image_url": {"url": url}}


def _bounded(lines: list[str], limit: int) -> tuple[str, bool]:
    """Join `lines` while they fit in `limit` characters; say if any were cut."""
    out: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > limit:
            if not out:
                out.append(line[:limit])
            return "\n".join(out), True
        out.append(line)
        used += cost
    return "\n".join(out), False


def _transcript_lines(transcript: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    segments = transcript.get("segments")
    for segment in segments if isinstance(segments, list) else []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker")
        label = (
            f"{_clock(segment.get('start_s'))} {speaker}"
            if speaker
            else _clock(segment.get("start_s"))
        )
        lines.append(f"[{label}] {text}")
    if not lines:
        bare = str(transcript.get("text") or "").strip()
        lines = [bare] if bare else [NO_SPEECH]
    return lines


def render(result: dict[str, Any], config: VideoConfig, number: int = 1) -> list[dict[str, Any]]:
    """Turn one service answer into the content parts that replace the video.

    Parameters
    ----------
    result : dict
        The service's parsed answer.
    config : VideoConfig
        Supplies the layout and the length bound.
    number : int
        The video's 1-based position in the request, for the header.

    Returns
    -------
    list[dict]
        A header text, per scene a label and its image parts (or one
        contact sheet), then `Transcript:` and on-screen text parts.

    Raises
    ------
    VideoError
        502 when the answer is not in the contract's shape.
    """
    scenes = result.get("scenes")
    if not isinstance(scenes, list):
        raise _bad("scenes is not a list")
    duration = num(result.get("duration_s"))
    noun = "scene" if len(scenes) == 1 else "scenes"
    parts = [_text(f"Video {number}: {duration:.1f}s, {len(scenes)} {noun}")]

    sheet = result.get("contact_sheet")
    if sheet is not None:
        notes = []
        for i, scene in enumerate(scenes, 1):
            if not isinstance(scene, dict):
                raise _bad("a scene is not an object")
            frames = scene.get("frames")
            stamps = (
                ", ".join(f"{num(f.get('t_s')):.1f}" for f in frames if isinstance(f, dict))
                if isinstance(frames, list)
                else ""
            )
            notes.append(
                f"[scene {i}, {num(scene.get('start_s')):.1f}-{num(scene.get('end_s')):.1f} s]"
                + (f" frames at {stamps} s" if stamps else "")
            )
        parts.append(_text("Contact sheet, tiles in time order:\n" + "\n".join(notes)))
        parts.append(_image(sheet))
    else:
        for i, scene in enumerate(scenes, 1):
            if not isinstance(scene, dict):
                raise _bad("a scene is not an object")
            parts.append(
                _text(
                    f"[scene {i}, {num(scene.get('start_s')):.1f}-{num(scene.get('end_s')):.1f} s]"
                )
            )
            frames = scene.get("frames")
            for frame in frames if isinstance(frames, list) else []:
                if isinstance(frame, dict) and frame.get("image") is not None:
                    parts.append(_image(frame["image"]))

    transcript = result.get("transcript")
    if isinstance(transcript, dict):
        body, cut = _bounded(_transcript_lines(transcript), config.max_transcript_chars)
        note = "\n[transcript truncated]" if cut else ""
        parts.append(_text(f"Transcript:\n{body}{note}"))
    elif config.transcribe:
        why = "no audio track" if result.get("has_audio") is False else "not available"
        parts.append(_text(f"Transcript: [{why}]"))

    ocr = [
        f"[scene {i}] {str(s.get('ocr_text')).strip()}"
        for i, s in enumerate(scenes, 1)
        if isinstance(s, dict) and str(s.get("ocr_text") or "").strip()
    ]
    if ocr:
        body, cut = _bounded(ocr, config.max_transcript_chars)
        note = "\n[on-screen text truncated]" if cut else ""
        parts.append(_text(f"On-screen text:\n{body}{note}"))
    return parts
