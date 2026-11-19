"""Normalized stream events: the one place OpenAI chunks are parsed.

An adapter never reads a raw chunk. `stream_events` turns the OpenAI chunk
stream into a short, ordered list of event types; the adapter maps each to
its dialect's wire format. What the helper owns, so no adapter re-derives it:

- tool-call accumulation: deltas keyed by the choice and the call's
  `index`, joined into one `ToolCallEnd` carrying the whole arguments string;
- block boundaries: a tool call is closed when the next one starts, when
  text resumes, when the choice finishes, or when the stream ends;
- usage arrives after the finish chunk (`stream_options.include_usage`),
  so it is held and delivered once, just before `End`;
- upstreams that send a whole tool call in one chunk, or reuse index 0 for
  successive calls, produce the same events as ones that stream it in pieces.

Event order for one choice: `Start`, then any of `TextDelta`,
`ReasoningDelta`, `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd` in the
order they arrived, `Finish`, then (once, for the stream) `Usage`, `End`.
`index` is the choice index; a request without `n` only ever has 0.

A source that raises propagates the exception out of `stream_events`;
the framework turns it into the dialect's stream-error frame.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "End",
    "Error",
    "Event",
    "Finish",
    "ReasoningDelta",
    "Start",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "Usage",
    "stream_events",
]


@dataclass(frozen=True)
class Start:
    """First chunk seen: the completion's identity."""

    id: str | None
    model: str | None
    created: int | None


@dataclass(frozen=True)
class TextDelta:
    index: int
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """Reasoning text (`reasoning_content` / `reasoning` on the delta)."""

    index: int
    text: str


@dataclass(frozen=True)
class ToolCallStart:
    """A tool call opened. `call_index` orders the calls within the choice."""

    index: int
    call_index: int
    id: str | None
    name: str | None


@dataclass(frozen=True)
class ToolCallDelta:
    """A piece of the call's arguments JSON text."""

    index: int
    call_index: int
    arguments: str


@dataclass(frozen=True)
class ToolCallEnd:
    """A tool call closed. `arguments` is the complete JSON text (maybe "")."""

    index: int
    call_index: int
    id: str | None
    name: str | None
    arguments: str


@dataclass(frozen=True)
class Finish:
    """The choice's finish reason, raw OpenAI: stop, length, tool_calls, ..."""

    index: int
    reason: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None
    raw: dict


@dataclass(frozen=True)
class Error:
    """An error the upstream reported inside the stream (HTTP status was 200)."""

    message: str
    code: Any = None


@dataclass(frozen=True)
class End:
    """The stream is over. The summary is choice 0's, for adapters that
    emit a trailer built from the whole answer."""

    text: str
    tool_calls: tuple[ToolCallEnd, ...]
    finish_reason: str | None
    usage: Usage | None


Event = (
    Start
    | TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | Finish
    | Usage
    | Error
    | End
)


@dataclass
class _Call:
    call_index: int
    id: str | None
    name: str | None
    parts: list[str] = field(default_factory=list)


@dataclass
class _Choice:
    open: dict[int, _Call] = field(default_factory=dict)  # upstream tool index -> call
    next_call: int = 0
    text: list[str] = field(default_factory=list)
    closed: list[ToolCallEnd] = field(default_factory=list)
    finish: str | None = None


def _dump(chunk: Any) -> dict:
    if isinstance(chunk, dict):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    return dump() if dump is not None else {}


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return Usage(prompt, completion, raw.get("total_tokens") or prompt + completion, cached, raw)


def _close(index: int, state: _Choice) -> list[ToolCallEnd]:
    ended = []
    for call in sorted(state.open.values(), key=lambda c: c.call_index):
        end = ToolCallEnd(index, call.call_index, call.id, call.name, "".join(call.parts))
        state.closed.append(end)
        ended.append(end)
    state.open.clear()
    return ended


async def stream_events(chunks: AsyncIterator[Any]) -> AsyncIterator[Event]:
    """Parse an OpenAI chunk stream (objects or dicts) into normalized events."""
    choices: dict[int, _Choice] = {}
    usage: Usage | None = None
    started = False

    async for chunk in chunks:
        data = _dump(chunk)
        if not started:
            started = True
            yield Start(data.get("id"), data.get("model"), data.get("created"))

        error = data.get("error")
        if error:
            body = error if isinstance(error, dict) else {"message": str(error)}
            yield Error(body.get("message") or "upstream error", body.get("code"))

        found = _usage(data.get("usage"))
        if found is not None:
            usage = found

        for choice in data.get("choices") or []:
            index = choice.get("index") or 0
            state = choices.setdefault(index, _Choice())
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                yield ReasoningDelta(index, reasoning)

            text = delta.get("content")
            if isinstance(text, str) and text:
                for end in _close(index, state):
                    yield end
                state.text.append(text)
                yield TextDelta(index, text)

            for piece in delta.get("tool_calls") or []:
                upstream = piece.get("index") or 0
                function = piece.get("function") or {}
                call_id, name = piece.get("id"), function.get("name")
                current = state.open.get(upstream)
                # A new id on a known index is a new call: some upstreams
                # reuse index 0 for every call they make.
                if current is not None and call_id and current.id and call_id != current.id:
                    for end in _close(index, state):
                        yield end
                    current = None
                if current is None:
                    # Starting a call closes the previous one: the next
                    # block cannot open while the last is still open.
                    for end in _close(index, state):
                        yield end
                    current = _Call(state.next_call, call_id, name)
                    state.next_call += 1
                    state.open[upstream] = current
                    yield ToolCallStart(index, current.call_index, call_id, name)
                else:
                    current.id = current.id or call_id
                    current.name = current.name or name
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    current.parts.append(arguments)
                    yield ToolCallDelta(index, current.call_index, arguments)

            reason = choice.get("finish_reason")
            if reason:
                for end in _close(index, state):
                    yield end
                state.finish = reason
                yield Finish(index, reason)

    for index, state in sorted(choices.items()):
        for end in _close(index, state):
            yield end
    if usage is not None:
        yield usage
    first = choices.get(0) or _Choice()
    yield End("".join(first.text), tuple(first.closed), first.finish, usage)
