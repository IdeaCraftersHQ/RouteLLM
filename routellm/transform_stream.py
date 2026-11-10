"""Restoring a transformed reply chunk by chunk.

A stream carries text in three places: `delta.content` (and
`reasoning_content`), each tool call's `function.arguments` (keyed by the
call's index) and a legacy `function_call.arguments`. Each of them, per
choice, gets its own restorer, because a token can straddle two chunks and
the two fields of one chunk are different texts.

Chunks map one to one, with text restored in place, except where a
restorer is holding back a possible partial token: that text comes out on
a later chunk, and whatever is still held when a choice finishes (or the
stream ends) goes out on a synthetic chunk placed before the finish, so
nothing is lost and no restored text follows a `finish_reason`.
"""

import copy
from collections.abc import Callable
from typing import Any

from routellm.transform import StreamRestorer

_TEXT = ("content", "reasoning_content")


def _fields(choice) -> list[tuple[Any, Any]]:
    """The `(key, text)` pairs a chunk's delta carries, key local to the choice."""
    delta = getattr(choice, "delta", None)
    if delta is None:
        return []
    found = []
    for name in _TEXT:
        value = getattr(delta, name, None)
        if isinstance(value, str):
            found.append((name, value))
    for position, call in enumerate(getattr(delta, "tool_calls", None) or ()):
        function = getattr(call, "function", None)
        args = getattr(function, "arguments", None)
        if isinstance(args, str):
            index = getattr(call, "index", None)
            found.append((("tool", position if index is None else index), args))
    legacy = getattr(delta, "function_call", None)
    if legacy is not None and isinstance(getattr(legacy, "arguments", None), str):
        found.append(("function_call", legacy.arguments))
    return found


def _set(choice, key, text: str) -> bool:
    """Write `text` into the field `key` of the choice's delta; False when absent."""
    delta = choice.delta
    if key in _TEXT:
        setattr(delta, key, text)
        return True
    if key == "function_call":
        delta.function_call.arguments = text
        return True
    for position, call in enumerate(getattr(delta, "tool_calls", None) or ()):
        index = getattr(call, "index", None)
        if (position if index is None else index) == key[1]:
            call.function.arguments = text
            return True
    return False


def _only(template, choice, key, text: str):
    """A copy of `template` carrying only `text` for `key` on `choice`."""
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    out = copy.deepcopy(template)
    out.choices = [c for c in out.choices if c.index == choice.index][:1]
    mine = out.choices[0]
    mine.finish_reason = None
    delta = mine.delta
    holder = None
    if isinstance(key, tuple):
        holder = next(
            (
                c
                for position, c in enumerate(getattr(delta, "tool_calls", None) or ())
                if (position if getattr(c, "index", None) is None else c.index) == key[1]
            ),
            None,
        )
    for name in ("content", "reasoning_content", "role", "function_call", "tool_calls"):
        if hasattr(delta, name):
            setattr(delta, name, None)
    if hasattr(out, "usage"):
        out.usage = None
    if isinstance(key, tuple):
        call = copy.deepcopy(holder) if holder is not None else None
        if call is None:
            call = ChatCompletionDeltaToolCall(index=key[1], function=Function(arguments=""))
        call.id = None
        if getattr(call.function, "name", None):
            call.function.name = None
        call.function.arguments = text
        delta.tool_calls = [call]
    elif key == "function_call":
        legacy = copy.deepcopy(getattr(template.choices[0].delta, "function_call", None))
        if legacy is not None:
            legacy.name = None
            legacy.arguments = text
            delta.function_call = legacy
    else:
        setattr(delta, key, text)
    return out


class StreamRestore:
    """Restores the chunks of one stream, holding back only partial tokens.

    Parameters
    ----------
    make : callable
        Returns a fresh `StreamRestorer` for each text field it meets.
    """

    def __init__(self, make: Callable[[], StreamRestorer]):
        self._make = make
        self._held: dict[tuple, StreamRestorer] = {}
        self._last = None

    def _restorer(self, index, key) -> StreamRestorer:
        slot = (index, key)
        if slot not in self._held:
            self._held[slot] = self._make()
        return self._held[slot]

    def feed(self, chunk) -> list:
        """Restore `chunk`; return the chunks to send in its place."""
        self._last = chunk
        before: list = []
        for choice in getattr(chunk, "choices", None) or ():
            index = getattr(choice, "index", 0)
            present = dict(_fields(choice))
            for key, text in present.items():
                _set(choice, key, self._restorer(index, key).feed(text))
            if getattr(choice, "finish_reason", None) is None:
                continue
            # The choice is done: release every field still holding text.
            for (held_index, key), restorer in list(self._held.items()):
                if held_index != index:
                    continue
                tail = restorer.finish()
                del self._held[(held_index, key)]
                if not tail:
                    continue
                if key in present:
                    self._append(choice, key, tail)
                else:
                    before.append(_only(chunk, choice, key, tail))
        return [*before, chunk]

    @staticmethod
    def _append(choice, key, tail: str) -> None:
        current = dict(_fields(choice)).get(key, "")
        _set(choice, key, current + tail)

    def finish(self) -> list:
        """The stream ended: chunks for whatever is still held, else none."""
        out = []
        for (index, key), restorer in list(self._held.items()):
            tail = restorer.finish()
            if tail and self._last is not None:
                choice = next((c for c in self._last.choices if c.index == index), None)
                if choice is not None:
                    out.append(_only(self._last, choice, key, tail))
        self._held.clear()
        return out


class RestoringIterator:
    """A synchronous stream whose chunks are restored on their way out.

    For `Controller.completion`, which hands the provider's stream back
    as it is. Carries `_hidden_params` like the stream it wraps and
    closes it with `close()`.
    """

    def __init__(self, stream, restore: StreamRestore):
        self._stream = stream
        self._source = iter(stream)
        self._restore = restore
        self._out: list = []
        self._drained = False
        self._hidden_params: dict[str, Any] = {}

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._out:
                return self._out.pop(0)
            if self._drained:
                raise StopIteration
            try:
                chunk = next(self._source)
            except StopIteration:
                self._drained = True
                self._out = self._restore.finish()
                continue
            self._out = self._restore.feed(chunk)

    def close(self) -> None:
        closer = getattr(self._stream, "close", None)
        if closer is not None:
            closer()
