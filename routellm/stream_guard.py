"""Bookkeeping every streamed response shares, whatever its wire format.

A stream that fails after its first chunk has already sent a 200, and a
caller that goes away cancels the generator mid-stream. Either way the
upstream must be closed and the audit trail told how the stream ended.
`StreamGuard` holds that, so the SSE framing of `/v1/chat/completions`
and the framing of a dialect adapter cannot drift apart.
"""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, NamedTuple

__all__ = ["StreamFrame", "StreamGuard"]


class StreamFrame(NamedTuple):
    """A framed stream: the bytes (or text) to send, and how to label them."""

    content: AsyncIterator[Any]
    media_type: str
    headers: dict[str, str]


class StreamGuard:
    """Tracks one stream's outcome and closes its upstream exactly once.

    Usage: call `saw` per chunk, `ok` when the stream ended cleanly,
    `failed` on an exception, and `await finish()` in a `finally`.
    `done(outcome, reason, tokens_in, tokens_out)`, when given, is called
    once from `finish`: `served`, or `failed` for an upstream error or a
    caller that went away.
    """

    def __init__(self, response: Any, request_id: str, done: Callable[..., None] | None = None):
        self.response = response
        self.request_id = request_id
        self.done = done
        self.outcome, self.reason, self.usage = "failed", "client_disconnect", None

    def saw(self, chunk: Any) -> None:
        self.usage = getattr(chunk, "usage", None) or self.usage

    def ok(self) -> None:
        self.outcome, self.reason = "served", None

    def failed(self, exc: Exception) -> None:
        """Record a failure. The exception can name the upstream and echo its
        text, so it is logged here under the request id and never sent."""
        self.outcome, self.reason = "failed", "stream_error"
        logging.getLogger(__name__).error(
            f"Stream failed after its first chunk (request id {self.request_id}): "
            f"{type(exc).__name__}: {exc!s}"
        )

    async def finish(self) -> None:
        # A client that disconnects cancels the generator mid-stream;
        # without this the upstream connection (and the provider's
        # generation) would run on until garbage collection.
        try:
            closer = getattr(self.response, "aclose", None)
            if closer is not None:
                await closer()
        finally:
            if self.done is not None:
                self.done(
                    self.outcome,
                    self.reason,
                    getattr(self.usage, "prompt_tokens", None),
                    getattr(self.usage, "completion_tokens", None),
                )
