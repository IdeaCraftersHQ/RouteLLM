"""A streamed completion that is only handed out once it has started.

litellm opens a stream lazily enough that a leaf can accept the call and
still fail on the first read. Routing has to learn that before it commits
to the leaf, because a sibling can only take over while nothing has been
sent to the client. `RoutedStream.open` therefore reads the first chunk
up front; a failure there is an ordinary leaf failure, and a stream that
survives it is replayed from that chunk.

After the first chunk the leaf is committed. A failure then reaches the
consumer as the exception it was, and the server turns it into an error
event; the sibling is never called.
"""

import copy
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def _close_upstream(*targets) -> None:
    """Close each distinct target that has a way to be closed.

    Prefers `aclose`, falls back to `close`; either may be sync or
    async. A failure to close is logged and swallowed: it must never
    replace the error that made closing necessary.
    """
    seen: list = []
    for target in targets:
        if target is None or any(target is s for s in seen):
            continue
        seen.append(target)
        closer = getattr(target, "aclose", None) or getattr(target, "close", None)
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning(f"Closing the upstream stream failed: {e!s}")


class RoutedStream:
    """An async iterator of chunks that reports once when it ends.

    `_hidden_params` mirrors what a non-streamed response carries, so
    the path attached to a stream is read the same way.

    Parameters
    ----------
    first : Any
        The chunk already read from `source`.
    source : AsyncIterator
        The upstream iterator, positioned after `first`.
    on_end : callable, optional
        Called once with `(chunks, error)` when the stream is exhausted
        (`error` None) or fails (`error` the exception). Never called for
        a consumer that stops early.
    stream : Any, optional
        The object `source` was taken from, when it is a different object
        and has its own way to be closed (a provider stream whose
        iterator is a generator, say). Closed along with `source`.
    restore : StreamRestore, optional
        Restores each chunk on its way out (a request transform's
        inverse). `on_end` still gets the chunks as the upstream sent
        them, the transformed form, so a trace never holds the restored
        text; only what the consumer reads is restored.

    The upstream is closed however the stream ends: exhausted, failed,
    or abandoned by `aclose()`.
    """

    def __init__(
        self,
        first,
        source,
        on_end: Callable[[list, Exception | None], Any] | None = None,
        stream=None,
        restore=None,
    ):
        self._pending = [first]
        self._restore = restore
        self._out: list = []
        self._drained = False
        self._chunks: list = []
        self._source = source
        self._stream = stream
        self._closed = False
        self._on_end = on_end
        self._ended = False
        self._hidden_params: dict[str, Any] = {}
        # The upstream's response headers, as litellm records them on its
        # own stream, so a header read off a stream reads the same way.
        upstream = getattr(stream, "_hidden_params", None)
        if isinstance(upstream, dict) and "additional_headers" in upstream:
            self._hidden_params["additional_headers"] = upstream["additional_headers"]

    @classmethod
    async def open(cls, stream, on_end=None, restore=None) -> "RoutedStream":
        """Read the first chunk of `stream` and return the primed stream.

        Raises whatever the first read raises, and `RuntimeError` for a
        stream that ends without a single chunk, so the caller can treat
        both as the leaf failing. In every one of those cases, a
        cancellation (a timeout around this call) included, the upstream
        is closed before the exception leaves.
        """
        source = stream.__aiter__()
        try:
            first = await source.__anext__()
        except StopAsyncIteration:
            await _close_upstream(source, stream)
            raise RuntimeError("upstream stream ended before its first chunk") from None
        except BaseException:
            await _close_upstream(source, stream)
            raise
        return cls(first, source, on_end, stream=stream, restore=restore)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if self._out:
                return self._out.pop(0)
            try:
                if self._drained:
                    raise StopAsyncIteration
                chunk = self._pending.pop() if self._pending else await self._source.__anext__()
            except StopAsyncIteration:
                if self._restore is not None and not self._drained:
                    # Text still held back goes out before the end.
                    self._drained = True
                    self._out = self._restore.finish()
                    continue
                self._finish(None)
                await self.aclose()
                raise
            except Exception as e:
                self._finish(e)
                await self.aclose()
                raise
            self._chunks.append(chunk)
            if self._restore is None:
                return chunk
            try:
                # A copy: `_chunks` is what the trace records, and it must
                # keep the form the upstream sent.
                self._out = self._restore.feed(copy.deepcopy(chunk))
            except Exception as e:
                self._finish(e)
                await self.aclose()
                raise

    def _finish(self, error: Exception | None) -> None:
        if self._ended:
            return
        self._ended = True
        if self._on_end is None:
            return
        try:
            self._on_end(self._chunks, error)
        except Exception as e:
            logger.error(f"Stream end hook failed: {e!s}")

    async def aclose(self) -> None:
        """Close the upstream stream, once, if it has a way to be closed."""
        if self._closed:
            return
        self._closed = True
        await _close_upstream(self._source, self._stream)
