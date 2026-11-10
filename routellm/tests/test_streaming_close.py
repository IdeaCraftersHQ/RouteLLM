"""An upstream stream is closed on every way out of a `RoutedStream`.

An unclosed stream holds its HTTP connection (and the provider's
generation) until garbage collection. These cases use a fake source that
counts `aclose` calls; no provider and no network.
"""

import asyncio

import pytest

from routellm.streaming import RoutedStream


class Source:
    """An async iterator over `items` that records how often it is closed."""

    def __init__(self, items=(), error=None, hang=False):
        self._items = list(items)
        self._error = error
        self._hang = hang
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._hang:
            await asyncio.Event().wait()
        if self._items:
            return self._items.pop(0)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def aclose(self):
        self.closed += 1


def _run(coro):
    return asyncio.run(coro)


def test_a_peek_that_times_out_closes_the_upstream():
    source = Source(hang=True)

    async def go():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(RoutedStream.open(source), timeout=0.05)

    _run(go())

    assert source.closed == 1


def test_an_empty_stream_closes_the_upstream():
    source = Source()

    async def go():
        with pytest.raises(RuntimeError):
            await RoutedStream.open(source)

    _run(go())

    assert source.closed == 1


def test_a_first_read_that_raises_closes_the_upstream():
    source = Source(error=ConnectionError("down"))

    async def go():
        with pytest.raises(ConnectionError):
            await RoutedStream.open(source)

    _run(go())

    assert source.closed == 1


def test_a_consumer_that_stops_early_closes_the_upstream_once():
    source = Source(items=["a", "b", "c"])

    async def go():
        stream = await RoutedStream.open(source)
        assert await stream.__anext__() == "a"
        await stream.aclose()
        await stream.aclose()

    _run(go())

    assert source.closed == 1


def test_a_failure_after_the_first_chunk_closes_the_upstream():
    source = Source(items=["a"], error=ConnectionError("dropped"))

    async def go():
        stream = await RoutedStream.open(source)
        with pytest.raises(ConnectionError):
            async for _ in stream:
                pass

    _run(go())

    assert source.closed == 1


def test_an_exhausted_stream_closes_the_upstream():
    source = Source(items=["a", "b"])

    async def go():
        stream = await RoutedStream.open(source)
        return [chunk async for chunk in stream]

    assert _run(go()) == ["a", "b"]
    assert source.closed == 1


def test_the_stream_object_is_closed_when_it_differs_from_its_iterator():
    class Wrapper:
        """Like a provider stream: iterates through a separate object."""

        def __init__(self):
            self.inner = Source(items=["a"])
            self.closed = 0

        def __aiter__(self):
            return self.inner

        async def aclose(self):
            self.closed += 1

    wrapper = Wrapper()

    async def go():
        stream = await RoutedStream.open(wrapper)
        await stream.aclose()

    _run(go())

    assert wrapper.closed == 1
    assert wrapper.inner.closed == 1


def test_a_source_that_only_has_a_sync_close_is_closed():
    class SyncClose(Source):
        aclose = None

        def close(self):
            self.closed += 1

    source = SyncClose(items=["a"])

    async def go():
        stream = await RoutedStream.open(source)
        await stream.aclose()

    _run(go())

    assert source.closed == 1


def test_a_failing_close_does_not_mask_the_original_error():
    class BadClose(Source):
        async def aclose(self):
            raise OSError("close failed")

    source = BadClose(error=ConnectionError("down"))

    async def go():
        with pytest.raises(ConnectionError):
            await RoutedStream.open(source)

    _run(go())
