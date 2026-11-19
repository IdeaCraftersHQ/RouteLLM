"""`stream_events`: OpenAI chunks to normalized events, no server involved."""

import asyncio

import pytest

from routellm.adapters.stream import (
    End,
    Error,
    Finish,
    ReasoningDelta,
    Start,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    stream_events,
)


def chunk(delta=None, finish=None, usage=None, index=0, **extra):
    out = {"id": "c1", "model": "m", "created": 7, **extra}
    out["choices"] = [{"index": index, "delta": delta or {}, "finish_reason": finish}]
    if usage:
        out["usage"] = usage
    return out


def tool(index, id=None, name=None, args=None):
    function = {}
    if name is not None:
        function["name"] = name
    if args is not None:
        function["arguments"] = args
    piece = {"index": index, "function": function}
    if id:
        piece["id"] = id
    return {"tool_calls": [piece]}


def run(chunks):
    async def source():
        for c in chunks:
            yield c

    async def collect():
        return [e async for e in stream_events(source())]

    return asyncio.run(collect())


def test_text_stream_orders_start_text_finish_usage_end():
    events = run(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": "he"}),
            chunk({"content": "llo"}),
            chunk({}, finish="stop"),
            {"id": "c1", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
    )
    assert events == [
        Start("c1", "m", 7),
        TextDelta(0, "he"),
        TextDelta(0, "llo"),
        Finish(0, "stop"),
        Usage(3, 2, 5, None, {"prompt_tokens": 3, "completion_tokens": 2}),
        End("hello", (), "stop", events[-2]),
    ]


def test_tool_call_deltas_accumulate_into_one_end():
    events = run(
        [
            chunk({"role": "assistant", **tool(0, "call_1", "read_file", "")}),
            chunk(tool(0, args='{"p"')),
            chunk(tool(0, args=': "a.txt"}')),
            chunk({}, finish="tool_calls"),
        ]
    )
    assert events[1] == ToolCallStart(0, 0, "call_1", "read_file")
    assert [e for e in events if isinstance(e, ToolCallDelta)] == [
        ToolCallDelta(0, 0, '{"p"'),
        ToolCallDelta(0, 0, ': "a.txt"}'),
    ]
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end == ToolCallEnd(0, 0, "call_1", "read_file", '{"p": "a.txt"}')
    # closed before the finish reason, so a block never outlives its choice
    assert events.index(end) < events.index(Finish(0, "tool_calls"))
    assert events[-1].tool_calls == (end,)


def test_parallel_calls_close_in_order_each_with_own_arguments():
    events = run(
        [
            chunk(tool(0, "a", "one", '{"x":1}')),
            chunk(tool(1, "b", "two", '{"y":2}')),
            chunk({}, finish="tool_calls"),
        ]
    )
    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert [(e.call_index, e.name, e.arguments) for e in ends] == [
        (0, "one", '{"x":1}'),
        (1, "two", '{"y":2}'),
    ]
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert events.index(ends[0]) < events.index(starts[1])


def test_a_whole_call_in_one_chunk_equals_a_streamed_one():
    whole = run([chunk(tool(0, "c", "f", '{"a":1}')), chunk({}, finish="tool_calls")])
    parts = run(
        [
            chunk(tool(0, "c", "f", "")),
            chunk(tool(0, args='{"a"')),
            chunk(tool(0, args=":1}")),
            chunk({}, finish="tool_calls"),
        ]
    )
    pick = lambda evs: [e for e in evs if isinstance(e, ToolCallStart | ToolCallEnd)]  # noqa: E731
    assert pick(whole) == pick(parts)


def test_an_upstream_that_reuses_index_zero_for_each_call_still_yields_two():
    events = run(
        [
            chunk(tool(0, "a", "one", "{}")),
            chunk(tool(0, "b", "two", "{}")),
            chunk({}, finish="tool_calls"),
        ]
    )
    assert [e.id for e in events if isinstance(e, ToolCallEnd)] == ["a", "b"]


def test_text_after_a_tool_call_closes_it():
    events = run([chunk(tool(0, "a", "f", "{}")), chunk({"content": "done"})])
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("ToolCallEnd") < kinds.index("TextDelta")


def test_unfinished_stream_still_closes_calls_and_ends():
    events = run([chunk(tool(0, "a", "f", '{"k":'))])
    assert isinstance(events[-2], ToolCallEnd) and events[-2].arguments == '{"k":'
    assert events[-1] == End("", (events[-2],), None, None)


def test_reasoning_text_is_its_own_event():
    events = run([chunk({"reasoning_content": "hmm"}), chunk({"content": "x"})])
    assert ReasoningDelta(0, "hmm") in events and TextDelta(0, "x") in events


def test_cached_tokens_are_read_from_usage_details():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    events = run([chunk({"content": "x"}, usage=usage)])
    assert next(e for e in events if isinstance(e, Usage)).cached_tokens == 8


def test_an_in_stream_upstream_error_is_an_event():
    events = run([chunk({"content": ""}, finish="error", error={"code": 429, "message": "slow"})])
    assert Error("slow", 429) in events


def test_objects_with_model_dump_are_read_like_dicts():
    class Obj:
        def model_dump(self):
            return chunk({"content": "hi"})

    assert TextDelta(0, "hi") in run([Obj()])


def test_a_failing_source_propagates():
    async def source():
        yield chunk({"content": "a"})
        raise RuntimeError("upstream died")

    async def drain():
        return [e async for e in stream_events(source())]

    with pytest.raises(RuntimeError, match="upstream died"):
        asyncio.run(drain())
