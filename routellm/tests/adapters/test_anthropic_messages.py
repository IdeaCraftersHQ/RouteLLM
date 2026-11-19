"""The Anthropic Messages adapter: litellm contract pins, mapping, the harness.

Three layers:

- `TestLitellmContract` pins what the adapter takes from litellm (the
  translator class, its method signatures and the shapes it returns). A
  litellm upgrade that moves or changes any of it fails here, by name,
  before it reaches a client.
- `TestDecode` / `TestEncode` / `TestStreamEvents` drive the adapter
  without a server: refusals, the request it builds, the answer it builds.
- `TestAnthropic` is the shared conformance suite; `TestSdk` and
  `TestWire` run the official `anthropic` client and raw requests against
  the harness server.
"""

import inspect
import json

import httpx
import pytest

from routellm.adapters import AdapterRefusal, Call
from routellm.adapters.anthropic import AnthropicAdapter
from routellm.tests.adapters.conformance import (
    HI,
    KEY,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
    ToolUse,
)

ADAPTER = AnthropicAdapter()
MESSAGES = ADAPTER.routes[0]


def call_for(body) -> Call:
    return Call(adapter=ADAPTER, route=MESSAGES, http_request=None, pipeline=None, body=body)


def base(**extra) -> dict:
    return {
        "model": "default",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def counted(**extra) -> dict:
    body = base(**extra)
    del body["max_tokens"]
    return body


def decode(body):
    call = call_for(body)
    return ADAPTER.decode(call), call


# -- litellm contract ---------------------------------------------------------


class TestLitellmContract:
    def test_translator_api_is_where_the_adapter_expects_it(self):
        from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter as T,
        )

        def params(fn):
            return set(inspect.signature(fn).parameters)

        assert {"anthropic_message_request", "custom_llm_provider"} <= params(
            T.translate_anthropic_to_openai
        )
        assert {"response", "tool_name_mapping"} <= params(T.translate_openai_response_to_anthropic)
        assert {"completion_stream", "model", "tool_name_mapping"} <= params(
            AnthropicStreamWrapper.__init__
        )
        assert params(T.translate_anthropic_thinking_to_reasoning_effort) == {"thinking"}
        assert "usage" in params(T.translate_openai_usage_to_anthropic_usage_delta)

    def test_translator_carries_top_k(self):
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter as T,
        )

        out, _ = T().translate_anthropic_to_openai(
            {
                "model": "routellm",
                "max_tokens": 10,
                "top_k": 4,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert out["top_k"] == 4

    def test_request_mapping(self):
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter as T,
        )

        out, names = T().translate_anthropic_to_openai(
            {
                "model": "routellm",
                "max_tokens": 10,
                "system": "sys",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "any"},
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "stop_sequences": ["x"],
                "metadata": {"user_id": "u1"},
            }
        )
        assert out["messages"][0] == {"role": "system", "content": "sys"}
        assert out["tools"][0]["function"]["name"] == "f"
        assert out["tools"][0]["function"]["parameters"] == {"type": "object"}
        assert out["tool_choice"] == "required"
        assert out["reasoning_effort"] == "medium"
        assert "thinking" not in out
        assert out["stop"] == ["x"]
        assert out["user"] == "u1"
        assert out["max_tokens"] == 10
        assert names == {}

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")],
    )
    def test_stop_reason_mapping(self, reason, expected):
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter as T,
        )

        assert T()._translate_openai_finish_reason_to_anthropic(reason) == expected

    def test_usage_mapping_splits_cached_tokens(self):
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter as T,
        )
        from litellm.types.utils import Usage

        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details={"cached_tokens": 4},
        )
        out = T.translate_openai_usage_to_anthropic_usage_delta(usage)
        assert out["input_tokens"] == 6
        assert out["output_tokens"] == 5
        assert out["cache_read_input_tokens"] == 4

    def test_stream_wrapper_event_order(self):
        import asyncio

        from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )

        async def go():
            async def src():
                for delta, finish in (({"content": "a"}, None), ({}, "stop")):
                    yield chunk({"index": 0, "delta": delta, "finish_reason": finish})
                yield chunk(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})

            return [e async for e in AnthropicStreamWrapper(src(), "m")]

        events = asyncio.run(go())
        assert [e["type"] for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[-2]["delta"]["stop_reason"] == "end_turn"
        assert events[-2]["usage"] == {"input_tokens": 1, "output_tokens": 2}


def chunk(*choices, usage=None):
    from litellm.types.utils import ModelResponseStream

    body = {"id": "c1", "created": 1, "model": "m", "choices": list(choices)}
    if usage:
        body["usage"] = usage
    return ModelResponseStream(**body)


def delta(finish=None, **fields):
    return {"index": 0, "delta": fields, "finish_reason": finish}


# -- decode -------------------------------------------------------------------


class TestDecode:
    def test_maps_the_fields_the_pipeline_carries(self):
        request, call = decode(
            base(
                system=[
                    {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}
                ],
                temperature=0.3,
                top_p=0.9,
                stop_sequences=["END"],
                metadata={"user_id": "u-1"},
                thinking={"type": "enabled", "budget_tokens": 1024},
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                tools=[{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
                stream=True,
            )
        )
        assert request.model == "default"
        assert request.max_tokens == 64
        assert request.temperature == 0.3 and request.top_p == 0.9
        assert request.stop == ["END"]
        assert request.user == "u-1"
        assert request.reasoning_effort == "low"
        assert request.tool_choice == "auto"
        assert request.parallel_tool_calls is False
        assert request.stream is True
        assert request.messages[0] == {
            "role": "system",
            "content": [{"type": "text", "text": "Be terse."}],
        }
        assert request.tools[0]["function"]["name"] == "f"
        assert call.state["model"] == "default"

    def test_the_clients_model_name_never_steers_the_translation(self):
        # A Claude model name would make litellm keep `thinking` verbatim.
        request, _ = decode(
            base(model="claude-sonnet-4-5", thinking={"type": "enabled", "budget_tokens": 2048})
        )
        assert request.model == "claude-sonnet-4-5"
        assert request.reasoning_effort == "medium"

    def test_blocks_become_chat_messages(self):
        request, _ = decode(
            base(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "QQ",
                                },
                            },
                            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hm", "signature": "sig"},
                            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "done"}],
                                "is_error": True,
                            }
                        ],
                    },
                ]
            )
        )
        user, assistant, tool = request.messages
        urls = [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]
        assert urls == ["data:image/png;base64,QQ", "https://x/y.png"]
        assert assistant["tool_calls"][0]["id"] == "toolu_1"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        assert assistant["thinking_blocks"] == [
            {"type": "thinking", "thinking": "hm", "signature": "sig"}
        ]
        assert tool == {"role": "tool", "tool_call_id": "toolu_1", "content": "done"}

    def test_cache_control_is_ignored_wherever_it_appears(self):
        cc = {"type": "ephemeral"}
        request, _ = decode(
            base(
                system=[{"type": "text", "text": "s", "cache_control": cc}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "q", "cache_control": cc}],
                    }
                ],
                tools=[{"name": "f", "input_schema": {"type": "object"}, "cache_control": cc}],
            )
        )
        assert "cache_control" not in json.dumps([request.messages, request.tools])

    def test_long_tool_names_are_shortened_and_remembered(self):
        long = "t" * 80
        request, call = decode(base(tools=[{"name": long, "input_schema": {"type": "object"}}]))
        short = request.tools[0]["function"]["name"]
        assert len(short) <= 64 and call.state["tool_names"] == {short: long}

    def test_structured_output_and_effort(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request, _ = decode(
            base(output_config={"format": {"type": "json_schema", "schema": schema}})
        )
        assert request.response_format["type"] == "json_schema"
        request, _ = decode(base(thinking={"type": "adaptive"}, output_config={"effort": "high"}))
        assert request.reasoning_effort == "high"

    def test_top_k_is_mapped(self):
        request, _ = decode(base(top_k=5))
        assert request.top_k == 5
        assert decode(base(top_k=0))[0].top_k == 0
        assert decode(base())[0].top_k is None

    def test_ignored_fields_are_accepted(self):
        decode(base(service_tier="auto"))

    def test_defer_loading_on_a_tool_is_accepted_and_not_forwarded(self):
        # Claude Code marks tools it would load lazily with `defer_loading`.
        request, _ = decode(
            base(tools=[{"name": "f", "input_schema": {"type": "object"}, "defer_loading": True}])
        )
        assert request.tools[0]["function"]["name"] == "f"
        assert "defer_loading" not in json.dumps(request.tools)

    def test_mid_conversation_system_message_is_accepted(self):
        # Claude Code sends its environment note as a `role: system` message
        # after the first user turn, text blocks with cache_control.
        note = {"type": "text", "text": "env note", "cache_control": {"type": "ephemeral"}}
        for content in ("env note", [note]):
            request, _ = decode(
                base(
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "system", "content": content},
                    ]
                )
            )
            dumped = json.dumps(request.model_dump(exclude_none=True))
            assert "env note" in dumped

    def test_context_management_is_accepted_and_nothing_of_it_is_forwarded(self):
        edits = {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"value": 9}}]}
        request, _ = decode(base(context_management=edits))
        assert "context_management" not in request.model_dump(exclude_none=True)
        assert "clear_tool_uses" not in json.dumps(request.model_dump())

    @pytest.mark.parametrize(
        ("patch", "param"),
        [
            ({"top_k": -1}, "top_k"),
            ({"top_k": "5"}, "top_k"),
            ({"top_k": True}, "top_k"),
            ({"mcp_servers": []}, "mcp_servers"),
            ({"container": "c"}, "container"),
            ({"metadata": {"user_id": "u", "tag": "x"}}, "metadata"),
            ({"tool_choice": {"type": "mystery"}}, "tool_choice"),
            ({"thinking": {"type": "mystery"}}, "thinking"),
            ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "tools"),
            ({"tools": [{"type": "bash_20250124", "name": "bash"}]}, "tools"),
            ({"output_config": {"mystery": 1}}, "output_config"),
            ({"system": [{"type": "image", "source": {}}]}, "system"),
            ({"messages": "hi"}, "messages"),
            ({"messages": [{"role": "tool", "content": "x"}]}, "messages"),
            (
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "system", "content": [{"type": "image", "source": {}}]},
                    ]
                },
                "messages",
            ),
            ({"messages": [{"role": "user", "content": [{"type": "mystery"}]}]}, "messages"),
            (
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "a", "x": 1}]}
                    ]
                },
                "messages",
            ),
            (
                {
                    "messages": [
                        {"role": "assistant", "content": [{"type": "server_tool_use", "id": "s"}]}
                    ]
                },
                "messages",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "document", "source": {"type": "text", "data": "t"}}
                            ],
                        }
                    ]
                },
                "messages",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": [{"type": "tool_reference", "tool_name": "x"}],
                                }
                            ],
                        }
                    ]
                },
                "messages",
            ),
        ],
    )
    def test_refuses_what_it_cannot_carry(self, patch, param):
        with pytest.raises(AdapterRefusal) as caught:
            decode(base(**patch))
        assert caught.value.param.split("[")[0].split(".")[0] == param
        assert caught.value.status == 400

    def test_a_field_the_translator_adds_is_refused_not_dropped(self, monkeypatch):
        import routellm.adapters.anthropic as module

        real = module._TRANSLATOR.translate_anthropic_to_openai

        def drifted(*args, **kwargs):
            out, names = real(*args, **kwargs)
            return {**out, "novel_field": 1}, names

        monkeypatch.setattr(module._TRANSLATOR, "translate_anthropic_to_openai", drifted)
        with pytest.raises(AdapterRefusal) as caught:
            decode(base())
        assert caught.value.param == "novel_field"

    def test_max_tokens_is_required(self):
        body = base()
        del body["max_tokens"]
        with pytest.raises(AdapterRefusal) as caught:
            decode(body)
        assert caught.value.param == "max_tokens"

    def test_document_blocks_travel_as_data_urls(self):
        request, _ = decode(
            base(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": "JVBER",
                                },
                            }
                        ],
                    }
                ]
            )
        )
        assert request.messages[0]["content"][0]["image_url"]["url"].startswith(
            "data:application/pdf;base64,"
        )


# -- encode -------------------------------------------------------------------


def completion(message=None, finish="stop", usage=None):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/model-a",
        "choices": [
            {
                "index": 0,
                "message": message or {"role": "assistant", "content": "ok"},
                "finish_reason": finish,
            }
        ],
        "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "routellm": {"path": [{"e": "a"}]},
    }


class TestEncode:
    def call(self, **state):
        call = call_for(base())
        call.state.update({"model": "default", "tool_names": {}, **state})
        return call

    def test_message_shape(self):
        out = ADAPTER.encode(completion(), self.call())
        assert out["type"] == "message" and out["role"] == "assistant"
        assert out["id"].startswith("msg_")
        assert out["model"] == "default"
        assert out["content"] == [{"type": "text", "text": "ok"}]
        assert out["stop_reason"] == "end_turn"
        assert out["stop_sequence"] is None
        assert out["usage"] == {"input_tokens": 3, "output_tokens": 2}
        assert "routellm" not in out

    @pytest.mark.parametrize(
        ("finish", "stop"),
        [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")],
    )
    def test_stop_reasons(self, finish, stop):
        assert ADAPTER.encode(completion(finish=finish), self.call())["stop_reason"] == stop

    def test_cached_tokens_become_cache_read(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        out = ADAPTER.encode(completion(usage=usage), self.call())
        assert out["usage"]["input_tokens"] == 6
        assert out["usage"]["cache_read_input_tokens"] == 4

    def test_tool_use_blocks_restore_long_names(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "short", "arguments": '{"a":1}'},
                }
            ],
        }
        out = ADAPTER.encode(
            completion(message, "tool_calls"), self.call(tool_names={"short": "t" * 80})
        )
        (block,) = out["content"]
        assert block == {"type": "tool_use", "id": "call_1", "name": "t" * 80, "input": {"a": 1}}

    def test_reasoning_becomes_a_thinking_block_with_a_string_signature(self):
        message = {"role": "assistant", "content": "yo", "reasoning_content": "hm"}
        out = ADAPTER.encode(completion(message), self.call())
        assert out["content"][0] == {"type": "thinking", "thinking": "hm", "signature": ""}
        assert out["content"][1]["type"] == "text"

    def test_signed_thinking_survives(self):
        message = {
            "role": "assistant",
            "content": "yo",
            "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "sig"}],
        }
        out = ADAPTER.encode(completion(message), self.call())
        assert out["content"][0]["signature"] == "sig"

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
        ],
    )
    def test_error_shape(self, status, kind):
        error = {"message": "m", "type": "x", "code": None, "param": None, "request_id": "r1"}
        out = ADAPTER.encode_error(status, error, self.call())
        assert out == {"type": "error", "error": {"type": kind, "message": "m"}, "request_id": "r1"}

    def test_error_without_request_id(self):
        error = {"message": "m", "type": "x", "code": None, "param": None, "request_id": None}
        assert "request_id" not in ADAPTER.encode_error(400, error, self.call())

    def test_stream_error_frame(self):
        error = {"message": "Upstream request failed", "type": "t", "code": None, "param": None}
        frame = ADAPTER.encode_stream_error({**error, "request_id": "r1"}, self.call())
        assert frame.startswith(b"event: error\ndata: ")
        data = json.loads(frame.split(b"data: ", 1)[1])
        assert data["type"] == "error" and data["error"]["type"] == "api_error"


# -- stream, no server --------------------------------------------------------


async def run_stream(chunks, **state):
    call = call_for(base(stream=True))
    call.state.update({"model": "default", "tool_names": {}, **state})

    async def source():
        for c in chunks:
            yield c

    frames = [f async for f in ADAPTER.encode_stream(source(), call)]
    events = []
    for frame in frames:
        head, data = frame.decode().rstrip("\n").split("\n", 1)
        payload = json.loads(data.removeprefix("data: "))
        assert head == f"event: {payload['type']}"
        events.append(payload)
    return events


class TestStreamEvents:
    @pytest.mark.asyncio
    async def test_ping_follows_message_start(self):
        events = await run_stream(
            [
                chunk(delta(content="a")),
                chunk(delta("stop")),
                chunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            ]
        )
        assert [e["type"] for e in events][:3] == ["message_start", "ping", "content_block_start"]
        assert events[-1]["type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_thinking_then_text(self):
        events = await run_stream(
            [
                chunk(delta(reasoning_content="hm")),
                chunk(delta(content="yo")),
                chunk(delta("stop")),
                chunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            ]
        )
        starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
        assert starts == ["thinking", "text"]
        kinds = [e["delta"]["type"] for e in events if e["type"] == "content_block_delta"]
        assert kinds == ["thinking_delta", "text_delta"]

    @pytest.mark.asyncio
    async def test_tool_call_streams_json_pieces(self):
        call = {"index": 0, "id": "call_1", "type": "function"}
        events = await run_stream(
            [
                chunk(delta(tool_calls=[{**call, "function": {"name": "short", "arguments": ""}}])),
                chunk(delta(tool_calls=[{"index": 0, "function": {"arguments": '{"p"'}}])),
                chunk(delta(tool_calls=[{"index": 0, "function": {"arguments": ': "a"}'}}])),
                chunk(delta("tool_calls")),
                chunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
            ],
            tool_names={"short": "long_name"},
        )
        (start,) = [e for e in events if e["type"] == "content_block_start"]
        assert start["content_block"]["type"] == "tool_use"
        assert start["content_block"]["id"] == "call_1"
        assert start["content_block"]["input"] == {}
        joined = "".join(
            e["delta"]["partial_json"] for e in events if e["type"] == "content_block_delta"
        )
        assert json.loads(joined) == {"p": "a"}
        stop = next(e for e in events if e["type"] == "message_delta")
        assert stop["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_dict_chunks_are_accepted(self):
        events = await run_stream(
            [
                {"id": "c", "created": 1, "model": "m", "choices": [delta(content="a")]},
                {"id": "c", "created": 1, "model": "m", "choices": [delta("stop")]},
            ]
        )
        assert events[-1]["type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_an_error_inside_the_stream_aborts_it(self):
        bad = {"id": "c", "created": 1, "model": "m", "choices": [], "error": {"message": "secret"}}
        with pytest.raises(RuntimeError):
            await run_stream([chunk(delta(content="a")), bad])

    @pytest.mark.asyncio
    async def test_a_failing_source_never_ends_cleanly(self):
        async def failing():
            yield chunk(delta(content="a"))
            raise RuntimeError("secret upstream detail")

        call = call_for(base(stream=True))
        call.state.update({"model": "default", "tool_names": {}})
        seen = []
        with pytest.raises(RuntimeError):
            async for frame in ADAPTER.encode_stream(failing(), call):
                seen.append(frame)
        assert b"message_stop" not in b"".join(seen)


# -- harness: conformance + wire ----------------------------------------------

HEADERS = {"anthropic-version": "2023-06-01"}


def _blocks(message) -> list[dict] | str:
    blocks: list[dict] = []
    if message.image_png_b64:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": message.image_png_b64,
                },
            }
        )
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    blocks += [
        {"type": "tool_use", "id": u.id, "name": u.name, "input": u.arguments}
        for u in message.tool_uses
    ]
    blocks += [
        {"type": "tool_result", "tool_use_id": r.id, "content": r.content}
        for r in message.tool_results
    ]
    return blocks


def sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        head, data = block.split("\n", 1)
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


class AnthropicDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        body = {
            "model": scenario.model,
            "max_tokens": 64,
            "messages": [{"role": m.role, "content": _blocks(m)} for m in scenario.messages],
            "stream": scenario.stream,
        }
        if scenario.system:
            body["system"] = scenario.system
        if scenario.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in scenario.tools
            ]
        return Http("POST", "/v1/messages", headers=dict(HEADERS), json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["x-api-key"] = key
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
        if not scenario.stream:
            body = response.json()
            assert body["type"] == "message" and body["role"] == "assistant"
            return Outcome(
                text="".join(b["text"] for b in body["content"] if b["type"] == "text"),
                tool_uses=[
                    ToolUse(b["id"], b["name"], b["input"])
                    for b in body["content"]
                    if b["type"] == "tool_use"
                ],
                finish=finish[body["stop_reason"]],
                input_tokens=body["usage"]["input_tokens"],
                output_tokens=body["usage"]["output_tokens"],
            )
        events = sse(response.text)
        assert events[0][0] == "message_start" and events[-1][0] == "message_stop"
        out, tools, stop = Outcome(), {}, None
        for name, data in events:
            if name == "content_block_start" and data["content_block"]["type"] == "tool_use":
                tools[data["index"]] = [data["content_block"], ""]
            elif name == "content_block_delta":
                d = data["delta"]
                if d["type"] == "text_delta":
                    out.text += d["text"]
                elif d["type"] == "input_json_delta":
                    tools[data["index"]][1] += d["partial_json"]
            elif name == "message_delta":
                stop = data["delta"]["stop_reason"]
                out.input_tokens = data["usage"]["input_tokens"]
                out.output_tokens = data["usage"]["output_tokens"]
        out.finish = finish[stop]
        out.tool_uses = [
            ToolUse(b["id"], b["name"], json.loads(args or "{}")) for b, args in tools.values()
        ]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        body = response.json()
        assert body["type"] == "error"
        assert set(body["error"]) == {"type", "message"}
        return body["error"]["message"]

    def unsupported(self) -> Http:
        http = self.build(Scenario((HI,)))
        http.json["mcp_servers"] = []
        return http


@pytest.mark.slow
class TestAnthropic(ConformanceSuite):
    driver = AnthropicDriver()


def post(harness, path, body, headers=None):
    return httpx.post(
        f"{harness.base}{path}",
        headers={**HEADERS, "x-api-key": KEY, **(headers or {})},
        json=body,
        timeout=60,
    )


@pytest.mark.slow
class TestWire:
    def test_both_credential_headers_work(self, harness):
        harness.upstream.mode = "echo"
        for headers in ({"x-api-key": KEY}, {"authorization": f"Bearer {KEY}"}):
            response = httpx.post(
                f"{harness.base}/v1/messages", headers=headers, json=base(), timeout=60
            )
            assert response.status_code == 200, response.text

    def test_version_and_beta_headers_are_accepted(self, harness):
        response = post(
            harness,
            "/v1/messages",
            base(),
            {"anthropic-beta": "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"},
        )
        assert response.status_code == 200

    def test_unauthorized_error_is_typed(self, harness):
        response = httpx.post(f"{harness.base}/v1/messages", json=base())
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    def test_fields_reach_the_upstream(self, harness):
        post(
            harness,
            "/v1/messages",
            base(
                system="Be terse.",
                stop_sequences=["END"],
                temperature=0.25,
                top_k=7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                ],
            ),
        )
        sent = harness.upstream.hits[-1][2]
        assert sent["stop"] == ["END"]
        assert sent["max_tokens"] == 64 and sent["temperature"] == 0.25
        assert sent["top_k"] == 7
        assert "cache_control" not in json.dumps(sent["messages"])

    def test_context_management_gets_a_200_and_stays_off_the_wire(self, harness):
        body = base(context_management={"edits": [{"type": "clear_tool_uses_20250919"}]})
        assert post(harness, "/v1/messages", body).status_code == 200
        assert "context_management" not in json.dumps(harness.upstream.hits[-1][2])

    def test_invalid_json_is_a_typed_400(self, harness):
        response = httpx.post(
            f"{harness.base}/v1/messages",
            headers={**HEADERS, "x-api-key": KEY},
            content=b"{",
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_stream_framing(self, harness):
        harness.upstream.mode = "echo"
        response = post(harness, "/v1/messages", base(stream=True))
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [n for n, _ in sse(response.text)]
        assert names[:3] == ["message_start", "ping", "content_block_start"]
        assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
        assert names.count("content_block_delta") >= 2

    def test_streaming_tool_block_starts_empty(self, harness):
        harness.upstream.mode = "tools"
        body = base(
            stream=True,
            tools=[{"name": "read_file", "input_schema": {"type": "object"}}],
        )
        events = sse(post(harness, "/v1/messages", body).text)
        (start,) = [d for n, d in events if n == "content_block_start"]
        assert start["content_block"] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {},
        }

    def test_a_stream_that_dies_ends_with_an_error_and_no_stop(self, harness):
        harness.upstream.mode = "die_after"
        response = post(harness, "/v1/messages", base(stream=True))
        names = [n for n, _ in sse(response.text)]
        assert names[-1] == "error"
        assert "message_stop" not in names
        assert "secret" not in response.text

    def test_count_tokens_is_an_authenticated_estimate(self, harness):
        url = "/v1/messages/count_tokens"
        assert httpx.post(f"{harness.base}{url}", json=counted()).status_code == 401
        small = post(harness, url, counted())
        assert small.status_code == 200
        n = small.json()["input_tokens"]
        assert isinstance(n, int) and n > 0
        bigger = post(
            harness,
            url,
            counted(
                system="You are a careful assistant." * 10,
                tools=[{"name": "f", "description": "d" * 200, "input_schema": {"type": "object"}}],
            ),
        )
        assert bigger.json()["input_tokens"] > n
        assert not harness.upstream.hits

    def test_count_tokens_refuses_unknown_fields(self, harness):
        refused = post(harness, "/v1/messages/count_tokens", counted(mystery=1))
        assert refused.status_code == 400
        assert refused.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.slow
class TestSdk:
    @pytest.fixture
    def client(self, harness):
        sdk = pytest.importorskip("anthropic")
        return sdk.Anthropic(base_url=harness.base, api_key=KEY, max_retries=0)

    def test_create(self, harness, client):
        harness.upstream.mode = "echo"
        message = client.messages.create(
            model="default", max_tokens=32, messages=[{"role": "user", "content": "hi"}]
        )
        assert message.content[0].text == "echo: hi"
        assert message.stop_reason == "end_turn"
        assert (message.usage.input_tokens, message.usage.output_tokens) == (3, 3)

    def test_stream_helper_assembles_the_final_message(self, harness, client):
        harness.upstream.mode = "echo"
        with client.messages.stream(
            model="default", max_tokens=32, messages=[{"role": "user", "content": "hi"}]
        ) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        assert text == "echo: hi" and final.content[0].text == "echo: hi"
        assert final.stop_reason == "end_turn"

    def test_tool_use_round_trip(self, harness, client):
        harness.upstream.mode = "tools"
        tool = {"name": "read_file", "input_schema": {"type": "object"}}
        message = client.messages.create(
            model="default",
            max_tokens=32,
            tools=[tool],
            messages=[{"role": "user", "content": "hi"}],
        )
        (use,) = [b for b in message.content if b.type == "tool_use"]
        assert use.name == "read_file" and use.input == {"p": "a.txt"}
        assert message.stop_reason == "tool_use"

    def test_count_tokens(self, harness, client):
        counted = client.messages.count_tokens(
            model="default", messages=[{"role": "user", "content": "hi there"}]
        )
        assert counted.input_tokens > 0

    def test_bad_key_raises_the_sdks_authentication_error(self, harness):
        sdk = pytest.importorskip("anthropic")
        bad = sdk.Anthropic(base_url=harness.base, api_key="sk-nope", max_retries=0)
        with pytest.raises(sdk.AuthenticationError):
            bad.messages.create(
                model="default", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
            )

    def test_refusal_raises_bad_request(self, harness):
        sdk = pytest.importorskip("anthropic")
        client = sdk.Anthropic(base_url=harness.base, api_key=KEY, max_retries=0)
        with pytest.raises(sdk.BadRequestError):
            client.messages.create(
                model="default",
                max_tokens=8,
                extra_body={"mcp_servers": []},
                messages=[{"role": "user", "content": "hi"}],
            )
