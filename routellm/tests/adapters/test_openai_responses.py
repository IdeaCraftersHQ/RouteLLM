"""The OpenAI Responses adapter: conformance, SDK, Codex, and litellm pins.

Three layers:
- `TestOpenAIResponses`: the shared conformance suite through a driver that
  speaks `POST /v1/responses`;
- dialect cases: Codex CLI's request, refusals, errors, the official SDK;
- `Test*Pin`: litellm's translator is wrapped, not copied; these tests pin
  the internal APIs and behaviours the adapter relies on, so an upgrade that
  changes one fails here rather than in a client.
"""

import asyncio
import inspect
import json
import re

import httpx
import pytest

from routellm.adapters import AdapterRefusal, Call, ChatCompletionRequest, Route
from routellm.adapters.openai_responses import OpenAIResponsesAdapter
from routellm.tests.adapters.conformance import (
    ECHO,
    KEY,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
    ToolUse,
)

slow = pytest.mark.slow


# -- the client side of the dialect ------------------------------------------


def sse(text: str) -> list[dict]:
    """The `data:` payloads of an SSE body, in order; each names its event."""
    events = []
    for raw in text.split("\n\n"):
        block = raw.strip()
        if not block:
            continue
        name = re.search(r"^event: (.+)$", block, re.M)
        data = json.loads(re.search(r"^data: (.+)$", block, re.M).group(1))
        assert name and name.group(1) == data["type"], block
        events.append(data)
    return events


def read_response(body: dict) -> Outcome:
    out = Outcome(text="")
    for item in body["output"]:
        if item["type"] == "message":
            out.text += "".join(p["text"] for p in item["content"] if p["type"] == "output_text")
        elif item["type"] == "function_call":
            out.tool_uses.append(
                ToolUse(item["call_id"], item["name"], json.loads(item["arguments"]))
            )
    if body["status"] == "incomplete":
        out.finish = "length"
    else:
        out.finish = "tool_calls" if out.tool_uses else "stop"
    usage = body.get("usage") or {}
    out.input_tokens, out.output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    return out


class ResponsesDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        items = []
        for m in scenario.messages:
            if m.tool_results:
                items += [
                    {"type": "function_call_output", "call_id": r.id, "output": r.content}
                    for r in m.tool_results
                ]
                continue
            kind = "input_text" if m.role == "user" else "output_text"
            content = [{"type": kind, "text": m.text}]
            if m.image_png_b64:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{m.image_png_b64}",
                    }
                )
            if m.text or m.image_png_b64:
                items.append({"type": "message", "role": m.role, "content": content})
            items += [
                {
                    "type": "function_call",
                    "call_id": u.id,
                    "name": u.name,
                    "arguments": json.dumps(u.arguments),
                }
                for u in m.tool_uses
            ]
        body = {"model": scenario.model, "input": items, "stream": scenario.stream}
        if scenario.system:
            body["instructions"] = scenario.system
        if scenario.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                }
                for t in scenario.tools
            ]
        return Http("POST", "/v1/responses", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            return read_response(response.json())
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse(response.text)
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        return read_response(events[-1]["response"])

    def parse_error(self, response: httpx.Response) -> str:
        return response.json()["error"]["message"]

    def unsupported(self) -> Http:
        return Http("POST", "/v1/responses", json={"model": "default", "input": "hi", "bogus": 1})


@slow
class TestOpenAIResponses(ConformanceSuite):
    driver = ResponsesDriver()


# -- Codex CLI ---------------------------------------------------------------


def codex_request(**over) -> dict:
    """What Codex CLI sends to a `wire_api = "responses"` provider.

    Always streaming, `store: false`, the full conversation in `input`
    (developer + user messages, prior reasoning / function calls / outputs),
    `include` asking for encrypted reasoning, `prompt_cache_key`, function
    tools plus the freeform `apply_patch` (`custom`) tool.
    """
    body = {
        "model": "default",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "<permissions>"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "Run a command",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "array", "items": {"type": "string"}}},
                    "required": ["command"],
                },
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Patch files",
                "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
            },
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "0f0c0c0c-0000-0000-0000-000000000000",
        "text": {"verbosity": "medium"},
    }
    body.update(over)
    return body


def decode(body: dict) -> tuple[ChatCompletionRequest, Call]:
    adapter = OpenAIResponsesAdapter()
    call = Call(adapter, adapter.routes[0], None, None, body=body)
    return adapter.decode(call), call


def refused(body: dict) -> AdapterRefusal:
    with pytest.raises(AdapterRefusal) as caught:
        decode(body)
    return caught.value


class TestCodexDecode:
    def test_codex_request_maps_to_chat(self):
        request, call = decode(codex_request())
        assert request.stream is True
        assert request.reasoning_effort == "medium"
        assert request.tool_choice == "auto"
        assert request.parallel_tool_calls is False
        roles = [m["role"] for m in request.messages]
        assert roles == ["system", "developer", "user"]
        assert request.messages[0]["content"] == "You are Codex."
        assert [t["function"]["name"] for t in request.tools] == ["shell", "apply_patch"]
        assert call.state["responses_request"]["model"] == "default"

    def test_codex_history_round_trips(self):
        history = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": '{"command":["ls"]}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            {
                "type": "custom_tool_call",
                "call_id": "call_2",
                "name": "apply_patch",
                "input": "*** Begin Patch",
            },
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "done"},
        ]
        request, _ = decode(codex_request(input=history))
        kinds = [(m["role"], bool(m.get("tool_calls"))) for m in request.messages]
        assert ("tool", False) in kinds and ("assistant", True) in kinds
        tool_ids = [m["tool_call_id"] for m in request.messages if m["role"] == "tool"]
        assert tool_ids == ["call_1", "call_2"]

    def test_accepted_and_ignored_fields_have_no_effect_on_the_request(self):
        plain, _ = decode(codex_request())
        bare = codex_request()
        for key in ("store", "include", "prompt_cache_key"):
            del bare[key]
        bare["text"] = {}
        bare["reasoning"] = {"effort": "medium"}
        other, _ = decode(bare)
        assert plain == other

    def test_text_format_maps_to_response_format(self):
        schema = {"type": "json_schema", "name": "x", "schema": {"type": "object"}, "strict": True}
        request, _ = decode(codex_request(text={"format": schema, "verbosity": "low"}))
        assert request.response_format["type"] == "json_schema"
        assert request.response_format["json_schema"]["name"] == "x"

    def test_sampling_and_limits_map(self):
        request, _ = decode(
            codex_request(temperature=0.2, top_p=0.9, max_output_tokens=50, user="u", metadata={})
        )
        assert (request.temperature, request.top_p, request.max_tokens) == (0.2, 0.9, 50)
        assert request.user == "u"

    def test_metadata_maps_because_litellm_does_not(self):
        request, _ = decode(codex_request(metadata={"data_class": "public"}))
        assert request.metadata == {"data_class": "public"}


class TestRefusals:
    def test_previous_response_id_is_refused(self):
        error = refused(codex_request(previous_response_id="resp_abc"))
        assert error.status == 400 and error.param == "previous_response_id"
        assert "state" in error.message

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("conversation", "conv_1"),
            ("background", True),
            ("max_tool_calls", 3),
            ("top_logprobs", 2),
            ("truncation", "auto"),
            ("prompt", {"id": "pmpt_1"}),
            ("context_management", [{"type": "compaction"}]),
            ("include", ["message.output_text.logprobs"]),
            ("bogus", 1),
        ],
    )
    def test_unsupported_fields_are_refused_by_name(self, field, value):
        assert refused(codex_request(**{field: value})).param == field

    def test_unknown_text_option_is_refused_by_name(self):
        assert refused(codex_request(text={"verbosity": "low", "bogus": 1})).param == "bogus"

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "local_shell"},
            {"type": "web_search"},
            {"type": "web_search_preview"},
            {"type": "mcp", "server_label": "s", "server_url": "http://x"},
            {"type": "file_search", "vector_store_ids": []},
            {"type": "computer_use_preview"},
            {"type": "image_generation"},
        ],
    )
    def test_hosted_tools_are_refused_not_dropped(self, tool):
        error = refused(codex_request(tools=[tool]))
        assert error.param == "tools" and tool["type"] in error.message

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "local_shell_call", "call_id": "c", "action": {}},
            {"type": "item_reference", "id": "msg_1"},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "computer_call", "call_id": "c"},
            {"type": "additional_tools", "tools": []},
        ],
    )
    def test_unsupported_input_items_are_refused_not_dropped(self, item):
        error = refused(codex_request(input=[item]))
        assert error.param == "input" and item["type"] in error.message

    def test_unsupported_content_part_is_refused(self):
        part = {"type": "input_video", "video_url": "x"}
        item = {"type": "message", "role": "user", "content": [part]}
        assert refused(codex_request(input=[item])).param == "input"

    def test_missing_or_empty_input_is_refused(self):
        body = codex_request()
        del body["input"]
        assert refused(body).param == "input"
        assert refused(codex_request(input=[])).param == "input"

    def test_stream_must_be_boolean(self):
        assert refused(codex_request(stream="yes")).param == "stream"


# -- the dialect on the wire -------------------------------------------------


def post(harness, body, key=KEY):
    return httpx.post(
        f"{harness.base}/v1/responses",
        headers={"authorization": f"Bearer {key}"},
        json=body,
        timeout=60,
    )


@slow
class TestWire:
    def test_codex_stream_lifecycle(self, harness):
        harness.upstream.mode = "echo"
        response = post(harness, codex_request())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse(response.text)
        types = [e["type"] for e in events]
        assert types[0] == "response.created" and types[-1] == "response.completed"
        assert "response.output_text.delta" in types
        assert types.index("response.output_item.added") < types.index("response.output_item.done")
        numbers = [e["sequence_number"] for e in events]
        assert numbers == sorted(set(numbers)), "sequence numbers strictly increase"
        ids = {e["response"]["id"] for e in events if "response" in e}
        assert len(ids) == 1 and ids.pop().startswith("resp_")
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        assert done[-1]["type"] == "message"
        assert done[-1]["content"][0]["text"] == ECHO
        usage = events[-1]["response"]["usage"]
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (3, 3, 6)

    def test_codex_tool_call_stream(self, harness):
        harness.upstream.mode = "tools"
        events = sse(post(harness, codex_request()).text)
        item = next(
            e["item"]
            for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        )
        assert item["call_id"] == "call_1" and item["name"] == "read_file"
        assert json.loads(item["arguments"]) == {"p": "a.txt"}
        # the whole reasoning request reached the chat upstream as chat fields
        sent = harness.upstream.hits[-1][2]
        assert sent["reasoning_effort"] == "medium"
        assert sent["parallel_tool_calls"] is False
        assert sent["messages"][1]["role"] in ("developer", "system")
        assert not {"include", "store", "prompt_cache_key", "input"} & set(sent)

    def test_non_stream_response_body_shape(self, harness):
        harness.upstream.mode = "echo"
        body = post(harness, codex_request(stream=False)).json()
        assert body["object"] == "response" and body["status"] == "completed"
        assert body["id"].startswith("resp_")
        assert body["output"][0]["content"][0]["text"] == ECHO
        assert body["usage"]["total_tokens"] == 6

    def test_refusal_is_the_responses_400_and_never_reaches_upstream(self, harness):
        response = post(harness, codex_request(previous_response_id="resp_abc"))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "previous_response_id"
        assert error["code"] and error["message"]
        assert not harness.upstream.hits

    def test_invalid_json_body_is_a_400(self, harness):
        response = httpx.post(
            f"{harness.base}/v1/responses",
            headers={"authorization": f"Bearer {KEY}"},
            content=b"{",
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_stream_failure_ends_in_response_failed_never_completed(self, harness):
        harness.upstream.mode = harness.other.mode = "die_after"
        response = post(harness, codex_request())
        events = sse(response.text)
        types = [e["type"] for e in events]
        assert "response.completed" not in types
        assert types[-1] == "response.failed"
        failed = events[-1]["response"]
        assert failed["status"] == "failed" and failed["error"]["message"]
        assert failed["id"] == events[0]["response"]["id"]

    def test_chat_completions_route_is_untouched(self, harness):
        harness.upstream.mode = "echo"
        response = httpx.post(
            f"{harness.base}/v1/chat/completions",
            headers={"authorization": f"Bearer {KEY}"},
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.json()["choices"][0]["message"]["content"] == ECHO


@slow
class TestOpenAISDK:
    @pytest.fixture
    def client(self, harness):
        openai = pytest.importorskip("openai")
        return openai.OpenAI(base_url=f"{harness.base}/v1", api_key=KEY, max_retries=0)

    def test_create(self, harness, client):
        harness.upstream.mode = "echo"
        response = client.responses.create(model="default", input="hi")
        assert response.output_text == ECHO
        assert response.status == "completed"
        assert response.usage.total_tokens == 6

    def test_create_with_instructions_and_tools(self, harness, client):
        harness.upstream.mode = "tools"
        tool = {
            "type": "function",
            "name": "read_file",
            "description": "Read",
            "parameters": {"type": "object", "properties": {"p": {"type": "string"}}},
            "strict": False,
        }
        response = client.responses.create(
            model="default", input="hi", instructions="Be terse.", tools=[tool]
        )
        call = next(i for i in response.output if i.type == "function_call")
        assert (call.name, json.loads(call.arguments)) == ("read_file", {"p": "a.txt"})
        assert harness.upstream.hits[-1][2]["messages"][0]["content"] == "Be terse."

    def test_stream(self, harness, client):
        harness.upstream.mode = "echo"
        stream = client.responses.create(model="default", input="hi", stream=True)
        events = list(stream)
        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert text == ECHO
        assert events[-1].type == "response.completed"
        assert events[-1].response.usage.output_tokens == 3

    def test_stream_helper_final_response(self, harness, client):
        harness.upstream.mode = "echo"
        with client.responses.stream(model="default", input="hi") as stream:
            final = stream.get_final_response()
        assert final.output_text == ECHO

    def test_previous_response_id_is_a_400(self, harness, client):
        import openai

        with pytest.raises(openai.BadRequestError) as caught:
            client.responses.create(model="default", input="hi", previous_response_id="resp_1")
        assert caught.value.param == "previous_response_id"
        assert not harness.upstream.hits

    def test_bad_key_is_401(self, harness):
        openai = pytest.importorskip("openai")
        bad = openai.OpenAI(base_url=f"{harness.base}/v1", api_key="sk-nope", max_retries=0)
        with pytest.raises(openai.AuthenticationError):
            bad.responses.create(model="default", input="hi")


# -- litellm pins -------------------------------------------------------------
#
# The adapter wraps litellm's Responses <-> chat translator and streaming
# iterator. They are internal APIs. Each test names what the adapter needs
# from them; a litellm upgrade that breaks one fails here, loudly.

from litellm.responses.litellm_completion_transformation.streaming_iterator import (  # noqa: E402
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (  # noqa: E402
    LiteLLMCompletionResponsesConfig as Translator,
)
from litellm.types.utils import ModelResponseStream  # noqa: E402


def chunk(**fields):
    return ModelResponseStream(id="chatcmpl-1", model="m", created=1, **fields)


def chunks_text():
    return [
        chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]),
        chunk(choices=[{"index": 0, "delta": {"content": "lo"}}]),
        chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        chunk(choices=[], usage={"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}),
    ]


class Wrapper:
    """The two things the iterator reads of a `CustomStreamWrapper`."""

    logging_obj = None

    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def etype(event) -> str:
    return getattr(event.type, "value", event.type)


def run_stream(chunks, request):
    async def go():
        iterator = LiteLLMCompletionStreamingIterator(
            model="default",
            litellm_custom_stream_wrapper=Wrapper(chunks),
            request_input="hi",
            responses_api_request=request,
        )
        return [event async for event in iterator]

    return asyncio.run(go())


class TestTranslatorPin:
    def test_request_transform_signature(self):
        params = inspect.signature(
            Translator.transform_responses_api_request_to_chat_completion_request
        ).parameters
        assert {"model", "input", "responses_api_request", "stream"} <= set(params)

    def test_response_transform_signature(self):
        params = inspect.signature(
            Translator.transform_chat_completion_response_to_responses_api_response
        ).parameters
        assert {"request_input", "responses_api_request", "chat_completion_response"} <= set(params)

    def test_iterator_signature(self):
        params = inspect.signature(LiteLLMCompletionStreamingIterator.__init__).parameters
        assert {
            "model",
            "litellm_custom_stream_wrapper",
            "request_input",
            "responses_api_request",
        } <= set(params)

    def test_request_output_stays_inside_the_chat_request(self):
        request = codex_request()
        out = Translator.transform_responses_api_request_to_chat_completion_request(
            model="default",
            input=request["input"],
            responses_api_request=request,
            stream=True,
        )
        assert set(out) <= set(ChatCompletionRequest.model_fields) | {"reasoning_summary"}
        assert out["stream_options"] == {"include_usage": True}

    def test_hosted_tool_and_item_types_are_silently_dropped_so_the_adapter_must_refuse(self):
        out = Translator.transform_responses_api_request_to_chat_completion_request(
            model="default",
            input=[
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "local_shell_call", "call_id": "c", "action": {}},
            ],
            responses_api_request={"tools": [{"type": "local_shell"}]},
        )
        assert "tools" not in out
        assert [m["role"] for m in out["messages"]] == ["user"]

    def test_metadata_is_not_mapped_by_litellm(self):
        out = Translator.transform_responses_api_request_to_chat_completion_request(
            model="default", input="hi", responses_api_request={"metadata": {"a": "b"}}
        )
        assert "metadata" not in out

    def test_text_verbosity_has_no_chat_equivalent(self):
        out = Translator.transform_responses_api_request_to_chat_completion_request(
            model="default",
            input="hi",
            responses_api_request={"text": {"verbosity": "low"}},
        )
        assert "response_format" not in out and "verbosity" not in out

    def test_custom_tool_becomes_a_function_tool_with_content_argument(self):
        out = Translator.transform_responses_api_request_to_chat_completion_request(
            model="default",
            input="hi",
            responses_api_request={"tools": [{"type": "custom", "name": "apply_patch"}]},
        )
        function = out["tools"][0]["function"]
        assert function["name"] == "apply_patch"
        assert function["parameters"]["required"] == ["content"]

    def test_non_stream_response_from_a_chat_dict(self):
        chat = {
            "id": "chatcmpl-9",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hey"},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "routellm": {"path": []},
        }
        response = Translator.transform_chat_completion_response_to_responses_api_response(
            request_input="hi", responses_api_request={}, chat_completion_response=chat
        )
        body = response.model_dump(mode="json")
        assert body["output"][0]["content"][0]["text"] == "hey"
        assert body["status"] == "completed"
        assert (body["usage"]["input_tokens"], body["usage"]["output_tokens"]) == (3, 4)

    def test_custom_tool_call_comes_back_as_custom_tool_call(self):
        chat = {
            "id": "chatcmpl-9",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_7",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"content": "*** Begin Patch"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        response = Translator.transform_chat_completion_response_to_responses_api_response(
            request_input="hi",
            responses_api_request={"tools": [{"type": "custom", "name": "apply_patch"}]},
            chat_completion_response=chat,
        )
        item = response.model_dump(mode="json")["output"][0]
        assert (item["type"], item["call_id"], item["input"]) == (
            "custom_tool_call",
            "call_7",
            "*** Begin Patch",
        )


class TestStreamingIteratorPin:
    def test_text_stream_event_order(self):
        events = run_stream(chunks_text(), {"model": "default", "input": "hi"})
        types = [etype(e) for e in events]
        assert types[:3] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
        ]
        assert types[-1] == "response.completed"
        deltas = [e.delta for e in events if etype(e) == "response.output_text.delta"]
        assert "".join(deltas) == "Hello"

    def test_works_without_a_logging_object(self):
        completed = run_stream(chunks_text(), {"model": "default", "input": "hi"})[-1]
        assert completed.response.usage.total_tokens == 6

    def test_custom_tool_call_streams_as_custom_tool_call(self):
        call = {"index": 0, "id": "call_7", "type": "function"}
        tool_chunks = [
            chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {**call, "function": {"name": "apply_patch", "arguments": ""}}
                            ],
                        },
                    }
                ]
            ),
            chunk(
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"content": "patch"}'}}
                            ]
                        },
                    }
                ]
            ),
            chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        ]
        request = {"tools": [{"type": "custom", "name": "apply_patch"}]}
        events = run_stream(tool_chunks, request)
        done = [
            e.item
            for e in events
            if etype(e) == "response.output_item.done" and getattr(e.item, "type", "") != "message"
        ]
        item = done[-1]
        item = item if isinstance(item, dict) else item.model_dump()
        assert item["type"] == "custom_tool_call" and item["input"] == "patch"

    def test_response_ids_are_litellm_encoded_so_the_adapter_normalizes_them(self):
        created = run_stream(chunks_text(), {"model": "default", "input": "hi"})[0]
        assert created.response.id != "resp_chatcmpl-1"


def test_adapter_declares_the_responses_route():
    adapter = OpenAIResponsesAdapter()
    assert adapter.name == "openai_responses"
    assert adapter.routes == (Route("POST", "/v1/responses"),)
