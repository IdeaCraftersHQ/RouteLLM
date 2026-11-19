"""The Cohere v2 Chat adapter: conformance suite, mapping, streams, the official SDK.

Layers: the shared conformance suite for `POST /v2/chat`; mapping and refusal
cases through the harness server (what the upstream received, what the
dialect refused); the raw SSE event order; a cross-check that litellm's own
Cohere v2 stream reader (the outbound side) understands what this adapter
emits; and the official `cohere` SDK pointed at the harness (skipped when not
installed).
"""

import json

import httpx
import pytest

from routellm.adapters import Call, Route
from routellm.adapters.cohere import CohereAdapter
from routellm.tests.adapters.conformance import (
    KEY,
    PNG_B64,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
    ToolUse,
    text_of,
)

pytestmark = pytest.mark.slow

AUTH = {"authorization": f"Bearer {KEY}"}
FINISH = {"COMPLETE": "stop", "MAX_TOKENS": "length", "TOOL_CALL": "tool_calls"}


def sse_frames(text: str) -> list[tuple[str, dict]]:
    """(event name, data) of every SSE frame; the name must equal the data's type."""
    frames = []
    for raw in text.strip().split("\n\n"):
        lines = raw.splitlines()
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), raw
        name, data = lines[0][7:], json.loads(lines[1][6:])
        assert data["type"] == name
        frames.append((name, data))
    return frames


# -- conformance --------------------------------------------------------------


class CohereDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        messages = []
        if scenario.system:
            messages.append({"role": "system", "content": scenario.system})
        for m in scenario.messages:
            if m.tool_results:
                messages += [
                    {"role": "tool", "tool_call_id": r.id, "content": r.content}
                    for r in m.tool_results
                ]
                continue
            content: str | list = m.text
            if m.image_png_b64:
                content = [
                    {"type": "text", "text": m.text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{m.image_png_b64}"},
                    },
                ]
            message = {"role": m.role, "content": content}
            if m.tool_uses:
                message["tool_calls"] = [
                    {
                        "id": u.id,
                        "type": "function",
                        "function": {"name": u.name, "arguments": json.dumps(u.arguments)},
                    }
                    for u in m.tool_uses
                ]
            messages.append(message)
        body = {"model": scenario.model, "messages": messages, "stream": scenario.stream}
        if scenario.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in scenario.tools
            ]
        return Http("POST", "/v2/chat", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            body = response.json()
            out = Outcome()
            message = body["message"]
            for block in message.get("content") or []:
                out.text += block.get("text", "")
            for call in message.get("tool_calls") or []:
                function = call["function"]
                out.tool_uses.append(
                    ToolUse(call["id"], function["name"], json.loads(function["arguments"]))
                )
            out.finish = FINISH[body["finish_reason"]]
            out.input_tokens = body["usage"]["tokens"]["input_tokens"]
            out.output_tokens = body["usage"]["tokens"]["output_tokens"]
            return out
        return self.read_stream(response.text)

    def read_stream(self, text: str) -> Outcome:
        out = Outcome()
        calls: dict[int, dict] = {}
        for name, data in sse_frames(text):
            delta = (data.get("delta") or {}).get("message") or {}
            if name == "content-delta":
                out.text += delta["content"].get("text", "")
            elif name == "tool-call-start":
                call = delta["tool_calls"]
                calls[data["index"]] = {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "args": call["function"]["arguments"],
                }
            elif name == "tool-call-delta":
                calls[data["index"]]["args"] += delta["tool_calls"]["function"]["arguments"]
            elif name == "message-end":
                out.finish = FINISH[data["delta"]["finish_reason"]]
                usage = data["delta"]["usage"]["tokens"]
                out.input_tokens, out.output_tokens = usage["input_tokens"], usage["output_tokens"]
        out.tool_uses = [
            ToolUse(c["id"], c["name"], json.loads(c["args"])) for _, c in sorted(calls.items())
        ]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        return response.json()["message"]

    def unsupported(self) -> Http:
        return Http(
            "POST",
            "/v2/chat",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "x"}],
                "connectors": [{"id": "web-search"}],
            },
        )


class TestCohereChat(ConformanceSuite):
    driver = CohereDriver()


# -- helpers ------------------------------------------------------------------


def post(harness, body: dict, headers=AUTH) -> httpx.Response:
    return httpx.post(f"{harness.base}/v2/chat", json=body, headers=headers, timeout=60)


def chat(harness, **body) -> httpx.Response:
    body.setdefault("model", "default")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return post(harness, body)


def sent(harness) -> dict:
    assert harness.upstream.hits, "the request never reached the upstream"
    return harness.upstream.hits[-1][2]


def refused(response: httpx.Response, harness, *words: str) -> None:
    assert response.status_code == 400, response.text
    message = response.json()["message"]
    assert all(w in message for w in words), message
    assert not harness.upstream.hits


def decode(body: dict):
    call = Call(
        adapter=CohereAdapter(), route=Route("POST", "/v2/chat"), http_request=None, pipeline=None
    )
    call.body = body
    return call.adapter.decode(call)


# -- sampling, tools, formats -------------------------------------------------


def test_sampling_fields_map_to_chat_fields(harness):
    response = chat(
        harness,
        temperature=0.2,
        p=0.9,
        k=40,
        seed=7,
        max_tokens=33,
        stop_sequences=["END", "STOP"],
        frequency_penalty=0.5,
        presence_penalty=0.25,
    )
    assert response.status_code == 200
    got = sent(harness)
    assert got["temperature"] == 0.2 and got["top_p"] == 0.9 and got["seed"] == 7
    assert got["max_tokens"] == 33 and got["stop"] == ["END", "STOP"]
    assert got["frequency_penalty"] == 0.5 and got["presence_penalty"] == 0.25


def test_k_is_carried_as_top_k():
    assert (
        decode({"model": "m", "messages": [{"role": "user", "content": "x"}], "k": 40}).top_k == 40
    )


@pytest.mark.parametrize("bad", [-1, 1.5, "7", True])
def test_k_must_be_a_whole_number(bad):
    from routellm.adapters import AdapterRefusal

    with pytest.raises(AdapterRefusal, match="k"):
        decode({"model": "m", "messages": [{"role": "user", "content": "x"}], "k": bad})


def test_stream_defaults_to_false(harness):
    response = chat(harness)
    assert response.headers["content-type"].startswith("application/json")
    assert sent(harness).get("stream") in (None, False)


def test_tools_and_tool_choice(harness):
    tool = {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
    }
    assert chat(harness, tools=[tool], tool_choice="REQUIRED").status_code == 200
    got = sent(harness)
    assert got["tool_choice"] == "required"
    assert got["tools"][0]["function"]["name"] == "f"
    assert "strict" not in got["tools"][0]["function"]
    chat(harness, tools=[tool], tool_choice="NONE", strict_tools=True)
    got = sent(harness)
    assert got["tool_choice"] == "none"
    assert got["tools"][0]["function"]["strict"] is True


def test_response_format(harness):
    chat(harness, response_format={"type": "json_object"})
    assert sent(harness)["response_format"] == {"type": "json_object"}
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    chat(harness, response_format={"type": "json_object", "json_schema": schema})
    got = sent(harness)["response_format"]
    assert got["type"] == "json_schema" and got["json_schema"]["schema"] == schema
    chat(harness, response_format={"type": "text"})
    assert "response_format" not in sent(harness)


def test_thinking(harness):
    chat(harness, thinking={"type": "enabled"})
    assert sent(harness)["reasoning_effort"] == "medium"
    chat(harness, thinking={"type": "disabled"})
    assert "reasoning_effort" not in sent(harness)


def test_content_blocks_and_history(harness):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be terse."}]},
        {"role": "user", "content": [{"type": "text", "text": "look"}]},
        {
            "role": "assistant",
            "tool_plan": "I will read it.",
            "content": [{"type": "thinking", "thinking": "hmm"}],
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
            "citations": [{"start": 0, "end": 1, "text": "x", "sources": []}],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "document", "document": {"id": "d1", "data": {"k": "v", "n": 2}}},
            ],
        },
        {"role": "user", "content": "go on"},
    ]
    assert chat(harness, messages=messages).status_code == 200
    got = sent(harness)["messages"]
    assert [m["role"] for m in got] == ["system", "user", "assistant", "tool", "user"]
    assert text_of(got[0]) == "Be terse."
    assert got[2]["content"] == "I will read it."
    assert got[2]["tool_calls"][0]["id"] == "c1"
    assert got[3]["tool_call_id"] == "c1"
    assert got[3]["content"] == 'ok\n<document id="d1">\nk: v\nn: 2\n</document>'


def test_documents_become_a_system_message_after_the_leading_ones(harness):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    documents = ["plain text", {"id": "mine", "data": {"title": "T", "text": "body", "n": [1]}}]
    assert chat(harness, messages=messages, documents=documents).status_code == 200
    got = sent(harness)["messages"]
    assert [m["role"] for m in got] == ["system", "system", "user"]
    assert got[0]["content"] == "Be terse."
    assert got[1]["content"] == (
        "Use the following documents as context for the conversation.\n\n"
        '<document id="doc_0">\nplain text\n</document>\n\n'
        '<document id="mine">\ntitle: T\ntext: body\nn: [1]\n</document>'
    )


def test_documents_without_system_come_first(harness):
    chat(harness, documents=[{"data": {"text": "t"}}])
    got = sent(harness)["messages"]
    assert [m["role"] for m in got] == ["system", "user"]
    assert '<document id="doc_0">\ntext: t\n</document>' in got[0]["content"]


def test_image_detail_is_kept(harness):
    block = {"type": "image_url", "image_url": {"url": "https://x.test/a.png", "detail": "low"}}
    chat(harness, messages=[{"role": "user", "content": [block]}])
    (part,) = sent(harness)["messages"][0]["content"]
    assert part["image_url"] == {"url": "https://x.test/a.png", "detail": "low"}


def test_image_data_url(harness):
    url = f"data:image/png;base64,{PNG_B64}"
    block = {"type": "image_url", "image_url": {"url": url}}
    chat(harness, messages=[{"role": "user", "content": [block]}])
    assert sent(harness)["messages"][0]["content"][0]["image_url"]["url"] == url


def test_ignored_fields_are_accepted(harness):
    response = chat(
        harness,
        priority=5,
        safety_mode="CONTEXTUAL",
        logprobs=False,
        citation_options={"mode": "disabled"},
    )
    assert response.status_code == 200
    got = sent(harness)
    assert "priority" not in got and "safety_mode" not in got


# -- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "words"),
    [
        ({"connectors": [{"id": "web-search"}]}, ("connectors",)),
        ({"raw_prompting": True}, ("raw_prompting",)),
        ({"logprobs": True}, ("logprobs",)),
        ({"safety_mode": "STRICT"}, ("safety_mode",)),
        ({"safety_mode": "OFF"}, ("safety_mode",)),
        ({"citation_options": {"mode": "accurate"}}, ("citation_options",)),
        (
            {"citation_options": {}, "thinking": {"type": "enabled", "token_budget": 10}},
            ("token_budget",),
        ),
        ({"tool_choice": "AUTO"}, ("tool_choice",)),
        ({"strict_tools": True}, ("strict_tools",)),
        ({"tools": [{"type": "web_search"}]}, ("tools[0]",)),
        ({"response_format": {"type": "text", "json_schema": {}}}, ("json_schema",)),
        ({"response_format": {"type": "xml"}}, ("response_format.type",)),
        ({"stop_sequences": "END"}, ("stop_sequences",)),
        ({"k": -3}, ("k",)),
        ({"messages": []}, ("messages",)),
        ({"messages": [{"role": "developer", "content": "x"}]}, ("role",)),
        ({"messages": [{"role": "user", "content": "x", "name": "n"}]}, ("name",)),
        ({"messages": [{"role": "user", "content": [{"type": "document"}]}]}, ("type",)),
        ({"messages": [{"role": "tool", "content": "x"}]}, ("tool_call_id",)),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": 1}]}]}, ("text",)),
        ({"documents": [{"data": "nope"}]}, ("documents[0].data",)),
    ],
)
def test_refusals(harness, body, words):
    refused(chat(harness, **body), harness, *words)


def test_model_is_required(harness):
    response = post(harness, {"messages": [{"role": "user", "content": "x"}]})
    refused(response, harness, "model")


# -- answers ------------------------------------------------------------------


def test_answer_shape(harness):
    harness.upstream.mode = "echo"
    body = chat(harness).json()
    assert body["id"] and body["finish_reason"] == "COMPLETE"
    assert body["message"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "echo: hi"}],
    }
    assert body["usage"] == {
        "billed_units": {"input_tokens": 3, "output_tokens": 3},
        "tokens": {"input_tokens": 3, "output_tokens": 3},
    }


def test_tool_call_answer_shape(harness):
    harness.upstream.mode = "tools"
    body = chat(harness).json()
    assert body["finish_reason"] == "TOOL_CALL"
    (call,) = body["message"]["tool_calls"]
    assert call == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"p": "a.txt"}'},
    }


def test_max_tokens_finish(harness):
    harness.upstream.reply = None
    adapter = CohereAdapter()
    response = {
        "id": "x",
        "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 1},
        },
    }
    out = adapter.encode(response, None)
    assert out["finish_reason"] == "MAX_TOKENS"
    assert out["usage"]["cached_tokens"] == 1


def test_error_shape(harness):
    response = httpx.post(f"{harness.base}/v2/chat", json={}, headers={})
    assert response.status_code == 401
    assert set(response.json()) <= {"message", "id"}


# -- streams ------------------------------------------------------------------


def stream(harness, **body) -> list[tuple[str, dict]]:
    response = chat(harness, stream=True, **body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in response.text
    return sse_frames(response.text)


def test_text_stream_event_order(harness):
    harness.upstream.mode = "echo"
    frames = stream(harness)
    names = [n for n, _ in frames]
    assert names[0] == "message-start" and names[-1] == "message-end"
    assert names[1] == "content-start" and names[-2] == "content-end"
    assert set(names[2:-2]) == {"content-delta"}
    start = frames[0][1]
    assert start["id"] and start["delta"]["message"]["role"] == "assistant"
    assert frames[1][1]["delta"]["message"]["content"] == {"type": "text", "text": ""}
    assert {d["index"] for n, d in frames if n.startswith("content")} == {0}
    text = "".join(
        d["delta"]["message"]["content"]["text"] for n, d in frames if n == "content-delta"
    )
    assert text == "echo: hi"
    end = frames[-1][1]["delta"]
    assert end["finish_reason"] == "COMPLETE"
    assert end["usage"]["tokens"] == {"input_tokens": 3, "output_tokens": 3}


def test_tool_stream_event_order(harness):
    harness.upstream.mode = "tools"
    frames = stream(harness)
    names = [n for n, _ in frames]
    assert names[0] == "message-start" and names[-1] == "message-end"
    assert names[1] == "tool-call-start" and names[-2] == "tool-call-end"
    assert set(names[2:-2]) == {"tool-call-delta"}
    assert "tool-plan-delta" not in names and "content-start" not in names
    start = frames[1][1]
    assert start["index"] == 0
    assert start["delta"]["message"]["tool_calls"] == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": ""},
    }
    pieces = "".join(
        d["delta"]["message"]["tool_calls"]["function"]["arguments"]
        for n, d in frames
        if n == "tool-call-delta"
    )
    assert json.loads(pieces) == {"p": "a.txt"}
    assert frames[-1][1]["delta"]["finish_reason"] == "TOOL_CALL"


def test_stream_error_after_start(harness):
    harness.upstream.mode = "stream_error_after"
    frames = stream(harness)
    name, data = frames[-1]
    assert name == "message-end"
    assert data["delta"]["finish_reason"] == "ERROR" and data["delta"]["error"]
    assert "usage" not in data["delta"]


def test_stream_died_midway_is_not_a_clean_end(harness):
    harness.upstream.mode = "die_after"
    response = chat(harness, stream=True)
    frames = sse_frames(response.text) if response.text.strip() else []
    ends = [d for n, d in frames if n == "message-end"]
    assert all(d["delta"]["finish_reason"] == "ERROR" for d in ends)


# -- litellm cross-check ------------------------------------------------------


def test_litellm_reads_our_text_deltas(harness):
    """litellm's Cohere v2 reader (the outbound side) takes our `content-delta` frames.

    Its `message-end` and tool-call branches read a different envelope than
    Cohere's wire (`event`/`data` keys, a list of calls), so only the text
    path is a meaningful cross-check.
    """
    from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator

    harness.upstream.mode = "echo"
    reader = CohereV2ModelResponseIterator(iter(()), sync_stream=True)
    texts = [
        reader.chunk_parser(data)["text"]
        for name, data in stream(harness)
        if name == "content-delta"
    ]
    assert "".join(texts) == "echo: hi"


def test_litellm_still_maps_the_v2_request_by_these_names():
    """The names this adapter inverts: top_p/p, stop/stop_sequences, max_tokens, seed."""
    from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig

    mapped = CohereV2ChatConfig().map_openai_params(
        {"top_p": 0.5, "stop": ["x"], "max_tokens": 9, "seed": 1, "tools": [{"a": 1}]},
        {},
        "command-a",
        False,
    )
    assert mapped == {
        "p": 0.5,
        "stop_sequences": ["x"],
        "max_tokens": 9,
        "seed": 1,
        "tools": [{"a": 1}],
    }


# -- the official SDK ---------------------------------------------------------


@pytest.fixture
def client(harness):
    cohere = pytest.importorskip("cohere")
    return cohere.ClientV2(api_key=KEY, base_url=harness.base)


def test_sdk_chat(harness, client):
    harness.upstream.mode = "echo"
    response = client.chat(model="default", messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "COMPLETE"
    assert response.message.content[0].text == "echo: hi"
    assert response.usage.tokens.input_tokens == 3


def test_sdk_chat_tool_call(harness, client):
    harness.upstream.mode = "tools"
    tool = {
        "type": "function",
        "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}},
    }
    response = client.chat(
        model="default", messages=[{"role": "user", "content": "hi"}], tools=[tool]
    )
    assert response.finish_reason == "TOOL_CALL"
    call = response.message.tool_calls[0]
    assert call.id == "call_1" and call.function.name == "read_file"
    assert json.loads(call.function.arguments) == {"p": "a.txt"}


def test_sdk_chat_stream(harness, client):
    harness.upstream.mode = "echo"
    events = list(client.chat_stream(model="default", messages=[{"role": "user", "content": "hi"}]))
    assert events[0].type == "message-start"
    assert events[-1].type == "message-end"
    assert events[-1].delta.finish_reason == "COMPLETE"
    text = "".join(e.delta.message.content.text for e in events if e.type == "content-delta")
    assert text == "echo: hi"


def test_sdk_chat_stream_tool_call(harness, client):
    harness.upstream.mode = "tools"
    tool = {
        "type": "function",
        "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}},
    }
    events = list(
        client.chat_stream(
            model="default", messages=[{"role": "user", "content": "hi"}], tools=[tool]
        )
    )
    start = next(e for e in events if e.type == "tool-call-start")
    assert start.delta.message.tool_calls.function.name == "read_file"
    args = "".join(
        e.delta.message.tool_calls.function.arguments for e in events if e.type == "tool-call-delta"
    )
    assert json.loads(args) == {"p": "a.txt"}


def test_sdk_refusal_is_a_bad_request(harness, client):
    from cohere.core.api_error import ApiError

    with pytest.raises(ApiError) as caught:
        client.chat(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            safety_mode="STRICT",
        )
    assert caught.value.status_code == 400
    assert "safety_mode" in str(caught.value.body)


def test_sdk_bad_key_is_unauthorized(harness):
    cohere = pytest.importorskip("cohere")
    from cohere.core.api_error import ApiError

    bad = cohere.ClientV2(api_key="sk-not-a-key", base_url=harness.base)
    with pytest.raises(ApiError) as caught:
        bad.chat(model="default", messages=[{"role": "user", "content": "hi"}])
    assert caught.value.status_code == 401
