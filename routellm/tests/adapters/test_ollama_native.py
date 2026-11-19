"""The Ollama native adapter: conformance suite, dialect cases, official client.

Three layers: the shared conformance suite for `/api/chat` and for
`/api/generate`; mapping cases through the harness server (what the
upstream received, what the dialect refused); and the official `ollama`
Python client pointed at the harness (skipped when not installed).
"""

import json

import httpx
import pytest

from routellm.adapters import Call, Route
from routellm.adapters.ollama import OllamaAdapter
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


# -- conformance --------------------------------------------------------------


class ChatDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        names = {u.id: u.name for m in scenario.messages for u in m.tool_uses}
        messages = []
        if scenario.system:
            messages.append({"role": "system", "content": scenario.system})
        for m in scenario.messages:
            if m.tool_results:
                messages += [
                    {"role": "tool", "tool_name": names[r.id], "content": r.content}
                    for r in m.tool_results
                ]
                continue
            message = {"role": m.role, "content": m.text}
            if m.image_png_b64:
                message["images"] = [m.image_png_b64]
            if m.tool_uses:
                message["tool_calls"] = [
                    {"id": u.id, "function": {"name": u.name, "arguments": u.arguments}}
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
        return Http("POST", "/api/chat", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            return self.read(response.json(), Outcome())
        out = Outcome()
        lines = [json.loads(line) for line in response.text.splitlines()]
        assert lines[-1]["done"] is True
        assert all(not line["done"] for line in lines[:-1])
        for line in lines:
            self.read(line, out)
        return out

    def read(self, body: dict, out: Outcome) -> Outcome:
        message = body["message"]
        out.text += message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call["function"]
            out.tool_uses.append(ToolUse(call["id"], function["name"], function["arguments"]))
        if body["done"]:
            out.finish = "tool_calls" if out.tool_uses else body["done_reason"]
            out.input_tokens = body["prompt_eval_count"]
            out.output_tokens = body["eval_count"]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        return response.json()["error"]

    def unsupported(self) -> Http:
        return Http(
            "POST",
            "/api/chat",
            json={"model": "default", "messages": [{"role": "user", "content": "x"}], "bogus": 1},
        )


class GenerateDriver(ChatDriver):
    def build(self, scenario: Scenario) -> Http:
        (message,) = scenario.messages
        body = {"model": scenario.model, "prompt": message.text, "stream": scenario.stream}
        if scenario.system:
            body["system"] = scenario.system
        if message.image_png_b64:
            body["images"] = [message.image_png_b64]
        return Http("POST", "/api/generate", json=body)

    def read(self, body: dict, out: Outcome) -> Outcome:
        out.text += body["response"]
        if body["done"]:
            out.finish = body["done_reason"]
            out.input_tokens = body["prompt_eval_count"]
            out.output_tokens = body["eval_count"]
        return out

    def unsupported(self) -> Http:
        return Http("POST", "/api/generate", json={"model": "default", "prompt": "x", "bogus": 1})


class TestOllamaChat(ConformanceSuite):
    driver = ChatDriver()


class TestOllamaGenerate(ConformanceSuite):
    driver = GenerateDriver()
    skip = {  # noqa: RUF012
        "multi_turn": "generate takes one prompt",
        "tool_call_response": "generate has no tools",
        "tool_result_request": "generate has no tools",
        "streaming_tool_call": "generate has no tools",
    }


# -- helpers ------------------------------------------------------------------


def post(harness, path: str, body: dict, headers=AUTH) -> httpx.Response:
    return httpx.post(f"{harness.base}{path}", json=body, headers=headers, timeout=60)


def chat(harness, **body) -> httpx.Response:
    body.setdefault("model", "default")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    body.setdefault("stream", False)
    return post(harness, "/api/chat", body)


def sent(harness) -> dict:
    assert harness.upstream.hits, "the request never reached the upstream"
    return harness.upstream.hits[-1][2]


def refused(response: httpx.Response, harness, *words: str) -> None:
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert all(w in error for w in words), error
    assert not harness.upstream.hits


def decode(path: str, body: dict):
    call = Call(
        adapter=OllamaAdapter(), route=Route("POST", path), http_request=None, pipeline=None
    )
    call.body = body
    return call.adapter.decode(call), call


# -- options, format, think ---------------------------------------------------


def test_options_map_to_chat_fields(harness):
    options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 7,
        "num_predict": 33,
        "stop": ["END", "STOP"],
        "frequency_penalty": 0.5,
        "presence_penalty": 0.25,
    }
    assert chat(harness, options=options).status_code == 200
    got = sent(harness)
    assert got["temperature"] == 0.2 and got["top_p"] == 0.9 and got["seed"] == 7
    assert got["max_tokens"] == 33 and got["stop"] == ["END", "STOP"]
    assert got["frequency_penalty"] == 0.5 and got["presence_penalty"] == 0.25


def test_stop_may_be_one_string(harness):
    assert chat(harness, options={"stop": "END"}).status_code == 200
    assert sent(harness)["stop"] == "END"


def test_unbounded_num_predict_sets_no_limit(harness):
    for value in (-1, -2):
        assert chat(harness, options={"num_predict": value}).status_code == 200
        assert "max_tokens" not in sent(harness)


def test_runtime_knobs_are_accepted_and_ignored(harness):
    options = {"num_ctx": 8192, "num_gpu": 1, "num_thread": 4}
    response = chat(harness, options=options, keep_alive="5m")
    assert response.status_code == 200
    assert "num_ctx" not in sent(harness) and "keep_alive" not in sent(harness)


@pytest.mark.parametrize("option", ["top_k", "min_p", "repeat_penalty", "mirostat", "bogus"])
def test_sampling_options_without_an_equivalent_are_refused(harness, option):
    refused(chat(harness, options={option: 1}), harness, option)


def test_options_must_be_an_object(harness):
    refused(chat(harness, options=[1]), harness, "options")


def test_format_json_is_json_mode(harness):
    assert chat(harness, format="json").status_code == 200
    assert sent(harness)["response_format"] == {"type": "json_object"}


def test_format_schema_is_json_schema(harness):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert chat(harness, format=schema).status_code == 200
    rf = sent(harness)["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema and rf["json_schema"]["name"]


def test_format_other_values_are_refused(harness):
    refused(chat(harness, format="yaml"), harness, "format")


@pytest.mark.parametrize(("think", "effort"), [(True, "medium"), ("low", "low"), ("high", "high")])
def test_think_is_reasoning_effort(harness, think, effort):
    assert chat(harness, think=think).status_code == 200
    assert sent(harness)["reasoning_effort"] == effort


def test_think_false_sets_no_effort(harness):
    assert chat(harness, think=False).status_code == 200
    assert "reasoning_effort" not in sent(harness)


def test_think_unknown_level_is_refused(harness):
    refused(chat(harness, think="extreme"), harness, "think")


# -- wire behaviour -----------------------------------------------------------


def test_stream_defaults_to_true_like_ollama(harness):
    harness.upstream.mode = "echo"
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
    response = post(harness, "/api/chat", body)
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert len(lines) > 1 and lines[-1]["done"] is True
    assert sent(harness)["stream"] is True


def test_non_stream_is_one_json_object(harness):
    harness.upstream.mode = "echo"
    response = chat(harness)
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["model"] == "default" and body["done"] is True and body["done_reason"] == "stop"
    assert body["message"] == {"role": "assistant", "content": "echo: hi"}
    assert body["created_at"].endswith("Z")
    for key in ("total_duration", "load_duration", "eval_duration", "prompt_eval_duration"):
        assert isinstance(body[key], int)


def test_the_client_model_name_is_echoed_and_latest_is_stripped(harness):
    response = chat(harness, model="default:latest")
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "default:latest"


def test_generate_raw_sends_the_prompt_alone(harness):
    body = {"model": "default", "prompt": "hi", "system": "Be terse.", "raw": True}
    assert post(harness, "/api/generate", {**body, "stream": False}).status_code == 200
    assert [m["role"] for m in sent(harness)["messages"]] == ["user"]


def test_generate_system_and_prompt(harness):
    body = {"model": "default", "prompt": "hi", "system": "Be terse.", "stream": False}
    assert post(harness, "/api/generate", body).status_code == 200
    got = [(m["role"], text_of(m)) for m in sent(harness)["messages"]]
    assert got == [("system", "Be terse."), ("user", "hi")]


@pytest.mark.parametrize("field", ["suffix", "template", "context"])
def test_generate_fields_without_an_equivalent_are_refused(harness, field):
    body = {"model": "default", "prompt": "hi", "stream": False, field: "x"}
    refused(post(harness, "/api/generate", body), harness, field)


def test_empty_prompt_preload_is_refused(harness):
    refused(post(harness, "/api/generate", {"model": "default", "prompt": ""}), harness, "prompt")
    refused(chat(harness, messages=[]), harness, "messages")


def test_error_is_a_bare_error_string(harness):
    harness.upstream.mode = harness.other.mode = "500"
    response = chat(harness)
    assert response.status_code == 502
    assert set(response.json()) == {"error"}


# -- messages -----------------------------------------------------------------


def test_tool_results_pair_with_calls_by_name_in_order(harness):
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"p": "a"}}},
                {"function": {"name": "read_file", "arguments": {"p": "b"}}},
                {"function": {"name": "ls", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_name": "ls", "content": "dir"},
        {"role": "tool", "tool_name": "read_file", "content": "A"},
        {"role": "tool", "tool_name": "read_file", "content": "B"},
    ]
    assert chat(harness, messages=messages).status_code == 200
    out = sent(harness)["messages"]
    calls = out[1]["tool_calls"]
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == 3 and all(ids)
    assert json.loads(calls[0]["function"]["arguments"]) == {"p": "a"}
    results = {m["content"]: m["tool_call_id"] for m in out if m["role"] == "tool"}
    assert results == {"dir": ids[2], "A": ids[0], "B": ids[1]}


def test_tool_result_with_no_call_is_refused(harness):
    messages = [{"role": "user", "content": "go"}, {"role": "tool", "content": "x"}]
    refused(chat(harness, messages=messages), harness, "tool")


def test_explicit_tool_call_id_wins(harness):
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c9", "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "c9", "content": "r"},
    ]
    assert chat(harness, messages=messages).status_code == 200
    assert sent(harness)["messages"][1]["tool_call_id"] == "c9"


def test_images_may_be_data_urls_and_are_sniffed(harness):
    message = {"role": "user", "content": "x", "images": [f"data:image/png;base64,{PNG_B64}"]}
    assert chat(harness, messages=[message]).status_code == 200
    url = sent(harness)["messages"][0]["content"][1]["image_url"]["url"]
    assert url == f"data:image/png;base64,{PNG_B64}"


def test_bad_images_are_refused(harness):
    for image in ("not base64 !!", "aGVsbG8="):  # junk, then base64 of "hello"
        message = {"role": "user", "content": "x", "images": [image]}
        refused(chat(harness, messages=[message]), harness, "images")


def test_unknown_message_key_and_role_are_refused(harness):
    refused(chat(harness, messages=[{"role": "user", "content": "x", "foo": 1}]), harness, "foo")
    refused(chat(harness, messages=[{"role": "robot", "content": "x"}]), harness, "role")


def test_prior_thinking_is_ignored(harness):
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "thinking": "hmm"},
        {"role": "user", "content": "c"},
    ]
    assert chat(harness, messages=messages).status_code == 200
    assert "hmm" not in json.dumps(sent(harness))


# -- encoders, without a server -----------------------------------------------


def completion(message: dict, finish: str = "stop") -> dict:
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }


def encode(response: dict, path: str = "/api/chat") -> dict:
    _, call = (
        decode(path, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
        if (path == "/api/chat")
        else decode(path, {"model": "m", "prompt": "x"})
    )
    return call.adapter.encode(response, call)


def test_length_maps_to_done_reason_length():
    body = encode(completion({"content": "cut"}, "length"))
    assert body["done_reason"] == "length" and body["eval_count"] == 2


def test_tool_calls_finish_is_stop_with_parsed_arguments():
    message = {
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
        ],
    }
    body = encode(completion(message, "tool_calls"))
    assert body["done_reason"] == "stop"
    assert body["message"]["tool_calls"] == [
        {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}
    ]


def test_reasoning_is_thinking():
    body = encode(completion({"content": "a", "reasoning_content": "why"}))
    assert body["message"]["thinking"] == "why"
    body = encode(completion({"content": "a", "reasoning_content": "why"}), "/api/generate")
    assert body["thinking"] == "why" and body["response"] == "a"


def test_decode_refuses_a_non_object_arguments_string():
    message = {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1"}}]}
    with pytest.raises(Exception, match="arguments"):
        decode("/api/chat", {"model": "m", "messages": [message]})


# -- model listing ------------------------------------------------------------


def test_tags_lists_tiers_and_routers(harness):
    url = f"{harness.base}/api/tags"
    response = httpx.get(url, headers=AUTH)
    assert response.status_code == 200
    names = [m["name"] for m in response.json()["models"]]
    assert "default" in names and any(n.startswith("router-random-") for n in names)
    first = response.json()["models"][0]
    assert first["model"] == first["name"] and first["details"]["family"] == "routellm"


def test_tags_show_version_need_a_key(harness):
    for method, path in (("GET", "/api/tags"), ("POST", "/api/show"), ("GET", "/api/version")):
        response = httpx.request(
            method, f"{harness.base}{path}", json={} if method == "POST" else None
        )
        assert response.status_code == 401
        assert set(response.json()) == {"error"}


def test_show_a_listed_model(harness):
    for name in ("default", "default:latest"):
        response = post(harness, "/api/show", {"model": name})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["details"]["family"] == "routellm"
        assert "completion" in body["capabilities"] and "tools" in body["capabilities"]
    assert post(harness, "/api/show", {"name": "default"}).status_code == 200


def test_show_unknown_model_is_404_in_ollama_shape(harness):
    response = post(harness, "/api/show", {"model": "nope"})
    assert response.status_code == 404
    assert response.json() == {"error": "model 'nope' not found"}


def test_show_needs_a_model_and_refuses_unknown_fields(harness):
    assert post(harness, "/api/show", {}).status_code == 400
    assert post(harness, "/api/show", {"model": "default", "x": 1}).status_code == 400


def test_version(harness):
    response = httpx.get(f"{harness.base}/api/version", headers=AUTH)
    assert response.status_code == 200
    version = response.json()["version"]
    assert version.count(".") == 2 and all(p.isdigit() for p in version.split("."))


def test_chat_completions_route_is_untouched(harness):
    harness.upstream.mode = "echo"
    response = httpx.post(
        f"{harness.base}/v1/chat/completions",
        headers=AUTH,
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.json()["choices"][0]["message"]["content"] == "echo: hi"


# -- the official client ------------------------------------------------------


@pytest.fixture
def client(harness):
    ollama = pytest.importorskip("ollama")
    return ollama.Client(host=harness.base, headers=AUTH)


def test_client_chat(client, harness):
    harness.upstream.mode = "echo"
    reply = client.chat(model="default", messages=[{"role": "user", "content": "hi"}])
    assert reply.message.content == "echo: hi" and reply.done and reply.done_reason == "stop"
    assert reply.prompt_eval_count == 3 and reply.eval_count == 3


def test_client_chat_stream(client, harness):
    harness.upstream.mode = "echo"
    parts = list(
        client.chat(model="default", messages=[{"role": "user", "content": "hi"}], stream=True)
    )
    assert "".join(p.message.content for p in parts) == "echo: hi"
    assert parts[-1].done and not any(p.done for p in parts[:-1])


def test_client_generate_with_options(client, harness):
    harness.upstream.mode = "echo"
    reply = client.generate(
        model="default", prompt="hi", system="Be terse.", options={"temperature": 0.1, "seed": 3}
    )
    assert reply.response == "echo: hi" and reply.done
    assert sent(harness)["temperature"] == 0.1 and sent(harness)["seed"] == 3


def test_client_tools(client, harness):
    harness.upstream.mode = "tools"

    def read_file(p: str) -> str:
        """Read a file.

        Args:
            p: path
        """
        return p

    reply = client.chat(
        model="default", messages=[{"role": "user", "content": "hi"}], tools=[read_file]
    )
    (call,) = reply.message.tool_calls
    assert call.function.name == "read_file" and call.function.arguments == {"p": "a.txt"}


def test_client_images(client, harness):
    harness.upstream.mode = "echo"
    message = {"role": "user", "content": "hi", "images": [PNG_B64]}
    client.chat(model="default", messages=[message])
    assert sent(harness)["messages"][0]["content"][1]["image_url"]["url"].endswith(PNG_B64)


def test_client_list_and_show(client):
    assert "default" in [m.model for m in client.list().models]
    assert client.show("default").details.family == "routellm"


def test_client_errors(client, harness):
    ollama = pytest.importorskip("ollama")
    with pytest.raises(ollama.ResponseError) as missing:
        client.show("nope")
    assert missing.value.status_code == 404
    anonymous = ollama.Client(host=harness.base)
    with pytest.raises(ollama.ResponseError) as denied:
        anonymous.chat(model="default", messages=[{"role": "user", "content": "hi"}])
    assert denied.value.status_code == 401
    with pytest.raises(ollama.ResponseError) as refused_:
        client.chat(
            model="default", messages=[{"role": "user", "content": "hi"}], options={"top_k": 5}
        )
    assert refused_.value.status_code == 400
