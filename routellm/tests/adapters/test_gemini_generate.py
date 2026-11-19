"""The Gemini dialect: conformance suite, SDK round trips, mapping, litellm pins.

Three layers:

- `TestGeminiDialect`: the shared conformance suite against a real server;
- server tests for what only this dialect has (key in the query, `/v1`
  prefix, JSON-array stream, `countTokens`, model listing, error shape)
  and the `google-genai` SDK pointed at the harness;
- unit tests of `decode` (no server) for the mapping and every refusal,
  plus pins of the litellm translator this adapter wraps: a litellm
  upgrade that changes those internals fails here, loudly.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from litellm.google_genai.adapters.transformation import (
    GoogleGenAIAdapter,
    GoogleGenAIStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from routellm.adapters import AdapterRefusal
from routellm.adapters.gemini import GeminiAdapter
from routellm.tests.adapters.conformance import (
    KEY,
    PNG_B64,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
    ToolUse,
)

pytestmark = pytest.mark.slow

HI = {"role": "user", "parts": [{"text": "hi"}]}


# -- the client side of the dialect -------------------------------------------


def _parts(message) -> list[dict]:
    parts = []
    if message.text:
        parts.append({"text": message.text})
    if message.image_png_b64:
        parts.append({"inlineData": {"mimeType": "image/png", "data": message.image_png_b64}})
    for use in message.tool_uses:
        parts.append({"functionCall": {"id": use.id, "name": use.name, "args": use.arguments}})
    for result in message.tool_results:
        # The scenario carries the tool's text; the call's name is the
        # one the history made for that id.
        parts.append(
            {"functionResponse": {"id": result.id, "response": {"output": result.content}}}
        )
    return parts


class GeminiDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        names = {u.id: u.name for m in scenario.messages for u in m.tool_uses}
        contents = []
        for message in scenario.messages:
            parts = _parts(message)
            for part in parts:
                response = part.get("functionResponse")
                if response:
                    response["name"] = names[response["id"]]
            contents.append(
                {"role": "model" if message.role == "assistant" else "user", "parts": parts}
            )
        body = {"contents": contents}
        if scenario.system:
            body["systemInstruction"] = {"parts": [{"text": scenario.system}]}
        if scenario.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in scenario.tools
                    ]
                }
            ]
        action = "streamGenerateContent" if scenario.stream else "generateContent"
        params = {"alt": "sse"} if scenario.stream else {}
        return Http("POST", f"/v1beta/models/{scenario.model}:{action}", params=params, json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["x-goog-api-key"] = key
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if scenario.stream:
            chunks = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
        else:
            chunks = [response.json()]
        return read_outcome(chunks)

    def parse_error(self, response: httpx.Response) -> str:
        error = response.json()["error"]
        assert isinstance(error["code"], int) and error["status"].isupper()
        return error["message"]

    def unsupported(self) -> Http:
        return Http(
            "POST", "/v1beta/models/default:generateContent", json={"contents": [HI], "labels": {}}
        )


def read_outcome(chunks: list[dict]) -> Outcome:
    out = Outcome()
    reason = None
    for chunk in chunks:
        for candidate in chunk.get("candidates") or []:
            for part in candidate["content"]["parts"]:
                out.text += part.get("text", "")
                call = part.get("functionCall")
                if call:
                    out.tool_uses.append(ToolUse(call.get("id"), call["name"], call["args"]))
            reason = candidate.get("finishReason") or reason
        usage = chunk.get("usageMetadata")
        if usage:
            out.input_tokens = usage["promptTokenCount"]
            out.output_tokens = usage["candidatesTokenCount"]
    out.finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason, reason)
    if out.tool_uses and out.finish == "stop":
        out.finish = "tool_calls"
    return out


class TestGeminiDialect(ConformanceSuite):
    driver = GeminiDriver()


# -- wire details --------------------------------------------------------------


def post(harness, path, body, *, key=KEY, headers=None, **params):
    headers = {"x-goog-api-key": key, **(headers or {})} if key else dict(headers or {})
    return httpx.post(f"{harness.base}{path}", headers=headers, params=params, json=body)


def test_key_in_query_string_authenticates(harness):
    response = post(harness, "/v1beta/models/default:generateContent", {"contents": [HI]}, key=None)
    assert response.status_code == 401
    response = httpx.post(
        f"{harness.base}/v1beta/models/default:generateContent",
        params={"key": KEY},
        json={"contents": [HI]},
    )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["role"] == "model"


def test_bearer_is_not_this_dialects_credential(harness):
    response = post(
        harness,
        "/v1beta/models/default:generateContent",
        {"contents": [HI]},
        key=None,
        headers={"authorization": f"Bearer {KEY}"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["status"] == "UNAUTHENTICATED"


@pytest.mark.parametrize("prefix", ["/v1", "/v1beta"])
def test_both_version_prefixes_serve_every_action(harness, prefix):
    harness.upstream.mode = "echo"
    body = {"contents": [HI]}
    text = post(harness, f"{prefix}/models/default:generateContent", body).json()
    assert text["candidates"][0]["content"]["parts"][0]["text"] == "echo: hi"
    stream = post(harness, f"{prefix}/models/default:streamGenerateContent", body, alt="sse")
    assert stream.text.startswith("data: ")
    count = post(harness, f"{prefix}/models/default:countTokens", body)
    assert count.json()["totalTokens"] > 0


def test_model_in_path_carries_routing_parameters(harness):
    post(harness, "/v1beta/models/default:generateContent", {"contents": [HI]})
    assert harness.upstream.hits or harness.other.hits
    response = post(harness, "/v1beta/models/models/default:generateContent", {"contents": [HI]})
    assert response.status_code == 200


def test_json_array_stream_without_alt_sse(harness):
    harness.upstream.mode = "echo"
    response = post(harness, "/v1beta/models/default:streamGenerateContent", {"contents": [HI]})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    chunks = response.json()
    assert isinstance(chunks, list) and len(chunks) > 1
    assert read_outcome(chunks).text == "echo: hi"
    assert read_outcome(chunks).input_tokens == 3


def test_sse_stream_ends_with_one_final_chunk_holding_finish_and_usage(harness):
    harness.upstream.mode = "echo"
    response = post(
        harness, "/v1beta/models/default:streamGenerateContent", {"contents": [HI]}, alt="sse"
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    finals = [c for c in chunks if c["candidates"][0].get("finishReason")]
    assert finals == [chunks[-1]]
    assert chunks[-1]["usageMetadata"]["totalTokenCount"] == 6
    assert sum("usageMetadata" in c for c in chunks) == 1
    assert chunks[0]["candidates"][0]["content"]["role"] == "model"


def test_streamed_tool_call_carries_the_upstream_call_id(harness):
    harness.upstream.mode = "tools"
    body = {
        "contents": [HI],
        "tools": [
            {"functionDeclarations": [{"name": "read_file", "parameters": {"type": "OBJECT"}}]}
        ],
    }
    response = post(harness, "/v1beta/models/default:streamGenerateContent", body, alt="sse")
    out = read_outcome(
        [json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: ")]
    )
    assert [(u.id, u.name, u.arguments) for u in out.tool_uses] == [
        ("call_1", "read_file", {"p": "a.txt"})
    ]


def test_midstream_failure_ends_without_a_clean_finish(harness):
    harness.upstream.mode = "die_after"
    response = post(
        harness, "/v1beta/models/default:streamGenerateContent", {"contents": [HI]}, alt="sse"
    )
    chunks = [json.loads(x[6:]) for x in response.text.splitlines() if x.startswith("data: ")]
    assert chunks[-1].get("error", {}).get("status") == "UNAVAILABLE"
    assert not any(
        c.get("candidates", [{}])[0].get("finishReason") for c in chunks if "error" not in c
    )


def test_error_shape_has_code_message_status(harness):
    response = post(harness, "/v1beta/models/default:generateContent", {"contents": [HI]}, key="x")
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": 401,
            "message": "Missing or invalid API key",
            "status": "UNAUTHENTICATED",
        }
    }
    bad = post(harness, "/v1beta/models/default:generateContent", {"contents": [HI], "x": 1})
    assert bad.status_code == 400
    assert bad.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert "x" in bad.json()["error"]["message"]
    not_json = httpx.post(
        f"{harness.base}/v1beta/models/default:generateContent",
        headers={"x-goog-api-key": KEY},
        content=b"{",
    )
    assert not_json.status_code == 400 and not_json.json()["error"]["status"] == "INVALID_ARGUMENT"


def test_safety_settings_are_accepted_and_have_no_effect(harness):
    body = {
        "contents": [HI],
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
    }
    assert post(harness, "/v1beta/models/default:generateContent", body).status_code == 200
    assert "safetySettings" not in json.dumps(harness.upstream.hits[-1][2])


def test_count_tokens_estimates_without_calling_the_upstream(harness):
    short = post(harness, "/v1beta/models/default:countTokens", {"contents": [HI]})
    assert short.status_code == 200
    n = short.json()["totalTokens"]
    assert isinstance(n, int) and n > 0
    long_text = {"role": "user", "parts": [{"text": "word " * 200}]}
    wrapped = {"generateContentRequest": {"model": "models/default", "contents": [long_text]}}
    assert (
        post(harness, "/v1beta/models/default:countTokens", wrapped).json()["totalTokens"] > n * 20
    )
    assert not harness.upstream.hits and not harness.other.hits
    both = {"contents": [HI], "generateContentRequest": {"contents": [HI]}}
    assert post(harness, "/v1beta/models/default:countTokens", both).status_code == 400
    assert (
        post(
            harness, "/v1beta/models/default:countTokens", {"contents": [HI]}, key=None
        ).status_code
        == 401
    )


def test_models_list_and_get(harness):
    assert httpx.get(f"{harness.base}/v1beta/models").status_code == 401
    listing = httpx.get(f"{harness.base}/v1beta/models", headers={"x-goog-api-key": KEY}).json()
    names = {m["name"] for m in listing["models"]}
    assert "models/default" in names
    entry = next(m for m in listing["models"] if m["name"] == "models/default")
    assert "generateContent" in entry["supportedGenerationMethods"]
    one = httpx.get(f"{harness.base}/v1beta/models/default", headers={"x-goog-api-key": KEY})
    assert one.json()["name"] == "models/default"
    missing = httpx.get(f"{harness.base}/v1/models/nope", headers={"x-goog-api-key": KEY})
    assert missing.status_code == 404 and missing.json()["error"]["status"] == "NOT_FOUND"


def test_openai_models_route_is_untouched(harness):
    response = httpx.get(f"{harness.base}/v1/models", headers={"authorization": f"Bearer {KEY}"})
    assert response.json()["object"] == "list"


# -- google-genai SDK ------------------------------------------------------------


@pytest.fixture
def sdk(harness):
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")

    def make(version="v1beta"):
        return genai.Client(
            api_key=KEY,
            http_options=types.HttpOptions(base_url=harness.base, api_version=version),
        ), types

    return make


def test_sdk_generate_content(harness, sdk):
    client, types = sdk()
    harness.upstream.mode = "echo"
    response = client.models.generate_content(
        model="default",
        contents="hi",
        config=types.GenerateContentConfig(
            system_instruction="Be terse.", temperature=0.2, max_output_tokens=50, top_p=0.9
        ),
    )
    assert response.text == "echo: hi"
    assert response.usage_metadata.prompt_token_count == 3
    sent = harness.upstream.hits[-1][2]
    assert sent["messages"][0] == {"role": "system", "content": "Be terse."}
    assert (sent["temperature"], sent["max_tokens"], sent["top_p"]) == (0.2, 50, 0.9)


def test_sdk_v1_api_version(harness, sdk):
    client, _ = sdk("v1")
    harness.upstream.mode = "echo"
    assert client.models.generate_content(model="default", contents="hi").text == "echo: hi"


def test_sdk_stream(harness, sdk):
    client, _ = sdk()
    harness.upstream.mode = "echo"
    chunks = list(client.models.generate_content_stream(model="default", contents="hi"))
    assert "".join(c.text or "" for c in chunks) == "echo: hi"
    assert chunks[-1].usage_metadata.total_token_count == 6
    assert chunks[-1].candidates[0].finish_reason.name == "STOP"


def test_sdk_function_calling_round_trip(harness, sdk):
    client, types = sdk()
    harness.upstream.mode = "tools"
    declaration = types.FunctionDeclaration(
        name="read_file",
        description="Read a file",
        parameters=types.Schema(
            type="OBJECT", properties={"p": types.Schema(type="STRING")}, required=["p"]
        ),
    )
    config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[declaration])])
    first = client.models.generate_content(model="default", contents="hi", config=config)
    (call,) = first.function_calls
    assert (call.name, call.args) == ("read_file", {"p": "a.txt"})
    sent = harness.upstream.hits[-1][2]["tools"][0]["function"]
    assert sent["parameters"]["type"] == "object"
    assert sent["parameters"]["properties"]["p"]["type"] == "string"

    harness.upstream.mode = "echo"
    contents = [
        types.Content(role="user", parts=[types.Part(text="hi")]),
        first.candidates[0].content,
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response={"output": "file body"}
                    )
                )
            ],
        ),
    ]
    client.models.generate_content(model="default", contents=contents, config=config)
    messages = harness.upstream.hits[-1][2]["messages"]
    tool_call = next(m for m in messages if m["role"] == "assistant")["tool_calls"][0]
    result = next(m for m in messages if m["role"] == "tool")
    assert result["tool_call_id"] == tool_call["id"] == "call_1"
    assert result["content"] == "file body"


def test_sdk_image_and_count_tokens_and_models(harness, sdk):
    import base64

    client, types = sdk()
    harness.upstream.mode = "echo"
    part = types.Part.from_bytes(data=base64.b64decode(PNG_B64), mime_type="image/png")
    client.models.generate_content(model="default", contents=[part, "hi"])
    url = harness.upstream.hits[-1][2]["messages"][-1]["content"][0]["image_url"]["url"]
    assert url == f"data:image/png;base64,{PNG_B64}"
    assert client.models.count_tokens(model="default", contents="hi there").total_tokens > 0
    assert "models/default" in [m.name for m in client.models.list()]
    assert client.models.get(model="default").name == "models/default"


def test_sdk_error_is_an_api_error_with_status(harness, sdk):
    from google.genai import errors

    client, _ = sdk()
    harness.upstream.mode = harness.other.mode = "500"
    with pytest.raises(errors.APIError) as caught:
        client.models.generate_content(model="default", contents="hi")
    assert caught.value.code == 502 and caught.value.status == "UNAVAILABLE"


# -- decode: mapping and refusals (no server) ----------------------------------


def decode(body, *, action="generateContent", model="default", query=None):
    adapter = GeminiAdapter()
    route = next(
        r for r in adapter.routes if r.path.startswith("/v1beta/") and r.path.endswith(f":{action}")
    )
    call = SimpleNamespace(
        route=route, path_params={"model": model}, query_params=query or {}, body=body, state={}
    )
    return adapter.decode(call), call


def refused(body, fragment=None, **kw):
    with pytest.raises(AdapterRefusal) as caught:
        decode(body, **kw)
    if fragment:
        assert fragment in caught.value.message
    return caught.value


def test_generation_config_maps_to_chat_fields():
    request, _ = decode(
        {
            "contents": [HI],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 99,
                "stopSequences": ["END"],
                "presencePenalty": 0.5,
                "frequencyPenalty": 0.25,
                "seed": 7,
                "candidateCount": 1,
                "thinkingConfig": {"thinkingBudget": 0},
            },
            "safetySettings": [],
        }
    )
    assert request.model == "default"
    assert (request.temperature, request.top_p, request.max_tokens) == (0.3, 0.8, 99)
    assert request.stop == ["END"]
    assert (request.presence_penalty, request.frequency_penalty, request.seed) == (0.5, 0.25, 7)


def test_snake_case_names_are_accepted_like_camel_case():
    request, _ = decode(
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "system_instruction": {"parts": [{"text": "sys"}]},
            "generation_config": {"max_output_tokens": 5, "stop_sequences": ["x"]},
        }
    )
    assert request.max_tokens == 5 and request.stop == ["x"]
    assert request.messages[0] == {"role": "system", "content": "sys"}


def test_system_instruction_parts_are_joined():
    request, _ = decode(
        {"contents": [HI], "systemInstruction": {"parts": [{"text": "a"}, {"text": "b"}]}}
    )
    assert request.messages[0] == {"role": "system", "content": "a\nb"}


def test_camel_case_inline_data_becomes_an_image():
    part = {"inlineData": {"mimeType": "image/jpeg", "data": "QUJD"}}
    request, _ = decode({"contents": [{"role": "user", "parts": [{"text": "see"}, part]}]})
    assert request.messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }


def test_url_safe_base64_image_data_becomes_standard_base64():
    part = {"inlineData": {"mimeType": "image/png", "data": "a-b_c"}}
    request, _ = decode({"contents": [{"role": "user", "parts": [part]}]})
    assert request.messages[0]["content"][0]["image_url"]["url"].endswith("base64,a+b/c")


def test_streaming_route_sets_stream_and_alt_picks_the_framing():
    request, call = decode({"contents": [HI]}, action="streamGenerateContent")
    assert request.stream is True and call.state["sse"] is False
    _, call = decode({"contents": [HI]}, action="streamGenerateContent", query={"alt": "sse"})
    assert call.state["sse"] is True
    assert decode({"contents": [HI]})[0].stream is None
    refused({"contents": [HI]}, "alt", action="streamGenerateContent", query={"alt": "proto"})


def test_parallel_calls_of_one_function_get_distinct_ids_and_pair_up():
    call = lambda p: {"functionCall": {"name": "read", "args": {"p": p}}}  # noqa: E731
    answer = lambda t: {"functionResponse": {"name": "read", "response": {"output": t}}}  # noqa: E731
    request, _ = decode(
        {
            "contents": [
                HI,
                {"role": "model", "parts": [call("a"), call("b")]},
                {"role": "user", "parts": [answer("A"), answer("B")]},
            ]
        }
    )
    calls = request.messages[1]["tool_calls"]
    assert len({c["id"] for c in calls}) == 2
    results = [m for m in request.messages if m["role"] == "tool"]
    assert [(r["tool_call_id"], r["content"]) for r in results] == [
        (calls[0]["id"], "A"),
        (calls[1]["id"], "B"),
    ]


def test_client_call_ids_are_kept_and_structured_responses_stay_json():
    request, _ = decode(
        {
            "contents": [
                HI,
                {
                    "role": "model",
                    "parts": [{"functionCall": {"id": "x1", "name": "f", "args": {}}}],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "x1",
                                "name": "f",
                                "response": {"a": 1, "b": 2},
                            }
                        }
                    ],
                },
            ]
        }
    )
    assert request.messages[1]["tool_calls"][0]["id"] == "x1"
    assert request.messages[2]["tool_call_id"] == "x1"
    assert json.loads(request.messages[2]["content"]) == {"a": 1, "b": 2}


def test_json_mode_and_schema():
    request, _ = decode(
        {"contents": [HI], "generationConfig": {"responseMimeType": "application/json"}}
    )
    assert request.response_format == {"type": "json_object"}
    schema = {"type": "OBJECT", "properties": {"n": {"type": "INTEGER"}}}
    request, _ = decode(
        {
            "contents": [HI],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
        }
    )
    sent = request.response_format["json_schema"]["schema"]
    assert request.response_format["type"] == "json_schema"
    assert sent["type"] == "object" and sent["properties"]["n"]["type"] == "integer"
    refused(
        {
            "contents": [HI],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
        },
        "responseSchema",
    )


def test_tool_config_modes():
    tools = [{"functionDeclarations": [{"name": "f"}, {"name": "g"}]}]

    def choice(config):
        return decode({"contents": [HI], "tools": tools, "toolConfig": config})[0].tool_choice

    assert choice({"functionCallingConfig": {"mode": "AUTO"}}) == "auto"
    assert choice({"functionCallingConfig": {"mode": "ANY"}}) == "required"
    assert choice({"functionCallingConfig": {"mode": "NONE"}}) == "none"
    one = choice({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["g"]}})
    assert one == {"type": "function", "function": {"name": "g"}}
    refused(
        {
            "contents": [HI],
            "tools": tools,
            "toolConfig": {
                "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f", "g"]}
            },
        },
        "allowedFunctionNames",
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"contents": [HI], "cachedContent": "cachedContents/1"}, "cachedContent"),
        ({"contents": [HI], "labels": {}}, "labels"),
        ({"contents": []}, "contents"),
        ({"foo": 1}, "foo"),
        ({"contents": [HI], "generationConfig": {"responseLogprobs": True}}, "responseLogprobs"),
        ({"contents": [HI], "generationConfig": {"candidateCount": 2}}, "candidateCount"),
        (
            {"contents": [HI], "generationConfig": {"responseMimeType": "text/x.enum"}},
            "responseMimeType",
        ),
        ({"contents": [HI], "tools": [{"googleSearch": {}}]}, "googleSearch"),
        (
            {
                "contents": [HI],
                "tools": [{"functionDeclarations": [{"name": "f", "response": {}}]}],
            },
            "response",
        ),
        ({"contents": [{"role": "robot", "parts": []}]}, "role"),
        (
            {"contents": [{"role": "user", "parts": [{"fileData": {"fileUri": "gs://b/o"}}]}]},
            "fileData",
        ),
        (
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "AA"}}],
                    }
                ]
            },
            "image",
        ),
        ({"contents": [{"role": "user", "parts": [{"executableCode": {}}]}]}, "executableCode"),
        (
            {"contents": [{"role": "user", "parts": [{"functionCall": {"name": "f"}}]}]},
            "functionCall",
        ),
        (
            {
                "contents": [HI],
                "systemInstruction": {
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA"}}]
                },
            },
            "systemInstruction",
        ),
        (
            {"contents": [HI], "toolConfig": {"functionCallingConfig": {"mode": "VALIDATED"}}},
            "mode",
        ),
    ],
)
def test_unsupported_input_is_refused_never_dropped(body, fragment):
    refused(body, fragment)


def test_thought_parts_and_signatures_are_ignored():
    request, _ = decode(
        {
            "contents": [
                HI,
                {
                    "role": "model",
                    "parts": [
                        {"text": "pondering", "thought": True},
                        {"text": "answer", "thoughtSignature": "c2ln"},
                    ],
                },
            ]
        }
    )
    assert request.messages[1]["content"] == "answer"


# -- litellm pins: the translator this adapter wraps ------------------------------
# These assert what litellm's `GoogleGenAIAdapter` does today. A litellm
# upgrade that changes it fails here. Where litellm lacks something the
# adapter adds, the pin says so, and goes red when litellm fixes it (the
# shim can then go).


def test_pin_request_translation():
    messages = GoogleGenAIAdapter().translate_generate_content_to_completion(
        model="m",
        contents=[
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "f", "args": {"a": 1}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"r": 1}}}]},
        ],
        config={"temperature": 0.5, "maxOutputTokens": 7, "topP": 0.1, "stopSequences": ["s"]},
        systemInstruction={"parts": [{"text": "sys"}]},
        tools=[{"functionDeclarations": [{"name": "f", "parameters": {"type": "OBJECT"}}]}],
        toolConfig={"functionCallingConfig": {"mode": "ANY"}},
    )
    assert [m["role"] for m in messages["messages"]] == ["system", "user", "assistant", "tool"]
    assert messages["messages"][2]["tool_calls"][0]["id"] == "call_f"
    assert messages["messages"][3]["tool_call_id"] == "call_f"
    assert messages["messages"][3]["content"] == json.dumps({"r": 1})
    assert (messages["temperature"], messages["max_tokens"], messages["top_p"]) == (0.5, 7, 0.1)
    assert messages["stop"] == ["s"]
    assert messages["tool_choice"] == "required"
    assert messages["tools"][0]["function"]["parameters"]["type"] == "object"


def test_pin_litellm_only_reads_snake_case_inline_data():
    translate = GoogleGenAIAdapter().translate_generate_content_to_completion
    blob = {"mime_type": "image/png", "data": "AA"}
    snake = translate("m", [{"role": "user", "parts": [{"inline_data": blob}]}])
    assert snake["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AA"
    camel = translate(
        "m",
        [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "AA"}}]}],
    )
    assert camel["messages"] == []  # dropped silently: the adapter normalizes the name


def test_pin_litellm_drops_what_the_adapter_refuses_or_maps_itself():
    out = GoogleGenAIAdapter().translate_generate_content_to_completion(
        "m",
        [HI],
        config={"topK": 3, "presencePenalty": 1.0, "seed": 1, "candidateCount": 2},
        toolConfig={"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}},
    )
    for key in ("top_k", "presence_penalty", "seed", "n"):
        assert key not in out
    assert out["tool_choice"] == "required"


def _response(**message):
    return ModelResponse(
        choices=[
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
        ],
        usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    )


def test_pin_response_translation():
    out = GoogleGenAIAdapter().translate_completion_to_generate_content(
        _response(
            content="yo",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
            ],
        )
    )
    candidate = out["candidates"][0]
    assert candidate["content"] == {
        "role": "model",
        "parts": [{"text": "yo"}, {"functionCall": {"name": "f", "args": {"a": 1}}}],
    }  # no call id: the adapter adds it
    assert candidate["finishReason"] == "STOP"
    assert out["usageMetadata"] == {
        "promptTokenCount": 4,
        "candidatesTokenCount": 2,
        "totalTokenCount": 6,
    }


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop", "STOP"),
        ("length", "MAX_TOKENS"),
        ("content_filter", "SAFETY"),
        ("tool_calls", "STOP"),
    ],
)
def test_pin_finish_reasons(reason, expected):
    assert GoogleGenAIAdapter()._map_finish_reason(reason) == expected


def _chunk(content=None, finish=None, usage=None, tool=None):
    if content is None and tool is None and finish is None:
        return ModelResponseStream(choices=[], usage=usage)
    delta = Delta(content=content, tool_calls=[tool] if tool else None)
    choice = StreamingChoices(index=0, delta=delta, finish_reason=finish)
    return ModelResponseStream(choices=[choice])


def _collect(chunks):
    async def source():
        for chunk in chunks:
            yield chunk

    async def run():
        return [item async for item in GoogleGenAIStreamWrapper(source())]

    return asyncio.run(run())


def test_pin_stream_translation_and_missing_trailing_usage():
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    items = _collect([_chunk("he"), _chunk("llo"), _chunk(finish="stop"), _chunk(usage=usage)])
    assert [i["candidates"][0]["content"]["parts"] for i in items[:2]] == [
        [{"text": "he"}],
        [{"text": "llo"}],
    ]
    assert items[2]["candidates"][0]["finishReason"] == "STOP"
    # usage rides a trailing choice-less chunk: litellm reports zeros, the adapter fixes it
    assert items[2]["usageMetadata"]["totalTokenCount"] == 0
    assert len(items) == 3


def test_pin_stream_tool_call_accumulates_and_flushes_empty_arguments():
    def tool(args, name=None, index=0):
        function = Function(name=name, arguments=args)
        return ChatCompletionDeltaToolCall(index=index, id="c1", function=function)

    items = _collect(
        [_chunk(tool=tool('{"a":', "f")), _chunk(tool=tool("1}")), _chunk(finish="tool_calls")]
    )
    assert items[0]["candidates"][0]["content"]["parts"] == [
        {"functionCall": {"name": "f", "args": {"a": 1}}}
    ]
    flushed = _collect([_chunk(tool=tool("", "g")), _chunk(finish="tool_calls")])
    assert flushed[-1]["candidates"][0]["content"]["parts"] == [
        {"functionCall": {"name": "g", "args": {}}}
    ]
