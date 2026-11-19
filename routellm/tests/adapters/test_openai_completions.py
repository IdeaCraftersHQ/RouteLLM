"""The OpenAI legacy Completions adapter: conformance suite, wire details, SDK."""

import json

import httpx
import pytest

from routellm.tests.adapters.conformance import (
    ECHO,
    KEY,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
)

pytestmark = pytest.mark.slow

PATH = "/v1/completions"


def post(harness, body, key=KEY):
    headers = {"authorization": f"Bearer {key}"} if key else {}
    return httpx.post(f"{harness.base}{PATH}", headers=headers, json=body, timeout=60)


def sse(response: httpx.Response) -> list:
    """The `data:` payloads of a stream, `[DONE]` kept as the string."""
    lines = [x[6:] for x in response.text.splitlines() if x.startswith("data: ")]
    return [x if x == "[DONE]" else json.loads(x) for x in lines]


class CompletionsDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        (message,) = scenario.messages
        body = {"model": scenario.model, "prompt": message.text, "stream": scenario.stream}
        if scenario.stream:
            body["stream_options"] = {"include_usage": True}
        return Http("POST", PATH, json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            body = response.json()
            assert body["object"] == "text_completion"
            choice = body["choices"][0]
            return Outcome(
                text=choice["text"],
                finish=choice["finish_reason"],
                input_tokens=body["usage"]["prompt_tokens"],
                output_tokens=body["usage"]["completion_tokens"],
            )
        out = Outcome()
        events = sse(response)
        assert events[-1] == "[DONE]"
        for event in events[:-1]:
            assert event["object"] == "text_completion"
            for choice in event["choices"]:
                out.text += choice["text"]
                out.finish = choice["finish_reason"] or out.finish
            if event.get("usage"):
                out.input_tokens = event["usage"]["prompt_tokens"]
                out.output_tokens = event["usage"]["completion_tokens"]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        error = response.json()["error"]
        assert {"message", "type", "param", "code"} <= set(error)
        return error["message"]

    def unsupported(self) -> Http:
        return Http("POST", PATH, json={"model": "default", "prompt": "hi", "suffix": "x"})


class TestOpenAICompletions(ConformanceSuite):
    driver = CompletionsDriver()
    skip = {  # noqa: RUF012
        "system_prompt": "a completion is one prompt, no system role",
        "multi_turn": "a completion is one prompt, no history",
        "image_input": "text completions have no image input",
        "tool_call_response": "text completions have no tools",
        "tool_result_request": "text completions have no tools",
        "streaming_tool_call": "text completions have no tools",
    }


# -- request mapping ----------------------------------------------------------


def test_parameters_reach_the_upstream_as_chat_fields(harness):
    response = post(
        harness,
        {
            "model": "default",
            "prompt": "hi",
            "max_tokens": 7,
            "stop": ["\n", "END"],
            "temperature": 0.25,
            "top_p": 0.5,
            "n": 2,
            "seed": 11,
            "user": "u1",
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "logit_bias": {"50256": -100},
        },
    )
    assert response.status_code == 200, response.text
    sent = harness.upstream.hits[-1][2]
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 7
    assert sent["stop"] == ["\n", "END"]
    assert (sent["temperature"], sent["top_p"], sent["n"], sent["seed"]) == (0.25, 0.5, 2, 11)
    # `user` is the routing session id in the pipeline; it is not forwarded.
    assert (sent["presence_penalty"], sent["frequency_penalty"]) == (0.5, 0.25)
    assert sent["logit_bias"] == {"50256": -100.0}
    assert "prompt" not in sent


def test_unset_parameters_are_not_manufactured(harness):
    post(harness, {"model": "default", "prompt": "hi"})
    sent = harness.upstream.hits[-1][2]
    assert "max_tokens" not in sent and "temperature" not in sent and "n" not in sent


def test_list_with_one_prompt_is_that_prompt(harness):
    harness.upstream.mode = "echo"
    response = post(harness, {"model": "default", "prompt": ["hi"]})
    assert response.json()["choices"][0]["text"] == ECHO


def test_model_carries_routing_parameters(harness):
    response = post(harness, {"model": "router-random-0.0", "prompt": "hi"})
    assert response.status_code == 200, response.text
    assert json.loads(response.headers["x-routellm-path"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", ["a", "b"]),
        ("prompt", [1, 2, 3]),
        ("prompt", [[1, 2], [3]]),
        ("prompt", []),
        ("prompt", 7),
        ("suffix", "tail"),
        ("logprobs", 0),
        ("logprobs", 3),
        ("best_of", 3),
        ("stream_options", {"include_usage": True}),
        ("frobnicate", True),
    ],
)
def test_refused_before_the_pipeline(harness, field, value):
    body = {"model": "default", "prompt": "hi", field: value}
    response = post(harness, body)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["message"] and error["type"] == "invalid_request_error"
    assert not harness.upstream.hits


def test_missing_prompt_and_model_are_refused(harness):
    for body in ({"model": "default"}, {"prompt": "hi"}, [], {"model": "default", "prompt": None}):
        response = post(harness, body)
        assert response.status_code == 400, body
        assert response.json()["error"]["message"]
    assert not harness.upstream.hits


def test_refusal_names_the_parameter(harness):
    error = post(harness, {"model": "default", "prompt": "hi", "suffix": "x"}).json()["error"]
    assert error["param"] == "suffix"


def test_best_of_one_and_null_suffix_are_accepted(harness):
    response = post(
        harness,
        {"model": "default", "prompt": "hi", "best_of": 1, "suffix": None, "logprobs": None},
    )
    assert response.status_code == 200, response.text


# -- response shape -----------------------------------------------------------


def test_response_is_a_text_completion(harness):
    harness.upstream.mode = "echo"
    body = post(harness, {"model": "default", "prompt": "hi"}).json()
    assert body["object"] == "text_completion"
    assert body["id"] and isinstance(body["created"], int) and body["model"]
    (choice,) = body["choices"]
    assert choice == {
        "text": ECHO,
        "index": 0,
        "logprobs": None,
        "finish_reason": "stop",
    }
    assert body["usage"]["total_tokens"] == 6
    assert "message" not in choice


def test_echo_prepends_the_prompt(harness):
    harness.upstream.mode = "echo"
    body = post(harness, {"model": "default", "prompt": "hi", "echo": True}).json()
    assert body["choices"][0]["text"] == "hi" + ECHO
    sent = harness.upstream.hits[-1][2]
    assert "echo" not in sent


def test_stream_chunks_and_terminator(harness):
    harness.upstream.mode = "echo"
    response = post(harness, {"model": "default", "prompt": "hi", "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse(response)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "text_completion" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert "".join(c["choices"][0]["text"] for c in chunks) == ECHO
    assert [c["choices"][0]["finish_reason"] for c in chunks][-1] == "stop"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    # The client did not ask for usage: no usage chunk.
    assert not any("usage" in c for c in chunks)
    assert harness.upstream.hits[-1][2]["stream"] is True


def test_stream_usage_only_when_asked(harness):
    body = {
        "model": "default",
        "prompt": "hi",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks = sse(post(harness, body))[:-1]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}


def test_stream_echo_is_the_first_text(harness):
    harness.upstream.mode = "echo"
    chunks = sse(post(harness, {"model": "default", "prompt": "hi", "echo": True, "stream": True}))
    text = [c["choices"][0]["text"] for c in chunks[:-1]]
    assert text[0] == "hi"
    assert "".join(text) == "hi" + ECHO


def test_errors_use_the_openai_shape(harness):
    assert httpx.post(f"{harness.base}{PATH}", json={}).json()["error"]["type"] == (
        "authentication_error"
    )
    harness.upstream.mode = harness.other.mode = "500"
    response = post(harness, {"model": "default", "prompt": "hi"})
    assert response.status_code == 502
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}


def test_stream_failure_has_no_clean_terminator(harness):
    harness.upstream.mode = harness.other.mode = "die_after"
    response = post(harness, {"model": "default", "prompt": "hi", "stream": True})
    assert "[DONE]" not in response.text


def test_chat_route_is_untouched(harness):
    harness.upstream.mode = "echo"
    response = httpx.post(
        f"{harness.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {KEY}"},
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.json()["choices"][0]["message"]["content"] == ECHO


# -- the official SDK ---------------------------------------------------------


def _client(harness):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url=f"{harness.base}/v1", api_key=KEY, max_retries=0)


def test_sdk_completion(harness):
    harness.upstream.mode = "echo"
    result = _client(harness).completions.create(model="default", prompt="hi", max_tokens=5)
    assert result.object == "text_completion"
    assert result.choices[0].text == ECHO
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 3


def test_sdk_streaming_with_usage(harness):
    harness.upstream.mode = "echo"
    stream = _client(harness).completions.create(
        model="default", prompt="hi", stream=True, stream_options={"include_usage": True}
    )
    chunks = list(stream)
    assert "".join(c.choices[0].text for c in chunks if c.choices) == ECHO
    assert chunks[-1].usage.completion_tokens == 3


def test_sdk_refusal_is_a_bad_request(harness):
    openai = pytest.importorskip("openai")
    with pytest.raises(openai.BadRequestError) as info:
        _client(harness).completions.create(model="default", prompt="hi", suffix="x")
    assert "suffix" in str(info.value)
