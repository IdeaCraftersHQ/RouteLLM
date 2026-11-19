"""The Bedrock InvokeModel adapter (Anthropic bodies): conformance, framing, boto3, auth.

The suite drives the real server with bearer auth. The event-stream frames
are read back by the independent decoder of the Converse tests (written from
the format description, CRCs checked) and cross-checked with botocore's.
"""

import base64
import json

import httpx
import pytest

from routellm.adapters import AdapterRefusal, Call
from routellm.adapters.bedrock_invoke import BedrockInvoke
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
from routellm.tests.adapters.test_anthropic_messages import _blocks
from routellm.tests.adapters.test_bedrock_converse import SIGV4, decode_stream

pytestmark = pytest.mark.slow

VERSION = "bedrock-2023-05-31"
FINISH = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
AUTH = {"authorization": f"Bearer {KEY}"}


def anthropic_body(**extra) -> dict:
    return {
        "anthropic_version": VERSION,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def chunk_events(response: httpx.Response) -> list[dict]:
    """The Anthropic events of a stream: frame headers checked, bytes unwrapped."""
    assert response.headers["content-type"] == "application/vnd.amazon.eventstream"
    events = []
    for headers, payload in decode_stream(response.content):
        assert headers == {
            ":event-type": "chunk",
            ":message-type": "event",
            ":content-type": "application/json",
        }
        wrapper = json.loads(payload)
        assert set(wrapper) == {"bytes"}
        events.append(json.loads(base64.b64decode(wrapper["bytes"], validate=True)))
    return events


class InvokeDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        body = anthropic_body(
            messages=[{"role": m.role, "content": _blocks(m)} for m in scenario.messages]
        )
        if scenario.system:
            body["system"] = scenario.system
        if scenario.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in scenario.tools
            ]
        verb = "invoke-with-response-stream" if scenario.stream else "invoke"
        return Http("POST", f"/model/{scenario.model}/{verb}", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
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
                finish=FINISH[body["stop_reason"]],
                input_tokens=body["usage"]["input_tokens"],
                output_tokens=body["usage"]["output_tokens"],
            )
        events = chunk_events(response)
        assert events[0]["type"] == "message_start" and events[-1]["type"] == "message_stop"
        out, tools, stop = Outcome(), {}, None
        for data in events:
            if data["type"] == "content_block_start":
                if data["content_block"]["type"] == "tool_use":
                    tools[data["index"]] = [data["content_block"], ""]
            elif data["type"] == "content_block_delta":
                d = data["delta"]
                if d["type"] == "text_delta":
                    out.text += d["text"]
                elif d["type"] == "input_json_delta":
                    tools[data["index"]][1] += d["partial_json"]
            elif data["type"] == "message_delta":
                stop = data["delta"]["stop_reason"]
                out.input_tokens = data["usage"]["input_tokens"]
                out.output_tokens = data["usage"]["output_tokens"]
        out.finish = FINISH[stop]
        out.tool_uses = [
            ToolUse(b["id"], b["name"], json.loads(args or "{}")) for b, args in tools.values()
        ]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        assert response.headers["x-amzn-errortype"]
        return response.json()["message"]

    def unsupported(self) -> Http:
        http = self.build(Scenario((HI,)))
        http.json["top_k"] = 5
        return http


class TestBedrockInvoke(ConformanceSuite):
    driver = InvokeDriver()


# -- decode, without a server ------------------------------------------------------


ADAPTER = BedrockInvoke()
INVOKE, STREAM, COUNT = ADAPTER.routes


def decode(body: dict, route=INVOKE, model="default"):
    call = Call(adapter=ADAPTER, route=route, http_request=None, pipeline=None, body=body)
    call.http_request = type("R", (), {"path_params": {"model_id": model}})()
    return ADAPTER.decode(call), call


def refused(body: dict) -> AdapterRefusal:
    with pytest.raises(AdapterRefusal) as info:
        decode(body)
    return info.value


def test_routes_and_operations():
    assert [(r.path.rsplit("/", 1)[1], r.operation) for r in ADAPTER.routes] == [
        ("invoke", "chat"),
        ("invoke-with-response-stream", "chat"),
        ("count-tokens", "count_tokens"),
    ]


def test_model_comes_from_the_path_and_stream_from_the_route():
    request, call = decode(anthropic_body(), model="default")
    assert request.model == "default" and not request.stream
    assert call.state["model"] == "default"
    request, _ = decode(
        anthropic_body(), route=STREAM, model="arn:aws:bedrock:x/inference-profile/y"
    )
    assert request.stream is True
    assert request.model == "arn:aws:bedrock:x/inference-profile/y"


def test_version_and_beta_are_accepted_and_ignored():
    decode(anthropic_body(anthropic_beta=["interleaved-thinking-2025-05-14", "x"]))
    decode({k: v for k, v in anthropic_body().items() if k != "anthropic_version"})
    decode(anthropic_body(anthropic_version="anything"))


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"anthropic_beta": "x"}, "anthropic_beta"),
        ({"anthropic_beta": [1]}, "anthropic_beta"),
        ({"anthropic_version": 2}, "anthropic_version"),
        ({"model": "other"}, "model"),
        ({"stream": True}, "stream"),
        ({"top_k": 5}, "top_k"),
        ({"mcp_servers": []}, "mcp_servers"),
    ],
)
def test_refusals_name_the_field(extra, param):
    assert refused(anthropic_body(**extra)).param == param


# -- auth ---------------------------------------------------------------------------


FLAG = "ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4"
PRESIGNED = {"X-Amz-Credential": "AKIAQ/20260101/us-east-1/bedrock/aws4_request"}


def test_credential_matches_the_converse_adapter(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert ADAPTER.credential({"authorization": "Bearer tok"}, {}) == "tok"
    assert ADAPTER.credential({"authorization": SIGV4.format(key="AKIAX", sig="ab")}, {}) is None
    assert ADAPTER.credential({}, PRESIGNED) is None
    monkeypatch.setenv(FLAG, "1")
    assert ADAPTER.credential({"authorization": SIGV4.format(key="AKIAX", sig="ab")}, {}) == "AKIAX"
    assert ADAPTER.credential({}, PRESIGNED) is None


@pytest.mark.parametrize("verb", ["invoke", "invoke-with-response-stream", "count-tokens"])
def test_sigv4_alone_and_query_credentials_are_refused(harness, verb):
    url = f"{harness.base}/model/default/{verb}"
    body = {"input": {"invokeModel": {"body": {}}}} if verb == "count-tokens" else anthropic_body()
    sigv4 = {"authorization": SIGV4.format(key=KEY, sig="00")}
    for kwargs in (
        {"headers": sigv4},
        {"params": {"X-Amz-Credential": f"{KEY}/d/r/s/aws4_request"}},
        {},
    ):
        response = httpx.post(url, json=body, **kwargs)
        assert response.status_code == 401
        assert response.headers["x-amzn-errortype"] == "UnrecognizedClientException"
    assert not harness.upstream.hits


def test_bearer_token_authenticates(harness):
    harness.upstream.mode = "echo"
    url = f"{harness.base}/model/default/invoke"
    assert httpx.post(url, json=anthropic_body(), headers=AUTH).status_code == 200
    wrong = httpx.post(url, json=anthropic_body(), headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401


# -- the wire -----------------------------------------------------------------------


def invoke(harness, body, verb="invoke", model="default"):
    return httpx.post(f"{harness.base}/model/{model}/{verb}", json=body, headers=AUTH, timeout=60)


def test_invoke_body_and_headers(harness):
    harness.upstream.mode = "echo"
    response = invoke(harness, anthropic_body(system="sys", temperature=0.5))
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "echo: hi"}]
    assert body["model"] == "default" and body["stop_reason"] == "end_turn"
    assert response.headers["x-amzn-bedrock-input-token-count"] == "3"
    assert response.headers["x-amzn-bedrock-output-token-count"] == "3"
    assert json.loads(response.headers["x-routellm-path"])
    sent = harness.upstream.hits[-1][2]
    assert sent["temperature"] == 0.5 and sent["messages"][0] == {
        "role": "system",
        "content": "sys",
    }


def test_model_id_with_colon_and_arn_reaches_routing(harness):
    harness.upstream.mode = "echo"
    for model in (
        "anthropic.claude-sonnet-4%3A0",
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Ainference-profile%2Fdefault",
    ):
        response = invoke(harness, anthropic_body(), model=model)
        # the router judges the name (here: not a routing model); the route matched
        # and the whole id, colons and slashes included, reached it
        assert response.status_code == 400, response.text
        assert response.headers["x-amzn-errortype"] == "ValidationException"
        assert model.replace("%3A", ":").replace("%2F", "/") in response.json()["message"]


def test_stream_frames_are_chunk_events_in_order(harness):
    harness.upstream.mode = "echo"
    response = invoke(harness, anthropic_body(), verb="invoke-with-response-stream")
    assert response.status_code == 200
    kinds = [e["type"] for e in chunk_events(response)]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    assert "ping" in kinds and "content_block_delta" in kinds
    assert harness.upstream.hits[-1][2]["stream"] is True


def test_stream_failure_is_an_exception_frame_without_terminator(harness):
    harness.upstream.mode = "die_after"
    response = invoke(harness, anthropic_body(), verb="invoke-with-response-stream")
    frames = decode_stream(response.content)
    last = frames[-1][0]
    assert last[":message-type"] == "exception"
    assert last[":exception-type"] == "internalServerException"
    kinds = [json.loads(base64.b64decode(json.loads(p)["bytes"]))["type"] for h, p in frames[:-1]]
    assert "message_stop" not in kinds
    assert "boom" not in response.text


def test_refusal_is_a_validation_exception(harness):
    response = invoke(harness, anthropic_body(top_k=3))
    assert response.status_code == 400
    assert response.headers["x-amzn-errortype"] == "ValidationException"
    assert "top_k" in response.json()["message"]
    assert not harness.upstream.hits


# -- count-tokens -------------------------------------------------------------------


def count(harness, blob, model="default", headers=AUTH):
    return httpx.post(
        f"{harness.base}/model/{model}/count-tokens",
        json={"input": {"invokeModel": {"body": blob}}},
        headers=headers,
        timeout=60,
    )


def test_count_tokens_accepts_base64_json_text_and_object(harness):
    body = anthropic_body(
        system="be brief", tools=[{"name": "f", "input_schema": {"type": "object"}}]
    )
    counts = set()
    for blob in (
        base64.b64encode(json.dumps(body).encode()).decode(),
        json.dumps(body),
        body,
    ):
        response = count(harness, blob)
        assert response.status_code == 200, response.text
        (n,) = response.json().values() if set(response.json()) == {"inputTokens"} else (None,)
        assert isinstance(n, int) and n > 0
        counts.add(n)
    assert len(counts) == 1
    assert not harness.upstream.hits


def test_count_tokens_grows_with_the_input(harness):
    small = count(harness, anthropic_body()).json()["inputTokens"]
    big_body = anthropic_body(messages=[{"role": "user", "content": "word " * 200}])
    assert count(harness, big_body).json()["inputTokens"] > small


@pytest.mark.parametrize(
    "payload",
    [
        {"input": {"converse": {"messages": []}}},
        {"input": {"invokeModel": {}}},
        {"input": {"invokeModel": {"body": "!!not base64!!"}}},
        {"input": {"invokeModel": {"body": base64.b64encode(b"[1]").decode()}}},
        {"input": {"invokeModel": {"body": {"model": "x", "messages": []}}}},
        {"input": {"invokeModel": {"body": anthropic_body(top_k=1)}}},
        {"other": 1},
    ],
)
def test_count_tokens_refusals(harness, payload):
    response = httpx.post(
        f"{harness.base}/model/default/count-tokens", json=payload, headers=AUTH, timeout=60
    )
    assert response.status_code == 400, response.text
    assert response.headers["x-amzn-errortype"] == "ValidationException"


# -- framing ------------------------------------------------------------------------


def test_frames_round_trip_through_botocore(harness):
    buffer_module = pytest.importorskip("botocore.eventstream")
    harness.upstream.mode = "echo"
    response = invoke(harness, anthropic_body(), verb="invoke-with-response-stream")
    buffer = buffer_module.EventStreamBuffer()
    buffer.add_data(response.content)
    theirs = [(m.headers, m.payload) for m in buffer]
    ours = decode_stream(response.content)
    assert theirs == [(h, p) for h, p in ours]
    assert all(h[":event-type"] == "chunk" for h, _ in theirs)


# -- boto3 --------------------------------------------------------------------------


@pytest.fixture
def runtime(harness):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    client = boto3.client(
        "bedrock-runtime",
        endpoint_url=harness.base,
        region_name="us-east-1",
        aws_access_key_id=KEY,
        aws_secret_access_key="not-checked",
        config=Config(retries={"max_attempts": 1, "mode": "standard"}),
    )

    def bearer_header(request, **_):
        request.headers["Authorization"] = f"Bearer {KEY}"

    client.meta.events.register("before-send.bedrock-runtime.*", bearer_header)
    return client


def test_boto3_invoke_model(harness, runtime):
    harness.upstream.mode = "echo"
    out = runtime.invoke_model(modelId="default", body=json.dumps(anthropic_body()))
    body = json.loads(out["body"].read())
    assert body["content"][0]["text"] == "echo: hi"
    assert out["contentType"] == "application/json"


def test_boto3_invoke_model_with_response_stream(harness, runtime):
    harness.upstream.mode = "echo"
    out = runtime.invoke_model_with_response_stream(
        modelId="default", body=json.dumps(anthropic_body())
    )
    events, text = [], ""
    for item in out["body"]:
        event = json.loads(item["chunk"]["bytes"])
        events.append(event["type"])
        if event["type"] == "content_block_delta":
            text += event["delta"]["text"]
    assert text == "echo: hi"
    assert events[0] == "message_start" and events[-1] == "message_stop"


def test_boto3_stream_failure_raises(harness, runtime):
    from botocore.exceptions import ClientError, EventStreamError

    harness.upstream.mode = "die_after"
    out = runtime.invoke_model_with_response_stream(
        modelId="default", body=json.dumps(anthropic_body())
    )
    with pytest.raises((EventStreamError, ClientError)) as info:
        list(out["body"])
    assert "internalServerException" in str(info.value) or "InternalServer" in str(info.value)


def test_boto3_error_code(harness, runtime):
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError) as info:
        runtime.invoke_model(modelId="default", body=json.dumps(anthropic_body(top_k=1)))
    assert info.value.response["Error"]["Code"] == "ValidationException"
    assert info.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400


def test_boto3_count_tokens(harness, runtime):
    if not hasattr(runtime, "count_tokens"):
        pytest.skip("this botocore has no bedrock-runtime count_tokens")
    out = runtime.count_tokens(
        modelId="default",
        input={"invokeModel": {"body": json.dumps(anthropic_body())}},
    )
    assert out["inputTokens"] > 0
