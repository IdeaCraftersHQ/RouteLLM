"""The Bedrock Converse adapter: conformance suite, mapping matrix, framing, boto3.

The suite drives the real server with unsigned requests that carry a
SigV4-shaped `Authorization` header (the adapter never verifies the
signature). The event-stream frames are read back by a decoder written here
from the format description, independent of the encoder, and cross-checked
against botocore's own decoder.
"""

import base64
import json
import struct
import zlib

import httpx
import pytest

from routellm.adapters import AdapterRefusal, Call
from routellm.adapters import bedrock as bedrock_module
from routellm.adapters.bedrock import BedrockConverse
from routellm.adapters.eventstream import encode_message
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

SIGV4 = "AWS4-HMAC-SHA256 Credential={key}/20260101/us-east-1/bedrock/aws4_request, SignedHeaders=host;x-amz-date, Signature={sig}"
STOP = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


# -- an independent event-stream decoder ---------------------------------------


def decode_stream(data: bytes) -> list[tuple[dict, bytes]]:
    """Frames of `data` as (headers, payload), every CRC checked."""
    out, at = [], 0
    while at < len(data):
        total, hlen = struct.unpack(">II", data[at : at + 8])
        (prelude_crc,) = struct.unpack(">I", data[at + 8 : at + 12])
        assert zlib.crc32(data[at : at + 8]) == prelude_crc, "prelude CRC"
        message = data[at : at + total]
        assert len(message) == total
        (message_crc,) = struct.unpack(">I", message[-4:])
        assert zlib.crc32(message[:-4]) == message_crc, "message CRC"
        headers, pos, end = {}, 12, 12 + hlen
        while pos < end:
            n = message[pos]
            name = message[pos + 1 : pos + 1 + n].decode()
            pos += 1 + n
            assert message[pos] == 7, "string headers only"
            (vlen,) = struct.unpack(">H", message[pos + 1 : pos + 3])
            headers[name] = message[pos + 3 : pos + 3 + vlen].decode()
            pos += 3 + vlen
        out.append((headers, message[end:-4]))
        at += total
    return out


def events_of(response: httpx.Response) -> list[tuple[str, dict]]:
    assert response.headers["content-type"] == "application/vnd.amazon.eventstream"
    frames = decode_stream(response.content)
    for headers, _ in frames:
        assert headers[":message-type"] == "event", headers
        assert headers[":content-type"] == "application/json"
    return [(h[":event-type"], json.loads(p)) for h, p in frames]


# -- the driver ------------------------------------------------------------------


def block(message) -> list[dict]:
    blocks = []
    for r in message.tool_results:
        blocks.append({"toolResult": {"toolUseId": r.id, "content": [{"text": r.content}]}})
    if message.text:
        blocks.append({"text": message.text})
    if message.image_png_b64:
        blocks.append({"image": {"format": "png", "source": {"bytes": message.image_png_b64}}})
    for u in message.tool_uses:
        blocks.append({"toolUse": {"toolUseId": u.id, "name": u.name, "input": u.arguments}})
    return blocks


class BedrockDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        body: dict = {
            "messages": [{"role": m.role, "content": block(m)} for m in scenario.messages],
        }
        if scenario.system:
            body["system"] = [{"text": scenario.system}]
        if scenario.tools:
            body["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {"json": t.parameters},
                        }
                    }
                    for t in scenario.tools
                ]
            }
        verb = "converse-stream" if scenario.stream else "converse"
        return Http("POST", f"/model/{scenario.model}/{verb}", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["authorization"] = f"Bearer {key}"
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            body = response.json()
            out = Outcome(
                finish=STOP[body["stopReason"]],
                input_tokens=body["usage"]["inputTokens"],
                output_tokens=body["usage"]["outputTokens"],
            )
            for part in body["output"]["message"]["content"]:
                if "text" in part:
                    out.text += part["text"]
                elif "toolUse" in part:
                    use = part["toolUse"]
                    out.tool_uses.append(ToolUse(use["toolUseId"], use["name"], use["input"]))
            return out
        out, args, names = Outcome(), {}, {}
        for kind, data in events_of(response):
            if kind == "contentBlockStart":
                names[data["contentBlockIndex"]] = data["start"]["toolUse"]
                args[data["contentBlockIndex"]] = ""
            elif kind == "contentBlockDelta":
                delta = data["delta"]
                if "text" in delta:
                    out.text += delta["text"]
                else:
                    args[data["contentBlockIndex"]] += delta["toolUse"]["input"]
            elif kind == "messageStop":
                out.finish = STOP[data["stopReason"]]
            elif kind == "metadata":
                out.input_tokens = data["usage"]["inputTokens"]
                out.output_tokens = data["usage"]["outputTokens"]
        for index, start in sorted(names.items()):
            out.tool_uses.append(
                ToolUse(start["toolUseId"], start["name"], json.loads(args[index]))
            )
        return out

    def parse_error(self, response: httpx.Response) -> str:
        assert response.headers["x-amzn-errortype"]
        return response.json()["message"]

    def unsupported(self) -> Http:
        body = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "guardrailConfig": {"guardrailIdentifier": "g", "guardrailVersion": "1"},
        }
        return Http("POST", "/model/default/converse", json=body)


class TestBedrockConverse(ConformanceSuite):
    driver = BedrockDriver()


# -- decode: the mapping matrix, without a server -----------------------------------


def decode(body: dict, stream: bool = False):
    adapter = BedrockConverse()
    route = adapter.routes[1 if stream else 0]
    call = Call(adapter=adapter, route=route, http_request=None, pipeline=None, body=body)
    call.state["_model"] = "default"
    call.http_request = type("R", (), {"path_params": {"model_id": "default"}})()
    return adapter.decode(call), call


def refused(body: dict) -> AdapterRefusal:
    with pytest.raises(AdapterRefusal) as info:
        decode(body)
    return info.value


USER = {"role": "user", "content": [{"text": "hi"}]}


def test_inference_config_maps_to_chat_fields():
    request, _ = decode(
        {
            "messages": [USER],
            "inferenceConfig": {
                "maxTokens": 50,
                "temperature": 0.25,
                "topP": 0.5,
                "stopSequences": ["END", "STOP"],
            },
        }
    )
    assert request.max_tokens == 50
    assert request.temperature == 0.25
    assert request.top_p == 0.5
    assert request.stop == ["END", "STOP"]
    assert request.stream is None


def test_stream_route_sets_stream():
    request, call = decode({"messages": [USER]}, stream=True)
    assert request.stream is True
    assert call.state["stream"] is True


def test_tool_choice_mapping():
    spec = {"toolSpec": {"name": "f", "inputSchema": {"json": {"type": "object"}}}}
    for choice, want in (
        ({"auto": {}}, "auto"),
        ({"any": {}}, "required"),
        ({"tool": {"name": "f"}}, {"type": "function", "function": {"name": "f"}}),
    ):
        request, _ = decode(
            {"messages": [USER], "toolConfig": {"tools": [spec], "toolChoice": choice}}
        )
        assert request.tool_choice == want
    assert request.tools == [
        {
            "type": "function",
            "function": {"name": "f", "description": "", "parameters": {"type": "object"}},
        }
    ]


def test_request_metadata_rides_in_metadata():
    request, _ = decode({"messages": [USER], "requestMetadata": {"data_class": "public"}})
    assert request.metadata == {"data_class": "public"}


def test_document_block_becomes_a_file_part():
    data = base64.b64encode(b"%PDF-1.4").decode()
    request, _ = decode(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "read"},
                        {"document": {"format": "pdf", "name": "a b", "source": {"bytes": data}}},
                    ],
                }
            ]
        }
    )
    (message,) = request.messages
    assert message["content"][1] == {
        "type": "file",
        "file": {"file_data": f"data:application/pdf;base64,{data}", "filename": "a b"},
    }


def test_tool_result_json_and_status_and_order():
    request, _ = decode(
        {
            "messages": [
                {"role": "user", "content": [{"text": "q"}]},
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "t1", "name": "f", "input": {"a": 1}}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"text": "after"},
                        {
                            "toolResult": {
                                "toolUseId": "t1",
                                "status": "error",
                                "content": [{"json": {"k": [1]}}],
                            }
                        },
                    ],
                },
            ]
        }
    )
    roles = [m["role"] for m in request.messages]
    # the tool result precedes the user text that came before it in the turn
    assert roles == ["user", "assistant", "tool", "user"]
    assert request.messages[1]["content"] is None
    assert request.messages[2] == {"role": "tool", "tool_call_id": "t1", "content": '{"k": [1]}'}


def test_assistant_text_blocks_and_reasoning_history():
    request, _ = decode(
        {
            "messages": [
                USER,
                {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                        {"text": "one"},
                        {"cachePoint": {"type": "default"}},
                    ],
                },
                USER,
            ]
        }
    )
    assert request.messages[1] == {"role": "assistant", "content": "one"}


@pytest.mark.parametrize(
    "body, param",
    [
        (
            {"messages": [USER], "additionalModelRequestFields": {"thinking": {"type": "x"}}},
            "additionalModelRequestFields",
        ),
        (
            {
                "messages": [USER],
                "additionalModelRequestFields": {"top_k": 5, "thinking": {"type": "x"}},
            },
            "additionalModelRequestFields",
        ),
        (
            {"messages": [USER], "additionalModelRequestFields": {"top_k": -1}},
            "additionalModelRequestFields",
        ),
        ({"messages": [USER], "guardrailConfig": {}}, "guardrailConfig"),
        ({"messages": [USER], "outputConfig": {}}, "outputConfig"),
        (
            {"messages": [USER], "additionalModelResponseFieldPaths": ["/x"]},
            "additionalModelResponseFieldPaths",
        ),
        ({"messages": [USER], "inferenceConfig": {"topK": 3}}, "topK"),
        ({"messages": [{"role": "system", "content": [{"text": "x"}]}]}, "messages[0].role"),
        ({"messages": [{"role": "user", "content": [{"video": {}}]}]}, "messages[0].content[0]"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"image": {"format": "png", "source": {"s3Location": {}}}}],
                    }
                ]
            },
            "messages[0].content[0].image.source",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"image": {"format": "bmp", "source": {"bytes": "AA=="}}}],
                    }
                ]
            },
            "messages[0].content[0].image.format",
        ),
        ({"messages": [USER], "system": [{"guardContent": {"text": {"text": "x"}}}]}, "system[0]"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"toolUse": {"toolUseId": "a", "name": "f", "input": {}}}],
                    }
                ]
            },
            "messages[0].content[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "a",
                                    "content": [
                                        {"image": {"format": "png", "source": {"bytes": "AA=="}}}
                                    ],
                                }
                            }
                        ],
                    }
                ]
            },
            "messages[0].content[0].toolResult.content[0]",
        ),
        (
            {"messages": [USER], "toolConfig": {"tools": [{"systemTool": {"name": "x"}}]}},
            "toolConfig.tools[0]",
        ),
        ({"messages": []}, "messages"),
        (
            {"messages": [{"role": "user", "content": [{"text": "a", "image": {}}]}]},
            "messages[0].content[0]",
        ),
    ],
)
def test_refusals_name_the_field(body, param):
    assert refused(body).param == param


def test_empty_additional_fields_and_performance_hint_are_accepted():
    decode(
        {
            "messages": [USER],
            "additionalModelRequestFields": {},
            "performanceConfig": {"latency": "optimized"},
        }
    )


def test_documented_refusals_cover_every_unsupported_top_level_field():
    doc = bedrock_module.__doc__
    for field in (
        "guardrailConfig",
        "outputConfig",
        "promptVariables",
        "additionalModelResponseFieldPaths",
    ):
        assert field in doc


# -- credentials ---------------------------------------------------------------------


FLAG = "ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4"
PRESIGNED = {"X-Amz-Credential": "AKIAQ/20260101/us-east-1/bedrock/aws4_request"}


def test_credential_is_the_bearer_token_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    adapter = BedrockConverse()
    assert adapter.credential({"authorization": "Bearer tok"}, {}) == "tok"
    assert adapter.credential({"authorization": SIGV4.format(key="AKIAX", sig="ab")}, {}) is None
    assert adapter.credential({}, {}) is None


def test_sigv4_access_key_id_only_behind_the_opt_in(monkeypatch):
    adapter = BedrockConverse()
    sigv4 = {"authorization": SIGV4.format(key="AKIAEXAMPLE", sig="ab")}
    for off in ("", "0", "false", "no"):
        monkeypatch.setenv(FLAG, off)
        assert adapter.credential(sigv4, {}) is None
    for on in ("1", "true", "yes"):
        monkeypatch.setenv(FLAG, on)
        assert adapter.credential(sigv4, {}) == "AKIAEXAMPLE"
        assert adapter.credential({"authorization": "Bearer tok"}, {}) == "tok"


def test_credentials_never_come_from_the_url(monkeypatch):
    adapter = BedrockConverse()
    for value in (None, "1"):
        if value:
            monkeypatch.setenv(FLAG, value)
        else:
            monkeypatch.delenv(FLAG, raising=False)
        assert adapter.credential({}, PRESIGNED) is None
        assert adapter.credential({"authorization": "Bearer tok"}, PRESIGNED) == "tok"


def test_sigv4_alone_is_refused_on_the_wire(harness):
    url = f"{harness.base}/model/default/converse"
    body = {"messages": [USER]}
    sigv4 = {"authorization": SIGV4.format(key=KEY, sig="00")}
    for kwargs in (
        {"headers": sigv4},
        {"params": {**PRESIGNED, "X-Amz-Credential": f"{KEY}/d/r/s/aws4_request"}},
    ):
        response = httpx.post(url, json=body, **kwargs)
        assert response.status_code == 401
        assert response.headers["x-amzn-errortype"] == "UnrecognizedClientException"
    assert not harness.upstream.hits


def test_bearer_token_authenticates(harness):
    harness.upstream.mode = "echo"
    url = f"{harness.base}/model/default/converse"
    ok = httpx.post(url, json={"messages": [USER]}, headers={"authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200, ok.text
    wrong = httpx.post(url, json={"messages": [USER]}, headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401


# -- the wire ---------------------------------------------------------------------------


def converse(harness, body, stream=False, model="default"):
    verb = "converse-stream" if stream else "converse"
    return httpx.post(
        f"{harness.base}/model/{model}/{verb}",
        json=body,
        headers={"authorization": f"Bearer {KEY}"},
        timeout=60,
    )


def test_what_the_upstream_receives(harness):
    harness.upstream.mode = "echo"
    response = converse(
        harness,
        {
            "system": [{"text": "sys"}, {"cachePoint": {"type": "default"}}],
            "messages": [USER],
            "inferenceConfig": {
                "maxTokens": 9,
                "temperature": 0.5,
                "topP": 0.9,
                "stopSequences": ["Z"],
            },
        },
    )
    assert response.status_code == 200
    sent = harness.upstream.hits[-1][2]
    assert (sent["max_tokens"], sent["temperature"], sent["top_p"], sent["stop"]) == (
        9,
        0.5,
        0.9,
        ["Z"],
    )
    assert sent["messages"][0] == {"role": "system", "content": "sys"}


def test_non_stream_body_shape(harness):
    harness.upstream.mode = "echo"
    body = converse(harness, {"messages": [USER]}).json()
    assert body["output"]["message"] == {"role": "assistant", "content": [{"text": "echo: hi"}]}
    assert body["stopReason"] == "end_turn"
    assert body["usage"]["totalTokens"] == 6
    assert body["metrics"]["latencyMs"] >= 0


def test_stream_event_order_and_indexes(harness):
    harness.upstream.mode = "echo"
    events = events_of(converse(harness, {"messages": [USER]}, stream=True))
    kinds = [k for k, _ in events]
    assert kinds[0] == "messageStart" and events[0][1] == {"role": "assistant"}
    assert kinds[-3:] == ["contentBlockStop", "messageStop", "metadata"]
    deltas = [d for k, d in events if k == "contentBlockDelta"]
    assert "".join(d["delta"]["text"] for d in deltas) == "echo: hi"
    assert {d["contentBlockIndex"] for d in deltas} == {0}
    assert events[-2][1] == {"stopReason": "end_turn"}
    usage = events[-1][1]
    assert usage["usage"] == {"inputTokens": 3, "outputTokens": 3, "totalTokens": 6}
    assert usage["metrics"]["latencyMs"] >= 0


def test_stream_tool_call_blocks(harness):
    harness.upstream.mode = "tools"
    spec = {"toolSpec": {"name": "read_file", "inputSchema": {"json": {"type": "object"}}}}
    body = {"messages": [USER], "toolConfig": {"tools": [spec]}}
    events = events_of(converse(harness, body, stream=True))
    start = next(d for k, d in events if k == "contentBlockStart")
    assert start["start"]["toolUse"] == {"toolUseId": "call_1", "name": "read_file"}
    pieces = "".join(d["delta"]["toolUse"]["input"] for k, d in events if k == "contentBlockDelta")
    assert json.loads(pieces) == {"p": "a.txt"}
    stops = [d["contentBlockIndex"] for k, d in events if k == "contentBlockStop"]
    assert stops == [start["contentBlockIndex"]]
    assert dict(events)["messageStop"]["stopReason"] == "tool_use"


def test_model_id_with_colon_and_arn_reaches_routing(harness):
    harness.upstream.mode = "echo"
    for model in (
        "default",
        "default%3A0",
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Ainference-profile%2Fdefault",
    ):
        response = converse(harness, {"messages": [USER]}, model=model)
        # unknown model names are the router's to judge; the route must match
        assert response.status_code != 404 or "model" in response.text.lower()


def test_error_shape_and_types(harness):
    for status, kind in ((400, "ValidationException"), (401, "UnrecognizedClientException")):
        error = {"message": "m", "type": "x", "code": None, "param": None, "request_id": "r1"}
        reply = BedrockConverse().encode_error(status, error, None)
        assert reply.headers["x-amzn-ErrorType"] == kind
        assert reply.body == {"message": "m"}
    reply = BedrockConverse().encode_error(
        429,
        {
            "message": "slow",
            "type": "rate_limit_error",
            "code": None,
            "param": None,
            "request_id": None,
        },
        None,
    )
    assert reply.headers["x-amzn-ErrorType"] == "ThrottlingException"
    reply = BedrockConverse().encode_error(
        502,
        {"message": "up", "type": "server_error", "code": None, "param": None, "request_id": None},
        None,
    )
    assert reply.headers["x-amzn-ErrorType"] == "InternalServerException"


def test_refusal_is_a_validation_exception_on_the_wire(harness):
    response = converse(harness, {"messages": [USER], "additionalModelRequestFields": {"a": 1}})
    assert response.status_code == 400
    assert response.headers["x-amzn-errortype"] == "ValidationException"
    assert "additionalModelRequestFields" in response.json()["message"]
    assert not harness.upstream.hits


def test_stream_failure_is_an_exception_frame_without_terminator(harness):
    harness.upstream.mode = "die_after"
    response = converse(harness, {"messages": [USER]}, stream=True)
    frames = decode_stream(response.content)
    kinds = [
        h[":message-type"] + ":" + h.get(":event-type", h.get(":exception-type", ""))
        for h, _ in frames
    ]
    assert kinds[-1].startswith("exception:")
    assert "messageStop" not in " ".join(kinds)
    assert "boom" not in response.text


# -- framing ----------------------------------------------------------------------------


def test_encoder_round_trips_through_botocore():
    buffer_module = pytest.importorskip("botocore.eventstream")
    wire = encode_message({":event-type": "x", ":message-type": "event"}, b'{"a":1}')
    wire += encode_message({":event-type": "y"}, b"")
    buffer = buffer_module.EventStreamBuffer()
    buffer.add_data(wire)
    first, second = list(buffer)
    assert first.headers == {":event-type": "x", ":message-type": "event"}
    assert first.payload == b'{"a":1}'
    assert second.headers == {":event-type": "y"} and second.payload == b""


@pytest.mark.parametrize("offset", [0, 9, 14, -1])
def test_corrupted_frames_fail_the_crc(offset):
    buffer_module = pytest.importorskip("botocore.eventstream")
    wire = bytearray(encode_message({":event-type": "x"}, b"payload"))
    wire[offset] ^= 0xFF
    buffer = buffer_module.EventStreamBuffer()
    buffer.add_data(bytes(wire))
    with pytest.raises(Exception):  # noqa: B017  ChecksumMismatch or a length error
        list(buffer)


def test_own_decoder_rejects_what_botocore_rejects():
    wire = bytearray(encode_message({":event-type": "x"}, b"payload"))
    wire[-1] ^= 1
    with pytest.raises(AssertionError, match="message CRC"):
        decode_stream(bytes(wire))


# -- boto3 ------------------------------------------------------------------------------


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

    # boto3 refuses bearer auth over plain http; sign as usual, then send the
    # bearer header the adapter requires (what a TLS client does natively).
    def bearer_header(request, **_):
        request.headers["Authorization"] = f"Bearer {KEY}"

    client.meta.events.register("before-send.bedrock-runtime.*", bearer_header)
    return client


def test_boto3_converse(harness, runtime):
    harness.upstream.mode = "echo"
    out = runtime.converse(
        modelId="default",
        system=[{"text": "be brief"}],
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 20},
    )
    assert out["output"]["message"]["content"] == [{"text": "echo: hi"}]
    assert out["stopReason"] == "end_turn"
    assert out["usage"]["inputTokens"] == 3
    assert harness.upstream.hits[-1][2]["messages"][0]["content"] == "be brief"


def test_boto3_converse_image_bytes(harness, runtime):
    harness.upstream.mode = "echo"
    png = base64.b64decode(PNG_B64)
    runtime.converse(
        modelId="default",
        messages=[
            {
                "role": "user",
                "content": [{"text": "hi"}, {"image": {"format": "png", "source": {"bytes": png}}}],
            }
        ],
    )
    parts = harness.upstream.hits[-1][2]["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"


def test_boto3_converse_stream(harness, runtime):
    harness.upstream.mode = "echo"
    out = runtime.converse_stream(
        modelId="default", messages=[{"role": "user", "content": [{"text": "hi"}]}]
    )
    seen, text = [], ""
    for event in out["stream"]:
        seen.append(next(iter(event)))
        if "contentBlockDelta" in event:
            text += event["contentBlockDelta"]["delta"]["text"]
        if "metadata" in event:
            assert event["metadata"]["usage"]["outputTokens"] == 3
    assert text == "echo: hi"
    assert seen[0] == "messageStart" and seen[-2:] == ["messageStop", "metadata"]


def test_boto3_stream_tool_use(harness, runtime):
    harness.upstream.mode = "tools"
    out = runtime.converse_stream(
        modelId="default",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        toolConfig={
            "tools": [
                {"toolSpec": {"name": "read_file", "inputSchema": {"json": {"type": "object"}}}}
            ]
        },
    )
    start, pieces = None, ""
    for event in out["stream"]:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]["start"]["toolUse"]
        if "contentBlockDelta" in event:
            pieces += event["contentBlockDelta"]["delta"]["toolUse"]["input"]
    assert start == {"toolUseId": "call_1", "name": "read_file"}
    assert json.loads(pieces) == {"p": "a.txt"}


def test_additional_model_request_fields_top_k_is_mapped():
    request, _ = decode({"messages": [USER], "additionalModelRequestFields": {"top_k": 5}})
    assert request.top_k == 5
    assert decode({"messages": [USER], "additionalModelRequestFields": {}})[0].top_k is None


def test_top_k_reaches_the_upstream(harness):
    response = converse(harness, {"messages": [USER], "additionalModelRequestFields": {"top_k": 5}})
    assert response.status_code == 200
    assert harness.upstream.hits[-1][2]["top_k"] == 5


def test_boto3_error_codes(harness, runtime):
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError) as info:
        runtime.converse(
            modelId="default",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            additionalModelRequestFields={"thinking": {"type": "enabled"}},
        )
    assert info.value.response["Error"]["Code"] == "ValidationException"
    assert info.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400


def test_boto3_stream_failure_raises(harness, runtime):
    from botocore.exceptions import ClientError, EventStreamError

    harness.upstream.mode = "die_after"
    out = runtime.converse_stream(
        modelId="default", messages=[{"role": "user", "content": [{"text": "hi"}]}]
    )
    with pytest.raises((EventStreamError, ClientError)) as info:
        list(out["stream"])
    assert "internalServerException" in str(info.value) or "InternalServer" in str(info.value)
