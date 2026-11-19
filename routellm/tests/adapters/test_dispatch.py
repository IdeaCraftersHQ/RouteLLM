"""The framework's own behaviour, against a stub pipeline (no server, no upstream).

What is under test is what every adapter relies on: the credential reaches
the authenticator, refusals and failures leave in the dialect's shape, the
stream is guarded like the SSE route, and nothing unsupported reaches the
pipeline.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from routellm.adapters import (
    Adapter,
    ChatCompletionRequest,
    Reply,
    Route,
    mount_adapters,
    refuse,
    refuse_unknown,
)
from routellm.adapters.stream import End, TextDelta


class Unauthorized(Exception):
    pass


class LoopDetected(Exception):
    pass


class Source:
    """An upstream chunk stream that can fail, and records being closed."""

    def __init__(self, chunks, fail=False):
        self.chunks, self.fail, self.closed = chunks, fail, False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for c in self.chunks:
            yield c
        if self.fail:
            raise RuntimeError("secret upstream detail")

    async def aclose(self):
        self.closed = True


class Chunk(dict):
    usage = None

    def model_dump(self):
        return dict(self)


class Pipeline:
    CONTROLLER = None
    AUDIT = None
    Unauthorized = Unauthorized
    LoopDetected = LoopDetected

    def __init__(self):
        self.tokens, self.requests, self.done, self.source = [], [], [], None
        self.result = None  # a JSONResponse to return instead of running

    def authenticate(self, http_request, token):
        self.tokens.append(token)
        if token != "good":
            raise Unauthorized
        http_request.state.principal = "p"

    def refuse_loops(self, http_request):
        if "x-routellm-hop" in http_request.headers:
            raise LoopDetected

    async def run_chat_completion(self, request, http_request, framing=None):
        self.requests.append(request)
        if self.result is not None:
            return self.result
        if request.stream:
            frame = framing(self.source, "rid", lambda *a: self.done.append(a), [{"e": "a"}])
            return StreamingResponse(
                frame.content, media_type=frame.media_type, headers=frame.headers
            )
        body = {"id": "c1", "choices": [], "routellm": {"path": [{"e": "a"}]}}
        return JSONResponse(body)

    async def list_models(self):
        return JSONResponse({"data": [{"id": "default"}]})


class Dialect(Adapter):
    name = "unit"
    routes = (
        Route("POST", "/unit/chat"),
        Route("POST", "/unit/count", operation="count"),
    )

    def credential(self, headers, query):
        return headers.get("x-api-key")

    def decode(self, call):
        body = call.body
        if body.get("explode"):
            return body["missing"]["key"]  # a client body the adapter cannot read
        refuse_unknown(
            body, {"model", "messages", "stream", "stream_options", "explode", "stream_headers"}
        )
        if body.get("messages") == "bad":
            ChatCompletionRequest(model=1, messages=[])  # raises ValidationError
        if body.get("messages") == "nope":
            refuse("no such thing", param="messages")
        call.state["stream_headers"] = body.get("stream_headers")
        return ChatCompletionRequest(
            **{k: v for k, v in body.items() if k not in {"explode", "stream_headers"}}
        )

    async def handle(self, call):
        if call.body.get("headers"):
            return Reply({"n": 1}, headers={"x-extra": "1"}, status=202)
        return {"tokens": len(json.dumps(call.body))}

    def encode(self, response, call):
        return {"wrapped": response["id"], "path": response["routellm"]["path"]}

    def stream_media_type(self, call):
        return "application/x-unit"

    def stream_headers(self, call):
        return call.state.get("stream_headers") or {}

    async def encode_events(self, events, call):
        async for event in events:
            if isinstance(event, TextDelta):
                yield event.text.encode()
            elif isinstance(event, End):
                yield b"|end"

    def encode_error(self, status, error, call):
        return {"err": error["message"], "type": error["type"], "param": error["param"]}

    def encode_stream_error(self, error, call):
        return b"|error:" + error["message"].encode()


@pytest.fixture
def world():
    pipeline = Pipeline()
    app = FastAPI()
    mount_adapters(app, pipeline, [Dialect()])
    return TestClient(app), pipeline


GOOD = {"x-api-key": "good"}
CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def text_chunk(text, **kw):
    return Chunk(id="c", choices=[{"index": 0, "delta": {"content": text}}], **kw)


def test_the_dialects_credential_is_what_reaches_the_authenticator(world):
    client, pipeline = world
    client.post("/unit/chat", json=CHAT, headers={"authorization": "Bearer good", **GOOD})
    assert pipeline.tokens == ["good"]
    client.post("/unit/chat", json=CHAT, headers={"authorization": "Bearer good"})
    assert pipeline.tokens[-1] is None


def test_unauthorized_leaves_in_the_dialect_shape_and_runs_nothing(world):
    client, pipeline = world
    r = client.post("/unit/chat", json=CHAT)
    assert r.status_code == 401
    assert r.json() == {"err": "Missing or invalid API key", "type": "authentication_error",
                        "param": None}  # fmt: skip
    assert pipeline.requests == []


def test_a_loop_is_508_in_the_dialect_shape(world):
    client, pipeline = world
    r = client.post("/unit/chat", json=CHAT, headers={**GOOD, "x-routellm-hop": "1"})
    assert r.status_code == 508 and "Loop detected" in r.json()["err"]
    assert pipeline.requests == []


def test_a_body_that_is_not_json_is_a_dialect_400(world):
    client, pipeline = world
    r = client.post("/unit/chat", content=b"{nope", headers=GOOD)
    assert r.status_code == 400 and r.json()["err"] == "Request body is not valid JSON"
    assert pipeline.requests == []


def test_an_unsupported_field_is_refused_and_named_never_dropped(world):
    client, pipeline = world
    r = client.post("/unit/chat", json={**CHAT, "safety": 1}, headers=GOOD)
    assert r.status_code == 400
    assert "safety" in r.json()["err"] and r.json()["param"] == "safety"
    assert pipeline.requests == []


def test_a_refusal_raised_by_the_adapter_is_a_400(world):
    client, _ = world
    r = client.post("/unit/chat", json={**CHAT, "messages": "nope"}, headers=GOOD)
    assert (r.status_code, r.json()["err"], r.json()["param"]) == (400, "no such thing", "messages")


def test_a_validation_error_in_decode_is_a_400_naming_the_field(world):
    client, pipeline = world
    r = client.post("/unit/chat", json={**CHAT, "messages": "bad"}, headers=GOOD)
    assert r.status_code == 400 and r.json()["param"] == "model"
    assert pipeline.requests == []


def test_a_body_the_adapter_cannot_read_is_a_generic_400(world):
    client, pipeline = world
    r = client.post("/unit/chat", json={**CHAT, "explode": 1}, headers=GOOD)
    assert r.status_code == 400 and r.json()["err"] == "Malformed request body"
    assert pipeline.requests == []


def test_a_completion_is_encoded_and_carries_the_decision_path_header(world):
    client, pipeline = world
    r = client.post("/unit/chat", json=CHAT, headers=GOOD)
    assert r.status_code == 200 and r.json() == {"wrapped": "c1", "path": [{"e": "a"}]}
    assert json.loads(r.headers["x-routellm-path"]) == [{"e": "a"}]
    assert pipeline.requests[0].model == "m"


@pytest.mark.parametrize(
    ("status", "type_"), [(400, "invalid_request_error"), (403, "permission_error"),
                          (429, "rate_limit_error"), (502, "server_error")]
)  # fmt: skip
def test_pipeline_errors_are_re_encoded_with_a_type_the_dialect_can_map(world, status, type_):
    client, pipeline = world
    pipeline.result = JSONResponse({"object": "error", "message": "m"}, status_code=status)
    r = client.post("/unit/chat", json=CHAT, headers=GOOD)
    assert r.status_code == status
    assert r.json() == {"err": "m", "type": type_, "param": None}


def test_retry_after_survives_re_encoding(world):
    client, pipeline = world
    pipeline.result = JSONResponse(
        {"object": "error", "message": "cooling"}, status_code=503, headers={"Retry-After": "7"}
    )
    r = client.post("/unit/chat", json=CHAT, headers=GOOD)
    assert r.status_code == 503 and r.headers["retry-after"] == "7"


def test_a_stream_is_framed_by_the_dialect_and_asks_for_usage(world):
    client, pipeline = world
    pipeline.source = Source([text_chunk("he"), text_chunk("llo")])
    r = client.post("/unit/chat", json={**CHAT, "stream": True}, headers=GOOD)
    assert r.text == "hello|end"
    assert r.headers["content-type"].startswith("application/x-unit")
    assert json.loads(r.headers["x-routellm-path"]) == [{"e": "a"}]
    assert pipeline.requests[0].stream_options == {"include_usage": True}
    assert pipeline.done == [("served", None, None, None)]
    assert pipeline.source.closed


def test_a_caller_stream_option_is_kept_beside_include_usage(world):
    client, pipeline = world
    pipeline.source = Source([])
    body = {**CHAT, "stream": True, "stream_options": {"x": 1}}
    client.post("/unit/chat", json=body, headers=GOOD)
    assert pipeline.requests[0].stream_options == {"x": 1, "include_usage": True}


def test_a_stream_that_fails_sends_the_generic_frame_and_no_terminator(world):
    client, pipeline = world
    pipeline.source = Source([text_chunk("he")], fail=True)
    r = client.post("/unit/chat", json={**CHAT, "stream": True}, headers=GOOD)
    assert r.text.startswith("he|error:Upstream request failed (request id rid)")
    assert "|end" not in r.text and "secret" not in r.text
    assert pipeline.done[0][:2] == ("failed", "stream_error")
    assert pipeline.source.closed


def test_a_custom_route_gets_auth_and_the_parsed_body(world):
    client, _ = world
    assert client.post("/unit/count", json={"a": 1}).status_code == 401
    r = client.post("/unit/count", json={"a": 1}, headers=GOOD)
    assert r.json() == {"tokens": len('{"a": 1}')}


def test_a_reply_carries_its_own_status_and_headers(world):
    client, _ = world
    r = client.post("/unit/count", json={"headers": 1}, headers=GOOD)
    assert (r.status_code, r.headers["x-extra"], r.json()) == (202, "1", {"n": 1})


def test_call_models_reads_the_servers_model_list():
    import asyncio

    from routellm.adapters import Call

    call = Call(Dialect(), Route("GET", "/x"), http_request=None, pipeline=Pipeline())  # type: ignore[arg-type]
    assert asyncio.run(call.models()) == [{"id": "default"}]


def test_stream_headers_reach_the_streaming_response(world):
    client, pipeline = world
    pipeline.source = Source([text_chunk("hi")])
    body = {**CHAT, "stream": True, "stream_headers": {"x-ignored": "a,b"}}
    r = client.post("/unit/chat", json=body, headers=GOOD)
    assert r.status_code == 200 and r.text == "hi|end"
    assert r.headers["x-ignored"] == "a,b"
    assert json.loads(r.headers["x-routellm-path"]) == [{"e": "a"}]


def test_the_decision_path_header_wins_over_a_stream_header(world):
    client, pipeline = world
    pipeline.source = Source([text_chunk("hi")])
    body = {**CHAT, "stream": True, "stream_headers": {"X-Routellm-Path": "forged"}}
    r = client.post("/unit/chat", json=body, headers=GOOD)
    assert json.loads(r.headers["x-routellm-path"]) == [{"e": "a"}]


@pytest.mark.parametrize(
    "headers",
    [
        {"x-a": "b\r\nx-evil: 1"},
        {"x-a": "b\nc"},
        {"x-a\r\nx-evil": "1"},
        {"bad name": "1"},
        {"": "1"},
        {"x-é": "1"},
        {"x-a": "caf\u00e9"},
    ],
)
def test_invalid_stream_headers_are_a_server_error_never_sent(world, headers, caplog):
    client, pipeline = world
    pipeline.source = Source([text_chunk("hi")])
    body = {**CHAT, "stream": True, "stream_headers": headers}
    r = client.post("/unit/chat", json=body, headers=GOOD)
    assert r.status_code == 500
    assert "x-evil" not in r.headers and pipeline.requests == []
    assert "evil" not in caplog.text and "caf" not in caplog.text


def test_stream_headers_are_not_asked_of_a_non_streamed_request(world):
    client, _ = world
    body = {**CHAT, "stream_headers": {"bad name": "1"}}
    assert client.post("/unit/chat", json=body, headers=GOOD).status_code == 200
