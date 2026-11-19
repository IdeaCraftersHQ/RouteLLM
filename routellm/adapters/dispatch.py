"""Run one adapter request through the chat pipeline, in process.

Order, which mirrors `/v1/chat/completions`: credential, authenticate,
loop refusal, then the body is read and decoded, then the shared
`run_chat_completion` (request class, routing, transforms, cache,
resilience, audit trail). Every refusal and failure leaves as the
dialect's own error shape. Nothing here goes over HTTP.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from routellm.adapters.base import CHAT, Adapter, AdapterRefusal, Call, Reply, Route
from routellm.audit_http import Trail, settle
from routellm.auth import ClientNotListed
from routellm.stream_guard import StreamFrame, StreamGuard

logger = logging.getLogger(__name__)

#: Carries the routing decision on every adapter response, streamed or not.
PATH_HEADER = "x-routellm-path"

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_HEADER_VALUE = re.compile(r"[\x20-\x7e]*")

_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def error_info(status: int, body: Any = None, **fields: Any) -> dict:
    """The normalized error every `encode_error` receives.

    Reads the OpenAI shapes the pipeline produces (`{"object": "error",
    "message": ...}`, or `{"error": {...}}`) and fills the keys an adapter
    can rely on: `message`, `type`, `code`, `param`, `request_id`.
    """
    source = dict(body) if isinstance(body, dict) else {}
    if isinstance(source.get("error"), dict):
        source = {**source, **source["error"]}
    info = {
        "message": source.get("message") or "Request failed",
        "type": source.get("type") or _TYPES.get(status, "server_error"),
        "code": source.get("code"),
        "param": source.get("param"),
        "request_id": source.get("request_id"),
    }
    return {**info, **fields}


def _reply(reply: Any, status: int, extra: dict[str, str] | None = None) -> Response:
    if not isinstance(reply, Reply):
        reply = Reply(reply)
    headers = {**(extra or {}), **reply.headers}
    code = reply.status or status
    if reply.media_type == "application/json":
        return JSONResponse(reply.body, status_code=code, headers=headers)
    return Response(reply.body, status_code=code, headers=headers, media_type=reply.media_type)


def _refusal(
    adapter: Adapter,
    call: Call,
    status: int,
    error: dict,
    reason: str,
    *,
    outcome: str = "refused",
    headers: dict[str, str] | None = None,
) -> Response:
    """Answer in the dialect's error shape and write the audit event."""
    response = _reply(adapter.encode_error(status, error, call), status, headers)
    trail = None
    if call.route.operation == CHAT:
        trail = Trail.start(call.pipeline.AUDIT, call.http_request, None, "chat")
    return settle(trail, response, outcome, status, reason)


async def serve(adapter: Adapter, route: Route, http_request: Request, pipeline: Any) -> Response:
    """Answer one request to one of an adapter's routes."""
    call = Call(adapter=adapter, route=route, http_request=http_request, pipeline=pipeline)
    chat = route.operation == CHAT

    try:
        pipeline.authenticate(
            http_request, adapter.credential(http_request.headers, http_request.query_params)
        )
        if chat:
            pipeline.refuse_loops(http_request)
    except pipeline.Unauthorized:
        error = error_info(401, {"message": "Missing or invalid API key", "code": "unauthorized"})
        return _refusal(adapter, call, 401, error, "unauthorized")
    except ClientNotListed as exc:
        message = f"Header {exc.header} names a client not listed for this key"
        error = error_info(403, {"message": message, "code": "client_not_listed"})
        return _refusal(adapter, call, 403, error, "client_not_listed")
    except pipeline.LoopDetected:
        message = "Loop detected: request already carries the routing hop header"
        error = error_info(508, {"message": message, "code": "loop_detected"})
        return _refusal(adapter, call, 508, error, "loop_detected")

    try:
        if route.body == "json":
            try:
                call.body = json.loads(await http_request.body())
            except ValueError:
                raise AdapterRefusal(
                    "Request body is not valid JSON", code="invalid_json"
                ) from None
        if not chat:
            result = await adapter.handle(call)
            return _reply(result, 200)
        call.request = adapter.decode(call)
    except AdapterRefusal as exc:
        error = error_info(
            exc.status, {"message": exc.message, "code": exc.code, "param": exc.param}
        )
        return _refusal(adapter, call, exc.status, error, exc.code)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        param = ".".join(str(p) for p in first.get("loc", ())) or None
        error = error_info(
            400, {"message": f"Invalid request: {first.get('msg', 'invalid')}", "param": param}
        )
        return _refusal(adapter, call, 400, error, "invalid_request")
    except Exception as exc:
        # A client body the adapter could not read. The type is logged,
        # never the body: prompts are the caller's data.
        logger.error(f"adapter {adapter.name} could not decode a request: {type(exc).__name__}")
        error = error_info(400, {"message": "Malformed request body", "code": "malformed"})
        return _refusal(adapter, call, 400, error, "malformed")

    request = call.request
    if request.stream:
        # Usage comes back in the last chunk only when asked for, and
        # every dialect reports it.
        options = dict(request.stream_options or {})
        options.setdefault("include_usage", True)
        call.request = request = request.model_copy(update={"stream_options": options})

    extra = {}
    if request.stream:
        try:
            extra = _stream_headers(adapter, call)
        except Exception as exc:
            logger.error(f"adapter {adapter.name} returned invalid stream headers: {exc}")
            error = error_info(500, {"message": "Internal server error"})
            return _refusal(adapter, call, 500, error, "invalid_stream_headers", outcome="failed")

    response = await pipeline.run_chat_completion(
        request, http_request, framing=_framing(call, extra)
    )
    return _reshape(adapter, call, response)


def _stream_headers(adapter: Adapter, call: Call) -> dict[str, str]:
    """The adapter's extra stream headers, checked so none can inject another.

    Raises `ValueError` naming the offending header, never its value.
    """
    headers = {}
    for name, value in (adapter.stream_headers(call) or {}).items():
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
            raise ValueError("invalid stream header name")
        if not isinstance(value, str) or not _HEADER_VALUE.fullmatch(value):
            raise ValueError(f"invalid value for stream header {name}")
        if name.lower() != PATH_HEADER:
            headers[name] = value
    return headers


def _framing(call: Call, extra: dict[str, str] | None = None):
    adapter = call.adapter

    def framing(res, request_id, done, path) -> StreamFrame:
        headers = dict(extra or {})
        if path:
            headers[PATH_HEADER] = json.dumps(path, separators=(",", ":"))
        content = _stream(adapter, call, res, request_id, done)
        return StreamFrame(content, adapter.stream_media_type(call), headers)

    return framing


async def _stream(adapter: Adapter, call: Call, res, request_id: str, done):
    """The adapter's frames, with the bookkeeping `/v1/chat/completions` has.

    Same `StreamGuard` as the SSE route: usage is read from the chunks, a
    caller that goes away still closes the upstream, a failure after the
    first chunk logs the cause server-side and sends the generic
    `encode_stream_error` frame, and the stream ends with no clean
    terminator.
    """
    guard = StreamGuard(res, request_id, done)

    async def chunks():
        async for chunk in res:
            guard.saw(chunk)
            yield chunk

    try:
        async for frame in adapter.encode_stream(chunks(), call):
            yield frame
        guard.ok()
    except Exception as exc:
        guard.failed(exc)
        error = error_info(
            502,
            {
                "message": f"Upstream request failed (request id {request_id})",
                "type": "server_error",
                "request_id": request_id,
            },
        )
        frame = adapter.encode_stream_error(error, call)
        if frame:
            yield frame
    finally:
        await guard.finish()


def _reshape(adapter: Adapter, call: Call, response: Response) -> Response:
    """Re-encode the pipeline's OpenAI response in the dialect's shape."""
    if isinstance(response, StreamingResponse) or not isinstance(response, JSONResponse):
        return response
    body = json.loads(response.body)
    status = response.status_code
    extra = {}
    if "retry-after" in response.headers:
        extra["Retry-After"] = response.headers["retry-after"]
    if status != 200:
        return _reply(adapter.encode_error(status, error_info(status, body), call), status, extra)
    path = (body.get("routellm") or {}).get("path")
    if path:
        extra[PATH_HEADER] = json.dumps(path, separators=(",", ":"))
    return _reply(adapter.encode(body, call), 200, extra)
