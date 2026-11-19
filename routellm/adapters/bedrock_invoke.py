"""AWS Bedrock InvokeModel dialect, Anthropic bodies: `POST /model/{modelId}/invoke`,
`/invoke-with-response-stream` and `/count-tokens`.

Claude Code in Bedrock mode (`CLAUDE_CODE_USE_BEDROCK=1`) talks this, not
Converse: set `ANTHROPIC_BEDROCK_BASE_URL` to the server and
`AWS_BEARER_TOKEN_BEDROCK` to a routellm key. The same goes for the
`anthropic` SDK's Bedrock client and boto3 `invoke_model`,
`invoke_model_with_response_stream` and `count_tokens` with an Anthropic
Messages body.

The body is an Anthropic Messages request without `model` (the model id is
in the path and drives routing exactly as on Converse) and without `stream`
(the route says it). It is handed to the Anthropic adapter, whose
validation, translation and answers are used as they are; see its module
for the fields mapped, ignored and refused. What differs here:

- `anthropic_version` (`bedrock-2023-05-31`): accepted, any string, ignored.
- `anthropic_beta` (list of strings): accepted and ignored, as the Anthropic
  adapter does the `anthropic-beta` header; a value that is not a list of
  strings is refused.
- `model` or `stream` in the body: refused (Bedrock rejects both).

Routes
======

`/invoke`: the Anthropic message as JSON, plus the
`x-amzn-bedrock-input-token-count` and `x-amzn-bedrock-output-token-count`
headers.

`/invoke-with-response-stream`: `application/vnd.amazon.eventstream`. Each
Anthropic stream event (`message_start`, `ping`, `content_block_*`,
`message_delta`, `message_stop`) is one frame, headers `:event-type` `chunk`,
`:message-type` `event`, `:content-type` `application/json`, payload
`{"bytes": "<base64 of the event's JSON>"}`. A failure after the stream
began ends it with an `internalServerException` frame and no
`message_stop`.

`/count-tokens`: `{"input": {"invokeModel": {"body": <blob>}}}` where the
blob is the Anthropic body: base64 of its JSON as boto3 and the REST API
send it, or (leniently) the JSON text or object itself. Answer
`{"inputTokens": n}`, an ESTIMATE from the Anthropic adapter's counter
(`max_tokens` in the body is ignored). The `converse` input form is refused.

Auth and errors are the Converse adapter's: bearer only by default, the
`ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4` opt-in, no credentials from the
URL; errors are `{"message": ...}` with `x-amzn-ErrorType`.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Reply,
    Route,
    eventstream,
    refuse,
    register,
)
from routellm.adapters.anthropic import AnthropicAdapter, message_events
from routellm.adapters.bedrock import BedrockConverse

_ANTHROPIC = AnthropicAdapter()
_CONVERSE = BedrockConverse()
_STREAM_SUFFIX = "/invoke-with-response-stream"


def _anthropic_body(body: Any, model_id: str, *, where: str = "request") -> dict:
    """The Anthropic Messages body of a Bedrock one: model from the path, wrapper keys dropped."""
    if not isinstance(body, dict):
        refuse(f"{where} body must be a JSON object")
    for key in ("model", "stream"):
        if key in body:
            refuse(
                f"{key} does not belong in a Bedrock InvokeModel body",
                param=key,
            )
    beta = body.get("anthropic_beta")
    if beta is not None and not (
        isinstance(beta, list) and all(isinstance(item, str) for item in beta)
    ):
        refuse("anthropic_beta must be a list of strings", param="anthropic_beta")
    version = body.get("anthropic_version")
    if version is not None and not isinstance(version, str):
        refuse("anthropic_version must be a string", param="anthropic_version")
    wire = {k: v for k, v in body.items() if k not in ("anthropic_version", "anthropic_beta")}
    wire["model"] = model_id
    return wire


def _blob(value: Any) -> Any:
    """The JSON a CountTokens `body` blob carries: base64 text, JSON text or an object."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        refuse("input.invokeModel.body is required", param="input.invokeModel.body")
    text = value.lstrip()
    if not text.startswith("{"):
        try:
            text = base64.b64decode(value, validate=True).decode()
        except (binascii.Error, UnicodeDecodeError):
            refuse(
                "input.invokeModel.body must be base64 of a JSON object",
                param="input.invokeModel.body",
            )
    try:
        return json.loads(text)
    except ValueError:
        refuse("input.invokeModel.body is not valid JSON", param="input.invokeModel.body")


@register
class BedrockInvoke(Adapter):
    name = "bedrock-invoke"
    # `:path` so a model id that is an ARN (`inference-profile/...`) matches.
    routes = (
        Route("POST", "/model/{model_id:path}/invoke"),
        Route("POST", "/model/{model_id:path}" + _STREAM_SUFFIX),
        Route("POST", "/model/{model_id:path}/count-tokens", operation="count_tokens"),
    )

    def credential(self, headers: Mapping[str, str], query: Mapping[str, str]) -> str | None:
        return _CONVERSE.credential(headers, query)

    # -- request ---------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        call.body = _anthropic_body(call.body, call.path_params["model_id"])
        request = _ANTHROPIC.decode(call)
        streaming = call.route.path.endswith(_STREAM_SUFFIX)
        call.state["stream"] = streaming
        return request.model_copy(update={"stream": True}) if streaming else request

    async def handle(self, call: Call) -> Any:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        unknown = sorted(set(body) - {"input"})
        if unknown:
            refuse(
                f"Unsupported field(s) in CountTokens request: {', '.join(unknown)}",
                param=unknown[0],
            )
        source = body.get("input")
        if not isinstance(source, dict) or set(source) != {"invokeModel"}:
            refuse("CountTokens input must be {invokeModel: {body}}", param="input")
        invoke = source["invokeModel"]
        if not isinstance(invoke, dict) or set(invoke) != {"body"}:
            refuse("input.invokeModel must be {body}", param="input.invokeModel")
        counted = _anthropic_body(
            _blob(invoke["body"]), call.path_params["model_id"], where="input.invokeModel.body"
        )
        counted.pop("max_tokens", None)
        call.body = counted
        answer = await _ANTHROPIC.handle(call)
        return {"inputTokens": answer["input_tokens"]}

    # -- response --------------------------------------------------------------

    def encode(self, response: dict, call: Call) -> Reply:
        message = _ANTHROPIC.encode(response, call)
        usage = message.get("usage") or {}
        headers = {
            "x-amzn-bedrock-input-token-count": str(usage.get("input_tokens", 0)),
            "x-amzn-bedrock-output-token-count": str(usage.get("output_tokens", 0)),
        }
        return Reply(message, headers=headers)

    def stream_media_type(self, call: Call) -> str:
        return eventstream.MEDIA_TYPE

    async def encode_stream(self, chunks: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        async for event in message_events(chunks, call):
            raw = json.dumps(event, separators=(",", ":")).encode()
            yield eventstream.event("chunk", {"bytes": base64.b64encode(raw).decode()})

    # -- errors ----------------------------------------------------------------

    def encode_error(self, status: int, error: dict, call: Call) -> Reply:
        return _CONVERSE.encode_error(status, error, call)

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return _CONVERSE.encode_stream_error(error, call)
