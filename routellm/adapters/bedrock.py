"""AWS Bedrock Converse dialect: `POST /model/{modelId}/converse` and `/converse-stream`.

For boto3 `bedrock-runtime` clients (`converse`, `converse_stream`) and any
SDK that speaks Converse. The mapping inverts litellm's outbound Bedrock
Converse transformation (`litellm.llms.bedrock.chat.converse_transformation`
and the `AWSEventStreamDecoder` of `invoke_handler`): litellm has no inbound
translator. `modelId` is the model field, so routing parameters ride in it
as on `/v1/chat/completions`.

Claude Code in Bedrock mode does not use Converse: it sends InvokeModel with
an Anthropic Messages body, served by `bedrock_invoke`.

Auth: bearer only by default. Send `Authorization: Bearer <token>` (the
`AWS_BEARER_TOKEN_BEDROCK` value, which boto3 uses over TLS); the token is
looked up as an ordinary key in the keys file. A request signed with SigV4
and carrying no bearer token is a 401 `UnrecognizedClientException`: this
server stores only hashes of keys, so it cannot recompute a signature, and
an access key id is not a secret (it shows up in logs, URLs and error
messages), so trusting it would let anyone who has seen one impersonate its
principal.

Operator opt-in, off by default: set `ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4`
to `1`, `true` or `yes` (read on every request) and the access key id of a
SigV4 `Authorization` header is looked up as the key (list
`sha256(<access key id>)`). The signature is still NOT verified. Use it only
on a network where every caller is trusted. Credentials are never read from
the URL: a presigned `X-Amz-Credential` query parameter is ignored in both
modes.

Mapped
    system[].text                       one system message
    messages[].content[].text           text part
    ...image {format, source.bytes}     `image_url` data URL (png, jpeg, gif, webp)
    ...document {format, name, bytes}   `file` part (pdf csv doc docx xls xlsx html txt md)
    ...toolUse (assistant)              assistant `tool_calls`
    ...toolResult (user)                `role: tool` message, before the turn's other blocks
    toolConfig.tools[].toolSpec         function tools (`strict` carried)
    toolConfig.toolChoice               auto -> auto, any -> required, tool -> that function
    inferenceConfig                     maxTokens, temperature, topP, stopSequences
    requestMetadata                     `metadata` (`data_class`, `operation`)
    route (`converse-stream`)           `stream`
    additionalModelRequestFields        `top_k` when it is the only key, else only when empty

Answer: `output.message` (text, reasoning, toolUse blocks), `stopReason`
(stop -> end_turn, length -> max_tokens, tool_calls -> tool_use,
content_filter -> content_filtered), `usage` (cache reads split out of
`inputTokens` as Bedrock does), `metrics.latencyMs`. `stop_sequence` is never
reported: the upstream does not say which stop matched.

Ignored (no effect on the answer)
    cachePoint blocks (messages, system, tools)
    reasoningContent blocks in assistant history
    toolResult.status (an error result keeps its text)
    performanceConfig

Refused (400 ValidationException naming the field)
    additionalModelRequestFields with any key but `top_k`: no chat field
        carries provider-specific request fields (thinking, ...), and `top_k`
        beside another key would drop the other one
    additionalModelResponseFieldPaths, guardrailConfig, promptVariables,
        outputConfig, anything else unknown
    guardContent; video, audio, search and citation blocks; s3Location
        sources; image or document inside a toolResult; toolUse or
        reasoning in a user turn; systemTool entries; a message role
        other than user or assistant; a block with several keys

Errors are `{"message": ...}` with `x-amzn-ErrorType` (ValidationException,
UnrecognizedClientException, AccessDeniedException, ResourceNotFoundException,
ThrottlingException, InternalServerException, ...), the HTTP status kept.
A stream that fails after it began ends with an `internalServerException`
frame and no `messageStop`.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Reply,
    Route,
    bearer,
    eventstream,
    refuse,
    refuse_unknown,
    register,
    top_k_value,
)
from routellm.adapters.stream import (
    End,
    Error,
    Finish,
    ReasoningDelta,
    Start,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)

_IMAGE_FORMATS = {"png", "jpeg", "gif", "webp"}
_DOCUMENT_TYPES = {
    "pdf": "application/pdf",
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt": "text/plain",
    "md": "text/markdown",
}
_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filtered",
}
_ERROR_TYPES = {
    400: "ValidationException",
    401: "UnrecognizedClientException",
    403: "AccessDeniedException",
    404: "ResourceNotFoundException",
    408: "ModelTimeoutException",
    429: "ThrottlingException",
    503: "ServiceUnavailableException",
    504: "ModelTimeoutException",
}
#: Operator opt-in: trust the access key id of an unverified SigV4 header.
UNVERIFIED_SIGV4_ENV = "ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4"
_SIGV4 = re.compile(r"^AWS4-HMAC-SHA256\s.*?Credential=([^/,\s]+)/", re.IGNORECASE)

_REQUEST = {
    "messages",
    "system",
    "inferenceConfig",
    "toolConfig",
    "additionalModelRequestFields",
    "requestMetadata",
}
_REQUEST_IGNORED = {"performanceConfig"}
_INFERENCE = {"maxTokens", "temperature", "topP", "stopSequences"}


def _only(block: Any, where: str) -> tuple[str, Any]:
    """The one `(kind, value)` of a Converse union block."""
    if not isinstance(block, Mapping) or len(block) != 1:
        refuse("A content block must be an object with exactly one key", param=where)
    ((kind, value),) = block.items()
    return kind, value


def _data_url(mime: str, source: Any, where: str) -> str:
    if not isinstance(source, Mapping) or not isinstance(source.get("bytes"), str):
        refuse("Only inline `bytes` sources are supported (no s3Location)", param=f"{where}.source")
    return f"data:{mime};base64,{source['bytes']}"


def _image(value: Any, where: str) -> dict:
    fmt = (value or {}).get("format")
    if fmt not in _IMAGE_FORMATS:
        refuse(
            f"Unsupported image format; use one of {sorted(_IMAGE_FORMATS)}",
            param=f"{where}.format",
        )
    url = _data_url(f"image/{fmt}", value.get("source"), where)
    return {"type": "image_url", "image_url": {"url": url}}


def _document(value: Any, where: str) -> dict:
    fmt = (value or {}).get("format")
    if fmt not in _DOCUMENT_TYPES:
        refuse(
            f"Unsupported document format; use one of {sorted(_DOCUMENT_TYPES)}",
            param=f"{where}.format",
        )
    url = _data_url(_DOCUMENT_TYPES[fmt], value.get("source"), where)
    return {"type": "file", "file": {"file_data": url, "filename": value.get("name") or "document"}}


def _compact(parts: list[dict]) -> str | list[dict]:
    """A lone text part is a plain string; anything else stays a part list."""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def _tool_result(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping) or not value.get("toolUseId"):
        refuse("toolResult needs a toolUseId", param=f"{where}.toolUseId")
    parts = []
    for i, item in enumerate(value.get("content") or []):
        kind, inner = _only(item, f"{where}.content[{i}]")
        if kind == "text":
            parts.append({"type": "text", "text": inner})
        elif kind == "json":
            parts.append({"type": "text", "text": json.dumps(inner)})
        else:
            refuse(
                f"toolResult content `{kind}` is not supported (text and json only)",
                param=f"{where}.content[{i}]",
            )
    return {
        "role": "tool",
        "tool_call_id": value["toolUseId"],
        "content": _compact(parts) if parts else "",
    }


def _user_turn(content: list, at: str) -> list[dict]:
    tools, parts = [], []
    for i, block in enumerate(content):
        where = f"{at}.content[{i}]"
        kind, value = _only(block, where)
        if kind == "text":
            parts.append({"type": "text", "text": value})
        elif kind == "image":
            parts.append(_image(value, f"{where}.image"))
        elif kind == "document":
            parts.append(_document(value, f"{where}.document"))
        elif kind == "toolResult":
            tools.append(_tool_result(value, f"{where}.toolResult"))
        elif kind != "cachePoint":
            refuse(f"Content block `{kind}` is not supported in a user turn", param=where)
    out = tools
    if parts:
        out = [*tools, {"role": "user", "content": _compact(parts)}]
    return out


def _assistant_turn(content: list, at: str) -> dict:
    texts, calls = [], []
    for i, block in enumerate(content):
        where = f"{at}.content[{i}]"
        kind, value = _only(block, where)
        if kind == "text":
            texts.append({"type": "text", "text": value})
        elif kind == "toolUse":
            if (
                not isinstance(value, Mapping)
                or not value.get("toolUseId")
                or not value.get("name")
            ):
                refuse("toolUse needs toolUseId and name", param=f"{where}.toolUse")
            arguments = json.dumps(value.get("input") or {})
            call = {
                "id": value["toolUseId"],
                "type": "function",
                "function": {"name": value["name"], "arguments": arguments},
            }
            calls.append(call)
        elif kind not in ("cachePoint", "reasoningContent"):
            refuse(f"Content block `{kind}` is not supported in an assistant turn", param=where)
    message: dict[str, Any] = {"role": "assistant", "content": _compact(texts) if texts else None}
    if calls:
        message["tool_calls"] = calls
    return message


def _messages(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        refuse("messages must be a non-empty list", param="messages")
    out = []
    for i, message in enumerate(raw):
        at = f"messages[{i}]"
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            refuse("A message needs a role and a content list", param=at)
        refuse_unknown(message, {"role", "content"}, where=at)
        role = message.get("role")
        if role == "user":
            out += _user_turn(message["content"], at)
        elif role == "assistant":
            out.append(_assistant_turn(message["content"], at))
        else:
            refuse("role must be user or assistant", param=f"{at}.role")
    return out


def _system(raw: Any) -> list[dict]:
    texts = []
    for i, block in enumerate(raw or []):
        kind, value = _only(block, f"system[{i}]")
        if kind == "text":
            texts.append(value)
        elif kind != "cachePoint":
            refuse(f"System block `{kind}` is not supported", param=f"system[{i}]")
    return [{"role": "system", "content": "\n\n".join(texts)}] if texts else []


def _tools(config: Any) -> dict:
    refuse_unknown(config, {"tools", "toolChoice"}, where="toolConfig")
    out: dict[str, Any] = {}
    tools = []
    for i, entry in enumerate(config.get("tools") or []):
        kind, spec = _only(entry, f"toolConfig.tools[{i}]")
        if kind == "cachePoint":
            continue
        if kind != "toolSpec":
            refuse(
                f"Tool `{kind}` is not supported (toolSpec only)", param=f"toolConfig.tools[{i}]"
            )
        refuse_unknown(
            spec,
            {"name", "description", "inputSchema", "strict"},
            where=f"toolConfig.tools[{i}].toolSpec",
        )
        function = {
            "name": spec.get("name"),
            "description": spec.get("description", ""),
            "parameters": (spec.get("inputSchema") or {}).get("json")
            or {"type": "object", "properties": {}},
        }
        if "strict" in spec:
            function["strict"] = spec["strict"]
        tools.append({"type": "function", "function": function})
    if tools:
        out["tools"] = tools
    choice = config.get("toolChoice")
    if choice is not None:
        kind, value = _only(choice, "toolConfig.toolChoice")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool" and isinstance(value, Mapping) and value.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": value["name"]}}
        else:
            refuse("toolChoice must be auto, any or tool{name}", param="toolConfig.toolChoice")
    return out


def _error_type(status: int) -> str:
    return _ERROR_TYPES.get(status, "InternalServerException")


@register
class BedrockConverse(Adapter):
    name = "bedrock-converse"
    # `:path` so a model id that is an ARN (`inference-profile/...`) matches.
    routes = (
        Route("POST", "/model/{model_id:path}/converse"),
        Route("POST", "/model/{model_id:path}/converse-stream"),
    )

    def credential(self, headers: Mapping[str, str], query: Mapping[str, str]) -> str | None:
        token = bearer(headers)
        if token:
            return token
        if os.environ.get(UNVERIFIED_SIGV4_ENV, "").strip().lower() in ("1", "true", "yes"):
            found = _SIGV4.match(headers.get("authorization") or "")
            return found.group(1) if found else None
        return None

    # -- request ---------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        refuse_unknown(body, _REQUEST, ignored=_REQUEST_IGNORED, where="Converse request")
        extra = body.get("additionalModelRequestFields")
        if extra and (not isinstance(extra, dict) or set(extra) != {"top_k"}):
            refuse(
                "additionalModelRequestFields has no equivalent here; "
                "send it empty or with top_k alone",
                param="additionalModelRequestFields",
            )
        fields: dict[str, Any] = {
            "model": call.path_params["model_id"],
            "messages": [*_system(body.get("system")), *_messages(body.get("messages"))],
        }
        if extra:
            fields["top_k"] = top_k_value(
                extra["top_k"], "additionalModelRequestFields", "additionalModelRequestFields.top_k"
            )
        config = body.get("inferenceConfig") or {}
        refuse_unknown(config, _INFERENCE, where="inferenceConfig")
        for source, target in (
            ("maxTokens", "max_tokens"),
            ("temperature", "temperature"),
            ("topP", "top_p"),
            ("stopSequences", "stop"),
        ):
            if source in config:
                fields[target] = config[source]
        if body.get("toolConfig"):
            fields.update(_tools(body["toolConfig"]))
        if body.get("requestMetadata"):
            fields["metadata"] = body["requestMetadata"]
        streaming = call.route.path.endswith("/converse-stream")
        if streaming:
            fields["stream"] = True
        call.state.update(stream=streaming, started=time.monotonic())
        return ChatCompletionRequest(**fields)

    # -- response --------------------------------------------------------------

    def _latency(self, call: Call) -> int:
        return int((time.monotonic() - call.state.get("started", time.monotonic())) * 1000)

    def _usage(self, prompt: int, completion: int, total: int | None, cached: int | None) -> dict:
        usage = {
            "inputTokens": max(prompt - (cached or 0), 0),
            "outputTokens": completion,
            "totalTokens": total if total is not None else prompt + completion,
        }
        if cached:
            usage["cacheReadInputTokens"] = cached
        return usage

    def encode(self, response: dict, call: Call) -> dict:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content: list[dict] = []
        reasoning = message.get("reasoning_content")
        if reasoning:
            content.append({"reasoningContent": {"reasoningText": {"text": reasoning}}})
        if message.get("content"):
            content.append({"text": message["content"]})
        for tool in message.get("tool_calls") or []:
            function = tool.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                arguments = {}
            content.append(
                {
                    "toolUse": {
                        "toolUseId": tool.get("id"),
                        "name": function.get("name"),
                        "input": arguments,
                    }
                }
            )
        raw = response.get("usage") or {}
        details = raw.get("prompt_tokens_details") or {}
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": _STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
            "usage": self._usage(
                raw.get("prompt_tokens") or 0,
                raw.get("completion_tokens") or 0,
                raw.get("total_tokens"),
                details.get("cached_tokens"),
            ),
            "metrics": {"latencyMs": self._latency(call)},
        }

    def stream_media_type(self, call: Call) -> str:
        return eventstream.MEDIA_TYPE

    async def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        index = -1  # contentBlockIndex of the latest block
        open_text: str | None = None  # "text" / "reasoning" while such a block is open
        tool_deltas: set[int] = set()
        tool_blocks: dict[int, int] = {}  # call_index -> contentBlockIndex
        stopped = False

        def close_text() -> list[bytes]:
            nonlocal open_text
            if open_text is None:
                return []
            open_text = None
            return [eventstream.event("contentBlockStop", {"contentBlockIndex": index})]

        def stop(reason: str) -> bytes:
            nonlocal stopped
            stopped = True
            return eventstream.event("messageStop", {"stopReason": reason})

        async for event in events:
            if isinstance(event, Start):
                yield eventstream.event("messageStart", {"role": "assistant"})
            elif isinstance(event, (TextDelta, ReasoningDelta)):
                kind = "text" if isinstance(event, TextDelta) else "reasoning"
                if open_text != kind:
                    for frame in close_text():
                        yield frame
                    index += 1
                    open_text = kind
                delta = (
                    {"text": event.text}
                    if kind == "text"
                    else {"reasoningContent": {"text": event.text}}
                )
                yield eventstream.event(
                    "contentBlockDelta", {"contentBlockIndex": index, "delta": delta}
                )
            elif isinstance(event, ToolCallStart):
                for frame in close_text():
                    yield frame
                index += 1
                tool_blocks[event.call_index] = index
                start = {
                    "toolUse": {
                        "toolUseId": event.id or f"call_{event.call_index}",
                        "name": event.name or "",
                    }
                }
                yield eventstream.event(
                    "contentBlockStart", {"contentBlockIndex": index, "start": start}
                )
            elif isinstance(event, ToolCallDelta):
                tool_deltas.add(event.call_index)
                delta = {"toolUse": {"input": event.arguments}}
                payload = {"contentBlockIndex": tool_blocks[event.call_index], "delta": delta}
                yield eventstream.event("contentBlockDelta", payload)
            elif isinstance(event, ToolCallEnd):
                block = tool_blocks[event.call_index]
                if event.call_index not in tool_deltas:
                    # A call without arguments still needs a parseable input.
                    delta = {"toolUse": {"input": "{}"}}
                    yield eventstream.event(
                        "contentBlockDelta", {"contentBlockIndex": block, "delta": delta}
                    )
                yield eventstream.event("contentBlockStop", {"contentBlockIndex": block})
            elif isinstance(event, Finish):
                for frame in close_text():
                    yield frame
                yield stop(_STOP_REASONS.get(event.reason, "end_turn"))
            elif isinstance(event, Error):
                yield eventstream.exception("internalServerException", "Upstream request failed")
                return
            elif isinstance(event, End):
                for frame in close_text():
                    yield frame
                if not stopped:
                    yield stop("end_turn")
                usage = event.usage
                counts = self._usage(
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                    usage.total_tokens if usage else None,
                    usage.cached_tokens if usage else None,
                )
                yield eventstream.event(
                    "metadata", {"usage": counts, "metrics": {"latencyMs": self._latency(call)}}
                )

    # -- errors ----------------------------------------------------------------

    def encode_error(self, status: int, error: dict, call: Call) -> Reply:
        headers = {"x-amzn-ErrorType": _error_type(status)}
        if error.get("request_id"):
            headers["x-amzn-RequestId"] = error["request_id"]
        return Reply({"message": error["message"]}, headers=headers)

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return eventstream.exception("internalServerException", error["message"])
