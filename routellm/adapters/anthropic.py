"""Anthropic Messages dialect: `POST /v1/messages`, `POST /v1/messages/count_tokens`.

Claude Code and the `anthropic` SDK reach routellm by setting
`ANTHROPIC_BASE_URL` to the server and `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`) to a routellm key. The model field carries routing
parameters exactly as on `/v1/chat/completions`.

The Anthropic <-> chat completions mapping is litellm's, not re-derived
here: this module is glue between the adapter contract and
`litellm.llms.anthropic.pass_through.adapters` (the translator litellm's own
`/v1/messages` bridge runs). It adds only what that translator lacks:
validation that refuses instead of dropping, `count_tokens`, the error
shape of the contract, `ping`, `disable_parallel_tool_use`, and a string
`signature` on reasoning without one. The tests pin litellm's behaviour, so
a litellm upgrade that moves or changes what is used fails there.

Always sent to litellm with a neutral model name: the client's model name
(`claude-...`) must not change how the request is translated, because the
routed model is chosen later and is not Anthropic's.

Request fields
==============

Mapped (litellm translator):

- `messages`: `text`, `image` (base64 or url), `document` (base64 or url, as
  a data/url part), `tool_use` (assistant `tool_calls`), `tool_result`
  (`role: tool`), `thinking` and `redacted_thinking` (kept with their
  signatures as `thinking_blocks` on the assistant turn).
- `system` (string or text blocks), `max_tokens`, `temperature`, `top_p`,
  `stream`, `stop_sequences` (-> `stop`), `metadata.user_id` (-> `user`).
- `tools` (custom tools; names over 64 characters are shortened and restored
  in answers), `tool_choice` (`auto`, `any` -> required, `tool`, `none`).
- `thinking` (`enabled` budget bucketed, `disabled`, `adaptive`) ->
  `reasoning_effort`; `output_config.effort` with adaptive thinking.
- `output_config.format` / `output_format` -> `response_format`.

Mapped here: `tool_choice.disable_parallel_tool_use` ->
`parallel_tool_calls: false`.

Accepted and ignored (no effect on the answer):

- `cache_control` on any block, tool or system part: the routed model is not
  known when the request is translated, so it is dropped. Cached tokens a
  provider reports still appear as `cache_read_input_tokens`.
- `context_management` (any shape; logged at debug): server-side context
  editing is not available through chat completions, and clients keep their
  own compaction, which Claude Code sends this field alongside.
- `service_tier`; `thinking.display`; `tool_result.is_error` (the result text
  still reaches the model); `eager_input_streaming` on a tool.
- `output_config.effort` without adaptive thinking.
- Headers `anthropic-version` and `anthropic-beta` (any value).

Refused with a 400 `invalid_request_error`, naming the field:

- any other top-level field, among them `top_k`, `mcp_servers`, `container`;
- a missing `max_tokens`; `metadata` keys other than `user_id`;
- block types other than those mapped (`server_tool_use`, `search_result`,
  `web_search_tool_result`, `tool_reference`, ...), unknown keys in a block,
  `document` sources other than base64/url (`text`, `content`, `file`);
- hosted tools (`web_search_*`, `bash_*`, `computer_*`, `text_editor_*`,
  ...): only custom tools (name and `input_schema`) are carried;
- a `role` other than `user` / `assistant`; an unknown `tool_choice` or
  `thinking` type.

Answers
=======

- Message: `content` text, thinking (`signature` is `""` when the provider
  gave none) and `tool_use` blocks; `usage` as `input_tokens` (without cached
  tokens), `output_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens`; `model` echoes the client's model field (the
  routing decision is in the `x-routellm-path` header).
- `stop_reason`: `end_turn`, `max_tokens`, `tool_use`, `refusal`
  (`content_filter`). A stop sequence ends as `end_turn` with
  `stop_sequence: null`: chat completions do not say which one matched.
- Stream: `message_start` (usage zero: input tokens are not known until the
  provider answers), `ping`, `content_block_start` / `content_block_delta`
  (`text_delta`, `thinking_delta`, `input_json_delta`) /
  `content_block_stop`, `message_delta` (`stop_reason` and the usage),
  `message_stop`. A failure after the stream started ends it with an `error`
  event and no `message_stop`.
- Errors: `{"type": "error", "error": {"type", "message"}, "request_id"}`;
  types `invalid_request_error`, `authentication_error`, `permission_error`,
  `not_found_error`, `rate_limit_error`, `api_error`, `overloaded_error`
  (503 from cooled-down upstreams).

`count_tokens`
==============

`{"input_tokens": n}` is an ESTIMATE: the request's messages, system and
tools are translated like a real one and counted with litellm's default
tokenizer. The routed model is not chosen (nothing is sent upstream), so its
own tokenizer is not used; expect a few percent of drift. Authenticated like
`/v1/messages`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import litellm
from litellm.anthropic_interface.exceptions.exception_mapping_utils import (
    AnthropicExceptionMapping,
)
from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import AnthropicStreamWrapper
from litellm.llms.anthropic.pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import ModelResponse, ModelResponseStream

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Route,
    bearer,
    refuse,
    refuse_unknown,
    register,
)

__all__ = ["AnthropicAdapter", "message_events"]

#: The model name the translator sees, whatever the client asked for.
logger = logging.getLogger(__name__)

_NEUTRAL_MODEL = "routellm"

_TRANSLATOR = LiteLLMAnthropicMessagesAdapter()

_SUPPORTED = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "metadata",
        "stop_sequences",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "output_format",
    }
)
_IGNORED = frozenset({"service_tier", "context_management"})
#: What `count_tokens` accepts: the fields that change the token count, and
#: the ones a client sends alongside them.
_COUNTED = frozenset(
    {"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config"}
)

_CACHE = "cache_control"
_SOURCES: dict[str, frozenset[str]] = {
    "base64": frozenset({"type", "media_type", "data"}),
    "url": frozenset({"type", "url"}),
}
_USER_BLOCKS: dict[str, frozenset[str]] = {
    "text": frozenset({"type", "text", _CACHE}),
    "image": frozenset({"type", "source", _CACHE}),
    "document": frozenset({"type", "source", _CACHE}),
    "tool_result": frozenset({"type", "tool_use_id", "content", "is_error", _CACHE}),
}
_ASSISTANT_BLOCKS: dict[str, frozenset[str]] = {
    "text": frozenset({"type", "text", _CACHE}),
    "tool_use": frozenset({"type", "id", "name", "input", _CACHE}),
    "thinking": frozenset({"type", "thinking", "signature"}),
    "redacted_thinking": frozenset({"type", "data"}),
}
_RESULT_BLOCKS = ("text", "image", "document")
_TOOL_KEYS = frozenset(
    {"name", "description", "input_schema", "type", _CACHE, "strict", "eager_input_streaming"}
)
_CHOICES = ("auto", "any", "tool", "none")
_THINKING = ("enabled", "disabled", "adaptive")


# -- validation: refuse what litellm would silently drop ----------------------


def _keys(obj: Mapping, allowed: frozenset[str], path: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        refuse(f"Unsupported field(s) in {path}: {', '.join(extra)}", param=f"{path}.{extra[0]}")


def _source(source: Any, path: str) -> None:
    if not isinstance(source, dict) or source.get("type") not in _SOURCES:
        refuse(f"{path} must be a base64 or url source", param=path)
    _keys(source, _SOURCES[source["type"]], path)


def _block(block: Any, allowed: Mapping[str, frozenset[str]], path: str) -> str:
    if not isinstance(block, dict):
        refuse(f"{path} must be an object", param=path)
    kind = block.get("type")
    if kind not in allowed:
        refuse(f"Unsupported content block type {kind!r} in {path}", param=f"{path}.type")
    _keys(block, allowed[kind], path)
    if kind in ("image", "document"):
        _source(block.get("source"), f"{path}.source")
    return kind


def _tool_result(block: dict, path: str) -> None:
    content = block.get("content")
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        refuse(f"{path}.content must be a string or a list of blocks", param=f"{path}.content")
    allowed = {k: _USER_BLOCKS[k] for k in _RESULT_BLOCKS}
    for j, part in enumerate(content):
        _block(part, allowed, f"{path}.content[{j}]")


def _messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        refuse("messages must be a non-empty list", param="messages")
    for i, message in enumerate(messages):
        path = f"messages[{i}]"
        role = message.get("role") if isinstance(message, dict) else None
        if role not in ("user", "assistant"):
            refuse(f"{path}.role must be 'user' or 'assistant'", param=f"{path}.role")
        _keys(message, frozenset({"role", "content"}), path)
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            refuse(f"{path}.content must be a string or a list of blocks", param=f"{path}.content")
        allowed = _USER_BLOCKS if role == "user" else _ASSISTANT_BLOCKS
        for j, block in enumerate(content):
            kind = _block(block, allowed, f"{path}.content[{j}]")
            if kind == "tool_result":
                _tool_result(block, f"{path}.content[{j}]")


def _system(system: Any) -> None:
    if system is None or isinstance(system, str):
        return
    if not isinstance(system, list):
        refuse("system must be a string or a list of text blocks", param="system")
    for i, block in enumerate(system):
        _block(block, {"text": _USER_BLOCKS["text"]}, f"system[{i}]")


def _tools(tools: Any) -> None:
    if not isinstance(tools, list):
        refuse("tools must be a list", param="tools")
    for i, tool in enumerate(tools):
        path = f"tools[{i}]"
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            refuse(f"{path} needs a name", param=path)
        if tool.get("type", "custom") != "custom":
            refuse(f"Hosted tool type {tool['type']!r} is not supported", param=f"{path}.type")
        _keys(tool, _TOOL_KEYS, path)


def _tool_choice(choice: Any) -> None:
    if not isinstance(choice, dict) or choice.get("type") not in _CHOICES:
        refuse(f"tool_choice.type must be one of {', '.join(_CHOICES)}", param="tool_choice")
    _keys(choice, frozenset({"type", "name", "disable_parallel_tool_use"}), "tool_choice")
    if choice["type"] == "tool" and not isinstance(choice.get("name"), str):
        refuse("tool_choice of type 'tool' needs a name", param="tool_choice.name")


def _thinking(thinking: Any) -> None:
    if not isinstance(thinking, dict) or thinking.get("type") not in _THINKING:
        refuse(f"thinking.type must be one of {', '.join(_THINKING)}", param="thinking")
    _keys(thinking, frozenset({"type", "budget_tokens", "display"}), "thinking")
    if thinking["type"] == "enabled" and not isinstance(thinking.get("budget_tokens"), int):
        refuse("thinking of type 'enabled' needs budget_tokens", param="thinking.budget_tokens")


def _validate(body: Mapping[str, Any]) -> None:
    if not isinstance(body.get("model"), str):
        refuse("model is required", param="model")
    _messages(body.get("messages"))
    _system(body.get("system"))
    if "tools" in body:
        _tools(body["tools"])
    if body.get("tool_choice") is not None:
        _tool_choice(body["tool_choice"])
    if body.get("thinking") is not None:
        _thinking(body["thinking"])
    metadata = body.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict) or set(metadata) - {"user_id"}:
            refuse("metadata only carries user_id", param="metadata")
    config = body.get("output_config")
    if config is not None:
        if not isinstance(config, dict) or set(config) - {"format", "effort"}:
            refuse("output_config only carries format and effort", param="output_config")


# -- translation --------------------------------------------------------------


def _translate(body: Mapping[str, Any]) -> tuple[dict, dict[str, str]]:
    """Validate, run litellm's translator, return (chat request fields, tool name map)."""
    _validate(body)
    if "context_management" in body:
        logger.debug("anthropic adapter: context_management ignored")
    wire = {k: v for k, v in body.items() if k not in _IGNORED}
    wire["model"] = _NEUTRAL_MODEL
    out, names = _TRANSLATOR.translate_anthropic_to_openai(wire)
    out = dict(out)
    out["model"] = body["model"]
    choice = body.get("tool_choice") or {}
    if choice.get("disable_parallel_tool_use") is True:
        out["parallel_tool_calls"] = False
    # The translator forwards fields the pipeline has no place for (a litellm
    # upgrade may add one): refuse them rather than drop them.
    stray = sorted(set(out) - set(ChatCompletionRequest.model_fields))
    if stray:
        refuse(f"Unsupported field(s) after translation: {', '.join(stray)}", param=stray[0])
    return out, names


def _chunks(source: AsyncIterator[Any]) -> AsyncIterator[ModelResponseStream]:
    async def convert():
        async for chunk in source:
            as_dict = isinstance(chunk, dict)
            if chunk.get("error") if as_dict else getattr(chunk, "error", None):
                # The text can name the upstream; the framework logs the
                # cause and sends its own generic frame.
                raise RuntimeError("upstream reported an error inside the stream")
            if isinstance(chunk, ModelResponseStream):
                yield chunk
            else:
                yield ModelResponseStream(**(chunk if as_dict else chunk.model_dump()))

    return convert()


async def message_events(chunks: AsyncIterator, call: Call) -> AsyncIterator[dict]:
    """The Messages stream events of a chat chunk stream, as dicts.

    Shared with dialects that carry the same events in another framing.
    `call.state` holds `model` and `tool_names`, as `decode` left them.
    """
    wrapper = AnthropicStreamWrapper(
        _chunks(chunks),
        call.state["model"],
        tool_name_mapping=call.state.get("tool_names") or None,
    )
    # Iterated directly, not through litellm's SSE wrapper: that one turns
    # a failure into an event carrying the upstream's own message.
    async for event in wrapper:
        yield event
        if event["type"] == "message_start":
            yield {"type": "ping"}


def _sse(event: Mapping[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@register
class AnthropicAdapter(Adapter):
    name = "anthropic"
    routes = (
        Route("POST", "/v1/messages"),
        Route("POST", "/v1/messages/count_tokens", operation="count_tokens"),
    )

    def credential(self, headers: Mapping[str, str], query: Mapping[str, str]) -> str | None:
        return headers.get("x-api-key") or bearer(headers)

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        refuse_unknown(body, _SUPPORTED, ignored=_IGNORED, where="Messages request")
        if not isinstance(body.get("max_tokens"), int):
            refuse("max_tokens is required", param="max_tokens")
        fields, names = _translate(body)
        call.state.update(model=body["model"], tool_names=names)
        return ChatCompletionRequest(**fields)

    async def handle(self, call: Call) -> Any:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        refuse_unknown(body, _COUNTED, where="count_tokens request")
        fields, _ = _translate({**body, "max_tokens": 1})
        count = litellm.token_counter(
            messages=fields["messages"], tools=fields.get("tools") or None
        )
        return {"input_tokens": int(count)}

    def encode(self, response: dict, call: Call) -> Any:
        completion = {k: v for k, v in response.items() if k != "routellm"}
        out = dict(
            _TRANSLATOR.translate_openai_response_to_anthropic(
                ModelResponse(**completion), call.state.get("tool_names") or None
            )
        )
        out["model"] = call.state["model"]
        if not str(out["id"]).startswith("msg_"):
            out["id"] = f"msg_{out['id']}"
        for block in out["content"]:
            if block.get("type") == "thinking" and block.get("signature") is None:
                block["signature"] = ""
        return out

    async def encode_stream(self, chunks: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        async for event in message_events(chunks, call):
            yield _sse(event)

    def encode_error(self, status: int, error: dict, call: Call) -> Any:
        kind = 529 if status == 503 else status
        return AnthropicExceptionMapping.create_error_response(
            kind, error["message"], error.get("request_id")
        )

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        body = AnthropicExceptionMapping.create_error_response(
            500, error["message"], error.get("request_id")
        )
        return f"event: error\ndata: {json.dumps(body)}\n\n".encode()
