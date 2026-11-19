"""OpenAI Responses API: `POST /v1/responses`, for Codex CLI and `client.responses`.

Thin glue over litellm's own Responses <-> chat translator, not a second
mapping. litellm does the work:

- request: `LiteLLMCompletionResponsesConfig
  .transform_responses_api_request_to_chat_completion_request` (input items,
  instructions, tools, tool choice, reasoning effort, text format);
- answer: `...transform_chat_completion_response_to_responses_api_response`;
- stream: `LiteLLMCompletionStreamingIterator`, fed the pipeline's chunks.

This module adds only what litellm lacks, and what its bridge would otherwise
drop without a word:

- refusal of every field, tool type, input item and content part the chat
  pipeline cannot carry (litellm silently drops `local_shell`, `local_shell_call`,
  `text.verbosity`, ...);
- `metadata` (litellm leaves it to its own kwargs);
- stable `resp_<chat id>` response ids, the same streamed or not (litellm
  encodes routing hints into streamed ids), and `sequence_number` on every
  event;
- SSE framing, `response.failed` for a stream that dies after it started, the
  OpenAI error envelope.

Contract tests in `routellm/tests/adapters/test_openai_responses.py` pin the
litellm internals used here.

Request fields
==============

Mapped: `model` (carries routing parameters), `input` (string, or items),
`instructions`, `tools`, `tool_choice`, `parallel_tool_calls`, `temperature`,
`top_p`, `max_output_tokens`, `stream`, `reasoning.effort`
(`reasoning_effort`), `text.format` (`response_format`), `metadata`, `user`.

Ignored, no effect on the answer: `store` (nothing is stored: a response
cannot be fetched later), `include` of `reasoning.encrypted_content` (no
encrypted reasoning is produced), `prompt_cache_key`, `prompt_cache_retention`,
`service_tier`, `safety_identifier`, `client_metadata`, `stream_options`,
`reasoning.summary`, `reasoning.generate_summary`, `text.verbosity`,
`truncation: "disabled"`, `background: false`.

Refused with a 400 naming the field: `previous_response_id` (the server keeps
no response state: send the whole conversation in `input`), `conversation`,
`background: true`, `max_tool_calls`, `top_logprobs`, `truncation: "auto"`,
`prompt`, `context_management`, any other `include`, any other field.

Tools: `function`, `custom` (Codex's freeform `apply_patch`; answered as a
`custom_tool_call`) and `namespace` of those. Refused: hosted tools
(`web_search`, `file_search`, `mcp`, `local_shell`, `computer_use`,
`image_generation`, ...).

Input items: messages (roles `user`, `assistant`, `system`, `developer`) with
`input_text`, `output_text`, `text`, `input_image` and `input_file` parts,
`function_call`, `function_call_output`, `custom_tool_call`,
`custom_tool_call_output`, `reasoning`. Refused: `local_shell_call`,
`item_reference`, `web_search_call`, `computer_call`, `additional_tools`, ...

Codex CLI
=========

Codex always streams with `store: false`, sends developer messages, prior
reasoning items, function and custom tool calls in `input`, `include:
["reasoning.encrypted_content"]`, `prompt_cache_key` and `text.verbosity`:
all accepted. A model Codex runs with its `local_shell` or hosted
`web_search` tool is refused explicitly; with the default function `shell`
tool it works. Point `base_url` at `<server>/v1` with `wire_api =
"responses"`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig as Translator,
)
from litellm.types.utils import ModelResponseStream

from routellm.adapters.base import Adapter, Call, Route, refuse, refuse_unknown
from routellm.adapters.registry import register
from routellm.chat_request import ChatCompletionRequest

SUPPORTED = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "max_output_tokens",
        "stream",
        "reasoning",
        "text",
        "metadata",
        "user",
    }
)
IGNORED = frozenset(
    {
        "store",
        "include",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "safety_identifier",
        "client_metadata",
        "stream_options",
        "truncation",
        "background",
    }
)
#: Refused by name, with the reason the client reads.
REFUSED = {
    "previous_response_id": (
        "previous_response_id is not supported: this server keeps no response state;"
        " send the whole conversation in `input`"
    ),
    "conversation": "conversation is not supported: this server keeps no conversation state",
    "max_tool_calls": "max_tool_calls is not supported",
    "top_logprobs": "top_logprobs is not supported",
    "prompt": "prompt (stored prompt templates) is not supported",
    "context_management": "context_management is not supported",
}
INCLUDE = frozenset({"reasoning.encrypted_content"})
TOOL_TYPES = frozenset({"function", "custom", "namespace"})
ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "reasoning",
    }
)
PART_TYPES = frozenset({"input_text", "output_text", "text", "input_image", "input_file"})
ROLES = frozenset({"user", "assistant", "system", "developer"})
TEXT_KEYS = frozenset({"format", "verbosity"})
REASONING_KEYS = frozenset({"effort", "summary", "generate_summary"})
_CHAT_FIELDS = frozenset(ChatCompletionRequest.model_fields)


def _is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, dict) for v in value)


def _check_tools(tools: Any, where: str = "tools") -> None:
    if not _is_list_of_dicts(tools):
        refuse("tools must be a list of tool objects", param="tools")
    for tool in tools:
        kind = tool.get("type")
        if kind not in TOOL_TYPES:
            refuse(
                f"Tool type '{kind}' is not supported (function and custom tools only)",
                param="tools",
            )
        if kind == "namespace":
            for nested in tool.get("tools") or []:
                if not isinstance(nested, dict) or nested.get("type") not in {"function", "custom"}:
                    refuse("A namespace tool holds function and custom tools only", param="tools")


def _check_content(content: Any) -> None:
    if content is None or isinstance(content, str):
        return
    if not _is_list_of_dicts(content):
        refuse("input message content must be a string or a list of parts", param="input")
    for part in content:
        if part.get("type") not in PART_TYPES:
            refuse(f"Content part type '{part.get('type')}' is not supported", param="input")


def _check_input(items: Any) -> None:
    if isinstance(items, str):
        if not items:
            refuse("input must not be empty", param="input")
        return
    if not _is_list_of_dicts(items) or not items:
        refuse("input must be a non-empty string or list of items", param="input")
    for item in items:
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind not in ITEM_TYPES:
            refuse(f"Input item type '{kind}' is not supported", param="input")
        if kind == "message":
            if item.get("role") not in ROLES:
                refuse(f"Message role '{item.get('role')}' is not supported", param="input")
            _check_content(item.get("content"))


def _check_fields(body: dict) -> None:
    for name, message in REFUSED.items():
        if body.get(name) is not None:
            refuse(message, param=name)
    refuse_unknown(body, SUPPORTED | IGNORED | set(REFUSED), where="responses request")
    if body.get("background"):
        refuse("background responses are not supported", param="background")
    if body.get("truncation") not in (None, "disabled"):
        refuse("truncation 'auto' is not supported", param="truncation")
    include = body.get("include")
    if include is not None and (not isinstance(include, list) or not set(include) <= INCLUDE):
        refuse("include supports only reasoning.encrypted_content", param="include")
    if not isinstance(body.get("model"), str) or not body["model"]:
        refuse("model is required", param="model")
    if body.get("stream") not in (None, True, False):
        refuse("stream must be a boolean", param="stream")
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            refuse("text must be an object", param="text")
        refuse_unknown(text, TEXT_KEYS, where="text")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        refuse_unknown(reasoning, REASONING_KEYS, where="reasoning")
    if "input" not in body:
        refuse("input is required", param="input")
    _check_input(body["input"])
    if body.get("tools") is not None:
        _check_tools(body["tools"])


def _response_id(chat_id: str | None) -> str:
    """`resp_<chat id>`: the same id streamed or not, nothing litellm-specific."""
    if not chat_id:
        return f"resp_{uuid.uuid4().hex}"
    return chat_id if chat_id.startswith("resp_") else f"resp_{chat_id.removeprefix('chatcmpl-')}"


class _Chunks:
    """What `LiteLLMCompletionStreamingIterator` reads of a `CustomStreamWrapper`.

    An async iterator of `ModelResponseStream` and a `logging_obj` (None: no
    litellm spend logging runs here; routellm audits the request itself).
    Remembers the first chunk's id, which names the response.
    """

    logging_obj = None

    def __init__(self, chunks: AsyncIterator, state: dict):
        self._chunks = chunks.__aiter__()
        self._state = state

    def __aiter__(self):
        return self

    async def __anext__(self) -> ModelResponseStream:
        chunk = await self._chunks.__anext__()
        if isinstance(chunk, dict):
            chunk = ModelResponseStream(**chunk)
        self._state.setdefault("chat_id", getattr(chunk, "id", None))
        self._state.setdefault("model", getattr(chunk, "model", None))
        self._state.setdefault("created_at", getattr(chunk, "created", None))
        return chunk


def _frame(data: dict) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@register
class OpenAIResponsesAdapter(Adapter):
    name = "openai_responses"
    routes = (Route("POST", "/v1/responses"),)

    # -- request ------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        _check_fields(body)

        translated = Translator.transform_responses_api_request_to_chat_completion_request(
            model=body["model"],
            input=body["input"],
            responses_api_request=body,
            stream=body.get("stream"),
        )
        if body.get("metadata") is not None:
            translated["metadata"] = body["metadata"]
        lost = sorted(k for k, v in translated.items() if v is not None and k not in _CHAT_FIELDS)
        if lost:
            # litellm carried something the chat pipeline has no field for.
            refuse(f"Unsupported field(s) in responses request: {', '.join(lost)}", param=lost[0])

        call.state["responses_request"] = body
        return ChatCompletionRequest(**translated)

    # -- response -----------------------------------------------------------

    def encode(self, response: dict, call: Call) -> Any:
        body = call.state["responses_request"]
        chat = {k: v for k, v in response.items() if k != "routellm"}
        result = Translator.transform_chat_completion_response_to_responses_api_response(
            request_input=body["input"], responses_api_request=body, chat_completion_response=chat
        )
        data = result.model_dump(mode="json")
        data["id"] = _response_id(response.get("id"))
        return data

    async def encode_stream(self, chunks: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        body = call.state["responses_request"]
        state = call.state["stream"] = {"sequence": 0}
        iterator = LiteLLMCompletionStreamingIterator(
            model=body["model"],
            litellm_custom_stream_wrapper=_Chunks(chunks, state),
            request_input=body["input"],
            responses_api_request=body,
        )
        async for event in iterator:
            state["sequence"] += 1
            data = event.model_dump(mode="json", exclude_none=True)
            if getattr(event, "response", None) is not None:
                data["response"] = event.response.model_dump(mode="json")
                data["response"]["id"] = _response_id(state.get("chat_id"))
            data["sequence_number"] = state["sequence"]
            yield _frame(data)

    # -- errors -------------------------------------------------------------

    def encode_error(self, status: int, error: dict, call: Call) -> Any:
        return {
            "error": {
                "message": error["message"],
                "type": error["type"],
                "param": error["param"],
                "code": error["code"],
            }
        }

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        state = call.state.get("stream") or {"sequence": 0}
        state["sequence"] += 1
        response = {
            "id": _response_id(state.get("chat_id")),
            "object": "response",
            "created_at": state.get("created_at") or int(time.time()),
            "model": state.get("model") or (call.body or {}).get("model"),
            "status": "failed",
            "output": [],
            "error": {"code": error["code"] or "server_error", "message": error["message"]},
        }
        return _frame(
            {"type": "response.failed", "response": response, "sequence_number": state["sequence"]}
        )
