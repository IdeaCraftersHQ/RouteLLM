"""Cohere v2 Chat API: `POST /v2/chat`.

For tools that speak Cohere's own API (the `cohere` SDK's `ClientV2`, or any
HTTP client following the v2 Chat reference). Inbound only: whatever routellm
routes to upstream is independent of this dialect.

litellm has no inbound Cohere translator, so the mapping is this module's own.
It inverts what litellm does for a Cohere v2 upstream
(`litellm/llms/cohere/chat/v2_transformation.py`: the v2 request is already
OpenAI-shaped, `top_p` <-> `p`, `stop` <-> `stop_sequences`, `max_tokens`,
`seed`, penalties and `tools` carry over by name; the answer's `tokens` usage
and `tool_calls` list map back, `CohereV2ModelResponseIterator` reads the
stream events). Wire shapes follow the Cohere v2 Chat reference as modelled by
the `cohere` Python SDK (`cohere.types`: `V2ChatResponse`, `Usage`,
`StreamedChatResponseV2` and its `content-*`, `tool-call-*`, `message-*`
events).

Auth: `Authorization: Bearer <key>`, as the SDK sends it. Errors are
`{"message": "<text>"}` (plus `id` when the request id is known); the status
is kept, and the SDK raises its own error class from it.

Requests
--------
  mapped    model, messages, tools, strict_tools, tool_choice, documents,
            response_format, max_tokens, stop_sequences, temperature, p, k,
            seed, frequency_penalty, presence_penalty, stream (default
            false, as in Cohere), thinking, plus the checked fields below
  ignored   priority (queue order at Cohere, no effect on the answer);
            on assistant messages `citations` (an earlier answer replayed by
            the client); `thinking` content blocks in assistant history
            (earlier reasoning is not replayed upstream)
  refused   everything else, with a 400 naming the field: connectors,
            raw_prompting, search_queries_only and the other v1 fields, an
            unknown top-level or message key, a content block that is not
            text or image_url (a tool result also takes document), a role
            other than system, user, assistant, tool

`messages`: `content` is a string or a list of blocks (system: text; user:
text, image_url with `url` and `detail`; assistant: text, thinking; tool:
text, document). An assistant `tool_calls` entry keeps its `id`, `name` and
JSON-string `arguments`; a `tool` message is answered by its `tool_call_id`
(Cohere calls carry ids, so no pairing is guessed). An assistant `tool_plan`
is the message's text, joined after its content with a newline.
`tools`: function tools only. `strict_tools: true` is `strict: true` on each
function. `tool_choice`: `REQUIRED` is `required`, `NONE` is `none`.
`response_format`: `{"type": "json_object"}` is json mode; with `json_schema`
it is a `json_schema` format carrying that schema; `{"type": "text"}` sets
nothing, and a schema with it is refused.
`stop_sequences` is `stop`, `p` is `top_p`, `k` is `top_k` (a whole number
from 0).
`thinking`: `{"type": "disabled"}` sets nothing; `{"type": "enabled"}` is
`reasoning_effort: medium`; a `token_budget` is refused (no equivalent).

Checked fields (accepted only when they change nothing):
  citation_options  `{"mode": "disabled"}` only. routellm produces no
                    citations, so asking for them is refused
  safety_mode       `CONTEXTUAL` only (Cohere's default); `STRICT` and `OFF`
                    configure a moderation layer routellm does not have
  logprobs          `false` only

`documents` (strings, or `{"id", "data"}` objects whose `data` is a map of
text fields) have no chat completions field. They are carried as context in
the prompt: one system message, placed after the leading system messages
(first when there are none), reading "Use the following documents as context
for the conversation." then one block per document,
`<document id="...">` with a `key: value` line per data field (non-string
values as JSON), the id being the given one or `doc_<n>`. The same rendering
is used for a `document` block inside a tool result, which becomes part of the
tool message's text. Because no citations come back, answers carry no
`citations`.

Responses
---------
`id`, `finish_reason` (`stop` is `COMPLETE`, `length` is `MAX_TOKENS`,
`tool_calls` is `TOOL_CALL`, `content_filter` and `error` are `ERROR`;
`STOP_SEQUENCE` is never reported, chat completions does not say which stop
ended the answer), `message` (`role`, `content` as text and `thinking`
blocks, `tool_calls`) and `usage` (`billed_units` and `tokens`, both the
upstream's counts; routellm bills nothing, and `cached_tokens` when the
upstream reports it). Text that comes with tool calls stays a text content
block: `tool_plan` is never produced, in whole answers or streams, because a
stream cannot know ahead that text is a plan.

Streams are SSE, `event: <type>` and `data: <json>` frames, without a
`[DONE]` line, in Cohere's order: `message-start`, then per block
`content-start` / `content-delta` / `content-end` (a `thinking` block when the
upstream reasons, then a `text` block) and `tool-call-start` /
`tool-call-delta` / `tool-call-end` (tool calls are indexed apart from
content blocks), then `message-end` carrying `finish_reason` and `usage`.
`tool-plan-delta`, `citation-start` and `citation-end` are never sent. A
failure after the stream started is a final `message-end` with `error` and
`finish_reason: ERROR`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Route,
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
    Usage,
)

__all__ = ["CohereAdapter"]

CHAT_PATH = "/v2/chat"

_FIELDS = {
    "model",
    "messages",
    "stream",
    "tools",
    "strict_tools",
    "tool_choice",
    "documents",
    "response_format",
    "max_tokens",
    "stop_sequences",
    "temperature",
    "p",
    "k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "thinking",
}
_CHECKED = {"citation_options", "safety_mode", "logprobs"}
_IGNORED = {"priority"}

_ROLE_FIELDS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {"role", "content", "tool_calls", "tool_plan"},
    "tool": {"role", "tool_call_id", "content"},
}
_TOOL_CHOICES = {"REQUIRED": "required", "NONE": "none"}
_FINISH = {
    "stop": "COMPLETE",
    "length": "MAX_TOKENS",
    "tool_calls": "TOOL_CALL",
    "function_call": "TOOL_CALL",
    "content_filter": "ERROR",
    "error": "ERROR",
}

_PRELUDE = "Use the following documents as context for the conversation."


# -- request ------------------------------------------------------------------


def _blocks(content: Any, where: str, allowed: set[str]) -> list[dict]:
    """The content blocks of a message: a string is one text block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        refuse(f"{where} must be a string or a list of content blocks", where)
    for i, block in enumerate(content):
        at = f"{where}[{i}]"
        if not isinstance(block, dict):
            refuse(f"{at} must be an object", at)
        if block.get("type") not in allowed:
            refuse(f"{at}.type must be one of {', '.join(sorted(allowed))}", f"{at}.type")
    return content


def _text(block: dict, at: str) -> str:
    text = block.get("text")
    if not isinstance(text, str):
        refuse(f"{at}.text must be a string", f"{at}.text")
    return text


def _render_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _render_document(doc: Any, fallback_id: str | None, where: str) -> str:
    """A document as prompt text: see the module docstring."""
    if isinstance(doc, str):
        data, doc_id = None, fallback_id
        body = doc
    elif isinstance(doc, dict):
        refuse_unknown(doc, {"id", "data"}, where=where)
        data = doc.get("data")
        if not isinstance(data, dict):
            refuse(f"{where}.data must be an object", f"{where}.data")
        doc_id = doc.get("id") or fallback_id
        body = "\n".join(f"{key}: {_render_value(value)}" for key, value in data.items())
    else:
        refuse(f"{where} must be a string or an object with data", where)
    opening = f'<document id="{doc_id}">' if doc_id else "<document>"
    return f"{opening}\n{body}\n</document>"


def _documents_message(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        refuse("documents must be a list", "documents")
    if not raw:
        return None
    parts = [_render_document(d, f"doc_{n}", f"documents[{n}]") for n, d in enumerate(raw)]
    return {"role": "system", "content": _PRELUDE + "\n\n" + "\n\n".join(parts)}


def _user_content(content: Any, where: str) -> str | list[dict]:
    if isinstance(content, str):
        return content
    parts: list[dict] = []
    for i, block in enumerate(_blocks(content, where, {"text", "image_url"})):
        at = f"{where}[{i}]"
        if block["type"] == "text":
            refuse_unknown(block, {"type", "text"}, where=at)
            parts.append({"type": "text", "text": _text(block, at)})
            continue
        refuse_unknown(block, {"type", "image_url"}, where=at)
        image = block.get("image_url")
        if not isinstance(image, dict) or not isinstance(image.get("url"), str):
            refuse(f"{at}.image_url.url must be a string", f"{at}.image_url.url")
        refuse_unknown(image, {"url", "detail"}, where=f"{at}.image_url")
        parts.append(
            {"type": "image_url", "image_url": {k: v for k, v in image.items() if v is not None}}
        )
    return parts


def _system_content(content: Any, where: str) -> str | list[dict]:
    if isinstance(content, str):
        return content
    blocks = _blocks(content, where, {"text"})
    return [{"type": "text", "text": _text(b, f"{where}[{i}]")} for i, b in enumerate(blocks)]


def _tool_content(content: Any, where: str) -> str:
    if isinstance(content, str):
        return content
    out = []
    for i, block in enumerate(_blocks(content, where, {"text", "document"})):
        at = f"{where}[{i}]"
        if block["type"] == "text":
            out.append(_text(block, at))
        else:
            out.append(_render_document(block.get("document"), None, f"{at}.document"))
    return "\n".join(out)


def _tool_calls(raw: Any, where: str) -> list[dict]:
    if not isinstance(raw, list):
        refuse(f"{where} must be a list", where)
    calls = []
    for i, item in enumerate(raw):
        at = f"{where}[{i}]"
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            refuse(f"{at} needs a function object", at)
        refuse_unknown(item, {"id", "type", "function"}, where=at)
        if item.get("type", "function") != "function":
            refuse(f"{at}.type must be function", f"{at}.type")
        function = item["function"]
        refuse_unknown(function, {"name", "arguments"}, where=f"{at}.function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            refuse(f"{at}.function.name is required", f"{at}.function.name")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = "{}"
        elif isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            refuse(f"{at}.function.arguments must be a JSON string", f"{at}.function.arguments")
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            refuse(f"{at}.id is required", f"{at}.id")
        calls.append(
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        )
    return calls


def _assistant(message: dict, where: str) -> dict:
    texts = []
    content = message.get("content")
    if content is not None:
        blocks = _blocks(content, f"{where}.content", {"text", "thinking"})
        texts = [
            _text(b, f"{where}.content[{i}]") for i, b in enumerate(blocks) if b["type"] == "text"
        ]
    plan = message.get("tool_plan")
    if plan is not None:
        if not isinstance(plan, str):
            refuse(f"{where}.tool_plan must be a string", f"{where}.tool_plan")
        if plan:
            texts.append(plan)
    entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
    if message.get("tool_calls"):
        entry["tool_calls"] = _tool_calls(message["tool_calls"], f"{where}.tool_calls")
    elif entry["content"] is None:
        entry["content"] = ""
    return entry


def _messages(raw: Any, documents: dict | None) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        refuse("messages must be a non-empty list", "messages")
    out: list[dict] = []
    for index, message in enumerate(raw):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            refuse(f"{where} must be an object", where)
        role = message.get("role")
        if role not in _ROLE_FIELDS:
            refuse(
                f"{where}.role must be one of {', '.join(sorted(_ROLE_FIELDS))}", f"{where}.role"
            )
        ignored = {"citations"} if role == "assistant" else set()
        refuse_unknown(message, _ROLE_FIELDS[role], ignored=ignored, where=where)
        if role == "assistant":
            out.append(_assistant(message, where))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                refuse(f"{where}.tool_call_id is required", f"{where}.tool_call_id")
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _tool_content(message.get("content", ""), f"{where}.content"),
                }
            )
        else:
            parse = _system_content if role == "system" else _user_content
            out.append(
                {"role": role, "content": parse(message.get("content", ""), f"{where}.content")}
            )
    if documents is not None:
        leading = 0
        while leading < len(out) and out[leading]["role"] == "system":
            leading += 1
        out.insert(leading, documents)
    return out


def _tools(raw: Any, strict: Any) -> list[dict]:
    if not isinstance(raw, list):
        refuse("tools must be a list", "tools")
    if strict is not None and not isinstance(strict, bool):
        refuse("strict_tools must be a boolean", "strict_tools")
    out = []
    for i, tool in enumerate(raw):
        at = f"tools[{i}]"
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            refuse(f"{at} must be a function tool", at)
        refuse_unknown(tool, {"type", "function"}, where=at)
        if tool.get("type", "function") != "function":
            refuse(f"{at}.type must be function", f"{at}.type")
        function = dict(tool["function"])
        refuse_unknown(function, {"name", "description", "parameters"}, where=f"{at}.function")
        if strict:
            function["strict"] = True
        out.append({"type": "function", "function": function})
    return out


def _response_format(raw: Any) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        refuse("response_format must be an object", "response_format")
    refuse_unknown(raw, {"type", "json_schema"}, where="response_format")
    kind, schema = raw.get("type"), raw.get("json_schema")
    if kind == "text":
        if schema is not None:
            refuse("response_format.json_schema is not allowed with type text", "response_format")
        return None
    if kind != "json_object":
        refuse('response_format.type must be "text" or "json_object"', "response_format.type")
    if schema is None:
        return {"type": "json_object"}
    if not isinstance(schema, dict):
        refuse("response_format.json_schema must be an object", "response_format.json_schema")
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }


def _thinking(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        refuse("thinking must be an object", "thinking")
    refuse_unknown(raw, {"type", "token_budget"}, where="thinking")
    if raw.get("token_budget") is not None:
        refuse("thinking.token_budget has no chat completions equivalent", "thinking.token_budget")
    kind = raw.get("type")
    if kind == "disabled":
        return None
    if kind == "enabled":
        return "medium"
    refuse('thinking.type must be "enabled" or "disabled"', "thinking.type")


def _check(body: dict) -> None:
    """Refuse the checked fields unless their value changes nothing."""
    options = body.get("citation_options")
    if options is not None and not (
        isinstance(options, dict)
        and set(options) <= {"mode"}
        and options.get("mode") in (None, "disabled")
    ):
        refuse(
            'citation_options other than {"mode": "disabled"}: routellm returns no citations',
            "citation_options",
        )
    if body.get("safety_mode") not in (None, "CONTEXTUAL"):
        refuse("safety_mode other than CONTEXTUAL has no equivalent", "safety_mode")
    if body.get("logprobs") not in (None, False):
        refuse("logprobs are not supported", "logprobs")


def _stop(raw: Any) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        refuse("stop_sequences must be a list of strings", "stop_sequences")
    return raw


# -- response -----------------------------------------------------------------


def _finish(reason: str | None) -> str:
    return _FINISH.get(reason or "", "COMPLETE")


def _usage(prompt: int, completion: int, cached: int | None = None) -> dict:
    counts = {"input_tokens": prompt, "output_tokens": completion}
    usage: dict[str, Any] = {"billed_units": dict(counts), "tokens": dict(counts)}
    if cached is not None:
        usage["cached_tokens"] = cached
    return usage


def _sse(event: str, payload: dict) -> bytes:
    data = json.dumps({"type": event, **payload}, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n".encode()


def _tool_call_out(call_id: str | None, name: str | None, arguments: Any, n: int) -> dict:
    return {
        "id": call_id or f"call_{n}",
        "type": "function",
        "function": {"name": name or "", "arguments": arguments or "{}"},
    }


@register
class CohereAdapter(Adapter):
    name = "cohere"
    routes = (Route("POST", CHAT_PATH),)

    # -- request --------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        refuse_unknown(body, _FIELDS | _CHECKED, ignored=_IGNORED, where="request")
        _check(body)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            refuse("model is required", "model")
        stream = body.get("stream")
        if stream is not None and not isinstance(stream, bool):
            refuse("stream must be a boolean", "stream")
        fields: dict[str, Any] = {
            "model": model,
            "messages": _messages(body.get("messages"), _documents_message(body.get("documents"))),
            "stream": bool(stream),
        }
        for source, target in (
            ("max_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("p", "top_p"),
            ("seed", "seed"),
            ("frequency_penalty", "frequency_penalty"),
            ("presence_penalty", "presence_penalty"),
        ):
            if body.get(source) is not None:
                fields[target] = body[source]
        if body.get("k") is not None:
            fields["top_k"] = top_k_value(body["k"], "k")
        if body.get("stop_sequences") is not None:
            fields["stop"] = _stop(body["stop_sequences"])
        if body.get("tools"):
            fields["tools"] = _tools(body["tools"], body.get("strict_tools"))
        elif body.get("strict_tools"):
            refuse("strict_tools needs tools", "strict_tools")
        if (choice := body.get("tool_choice")) is not None:
            if choice not in _TOOL_CHOICES:
                refuse("tool_choice must be REQUIRED or NONE", "tool_choice")
            fields["tool_choice"] = _TOOL_CHOICES[choice]
        if (response_format := _response_format(body.get("response_format"))) is not None:
            fields["response_format"] = response_format
        if (effort := _thinking(body.get("thinking"))) is not None:
            fields["reasoning_effort"] = effort
        return ChatCompletionRequest(**fields)

    # -- response -------------------------------------------------------------

    def encode(self, response: dict, call: Call) -> Any:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        content: list[dict] = []
        if thinking := message.get("reasoning_content") or message.get("reasoning"):
            content.append({"type": "thinking", "thinking": thinking})
        if text := message.get("content"):
            content.append({"type": "text", "text": text})
        out_message: dict[str, Any] = {"role": "assistant", "content": content}
        calls = [
            _tool_call_out(
                c.get("id"),
                (c.get("function") or {}).get("name"),
                (c.get("function") or {}).get("arguments"),
                n,
            )
            for n, c in enumerate(message.get("tool_calls") or [])
        ]
        if calls:
            out_message["tool_calls"] = calls
        return {
            "id": response.get("id") or uuid.uuid4().hex,
            "finish_reason": _finish(choice.get("finish_reason")),
            "message": out_message,
            "usage": _usage(
                usage.get("prompt_tokens") or 0,
                usage.get("completion_tokens") or 0,
                details.get("cached_tokens"),
            ),
        }

    async def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        message_id: str | None = None
        started = False
        block: str | None = None  # the open content block: text or thinking
        content_index = -1
        opened: set[int] = set()  # tool calls with a tool-call-start sent
        finish: str | None = None
        usage: Usage | None = None
        tool_calls = False

        def start() -> list[bytes]:
            nonlocal message_id, started
            if started:
                return []
            started = True
            message_id = message_id or uuid.uuid4().hex
            empty = {"role": "assistant", "content": [], "tool_plan": "", "tool_calls": []}
            empty["citations"] = []
            return [_sse("message-start", {"id": message_id, "delta": {"message": empty}})]

        def close() -> list[bytes]:
            nonlocal block
            if block is None:
                return []
            block = None
            return [_sse("content-end", {"index": content_index})]

        def open_block(kind: str) -> list[bytes]:
            nonlocal block, content_index
            if block == kind:
                return []
            frames = close()
            block, content_index = kind, content_index + 1
            first = {"type": kind, kind: ""}
            frames.append(
                _sse(
                    "content-start",
                    {"index": content_index, "delta": {"message": {"content": first}}},
                )
            )
            return frames

        def open_tool(index: int, call_id: str | None, name: str | None) -> list[bytes]:
            if index in opened:
                return []
            opened.add(index)
            tool = _tool_call_out(call_id, name, "", index)
            tool["function"]["arguments"] = ""
            return [
                *close(),
                _sse(
                    "tool-call-start", {"index": index, "delta": {"message": {"tool_calls": tool}}}
                ),
            ]

        def tool_delta(index: int, arguments: str) -> list[bytes]:
            piece = {"function": {"arguments": arguments}}
            return [
                _sse(
                    "tool-call-delta", {"index": index, "delta": {"message": {"tool_calls": piece}}}
                )
            ]

        def end(error: str | None = None) -> list[bytes]:
            delta: dict[str, Any] = {"finish_reason": "ERROR" if error else _finish(finish)}
            if error is not None:
                delta["error"] = error
            elif usage is not None:
                delta["usage"] = _usage(
                    usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens
                )
            return [*start(), *close(), _sse("message-end", {"id": message_id, "delta": delta})]

        async for event in events:
            frames: list[bytes] = []
            if isinstance(event, Start):
                message_id = event.id
                frames = start()
            elif isinstance(event, TextDelta):
                frames = [*start(), *open_block("text")]
                frames.append(
                    _sse(
                        "content-delta",
                        {
                            "index": content_index,
                            "delta": {"message": {"content": {"text": event.text}}},
                        },
                    )
                )
            elif isinstance(event, ReasoningDelta):
                frames = [*start(), *open_block("thinking")]
                frames.append(
                    _sse(
                        "content-delta",
                        {
                            "index": content_index,
                            "delta": {"message": {"content": {"thinking": event.text}}},
                        },
                    )
                )
            elif isinstance(event, ToolCallStart):
                tool_calls = True
                frames = [*start(), *open_tool(event.call_index, event.id, event.name)]
            elif isinstance(event, ToolCallDelta):
                tool_calls = True
                frames = [*start(), *open_tool(event.call_index, None, None)]
                if event.arguments:
                    frames += tool_delta(event.call_index, event.arguments)
            elif isinstance(event, ToolCallEnd):
                tool_calls = True
                frames = [*start(), *open_tool(event.call_index, event.id, event.name)]
                frames.append(_sse("tool-call-end", {"index": event.call_index}))
            elif isinstance(event, Finish):
                finish = event.reason
            elif isinstance(event, Usage):
                usage = event
            elif isinstance(event, Error):
                for frame in end(event.message):
                    yield frame
                return
            elif isinstance(event, End):
                finish = finish or event.finish_reason or ("tool_calls" if tool_calls else None)
                frames = end()
            for frame in frames:
                yield frame

    def encode_error(self, status: int, error: dict, call: Call) -> Any:
        out = {"message": error["message"]}
        if error.get("request_id"):
            out["id"] = error["request_id"]
        return out

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return _sse("message-end", {"delta": {"finish_reason": "ERROR", "error": error["message"]}})
