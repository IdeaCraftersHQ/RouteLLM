"""Ollama native API: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/version`.

For tools that speak Ollama's own API rather than its OpenAI-compatible
one. Inbound only: whatever routellm routes to upstream (a real Ollama
included) is independent of this dialect.

The mapping inverts the outbound one litellm does for an Ollama upstream
(`litellm/llms/ollama`: `num_predict` <-> `max_tokens`, `format` <->
`response_format`, `think` <-> `reasoning_effort`, `images` <-> `image_url`
parts, object `arguments` <-> JSON string `arguments`); the wire shapes
follow Ollama's API docs.

Auth: the credential is the bearer token (clients that set a header, such
as `ollama.Client(headers=...)`). Errors are `{"error": "<message>"}`.

Requests
--------
`/api/chat`:
  mapped    model, messages (role, content, images, tool_calls, tool_name,
            tool_call_id), tools, format, options, stream, think
  ignored   keep_alive; per message `thinking` (earlier reasoning is not
            replayed upstream)
`/api/generate`:
  mapped    model, prompt, system, images, format, options, stream, think,
            raw (see below)
  ignored   keep_alive
Both: `stream` defaults to true, as in Ollama.
`options`:
  mapped    temperature, top_p, seed, stop (string or list), num_predict
            (> 0 is `max_tokens`; -1 and -2 mean no limit), plus
            frequency_penalty and presence_penalty
  ignored   resource knobs with no effect on the answer: num_ctx, num_batch,
            num_gpu, main_gpu, num_thread, num_keep, use_mmap, use_mlock,
            numa, low_vram, f16_kv, vocab_only
  refused   every other option, top_k included: chat completions has no
            such field, and dropping a sampling parameter changes the answer
`format`: "json" is `response_format` json_object; a JSON schema object is
  `response_format` json_schema. Anything else is refused.
`think`: true is `reasoning_effort: medium`, "low"/"medium"/"high" pass
  through, false sets nothing.
Model names: a trailing `:latest` is dropped before routing, and the
client's own spelling is echoed in answers.
Images: base64 strings (or data URLs) of png, jpeg, gif or webp; the type
is read from the bytes.
Tool calls: Ollama calls carry no id. A call with no `id` gets a stable
one (`call_<message>_<position>`); a `tool` message is paired with an
earlier unanswered call by `tool_call_id`, else by `tool_name`, else the
oldest unanswered call; one with no call to pair is refused.
`raw: true` sends the prompt as the only user message and ignores
`system`, as Ollama does; the chat template still applies upstream, so
this is as close to raw as chat completions gets.
  refused   any other top-level field (generate: suffix, template, context;
            chat/generate: logprobs, ...), unknown message keys, an empty
            prompt or message list (Ollama's "load the model" call)

Responses
---------
Chat answers are `message` objects, generate answers a `response` string;
`done` and `done_reason` (`stop`, or `length`; tool calls end in `stop`,
as in Ollama). Streams are NDJSON: text, reasoning (`thinking`) and tool
calls as they arrive with `done: false`, then one `done: true` line with
the counts. Tool calls carry `id`. `prompt_eval_count` / `eval_count` are
the upstream's usage. Durations are best effort, measured by routellm:
`total_duration` and `eval_duration` are the wall time of the request,
`load_duration` and `prompt_eval_duration` are 0. A failure after the
stream started is a final `{"error": ...}` line, never a `done` line.

`/api/tags` and `/api/show` list the server's tiers and routers as models;
`/api/show` of an unlisted model is a 404. `/api/version` reports the
Ollama API level this dialect follows, not routellm's own version.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Reply,
    Route,
    refuse,
    refuse_unknown,
    register,
)
from routellm.adapters.stream import (
    End,
    Error,
    Finish,
    ReasoningDelta,
    TextDelta,
    ToolCallEnd,
    Usage,
)

__all__ = ["OLLAMA_API_VERSION", "OllamaAdapter"]

#: The Ollama release whose API this dialect follows.
OLLAMA_API_VERSION = "0.12.6"

CHAT_PATH = "/api/chat"
GENERATE_PATH = "/api/generate"

_CHAT_FIELDS = {"model", "messages", "tools", "format", "options", "stream", "think"}
_GENERATE_FIELDS = {"model", "prompt", "system", "images", "format", "options", "stream", "think"}
_COMMON_IGNORED = {"keep_alive"}
_GENERATE_ONLY = {"raw"}
_MESSAGE_FIELDS = {"role", "content", "images", "tool_calls", "tool_name", "tool_call_id"}
_ROLES = {"system", "user", "assistant", "tool"}

_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}
_IGNORED_OPTIONS = {
    "num_ctx",
    "num_batch",
    "num_gpu",
    "main_gpu",
    "num_thread",
    "num_keep",
    "use_mmap",
    "use_mlock",
    "numa",
    "low_vram",
    "f16_kv",
    "vocab_only",
}
_EFFORTS = {"low", "medium", "high"}

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_DETAILS = {
    "parent_model": "",
    "format": "routellm",
    "family": "routellm",
    "families": ["routellm"],
    "parameter_size": "",
    "quantization_level": "",
}


# -- request ------------------------------------------------------------------


def _model(name: Any) -> str:
    if not isinstance(name, str) or not name:
        refuse("model is required", "model")
    return name.removesuffix(":latest") or name


def _sniff(raw: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image(value: Any, where: str) -> dict:
    if not isinstance(value, str):
        refuse(f"{where} must be a base64 string", where)
    data = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        refuse(f"{where} is not base64", where)
    mime = _sniff(raw)
    if mime is None:
        refuse(f"{where} is not a png, jpeg, gif or webp image", where)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _content(text: Any, images: Any, where: str) -> str | list[dict]:
    if text is None:
        text = ""
    if not isinstance(text, str):
        refuse(f"{where}.content must be a string", f"{where}.content")
    if not images:
        return text
    if not isinstance(images, list):
        refuse(f"{where}.images must be a list", f"{where}.images")
    parts = [{"type": "text", "text": text}] if text else []
    parts += [_image(v, f"{where}.images[{i}]") for i, v in enumerate(images)]
    return parts


def _arguments(value: Any, where: str) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            refuse(f"{where}.arguments must be a JSON object", f"{where}.arguments")
    if not isinstance(value, dict):
        refuse(f"{where}.arguments must be a JSON object", f"{where}.arguments")
    return json.dumps(value)


def _tool_calls(raw: Any, index: int, pending: list[tuple[str, str]]) -> list[dict]:
    where = f"messages[{index}].tool_calls"
    if not isinstance(raw, list):
        refuse(f"{where} must be a list", where)
    calls = []
    for position, item in enumerate(raw):
        at = f"{where}[{position}]"
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            refuse(f"{at} needs a function object", at)
        refuse_unknown(item, {"id", "type", "function"}, where=at)
        function = item["function"]
        refuse_unknown(function, {"name", "arguments", "index"}, where=f"{at}.function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            refuse(f"{at}.function.name is required", f"{at}.function.name")
        call_id = item.get("id") or f"call_{index}_{position}"
        pending.append((call_id, name))
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": _arguments(function.get("arguments"), at)},
            }
        )
    return calls


def _tool_message(message: dict, index: int, pending: list[tuple[str, str]]) -> dict:
    where = f"messages[{index}]"
    wanted_id, name = message.get("tool_call_id"), message.get("tool_name")
    for n, (call_id, call_name) in enumerate(pending):
        if (wanted_id and call_id == wanted_id) or (
            not wanted_id and (not name or call_name == name)
        ):
            del pending[n]
            call_id_ = call_id
            break
    else:
        if not wanted_id:
            refuse(f"{where}: tool message has no unanswered tool call to answer", where)
        call_id_ = wanted_id
    return {
        "role": "tool",
        "tool_call_id": call_id_,
        "content": _content(message.get("content"), None, where),
    }


def _chat_messages(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        refuse("messages must be a non-empty list", "messages")
    out: list[dict] = []
    pending: list[tuple[str, str]] = []
    for index, message in enumerate(raw):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            refuse(f"{where} must be an object", where)
        ignored = {"thinking"}
        refuse_unknown(message, _MESSAGE_FIELDS, ignored=ignored, where=where)
        role = message.get("role")
        if role not in _ROLES:
            refuse(f"{where}.role must be one of {', '.join(sorted(_ROLES))}", f"{where}.role")
        if role == "tool":
            out.append(_tool_message(message, index, pending))
            continue
        if message.get("images") and role != "user":
            refuse(f"{where}: images are accepted on user messages only", f"{where}.images")
        entry: dict[str, Any] = {
            "role": role,
            "content": _content(message.get("content"), message.get("images"), where),
        }
        if message.get("tool_calls"):
            if role != "assistant":
                refuse(f"{where}: tool_calls belong to assistant messages", f"{where}.tool_calls")
            entry["tool_calls"] = _tool_calls(message["tool_calls"], index, pending)
            if entry["content"] == "":
                entry["content"] = None
        out.append(entry)
    return out


def _options(raw: Any) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        refuse("options must be an object", "options")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _OPTIONS:
            if value is not None:
                out[_OPTIONS[key]] = value
        elif key == "num_predict":
            if isinstance(value, bool) or not isinstance(value, int) or value == 0:
                refuse("options.num_predict must be a non-zero integer", "options.num_predict")
            if value > 0:
                out["max_tokens"] = value
        elif key not in _IGNORED_OPTIONS:
            refuse(
                f"Unsupported option options.{key}: chat completions has no equivalent",
                f"options.{key}",
            )
    return out


def _format(value: Any) -> dict | None:
    if value in (None, ""):
        return None
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": value, "strict": True},
        }
    refuse('format must be "json" or a JSON schema object', "format")


def _think(value: Any) -> str | None:
    if value in (None, False):
        return None
    if value is True:
        return "medium"
    if isinstance(value, str) and value in _EFFORTS:
        return value
    refuse('think must be a boolean or "low", "medium" or "high"', "think")


def _generate_messages(body: dict) -> list[dict]:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        refuse("prompt must be a non-empty string (preloading a model is not supported)", "prompt")
    messages = []
    system = body.get("system")
    if system and not body.get("raw"):
        if not isinstance(system, str):
            refuse("system must be a string", "system")
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _content(prompt, body.get("images"), "prompt")})
    return messages


# -- response -----------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _line(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode() + b"\n"


def _done_reason(finish: str | None) -> str:
    return "length" if finish == "length" else "stop"


def _tool_call_out(call_id: str | None, name: str | None, arguments: Any) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except ValueError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    call = {"function": {"name": name or "", "arguments": arguments}}
    return {"id": call_id, **call} if call_id else call


def _frame(call: Call, text: str = "", **extra: Any) -> dict:
    """One answer object (a whole answer or a stream line) of the call's endpoint."""
    base = {"model": call.state["model"], "created_at": _now()}
    if call.state["mode"] == "generate":
        thinking = extra.pop("thinking", None)
        tool_calls = extra.pop("tool_calls", None)  # generate has none to carry
        del tool_calls
        out = {**base, "response": text}
        if thinking:
            out["thinking"] = thinking
        return {**out, **extra}
    message: dict[str, Any] = {"role": "assistant", "content": text}
    for key in ("thinking", "tool_calls"):
        if extra.get(key):
            message[key] = extra[key]
        extra.pop(key, None)
    return {**base, "message": message, **extra}


def _final(call: Call, finish: str | None, prompt: int, completion: int, **content: Any) -> dict:
    elapsed = int((time.monotonic() - call.state["started"]) * 1e9)
    return _frame(
        call,
        **content,
        done=True,
        done_reason=_done_reason(finish),
        total_duration=elapsed,
        load_duration=0,
        prompt_eval_count=prompt,
        prompt_eval_duration=0,
        eval_count=completion,
        eval_duration=elapsed,
    )


def _entry(model: dict) -> dict:
    name = model["id"]
    created = datetime.fromtimestamp(model.get("created") or 0, UTC)
    return {
        "name": name,
        "model": name,
        "modified_at": created.isoformat().replace("+00:00", "Z"),
        "size": 0,
        "digest": hashlib.sha256(name.encode()).hexdigest(),
        "details": dict(_DETAILS),
    }


@register
class OllamaAdapter(Adapter):
    name = "ollama"
    routes = (
        Route("POST", CHAT_PATH),
        Route("POST", GENERATE_PATH),
        Route("GET", "/api/tags", operation="tags", body="none"),
        Route("POST", "/api/show", operation="show"),
        Route("GET", "/api/version", operation="version", body="none"),
    )

    # -- request --------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object")
        generate = call.route.path == GENERATE_PATH
        supported = (_GENERATE_FIELDS | _GENERATE_ONLY) if generate else _CHAT_FIELDS
        refuse_unknown(
            body, supported, ignored=_COMMON_IGNORED, where="generate" if generate else "chat"
        )
        call.state.update(
            mode="generate" if generate else "chat",
            model=body.get("model"),
            started=time.monotonic(),
        )
        fields: dict[str, Any] = {
            "model": _model(body.get("model")),
            "messages": _generate_messages(body)
            if generate
            else _chat_messages(body.get("messages")),
            "stream": body.get("stream") is not False,
            **_options(body.get("options")),
        }
        if not generate and body.get("tools"):
            fields["tools"] = self._tools(body["tools"])
        if (response_format := _format(body.get("format"))) is not None:
            fields["response_format"] = response_format
        if (effort := _think(body.get("think"))) is not None:
            fields["reasoning_effort"] = effort
        return ChatCompletionRequest(**fields)

    @staticmethod
    def _tools(raw: Any) -> list[dict]:
        if not isinstance(raw, list):
            refuse("tools must be a list", "tools")
        for i, tool in enumerate(raw):
            if (
                not isinstance(tool, dict)
                or tool.get("type", "function") != "function"
                or not isinstance(tool.get("function"), dict)
            ):
                refuse(f"tools[{i}] must be a function tool", f"tools[{i}]")
        return [{**tool, "type": "function"} for tool in raw]

    # -- custom routes --------------------------------------------------------

    async def handle(self, call: Call) -> Any:
        models = await call.models()
        if call.route.operation == "version":
            return {"version": OLLAMA_API_VERSION}
        if call.route.operation == "tags":
            return {"models": [_entry(m) for m in models]}
        refuse_unknown(call.body or {}, {"model", "name"}, ignored={"verbose"}, where="show")
        name = (call.body or {}).get("model") or (call.body or {}).get("name")
        if not isinstance(name, str) or not name:
            refuse("model is required", "model")
        wanted = _model(name)
        found = next((m for m in models if m["id"] == wanted), None)
        if found is None:
            return Reply({"error": f"model '{name}' not found"}, status=404)
        return {
            "license": "",
            "modelfile": f"# routellm\nFROM {found['id']}\n",
            "parameters": "",
            "template": "",
            "details": dict(_DETAILS),
            "model_info": {"general.architecture": "routellm", "routellm.id": found["id"]},
            "capabilities": ["completion", "tools", "vision"],
            "modified_at": _entry(found)["modified_at"],
        }

    # -- response -------------------------------------------------------------

    def encode(self, response: dict, call: Call) -> Any:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        calls = [
            _tool_call_out(
                c.get("id"),
                (c.get("function") or {}).get("name"),
                (c.get("function") or {}).get("arguments"),
            )
            for c in message.get("tool_calls") or []
        ]
        return _final(
            call,
            choice.get("finish_reason"),
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            text=message.get("content") or "",
            thinking=message.get("reasoning_content") or message.get("reasoning"),
            tool_calls=calls,
        )

    def stream_media_type(self, call: Call) -> str:
        return "application/x-ndjson"

    async def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        finish: str | None = None
        prompt = completion = 0
        async for event in events:
            if isinstance(event, TextDelta):
                yield _line(_frame(call, event.text, done=False))
            elif isinstance(event, ReasoningDelta):
                yield _line(_frame(call, thinking=event.text, done=False))
            elif isinstance(event, ToolCallEnd):
                tool = _tool_call_out(event.id, event.name, event.arguments)
                yield _line(_frame(call, tool_calls=[tool], done=False))
            elif isinstance(event, Finish):
                finish = event.reason
            elif isinstance(event, Usage):
                prompt, completion = event.prompt_tokens, event.completion_tokens
            elif isinstance(event, Error):
                yield _line({"error": event.message})
                return
            elif isinstance(event, End):
                yield _line(_final(call, finish, prompt, completion))

    def encode_error(self, status: int, error: dict, call: Call) -> Any:
        return {"error": error["message"]}

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return _line({"error": error["message"]})
