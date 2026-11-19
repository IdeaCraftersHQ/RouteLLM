"""Google Gemini dialect: `generateContent`, `streamGenerateContent`, `countTokens`, models.

For `gemini-cli` and the `google-genai` SDK. The Gemini <-> chat mapping is
litellm's (`litellm.google_genai.adapters`): `GoogleGenAIAdapter` translates
the request and the answer, `GoogleGenAIStreamWrapper` the stream. This
module is glue: it validates the body (litellm silently drops what it does
not know; here that is a refusal), calls the translator, and adds the few
things litellm lacks. `test_gemini_generate.py` pins the litellm behaviour
relied on, so an upgrade that changes it fails there.

Routes (also under `/v1`; `GET /v1/models` stays the OpenAI model list):
`POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`
(`?alt=sse` for SSE, otherwise a JSON array), `:countTokens`;
`GET /v1beta/models`, `GET /v1beta/models/{model}`.
Auth: `x-goog-api-key` header or `?key=`. The model in the path carries
routing parameters, as the model field does on `/v1/chat/completions`.
Errors: `{"error": {"code", "message", "status"}}`.

What litellm lacks and this module adds: camelCase `inlineData` (litellm
reads only `inline_data`) and URL-safe base64 made standard; the call id on
`functionCall` parts, streamed or not; unique tool call ids (litellm names
every call `call_<name>`) with the client's ids kept; the real usage on the
stream's last chunk (litellm reads it from a chunk that has none);
`presencePenalty`, `frequencyPenalty`, `seed`; JSON mode without a schema;
a single `allowedFunctionNames`; an unwrapped `{"output": text}` tool
result; `countTokens`; models; errors.

Mapped
    contents[].parts: text, inlineData (image/*), functionCall,
    functionResponse; roles user, model, function; systemInstruction (text
    parts joined); tools[].functionDeclarations (name, description,
    parameters, parametersJsonSchema); toolConfig.functionCallingConfig
    (mode AUTO/ANY/NONE, one allowedFunctionNames); generationConfig:
    temperature, topP, topK, maxOutputTokens, stopSequences, responseMimeType
    (text/plain, application/json), responseSchema / responseJsonSchema
    (object), presencePenalty, frequencyPenalty, seed, candidateCount=1.
    Answers: finishReason (STOP, MAX_TOKENS, SAFETY), usageMetadata
    (prompt, candidates, total), modelVersion.
    snake_case spellings of the structural names are accepted.
Accepted, no effect
    safetySettings, generationConfig.thinkingConfig, generationConfig.mediaResolution,
    parts' thoughtSignature, `thought: true` parts (reasoning the model
    produced earlier is not sent back).
Refused (400 INVALID_ARGUMENT)
    every other body, generationConfig, tool, tool config, content or part
    field: cachedContent, labels, fileData, executableCode, googleSearch
    and other built-in tools, responseLogprobs, responseModalities,
    candidateCount > 1, non-image inlineData, several allowedFunctionNames,
    non-object response schemas, `alt` other than `sse`/`json`.
Limits
    A user turn mixing functionResponse and other parts is sent as the user
    message then the tool messages, in that order. Streamed tool call ids
    are paired with the calls in the order the upstream opened them.
    `countTokens` is an estimate (litellm's default tokenizer), not the
    routed model's count. Model listing ignores `pageSize`/`pageToken`.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import litellm
from litellm.google_genai.adapters.transformation import (
    GoogleGenAIAdapter,
    GoogleGenAIStreamWrapper,
)
from litellm.types.utils import ModelResponse, ModelResponseStream

from routellm.adapters import (
    Adapter,
    AdapterRefusal,
    Call,
    ChatCompletionRequest,
    Route,
    refuse,
    refuse_unknown,
    register,
    top_k_value,
)

__all__ = ["GeminiAdapter"]

_TRANSLATOR = GoogleGenAIAdapter()

_VERSIONS = ("/v1beta", "/v1")
_METHODS = ["generateContent", "streamGenerateContent", "countTokens"]

_BODY = {"contents", "systemInstruction", "tools", "toolConfig", "generationConfig"}
_BODY_IGNORED = {"safetySettings"}
_GENERATION = {
    "temperature",
    "topP",
    "topK",
    "maxOutputTokens",
    "stopSequences",
    "responseMimeType",
    "responseSchema",
    "responseJsonSchema",
    "presencePenalty",
    "frequencyPenalty",
    "seed",
    "candidateCount",
}
_GENERATION_IGNORED = {"thinkingConfig", "mediaResolution"}
_PART = {"text", "inlineData", "functionCall", "functionResponse"}
_PART_IGNORED = {"thought", "thoughtSignature"}
_ROLES = {None: "user", "user": "user", "function": "user", "model": "model"}
_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    409: "ABORTED",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    501: "UNIMPLEMENTED",
    502: "UNAVAILABLE",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}
_RESULT_KEYS = ("output", "result")


def _camel(key: str) -> str:
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), key)


def _object(value: Any, where: str) -> dict:
    """`value` as a dict with camelCase keys; anything else is refused."""
    if not isinstance(value, dict):
        refuse(f"{where} must be an object", param=where)
    return {_camel(k): v for k, v in value.items()}


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        refuse(f"{where} must be a list", param=where)
    return value


def _text_parts(content: dict, where: str) -> str:
    parts = _list(_object(content, where).get("parts", []), f"{where}.parts")
    texts = []
    for raw in parts:
        part = _object(raw, f"{where}.parts")
        if set(part) - {"text"}:
            refuse(f"{where} carries text parts only", param=where)
        texts.append(part.get("text", ""))
    return "\n".join(texts)


class _Contents:
    """The request's contents, checked and renamed for litellm, plus what litellm drops.

    `calls` and `results` list the client's tool call ids (or None) and
    the result payloads in order; they are matched to litellm's messages
    afterwards.
    """

    def __init__(self, contents: Any):
        self.items: list[dict] = []
        self.calls: list[str | None] = []
        self.results: list[tuple[str, str | None, Any]] = []
        contents = _list(contents, "contents")
        if not contents:
            refuse("contents must not be empty", param="contents")
        for content in contents:
            self._content(_object(content, "contents"))

    def _content(self, content: dict) -> None:
        refuse_unknown(content, {"role", "parts"}, where="contents")
        role = content.get("role")
        if role not in _ROLES:
            refuse(f"Unsupported role {role!r} in contents", param="contents.role")
        role = _ROLES[role]
        parts = []
        for raw in _list(content.get("parts", []), "contents.parts"):
            part = _object(raw, "contents.parts")
            refuse_unknown(part, _PART, ignored=_PART_IGNORED, where="a part")
            if part.get("thought"):
                continue
            kept = self._part(part, role)
            if kept:
                parts.append(kept)
        self.items.append({"role": role, "parts": parts})

    def _part(self, part: dict, role: str) -> dict | None:
        if "text" in part:
            return {"text": part["text"]}
        if "inlineData" in part:
            if role != "user":
                refuse("inlineData is accepted in user turns only", param="inlineData")
            blob = _object(part["inlineData"], "inlineData")
            mime = blob.get("mimeType") or ""
            if not mime.startswith("image/"):
                refuse(f"inlineData supports image/* only, got {mime!r}", param="inlineData")
            # The SDK sends URL-safe base64; a data URL carries the standard alphabet.
            data = str(blob.get("data", "")).replace("-", "+").replace("_", "/")
            return {"inline_data": {"mime_type": mime, "data": data}}
        if "functionCall" in part:
            if role != "model":
                refuse("functionCall is accepted in model turns only", param="functionCall")
            call = _object(part["functionCall"], "functionCall")
            self.calls.append(call.get("id"))
            return {
                "functionCall": {"name": _name(call, "functionCall"), "args": call.get("args", {})}
            }
        if "functionResponse" in part:
            if role != "user":
                refuse("functionResponse is accepted in user turns only", param="functionResponse")
            result = _object(part["functionResponse"], "functionResponse")
            name, response = _name(result, "functionResponse"), result.get("response", {})
            self.results.append((name, result.get("id"), response))
            return {"functionResponse": {"name": name, "response": response}}
        return None


def _name(body: dict, where: str) -> str:
    if not body.get("name"):
        refuse(f"{where} needs a name", param=where)
    return body["name"]


def _tools(tools: Any) -> list[dict]:
    out = []
    for raw in _list(tools, "tools"):
        tool = _object(raw, "tools")
        refuse_unknown(tool, {"functionDeclarations"}, where="a tool")
        declarations = []
        for entry in _list(tool.get("functionDeclarations", []), "functionDeclarations"):
            declaration = _object(entry, "functionDeclarations")
            refuse_unknown(
                declaration,
                {"name", "description", "parameters", "parametersJsonSchema"},
                where="a function declaration",
            )
            _name(declaration, "a function declaration")
            declarations.append(declaration)
        out.append({"functionDeclarations": declarations})
    return out


def _tool_config(config: Any) -> tuple[dict, str | None]:
    """The config for litellm and the single function an `ANY` mode is pinned to."""
    config = _object(config, "toolConfig")
    refuse_unknown(config, {"functionCallingConfig"}, where="toolConfig")
    calling = _object(config.get("functionCallingConfig", {}), "functionCallingConfig")
    refuse_unknown(calling, {"mode", "allowedFunctionNames"}, where="functionCallingConfig")
    mode = calling.get("mode") or "AUTO"
    if mode == "MODE_UNSPECIFIED":
        mode = "AUTO"
    if mode not in ("AUTO", "ANY", "NONE"):
        refuse(f"Unsupported function calling mode {mode!r}", param="toolConfig.mode")
    allowed = calling.get("allowedFunctionNames") or []
    if len(allowed) > 1:
        refuse("allowedFunctionNames with several names is not supported", param="toolConfig")
    pinned = allowed[0] if allowed and mode == "ANY" else None
    return {"functionCallingConfig": {"mode": mode}}, pinned


def _generation(config: Any) -> dict:
    config = _object(config, "generationConfig")
    refuse_unknown(config, _GENERATION, ignored=_GENERATION_IGNORED, where="generationConfig")
    if config.get("candidateCount") not in (None, 1):
        refuse("candidateCount other than 1 is not supported", param="candidateCount")
    mime = config.get("responseMimeType")
    if mime not in (None, "text/plain", "application/json"):
        refuse(f"Unsupported responseMimeType {mime!r}", param="responseMimeType")
    return config


def _link_tool_ids(messages: list[dict], contents: _Contents) -> None:
    """Give every tool call a unique id and pair each result with its call.

    litellm names a call `call_<name>` and a result by the same rule, which
    breaks two calls of one function. The client's ids win; otherwise ids
    are generated and results pair with the earliest unanswered call of
    their name. A sole `output`/`result` string is the tool's text.
    """
    taken = {i for i in contents.calls if i} | {i for _, i, _ in contents.results if i}
    pending: dict[str, list[str]] = {}
    calls, results = iter(contents.calls), iter(contents.results)

    def fresh(name: str) -> str:
        candidate, n = f"call_{name}", 1
        while candidate in taken:
            n += 1
            candidate = f"call_{name}_{n}"
        taken.add(candidate)
        return candidate

    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                name = call["function"]["name"]
                call["id"] = next(calls) or fresh(name)
                pending.setdefault(name, []).append(call["id"])
        elif message["role"] == "tool":
            name, given, response = next(results)
            queue = pending.get(name, [])
            if given:
                message["tool_call_id"] = given
                if given in queue:
                    queue.remove(given)
            else:
                message["tool_call_id"] = queue.pop(0) if queue else fresh(name)
            if isinstance(response, dict) and len(response) == 1:
                key, value = next(iter(response.items()))
                if key in _RESULT_KEYS and isinstance(value, str):
                    message["content"] = value


def _chat_request(model: str, body: Any) -> ChatCompletionRequest:
    """A generateContent body as a chat completion request, via litellm."""
    body = _object(body, "request body")
    refuse_unknown(body, _BODY, ignored=_BODY_IGNORED, where="generateContent request")
    contents = _Contents(body.get("contents"))
    extra: dict[str, Any] = {}
    if body.get("systemInstruction") is not None:
        extra["systemInstruction"] = {
            "parts": [{"text": _text_parts(body["systemInstruction"], "systemInstruction")}]
        }
    if body.get("tools"):
        extra["tools"] = _tools(body["tools"])
    pinned = None
    if body.get("toolConfig") is not None:
        extra["toolConfig"], pinned = _tool_config(body["toolConfig"])
    config = _generation(body.get("generationConfig", {}))

    request = _TRANSLATOR.translate_generate_content_to_completion(
        model=model, contents=contents.items, config=config, **extra
    )
    _link_tool_ids(request["messages"], contents)

    # litellm's translator drops topK, so it is mapped here.
    if config.get("topK") is not None:
        request["top_k"] = top_k_value(config["topK"], "topK")

    for source, target in (
        ("presencePenalty", "presence_penalty"),
        ("frequencyPenalty", "frequency_penalty"),
        ("seed", "seed"),
    ):
        if config.get(source) is not None:
            request[target] = config[source]
    if pinned:
        request["tool_choice"] = {"type": "function", "function": {"name": pinned}}
    if config.get("responseMimeType") == "application/json" and "response_format" not in request:
        if config.get("responseSchema") or config.get("responseJsonSchema"):
            refuse("responseSchema must describe an object", param="responseSchema")
        request["response_format"] = {"type": "json_object"}
    return ChatCompletionRequest(**request)


def _usage(raw: dict | None) -> dict:
    raw = raw or {}
    prompt, completion = raw.get("prompt_tokens") or 0, raw.get("completion_tokens") or 0
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": completion,
        "totalTokenCount": raw.get("total_tokens") or prompt + completion,
    }


def _number(parts: list[dict], ids: list[str | None]) -> None:
    """Put the upstream call ids on the `functionCall` parts, in order."""
    calls = (p["functionCall"] for p in parts if "functionCall" in p)
    for call, id_ in zip(calls, ids, strict=False):
        if id_:
            call["id"] = id_


def _error(status: int, message: str) -> dict:
    code = _STATUS.get(status, "INTERNAL" if status >= 500 else "INVALID_ARGUMENT")
    return {"error": {"code": status, "message": message, "status": code}}


def _estimate(request: ChatCompletionRequest) -> int:
    try:
        return int(litellm.token_counter(messages=request.messages, tools=request.tools))
    except Exception:
        return max(1, len(json.dumps(request.messages)) // 4)


def _model(name: str) -> dict:
    return {
        "name": f"models/{name}",
        "baseModelId": name,
        "version": "routellm",
        "displayName": name,
        "supportedGenerationMethods": _METHODS,
    }


@register
class GeminiAdapter(Adapter):
    name = "gemini"
    routes = tuple(
        route
        for version in _VERSIONS
        for route in (
            Route("POST", f"{version}/models/{{model:path}}:generateContent"),
            Route("POST", f"{version}/models/{{model:path}}:streamGenerateContent"),
            Route("POST", f"{version}/models/{{model:path}}:countTokens", operation="count"),
            Route("GET", f"{version}/models/{{model:path}}", operation="model", body="none"),
            # `GET /v1/models` is the OpenAI model list.
            *(
                (Route("GET", f"{version}/models", operation="models", body="none"),)
                if version == "/v1beta"
                else ()
            ),
        )
    )

    def credential(self, headers, query):
        return headers.get("x-goog-api-key") or query.get("key")

    # -- request -----------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        model = _model_name(call)
        streaming = call.route.path.endswith(":streamGenerateContent")
        alt = call.query_params.get("alt")
        if alt not in (None, "sse", "json"):
            refuse(f"Unsupported alt {alt!r}: use sse or json", param="alt")
        call.state.update(model=model, sse=alt == "sse")
        request = _chat_request(model, call.body)
        return request.model_copy(update={"stream": True}) if streaming else request

    async def handle(self, call: Call):
        operation = call.route.operation
        if operation == "count":
            return self._count(call)
        names = sorted({m["id"] for m in await call.models()})
        if operation == "models":
            return {"models": [_model(n) for n in names]}
        name = _model_name(call)
        if name not in names:
            raise AdapterRefusal(
                f"Model {name!r} is not found", param="model", status=404, code="not_found"
            )
        return _model(name)

    def _count(self, call: Call) -> dict:
        body = _object(call.body, "request body")
        refuse_unknown(body, {"contents", "generateContentRequest"}, where="countTokens request")
        if ("contents" in body) == ("generateContentRequest" in body):
            refuse("Send either contents or generateContentRequest", param="contents")
        if "contents" in body:
            request = _chat_request(_model_name(call), {"contents": body["contents"]})
        else:
            inner = _object(body["generateContentRequest"], "generateContentRequest")
            inner.pop("model", None)
            request = _chat_request(_model_name(call), inner)
        return {"totalTokens": _estimate(request)}

    # -- response ----------------------------------------------------------

    def encode(self, response: dict, call: Call):
        data = {k: v for k, v in response.items() if k != "routellm"}
        out = _TRANSLATOR.translate_completion_to_generate_content(ModelResponse(**data))
        message = data["choices"][0].get("message") or {}
        ids = [c.get("id") for c in message.get("tool_calls") or []]
        _number(out["candidates"][0]["content"]["parts"], ids)
        out["modelVersion"] = call.state["model"]
        return out

    def stream_media_type(self, call: Call) -> str:
        return "text/event-stream" if call.state["sse"] else "application/json"

    async def encode_stream(self, chunks: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        seen = _Seen()
        wrapper = GoogleGenAIStreamWrapper(seen.tee(chunks))
        opened, pending = False, None
        finish = None
        async for item in wrapper:
            if pending is not None:
                yield self._frame(call, pending, first=not opened)
                opened = True
            candidate = item["candidates"][0]
            finish = candidate.pop("finishReason", None) or finish
            item.pop("usageMetadata", None)
            _number(candidate["content"]["parts"], seen.ids[seen.numbered :])
            seen.numbered += sum("functionCall" in p for p in candidate["content"]["parts"])
            pending = item
        # litellm's wrapper ends its stream on any upstream failure; the
        # framework needs to see it to send the error frame.
        if seen.failure is not None:
            raise seen.failure
        pending = pending or {
            "candidates": [{"content": {"parts": [], "role": "model"}, "index": 0}]
        }
        if finish:
            pending["candidates"][0]["finishReason"] = finish
        pending["usageMetadata"] = _usage(seen.usage)
        pending["modelVersion"] = call.state["model"]
        yield self._frame(call, pending, first=not opened)
        if not call.state["sse"]:
            yield b"\n]"

    def _frame(self, call: Call, item: dict, *, first: bool) -> bytes:
        payload = json.dumps(item).encode()
        call.state["opened"] = True
        if call.state["sse"]:
            return b"data: " + payload + b"\n\n"
        return (b"[" if first else b",\n") + payload

    def encode_error(self, status: int, error: dict, call: Call):
        return _error(status, error["message"])

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        payload = json.dumps(_error(502, error["message"])).encode()
        if call.state.get("sse"):
            return b"data: " + payload + b"\n\n"
        return (b",\n" if call.state.get("opened") else b"[") + payload


def _model_name(call: Call) -> str:
    return call.path_params["model"].removeprefix("models/")


class _Seen:
    """What litellm's stream wrapper drops or hides, read from the raw chunks."""

    def __init__(self):
        self.usage: dict | None = None
        self.ids: list[str] = []
        self.numbered = 0
        self.failure: Exception | None = None

    async def tee(self, chunks: AsyncIterator) -> AsyncIterator:
        try:
            async for chunk in chunks:
                data = chunk if isinstance(chunk, dict) else chunk.model_dump()
                if data.get("error"):
                    error = data["error"]
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    raise RuntimeError(message or "upstream error")
                if data.get("usage"):
                    self.usage = data["usage"]
                for choice in data.get("choices") or []:
                    for call in (choice.get("delta") or {}).get("tool_calls") or []:
                        if call.get("id") and call["id"] not in self.ids:
                            self.ids.append(call["id"])
                yield chunk if not isinstance(chunk, dict) else ModelResponseStream(**chunk)
        except Exception as exc:
            self.failure = exc
