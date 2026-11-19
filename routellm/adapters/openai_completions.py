"""OpenAI legacy Completions dialect: `POST /v1/completions`.

For older clients and FIM-style tools that still speak `text_completion`.
The request becomes one user chat message and runs the normal chat
pipeline (routing, transforms, cache, audit); the answer is turned back
into a `text_completion` object. An upstream that serves `/v1/completions`
natively (colibri does) is still reached through chat: the route is the
same as for every other dialect, so routing decisions, policies and the
audit trail apply unchanged.

Reused from litellm (no hand mapping): `ModelResponse`,
`LiteLLMResponseObjectHandler.convert_chat_to_text_completion` (chat
choices to `text_completion` choices, usage carried over),
`TextCompletionResponse` and `TextChoices` (the wire types). litellm has
no request-side helper for this direction (its `text_completion` calls a
completions endpoint, it does not map to chat), no echo, no logprobs
mapping (its converter returns none), and its stream wrapper wants
litellm chunks, not normalized events, so the request, echo and stream
framing are mapped here.

Mapped (to the chat request):
- `model`: routing parameters ride in it, as on `/v1/chat/completions`;
- `prompt`: a string, or a list with exactly one string, becomes a single
  `user` message;
- `max_tokens`, `stop`, `temperature`, `top_p`, `n`, `seed`, `user`,
  `presence_penalty`, `frequency_penalty`, `logit_bias`;
- `stream`; `stream_options.include_usage` decides whether the final
  usage chunk is sent (the upstream always reports usage).

Handled in the adapter:
- `echo`: the prompt is prepended to every choice's text (streamed: as the
  first chunk of each choice);
- `best_of`: accepted when null or 1 (no effect); other values refused.

Ignored (no effect on the answer, accepted): none. A reasoning trace the
upstream returns has no place in `text_completion` and is not sent.

Refused (400, OpenAI error shape, nothing reaches the upstream):
- `prompt` as a list of several strings, token ids or lists of token ids,
  an empty list, or a missing prompt. One request is one prompt; ask for
  alternatives with `n`;
- `suffix` (insert mode): chat has no equivalent, and rewording it as an
  instruction would answer a different question;
- `logprobs` (any non-null value): the text-completion logprobs shape is
  not available through chat;
- `best_of` other than 1, `stream_options` without `stream`, and every
  field not listed above.

Notes: an absent `max_tokens` means the upstream's default, not the
legacy 16 tokens. The response `id` and `model` are the upstream's.
`finish_reason` is `stop`, `length` or `content_filter`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    LiteLLMResponseObjectHandler,
)
from litellm.types.utils import ModelResponse, TextChoices, TextCompletionResponse

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Route,
    refuse,
    refuse_unknown,
    register,
)
from routellm.adapters.stream import End, Error, Finish, Start, TextDelta, Usage

#: Request fields copied as they are into the chat request.
_COPIED = (
    "max_tokens",
    "stop",
    "temperature",
    "top_p",
    "n",
    "seed",
    "user",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
)
_SUPPORTED = frozenset(
    {"model", "prompt", "stream", "stream_options", "echo", "best_of", "suffix", "logprobs"}
    | set(_COPIED)
)


def _prompt(body: dict) -> str:
    prompt = body.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    if isinstance(prompt, str):
        return prompt
    if prompt is None:
        refuse("`prompt` is required", param="prompt", code="invalid_request")
    if isinstance(prompt, list) and len(prompt) > 1 and all(isinstance(p, str) for p in prompt):
        refuse(
            "A list of several prompts is not supported: send one prompt per "
            "request and use `n` for alternatives",
            param="prompt",
        )
    refuse(
        "`prompt` must be a string or a list holding one string "
        "(token ids and empty lists are not supported)",
        param="prompt",
    )
    raise AssertionError("unreachable")


def _choice(text: str | None, index: int, finish: str | None = None) -> dict:
    """One `text_completion` choice, through litellm's `TextChoices`."""
    return TextChoices(text=text or "", index=index, finish_reason=finish).model_dump()


def _frame(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


@register
class OpenAICompletions(Adapter):
    name = "openai-completions"
    routes = (Route("POST", "/v1/completions"),)

    def decode(self, call: Call) -> ChatCompletionRequest:
        body = call.body
        if not isinstance(body, dict):
            refuse("Request body must be a JSON object", code="invalid_request")
        refuse_unknown(body, _SUPPORTED, where="completions request")
        if not isinstance(body.get("model"), str) or not body["model"]:
            refuse("`model` is required", param="model", code="invalid_request")
        prompt = _prompt(body)
        if body.get("suffix") is not None:
            refuse("`suffix` (insert mode) is not supported", param="suffix")
        if body.get("logprobs") is not None:
            refuse("`logprobs` is not available", param="logprobs")
        if body.get("best_of") not in (None, 1):
            refuse("`best_of` other than 1 is not supported", param="best_of")

        stream = body.get("stream") is True
        options = body.get("stream_options")
        include_usage = False
        if options is not None:
            if not stream:
                refuse("`stream_options` needs `stream: true`", param="stream_options")
            if not isinstance(options, dict):
                refuse("`stream_options` must be an object", param="stream_options")
            refuse_unknown(options, {"include_usage"}, where="stream_options")
            include_usage = options.get("include_usage") is True

        call.state["echo"] = prompt if body.get("echo") is True else None
        call.state["include_usage"] = include_usage
        fields: dict[str, Any] = {k: body[k] for k in _COPIED if body.get(k) is not None}
        return ChatCompletionRequest(
            model=body["model"],
            messages=[{"role": "user", "content": prompt}],
            stream=True if stream else None,
            **fields,
        )

    def encode(self, response: dict, call: Call):
        chat = ModelResponse(**{k: v for k, v in response.items() if k != "routellm"})
        converted = LiteLLMResponseObjectHandler.convert_chat_to_text_completion(
            chat, TextCompletionResponse()
        )
        body = converted.model_dump(exclude_none=True)
        echo = call.state.get("echo") or ""
        body["choices"] = [
            _choice(echo + (c.text or ""), c.index, c.finish_reason) for c in converted.choices
        ]
        return body

    async def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        echo = call.state.get("echo")
        identity: dict[str, Any] = {"id": None, "created": None, "model": None}
        echoed: set[int] = set()

        def chunk(choices: list[dict], **extra) -> bytes:
            return _frame({"object": "text_completion", **identity, "choices": choices, **extra})

        async for event in events:
            if isinstance(event, Start):
                identity = {"id": event.id, "created": event.created, "model": event.model}
            elif isinstance(event, (TextDelta, Finish)):
                if echo and event.index not in echoed:
                    echoed.add(event.index)
                    yield chunk([_choice(echo, event.index)])
                if isinstance(event, TextDelta):
                    yield chunk([_choice(event.text, event.index)])
                else:
                    yield chunk([_choice("", event.index, event.reason)])
            elif isinstance(event, Usage) and call.state.get("include_usage"):
                yield chunk(
                    [],
                    usage={
                        "prompt_tokens": event.prompt_tokens,
                        "completion_tokens": event.completion_tokens,
                        "total_tokens": event.total_tokens,
                    },
                )
            elif isinstance(event, Error):
                yield _frame({"error": {"message": event.message, "type": "server_error"}})
                return
            elif isinstance(event, End):
                yield b"data: [DONE]\n\n"

    def encode_error(self, status: int, error: dict, call: Call):
        return {
            "error": {
                "message": error["message"],
                "type": error["type"],
                "param": error["param"],
                "code": error["code"],
            }
        }

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return _frame(
            {
                "error": {
                    "message": error["message"],
                    "type": error["type"],
                    "param": error["param"],
                    "code": error["code"],
                }
            }
        )
