"""A deliberately small dialect that proves the adapter contract.

Not shipped: it lives with the tests and is loaded into the harness server
through `ROUTELLM_ADAPTER_MODULES`. Copy it as the starting point of a
real adapter.

The "reference" dialect:
- request: an OpenAI chat body at `POST /reference/v1/chat`; a key outside
  the chat fields is refused, never dropped;
- auth: the `x-reference-key` header (not a bearer token);
- response: the OpenAI body;
- stream: NDJSON, one event object per line, built from normalized stream
  events: `text`, `tool_call`, `finish`, `usage`, `end`;
- errors: `{"reference_error": {"message", "code"}}`;
- `GET /reference/v1/models`: a custom route, listing the server's models.
"""

import json
from collections.abc import AsyncIterator

from routellm.adapters import (
    Adapter,
    Call,
    ChatCompletionRequest,
    Route,
    refuse_unknown,
    register,
)
from routellm.adapters.stream import End, Finish, TextDelta, ToolCallEnd, Usage


def _line(**event) -> bytes:
    return json.dumps(event).encode() + b"\n"


@register
class ReferenceAdapter(Adapter):
    name = "reference"
    routes = (
        Route("POST", "/reference/v1/chat"),
        Route("GET", "/reference/v1/models", operation="models", body="none"),
    )

    def credential(self, headers, query):
        return headers.get("x-reference-key")

    def decode(self, call: Call) -> ChatCompletionRequest:
        refuse_unknown(call.body, set(ChatCompletionRequest.model_fields), where="reference chat")
        call.state["model"] = call.body.get("model")
        return ChatCompletionRequest(**call.body)

    async def handle(self, call: Call):
        return {"models": [m["id"] for m in await call.models()]}

    def encode(self, response: dict, call: Call):
        return response

    def stream_media_type(self, call: Call) -> str:
        return "application/x-ndjson"

    async def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        async for event in events:
            if isinstance(event, TextDelta):
                yield _line(event="text", text=event.text)
            elif isinstance(event, ToolCallEnd):
                yield _line(
                    event="tool_call", id=event.id, name=event.name, arguments=event.arguments
                )
            elif isinstance(event, Finish):
                yield _line(event="finish", reason=event.reason)
            elif isinstance(event, Usage):
                yield _line(
                    event="usage", input=event.prompt_tokens, output=event.completion_tokens
                )
            elif isinstance(event, End):
                yield _line(event="end")

    def encode_error(self, status: int, error: dict, call: Call):
        return {"reference_error": {"message": error["message"], "code": error["code"]}}

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        return _line(event="error", message=error["message"])
