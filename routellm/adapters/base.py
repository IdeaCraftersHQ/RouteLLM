"""The adapter contract: what a dialect implements, what the framework gives it.

A dialect is one `Adapter` subclass in one module of `routellm/adapters/`,
registered with `@register`. It translates a foreign request into a
`ChatCompletionRequest`, and the answer back; the framework runs the
request through the same pipeline as `/v1/chat/completions`. See
`docs/adapters.md`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fastapi import Request

from routellm.adapters.stream import stream_events
from routellm.chat_request import ChatCompletionRequest

__all__ = [
    "CHAT",
    "Adapter",
    "AdapterRefusal",
    "Call",
    "Reply",
    "Route",
    "bearer",
    "refuse",
    "refuse_unknown",
]

#: The operation of a route that runs the chat completion pipeline. Any
#: other operation is a custom route the adapter answers in `handle`.
CHAT = "chat"


@dataclass(frozen=True)
class Route:
    """One HTTP route an adapter owns.

    `path` is a Starlette path: `{name}` parameters and converters work
    (`/model/{model_id}/converse`, `/v1beta/models/{model:path}`). Two
    adapters claiming the same method and path is a startup error.

    `operation` is `chat` (the default) for a route that generates a
    completion, or any other name for a route the adapter answers itself
    in `handle` (listing models, counting tokens, a version probe).

    `body` is `json` (parsed before `decode`/`handle`) or `none`.
    """

    method: str
    path: str
    operation: str = CHAT
    body: str = "json"


class AdapterRefusal(Exception):
    """A request the dialect cannot carry, refused in the dialect's own 400 shape.

    Raise it from `decode` (or `handle`) via `refuse`. The framework hands
    it to `encode_error`; nothing reaches the pipeline.
    """

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        status: int = 400,
        code: str = "unsupported_field",
    ):
        super().__init__(message)
        self.message, self.param, self.status, self.code = message, param, status, code


@dataclass
class Reply:
    """A dialect body with the wire details it needs.

    Returning a bare body from `encode`, `encode_error` or `handle` means
    `Reply(body)`. Use `Reply` for response headers (`x-amzn-ErrorType`), a
    non-JSON `media_type`, or a status other than the one given.
    """

    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    status: int | None = None
    media_type: str = "application/json"


@dataclass
class Call:
    """One request in flight, shared by every adapter hook.

    `state` is the adapter's scratch space between `decode` and the encode
    hooks (the model name the client used, a path-selected mode); the
    framework never touches it. `request` is the decoded chat request once
    `decode` has returned.
    """

    adapter: Adapter
    route: Route
    http_request: Request
    pipeline: Any
    body: Any = None
    request: ChatCompletionRequest | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def headers(self) -> Mapping[str, str]:
        return self.http_request.headers

    @property
    def query_params(self) -> Mapping[str, str]:
        return self.http_request.query_params

    @property
    def path_params(self) -> dict[str, Any]:
        return self.http_request.path_params

    @property
    def principal(self) -> Any:
        return getattr(self.http_request.state, "principal", None)

    @property
    def controller(self) -> Any:
        return self.pipeline.CONTROLLER

    async def models(self) -> list[dict]:
        """The server's model list, as `/v1/models` returns it (`data`)."""
        import json

        response = await self.pipeline.list_models()
        return json.loads(response.body)["data"]


def refuse(message: str, param: str | None = None, *, code: str = "unsupported_field") -> None:
    """Refuse the request: raise `AdapterRefusal`, which becomes the dialect's 400."""
    raise AdapterRefusal(message, param=param, code=code)


def top_k_value(value: Any, param: str, label: str | None = None) -> int:
    """Return a dialect's `top_k` as the whole number from 0 chat completions takes.

    Refuses anything else (a bool, a float, a string, a negative) naming
    `param`, rather than leaving a malformed value to fail inside the
    pipeline.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        refuse(f"{label or param} must be a whole number of 0 or more", param=param)
    return value


def refuse_unknown(
    body: Mapping[str, Any],
    supported: set[str] | frozenset[str],
    *,
    ignored: set[str] | frozenset[str] = frozenset(),
    where: str = "request",
) -> None:
    """Refuse any key of `body` that is neither `supported` nor `ignored`.

    `supported` is what `decode` maps; `ignored` is what the dialect lets a
    client send that has no effect on the answer (documented in the
    adapter). Anything else is a field the pipeline cannot honour, and
    dropping it silently would answer a different question than the one
    asked.
    """
    unknown = sorted(set(body) - set(supported) - set(ignored))
    if unknown:
        refuse(
            f"Unsupported field(s) in {where}: {', '.join(unknown)}",
            param=unknown[0],
        )


def bearer(headers: Mapping[str, str]) -> str | None:
    """The token of an `Authorization: Bearer` header, or None."""
    from routellm.auth import bearer_token

    return bearer_token(headers.get("authorization"))


class Adapter:
    """Base class of every dialect. Subclass, set `name` and `routes`, implement hooks.

    Hooks that raise `NotImplementedError` here are the ones every chat
    dialect must implement: `decode`, `encode`, and one of `encode_events`
    or `encode_stream`. The rest have working defaults.
    """

    #: Unique id, also the module's role: `anthropic`, `gemini`, ...
    name: ClassVar[str]
    #: Every route the dialect owns.
    routes: ClassVar[tuple[Route, ...]]

    # -- auth ---------------------------------------------------------------

    def credential(self, headers: Mapping[str, str], query: Mapping[str, str]) -> str | None:
        """The key the caller presented, as the string a keys file lists.

        Whatever the dialect carries it in: the default reads the bearer
        token; override for `x-api-key`, `x-goog-api-key`, `?key=`, or an
        access key id. The framework hands the result to the same
        resolver `/v1/chat/completions` uses.
        """
        return bearer(headers)

    # -- request ------------------------------------------------------------

    def decode(self, call: Call) -> ChatCompletionRequest:
        """Translate `call.body` (plus path/query/headers) into a chat request.

        Routing parameters ride in the model field exactly as on
        `/v1/chat/completions`. Set `stream` when the dialect asks for a
        stream (a body flag, or the route). Stash anything the encoders
        need in `call.state`. Refuse with `refuse` / `refuse_unknown`.
        """
        raise NotImplementedError

    async def handle(self, call: Call) -> Any:
        """Answer a custom route (`route.operation != "chat"`).

        Authentication has run. Return a body or a `Reply`; refuse with
        `refuse`.
        """
        raise NotImplementedError

    # -- response -----------------------------------------------------------

    def encode(self, response: dict, call: Call) -> Any:
        """Translate a non-streamed chat completion into the dialect's body.

        `response` is the OpenAI body plus a `routellm: {path: [...]}` key.
        The framework also sends the decision path as the `x-routellm-path`
        header on every adapter response, so a dialect whose body cannot
        carry extra fields can ignore the key.
        """
        raise NotImplementedError

    def stream_media_type(self, call: Call) -> str:
        """The streamed response's media type."""
        return "text/event-stream"

    def stream_headers(self, call: Call) -> dict[str, str]:
        """Extra response headers for a streamed reply (default none).

        Called once after `decode`, before the first byte, so it can only
        use what `decode` left in `call.state`. A non-streamed reply sets
        its headers on a `Reply`. Names must be HTTP tokens, values printable
        ASCII with no CR or LF; anything else is a server error, never sent.
        `x-routellm-path` is the framework's and wins on a clash.
        """
        return {}

    def encode_events(self, events: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        """Map normalized events (`routellm.adapters.stream`) to wire bytes.

        An async generator. The usual implementation.
        """
        raise NotImplementedError

    def encode_stream(self, chunks: AsyncIterator, call: Call) -> AsyncIterator[bytes]:
        """Map the raw OpenAI chunk stream to wire bytes.

        Defaults to `encode_events(stream_events(chunks), call)`. Override
        only to read chunks directly. If the iterator raises, the framework
        sends `encode_stream_error` and ends the stream.
        """
        return self.encode_events(stream_events(chunks), call)

    def encode_error(self, status: int, error: dict, call: Call) -> Any:
        """Translate an error into the dialect's error body.

        `error` is OpenAI-style and always has the keys `message`, `type`,
        `code`, `param`, `request_id` (None when unknown). `status` is the
        HTTP status the response keeps unless a `Reply` overrides it. The
        default is the OpenAI shape; every dialect overrides it.
        """
        return {
            "error": {
                "message": error["message"],
                "type": error["type"],
                "code": error["code"],
                "param": error["param"],
            }
        }

    def encode_stream_error(self, error: dict, call: Call) -> bytes:
        """The frame that reports a failure after the stream started.

        Same `error` as `encode_error`; the message is generic and carries
        the request id. Return empty bytes when the wire format has no such
        frame (the stream then just ends, never with a clean terminator).
        """
        return b""
