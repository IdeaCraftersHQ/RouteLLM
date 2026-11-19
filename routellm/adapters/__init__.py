"""Dialect adapters: other LLM APIs in front of `/v1/chat/completions`.

Each module of this package holds one dialect: an `Adapter` subclass
decorated with `@register`. `mount_adapters` imports them all and serves
their routes, so a new dialect is one new module and no edit elsewhere.
See `docs/adapters.md`.
"""

from routellm.adapters.base import (
    CHAT,
    Adapter,
    AdapterRefusal,
    Call,
    Reply,
    Route,
    bearer,
    refuse,
    refuse_unknown,
)
from routellm.adapters.registry import adapters, discover, mount_adapters, register
from routellm.adapters.stream import stream_events
from routellm.chat_request import ChatCompletionRequest

__all__ = [
    "CHAT",
    "Adapter",
    "AdapterRefusal",
    "Call",
    "ChatCompletionRequest",
    "Reply",
    "Route",
    "adapters",
    "bearer",
    "discover",
    "mount_adapters",
    "refuse",
    "refuse_unknown",
    "register",
    "stream_events",
]
