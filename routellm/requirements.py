"""What one request needs, read off the request itself.

No router runs, no network is touched, and nothing here is inferred
from the model name: `derive` looks at the messages and the kwargs the
client actually sent, and returns a small frozen record the tier walk
checks against each side's `Capabilities`.

Four facts are read and nothing else::

    vision              an image part in some message's content
    tools               a non-empty `tools` or `functions`
    structured_output   a `response_format` of json_schema/json_object
    context_needed      the prompt's token count plus `max_tokens`

`stream`, `n`, `temperature`, `seed`, `user` and the rest never affect
requirements, which is asserted by a test rather than only stated here.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import litellm

logger = logging.getLogger(__name__)

__all__ = ["Requirements", "derive"]

#: Content part types that mean the request carries an image.
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image"})

#: `response_format` types that mean the request wants a schema.
_STRUCTURED_TYPES = frozenset({"json_schema", "json_object"})

#: Characters per token in the fallback count. Crude on purpose: it
#: only has to be good enough to compare against a context window.
_CHARS_PER_TOKEN = 4

#: Whether the token-counter fallback has already been logged. One
#: DEBUG line per process is plenty; the fallback is not an error.
_warned_fallback = False


@dataclass(frozen=True)
class Requirements:
    """What one request needs of whatever endpoint answers it.

    Attributes
    ----------
    vision : bool
        Whether some message carries an image part.
    tools : bool
        Whether the request defines tools or functions.
    structured_output : bool
        Whether the request asks for a JSON schema or JSON object.
    context_needed : int, optional
        Estimated prompt tokens plus `max_tokens`, or None when no
        count could be made. Advisory: it is compared against an
        endpoint's context window only when both numbers are known.
    """

    vision: bool = False
    tools: bool = False
    structured_output: bool = False
    context_needed: Optional[int] = None

    def is_empty(self) -> bool:
        """Return whether this request needs nothing in particular.

        Returns
        -------
        bool
            True when no capability is demanded and no context count
            was made. The tier walk short-circuits on this, which is
            what keeps a request that carries no images, no tools and
            no `response_format` on the existing code path by
            construction rather than by luck.
        """
        return (
            not self.vision
            and not self.tools
            and not self.structured_output
            and self.context_needed is None
        )


def derive(messages: Any, kwargs: dict[str, Any], model: Optional[str]) -> Requirements:
    """Read a request's requirements off its messages and kwargs.

    Never raises: a hand-rolled client may send message content in
    shapes the OpenAI schema does not describe, and a routing decision
    is not the place to reject them.

    `functions` is read alongside `tools` even though
    `ChatCompletionRequest` declares no `functions` field, so a
    `functions` body never reaches the controller through the server.
    The controller is also a public API the SDK path calls directly,
    where `functions` does arrive. The asymmetry is deliberate: adding
    `functions` to the server model would start forwarding a deprecated
    parameter to providers that reject it.

    Parameters
    ----------
    messages : Any
        The request's messages, in any shape a client may send.
    kwargs : dict
        The request kwargs, read for `tools`, `functions`,
        `response_format` and `max_tokens`.
    model : str, optional
        The REQUEST's model string, which is a tier name rather than a
        provider model. `litellm.token_counter` tolerates one it does
        not know, and the count only has to be good enough to compare
        against a context window.

    Returns
    -------
    Requirements
        What the request needs.
    """
    return Requirements(
        vision=_has_image(messages),
        tools=bool(kwargs.get("tools")) or bool(kwargs.get("functions")),
        structured_output=_wants_schema(kwargs.get("response_format")),
        context_needed=_context_needed(messages, kwargs, model),
    )


def _has_image(messages: Any) -> bool:
    """Return whether any message content carries an image part.

    A string `content` can never set this, whatever it says. Every
    access is guarded: `content` may be a str, a list of dicts, or a
    list of things that are neither.
    """
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in _IMAGE_PART_TYPES:
                return True

    return False


def _wants_schema(response_format: Any) -> bool:
    """Return whether a `response_format` asks for structured output."""
    if not isinstance(response_format, dict):
        return False
    return response_format.get("type") in _STRUCTURED_TYPES


def _context_needed(messages: Any, kwargs: dict[str, Any], model: Optional[str]) -> Optional[int]:
    """Return the estimated tokens this request occupies.

    The prompt's count plus whatever `max_tokens` reserves for the
    answer, since both have to fit inside the same window.
    """
    reserved = kwargs.get("max_tokens") or 0
    try:
        reserved = int(reserved)
    except (TypeError, ValueError):
        reserved = 0

    return _count_tokens(messages, model) + reserved


def _count_tokens(messages: Any, model: Optional[str]) -> int:
    """Return the prompt's token count, falling back to chars over four."""
    global _warned_fallback

    try:
        return int(litellm.token_counter(model=model, messages=messages))
    except Exception as exc:
        if not _warned_fallback:
            _warned_fallback = True
            logger.debug(
                "token_counter failed (%s); estimating context from "
                "characters over %d for the rest of this process",
                exc,
                _CHARS_PER_TOKEN,
            )
        return _rough_tokens(messages)


def _rough_tokens(messages: Any) -> int:
    """Return a characters-over-four estimate over every message."""
    if not isinstance(messages, list):
        return len(str(messages)) // _CHARS_PER_TOKEN

    total = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message
        total += len(str(content)) // _CHARS_PER_TOKEN
    return total


def _prompt_text(messages: Any) -> str:
    """Return the last message's content as a string a router can read.

    A vision request's content is a `list[dict]`, and the routers all
    expect a string, so the text parts are joined and everything else
    dropped. A string content passes through unchanged.

    Parameters
    ----------
    messages : Any
        The request's messages.

    Returns
    -------
    str
        The prompt text, empty when there is none to be had.
    """
    if not isinstance(messages, list) or not messages:
        return ""

    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else last
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return " ".join(text for text in parts if text)
