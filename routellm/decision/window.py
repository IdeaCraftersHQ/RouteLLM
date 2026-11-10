"""Turn request messages into the text a model and the recognizers read."""

from typing import Any

from routellm.decision.config import InputWindow

#: Recognizers scan whole conversations; this only bounds pathological input.
SCAN_CAP = 1_000_000


def message_text(message: Any) -> str:
    """Return the text of one message, text parts joined, other parts dropped."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def full_text(messages: list[dict[str, Any]]) -> str:
    """Return every message's text, any role, capped at `SCAN_CAP`."""
    text = "\n".join(t for t in (message_text(m) for m in messages) if t)
    return text[:SCAN_CAP]


def select(messages: list[dict[str, Any]], window: InputWindow) -> list[dict[str, Any]]:
    """Return the messages the window says the model reads."""
    if window.messages == "all":
        return list(messages)
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if window.messages == "user":
        return users
    return users[-1:]


def truncate(text: str, window: InputWindow) -> str:
    """Cut `text` to `max_chars`, keeping the configured end."""
    if len(text) <= window.max_chars:
        return text
    return text[: window.max_chars] if window.keep == "head" else text[-window.max_chars :]
