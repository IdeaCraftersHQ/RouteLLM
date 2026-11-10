"""Per-conversation class floor with a TTL."""

import hashlib
import time
from collections.abc import Callable
from typing import Any

from routellm.decision.window import message_text


def conversation_key(messages: list[dict[str, Any]], user: str | None = None) -> str | None:
    """Derive a stable key from the first non-system message and `user`.

    None when there is nothing to key on. Every turn of one
    conversation shares its opening message, so they share the key.
    """
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            continue
        text = message_text(message)
        if text:
            digest = hashlib.sha256(f"{user or ''}\x00{text}".encode()).hexdigest()
            return digest[:32]
    return None


class StickyStore:
    """Strictest class rank seen per conversation, expiring after `ttl_s`."""

    def __init__(
        self,
        ttl_s: float,
        max_entries: int = 10000,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ttl, self._max, self._clock = ttl_s, max_entries, clock
        self._seen: dict[str, tuple[int, float]] = {}

    def floor(self, key: str | None) -> int | None:
        """Return the strictest rank seen for `key`, or None."""
        if key is None:
            return None
        hit = self._seen.get(key)
        if hit is None:
            return None
        if hit[1] <= self._clock():
            del self._seen[key]
            return None
        return hit[0]

    def record(self, key: str | None, rank: int) -> None:
        """Remember `rank` for `key` if it is the strictest yet; refresh the TTL."""
        if key is None:
            return
        current = self.floor(key)
        rank = rank if current is None else max(rank, current)
        if key not in self._seen and len(self._seen) >= self._max:
            self._evict()
        self._seen[key] = (rank, self._clock() + self._ttl)

    def _evict(self) -> None:
        now = self._clock()
        for key in [k for k, (_, exp) in self._seen.items() if exp <= now]:
            del self._seen[key]
        while len(self._seen) >= self._max:
            del self._seen[min(self._seen, key=lambda k: self._seen[k][1])]
