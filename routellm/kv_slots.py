"""Upstream extension hints, and pinning a session to a KV slot.

Some OpenAI-compatible upstreams take request fields beyond the OpenAI
set. colibri (`coli serve`) takes `cache_slot`, which picks one of its
isolated KV contexts, and `x_colibri_ignore_leading_stop`. They are
hints: they change how fast or where an upstream answers, never what
the request means. So they are forwarded only to an endpoint that lists
them in `extensions:` (or whose `kind: colibri` preset does), and
dropped, with a DEBUG log of the names, for every other endpoint. A
request that is not routed to such an endpoint is served unchanged.

Session affinity: a colibri slot keeps the KV prefix of the last
conversation sent to it, so a harness whose turns share a long prefix
runs faster when every turn lands on the same slot. A caller names its
session, and this module maps it to a slot of the endpoint's
`kv_slots:` budget.

The mapping is LRU, not a hash of the session id:

- A hash is stateless and survives a restart of this server, but two
  live sessions collide on one slot with the birthday odds (about 62%
  for three sessions on four slots), and each then evicts the other's
  prefix on every turn.
- LRU gives each live session its own slot until there are more
  sessions than slots; then the least recently used session loses its
  slot to the newcomer, which is the session whose prefix is the
  coldest anyway. It keeps in memory, so a restart of this server
  reassigns slots once (one cold turn per session).
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

__all__ = [
    "EXTENSION_PARAMS",
    "KIND_EXTENSIONS",
    "MAX_KV_SLOTS",
    "QUEUE_WAIT_HEADER",
    "SESSION_HEADER",
    "Extensions",
    "SlotMap",
    "queue_wait_ms",
    "session_of",
]

logger = logging.getLogger(__name__)

#: Request fields that are upstream hints. The inbound request accepts
#: them, and an endpoint's `extensions:` may name only these.
EXTENSION_PARAMS = ("cache_slot", "x_colibri_ignore_leading_stop")

#: What an endpoint `kind:` forwards when it lists no `extensions:`.
KIND_EXTENSIONS: dict[str, tuple[str, ...]] = {"colibri": EXTENSION_PARAMS}

#: colibri's ceiling on `--kv-slots`.
MAX_KV_SLOTS = 16

#: Header naming the caller's session; `metadata.session_id` does the
#: same when a client cannot set headers. The header wins.
SESSION_HEADER = "X-RouteLLM-Session"

#: The response header colibri reports its queue wait in, in milliseconds.
QUEUE_WAIT_HEADER = "x-colibri-queue-wait-ms"

_MAX_SESSION_LEN = 256


class SlotMap:
    """Assign sessions to the slots of one endpoint, least recently used first.

    Holds at most `slots` sessions, as digests: a raw session id is never
    kept. Safe to call from many threads.
    """

    def __init__(self, slots: int):
        if not 1 <= slots <= MAX_KV_SLOTS:
            raise ValueError(f"slots must be 1..{MAX_KV_SLOTS}, got {slots}")
        self.slots = slots
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, int] = OrderedDict()

    def slot_for(self, session: str) -> int:
        """Return the slot `session` is pinned to, assigning one on first use."""
        key = hashlib.sha256(session.encode()).hexdigest()[:16]
        with self._lock:
            slot = self._sessions.get(key)
            if slot is not None:
                self._sessions.move_to_end(key)
                return slot
            if len(self._sessions) < self.slots:
                slot = len(self._sessions)
            else:
                _, slot = self._sessions.popitem(last=False)
            self._sessions[key] = slot
            return slot


class Extensions:
    """Decide, per call, which extension hints an endpoint receives."""

    def __init__(self) -> None:
        self._maps: dict[str, SlotMap] = {}
        self._lock = threading.Lock()

    def _map(self, endpoint) -> SlotMap:
        with self._lock:
            current = self._maps.get(endpoint.name)
            if current is None or current.slots != endpoint.kv_slots:
                current = self._maps[endpoint.name] = SlotMap(endpoint.kv_slots)
            return current

    def body(
        self, endpoint, hints: Mapping[str, Any] | None, session: str | None
    ) -> dict[str, Any]:
        """Return the extra request-body fields `endpoint` gets for this call.

        Parameters
        ----------
        endpoint : Endpoint
            The endpoint being called.
        hints : Mapping, optional
            The extension hints the caller sent, by name.
        session : str, optional
            The caller's session id. With the endpoint's `kv_slots:` it
            picks `cache_slot` when the caller sent none.

        Returns
        -------
        dict
            Fields to add to the body: the hints the endpoint lists, and
            a session-pinned `cache_slot`. A hint the endpoint does not
            list is dropped and its name logged at DEBUG, never its value.
        """
        allowed = endpoint.forwarded_extensions
        out = {k: v for k, v in (hints or {}).items() if k in allowed}
        lost = [k for k in (hints or {}) if k not in allowed]
        if lost:
            logger.debug(
                "endpoint %s does not take extension hints, dropped: %s", endpoint.name, lost
            )
        if session and "cache_slot" not in out and endpoint.kv_slots:
            out["cache_slot"] = self._map(endpoint).slot_for(session)
        return out


def session_of(headers: Mapping[str, str], metadata: Mapping[str, Any] | None) -> str | None:
    """Return the caller's session id: the session header, else `metadata.session_id`.

    Only a non-empty string of at most 256 characters counts.
    """
    value = headers.get(SESSION_HEADER)
    if value is None and metadata:
        value = metadata.get("session_id")
    if isinstance(value, str) and 0 < len(value) <= _MAX_SESSION_LEN:
        return value
    return None


def queue_wait_ms(res: Any) -> int | None:
    """Return the upstream's reported queue wait for `res`, in ms, if it sent a valid one.

    A value that is not a plain non-negative integer is ignored: it is an
    upstream's text and is echoed to the caller as a header.
    """
    hidden = getattr(res, "_hidden_params", None)
    headers = hidden.get("additional_headers") if isinstance(hidden, dict) else None
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if str(key).lower().endswith(QUEUE_WAIT_HEADER):
            text = str(value).strip()
            if text.isascii() and text.isdigit() and len(text) <= 12:
                return int(text)
    return None
