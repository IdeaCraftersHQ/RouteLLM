"""Audit events: one record per request of what was decided and why.

An event never carries content. It names the caller, the declared and
effective data class, the policy and entry tier, the endpoint that
answered (with its reach and jurisdiction), the outcome and the refusal
reason, cache status, token counts, latency, and a keyed HMAC-SHA256 of
the prompt. The prompt, the response, placeholder mappings and every key
stay out; a digest lets two events be tied to the same prompt by whoever
holds the HMAC key, and nobody else.

A non-streamed request writes one event (`phase: complete`). A stream
writes `phase: decision` once its route is final and before its first
byte reaches the caller, then `phase: outcome` when it ends.

Sinks implement `AuditSink`. `FileSink` is the default: one JSON object
per line, appended with `O_APPEND` to a 0600 file in a 0700 directory,
reopened by path on every write so an external rotator (logrotate,
newsyslog) can move the file and the next event starts a fresh one.

When a sink cannot take an event, a loaded policy refuses the request
(`AuditUnavailable`); with no policy the failure is logged and counted
and the request is served.

```yaml
audit:
  sink: file                          # the only built-in sink
  path: /var/log/routellm/audit.jsonl
  hmac_key_env: ROUTELLM_AUDIT_KEY    # or hmac_key_file: a 0600 file; never the key itself
```

With a `policy:` loaded, `audit:` is required and pinned to the system
config layer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from routellm.config import ConfigError

__all__ = [
    "AuditError",
    "AuditRecord",
    "AuditSink",
    "AuditUnavailable",
    "Auditor",
    "FileSink",
    "build_audit",
    "current_audit",
    "note",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CONFIG_KEYS = frozenset({"enabled", "sink", "path", "hmac_key_env", "hmac_key_file"})

#: Longest error-type name kept on a walk entry.
_MAX_NAME = 128


class AuditError(Exception):
    """A sink could not take an event."""


class AuditUnavailable(Exception):
    """An event could not be written and a loaded policy requires it."""


@runtime_checkable
class AuditSink(Protocol):
    """Where audit events go.

    `emit` takes one finished event (a JSON-serialisable mapping) and
    returns once it is durable in the sink's sense, or raises. It must
    not mutate the event. `close` releases anything the sink holds.
    """

    def emit(self, event: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class FileSink:
    """Append-only JSONL file, 0600 in a 0700 directory.

    Every write opens the path with `O_APPEND | O_NOFOLLOW`, writes the
    whole line in one `write`, and closes. Reopening per write is what
    lets logrotate/newsyslog work with no signal: after a rename the next
    event creates a fresh 0600 file at the path. A file or directory
    wider than that, or a symlink, is refused rather than repaired.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise AuditError(f"audit dir {self.path.parent}: {exc.strerror or exc}") from None
        wide = stat.S_IMODE(os.stat(self.path.parent).st_mode) & 0o077
        if wide:
            raise AuditError(f"audit dir {self.path.parent} must be 0700, it is wider")
        self._append(b"")  # fail at startup, not on the first request

    def emit(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        self._append(line.encode())

    def _append(self, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise AuditError(f"audit file {self.path}: {exc.strerror or exc}") from None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) & 0o077:
                raise AuditError(f"audit file {self.path} must be a regular file, 0600")
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        except OSError as exc:
            raise AuditError(f"audit file {self.path}: {exc.strerror or exc}") from None
        finally:
            os.close(fd)

    def close(self) -> None:
        return None


@dataclass
class AuditRecord:
    """What is known of one request; the controller adds to it as it routes.

    Created by the server when the request arrives and held in
    `current_audit`; `note` fills fields from anywhere below it.
    """

    request_id: str
    route: str = "chat"
    principal: str | None = None
    profile: str | None = None
    data_class: str | None = None
    operation: str | None = None
    prompt_hmac: str | None = None
    effective_class: str | None = None
    policy: str | None = None
    entry_tier: str | None = None
    cache: str | None = None
    streaming: bool = False
    walk: list[dict[str, Any]] | None = None
    targets: list[str] | None = None
    video: dict[str, Any] | None = None
    queue_wait_ms: int | None = None
    started: float = field(default_factory=time.monotonic)


#: The record of the request being served; None outside one.
current_audit: ContextVar[AuditRecord | None] = ContextVar("current_audit", default=None)


def note(**fields: Any) -> None:
    """Add what the router learned to the current request's record.

    A no-op outside a request or with audit off. Unknown names are ignored.
    """
    record = current_audit.get()
    if record is None:
        return
    for name, value in fields.items():
        if hasattr(record, name):
            setattr(record, name, value)


def _decision_view(decision: Any) -> dict[str, Any] | None:
    """The decision record without its free-form `detail` or error text."""
    if not isinstance(decision, Mapping):
        return None
    keep = ("data_class", "source", "intent", "area", "confidence")
    return {k: decision[k] for k in keep if k in decision}


def _walk_view(walk: Any) -> list[dict[str, Any]] | None:
    if not isinstance(walk, list):
        return None
    out = []
    for step in walk:
        if isinstance(step, Mapping):
            out.append(
                {k: str(step[k])[:_MAX_NAME] for k in ("endpoint", "outcome", "error") if k in step}
            )
    return out


class Auditor:
    """Builds events from request records and writes them to a sink.

    Parameters
    ----------
    sink : AuditSink
    key : bytes
        HMAC key for prompt digests.
    enforce : bool
        True when a policy is loaded: a failed write raises
        `AuditUnavailable`. False: log, count in `failures`, carry on.
    resolve : callable, optional
        `endpoint name -> (reach labels, jurisdiction)`; wired by the
        server once the registry exists.
    """

    def __init__(
        self,
        sink: AuditSink,
        key: bytes,
        *,
        enforce: bool,
        resolve: Callable[[str], tuple[list[str], str | None]] | None = None,
    ):
        self.sink = sink
        self._key = key
        self.enforce = enforce
        self.resolve = resolve
        self.failures = 0

    def digest(self, text: str) -> str:
        """Keyed HMAC-SHA256 of `text`, hex."""
        return hmac.new(self._key, text.encode("utf-8", "replace"), hashlib.sha256).hexdigest()

    def event(
        self,
        record: AuditRecord,
        outcome: str,
        *,
        phase: str = "complete",
        status: int | None = None,
        reason: str | None = None,
        res: Any = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> dict[str, Any]:
        """The event for `record`: fixed fields, no content."""
        hidden = getattr(res, "_hidden_params", None)
        hidden = hidden if isinstance(hidden, dict) else {}
        path = hidden.get("routellm_path")
        path = path if isinstance(path, list) and path else []
        first = path[0] if path else {}
        last = path[-1] if path else {}
        endpoint = hidden.get("routellm_endpoint")
        reach = jur = None
        if endpoint is not None and self.resolve is not None:
            try:
                reach, jur = self.resolve(endpoint)
            except Exception:  # an unresolvable name is recorded bare
                reach = jur = None
        gate = last.get("egress_gate") if isinstance(last, Mapping) else None
        routing: dict[str, Any] = {}
        if isinstance(first, Mapping):
            if (decision := _decision_view(first.get("decision"))) is not None:
                routing["decision"] = decision
            for key in ("policy_routing", "intent_ignored", "tier_from"):
                if key in first:
                    routing[key] = first[key]
        if isinstance(last, Mapping):
            for key in ("fallback_from",):
                if key in last:
                    routing[key] = last[key]
        walk = _walk_view(last.get("walk")) if isinstance(last, Mapping) else None
        walk = walk if walk is not None else _walk_view(record.walk)
        if walk:
            routing["walk"] = walk
        if record.queue_wait_ms is not None:
            routing["queue_wait_ms"] = record.queue_wait_ms
        if record.targets:
            routing["targets"] = list(record.targets)
        if record.video:
            routing["video"] = {
                k: v
                for k, v in record.video.items()
                if isinstance(v, int) and not isinstance(v, bool)
            }
        return {
            "v": SCHEMA_VERSION,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": record.request_id,
            "phase": phase,
            "route": record.route,
            "principal": record.principal,
            "profile": record.profile,
            "data_class": record.data_class,
            "operation": record.operation,
            "effective_class": record.effective_class,
            "policy": record.policy,
            "entry_tier": record.entry_tier,
            "endpoint": endpoint,
            "reach": sorted(reach) if reach is not None else None,
            "jur": jur,
            "treatment": {"egress_gate": gate["passed"]}
            if isinstance(gate, Mapping) and "passed" in gate
            else None,
            "outcome": outcome,
            "status": status,
            "reason": reason,
            "cache": record.cache,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": int((time.monotonic() - record.started) * 1000),
            "prompt_hmac": record.prompt_hmac,
            "routing": routing,
        }

    def emit(self, record: AuditRecord, outcome: str, **fields: Any) -> None:
        """Build and write one event.

        Raises
        ------
        AuditUnavailable
            The sink failed and a policy is loaded.
        """
        if fields.get("res") is not None and "tokens_in" not in fields:
            usage = getattr(fields["res"], "usage", None)
            fields["tokens_in"] = getattr(usage, "prompt_tokens", None)
            fields["tokens_out"] = getattr(usage, "completion_tokens", None)
        try:
            self.sink.emit(self.event(record, outcome, **fields))
        except Exception as exc:
            self.failures += 1
            logger.error(
                "audit event not written (request id %s): %s: %s",
                record.request_id,
                type(exc).__name__,
                exc,
            )
            if self.enforce:
                raise AuditUnavailable(record.request_id) from None


def _read_key(cfg: Mapping[str, Any], env: Mapping[str, str], resolve_path) -> bytes:
    env_name, file_name = cfg.get("hmac_key_env"), cfg.get("hmac_key_file")
    if env_name and file_name:
        raise ConfigError("`audit:` sets both hmac_key_env and hmac_key_file; set one of them")
    if not env_name and not file_name:
        raise ConfigError("`audit:` needs hmac_key_env or hmac_key_file to key the prompt digest")
    if env_name:
        value = env.get(str(env_name), "")
        if not value.strip():
            raise ConfigError(f"`audit.hmac_key_env`: ${env_name} is unset or empty")
        return value.strip().encode()
    path = resolve_path("audit.hmac_key_file", file_name) if resolve_path else Path(file_name)
    try:
        st = os.stat(path)
    except OSError as exc:
        raise ConfigError(f"`audit.hmac_key_file` {path}: {exc.strerror or exc}") from None
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) & 0o077:
        raise ConfigError(f"`audit.hmac_key_file` {path} must be a regular file, mode 0600")
    value = Path(path).read_bytes().strip()
    if not value:
        raise ConfigError(f"`audit.hmac_key_file` {path} is empty")
    return value


def build_audit(
    file_config: Mapping[str, Any] | None,
    policy: Any,
    resolve_path: Callable[[str, Any], Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> Auditor | None:
    """Return the auditor the `audit:` key configures, else None.

    Raises
    ------
    ConfigError
        A policy is loaded and audit is absent or disabled; the section
        is malformed; the HMAC key is missing, empty, inline in the
        config, or in a file wider than 0600; the sink cannot start.
    """
    env = os.environ if env is None else env
    raw = (file_config or {}).get("audit")
    if raw is not None and not isinstance(raw, Mapping):
        raise ConfigError("`audit:` must be a mapping")
    cfg = dict(raw or {})
    if "hmac_key" in cfg:
        raise ConfigError(
            "`audit.hmac_key`: never put the key in config; use hmac_key_env or _file"
        )
    if unknown := sorted(set(cfg) - CONFIG_KEYS):
        raise ConfigError(f"`audit:` unknown key(s) {', '.join(unknown)}")
    if raw is None or cfg.get("enabled") is False:
        if policy is not None:
            raise ConfigError(
                "a `policy:` is loaded, so `audit:` must be configured and enabled: "
                "the policy requires an audit trail"
            )
        return None
    sink_kind = cfg.get("sink", "file")
    if sink_kind != "file":
        raise ConfigError(f"`audit.sink`: unknown sink {sink_kind!r} (known: file)")
    path = cfg.get("path")
    if not isinstance(path, str) or not path:
        raise ConfigError("`audit.path` is required for the file sink")
    key = _read_key(cfg, env, resolve_path)
    resolved = resolve_path("audit.path", path) if resolve_path else Path(path)
    try:
        sink = FileSink(resolved)
    except AuditError as exc:
        raise ConfigError(f"`audit:` {exc}") from None
    return Auditor(sink, key, enforce=policy is not None)
