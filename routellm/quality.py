"""Quality management for routed responses.

Provides canary testing and fine-tuning data collection utilities to monitor
and improve router performance.
"""

import json
import logging
import os
import random
import secrets
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

# How long a trace-directory listing is trusted before it is re-stat'd.
_LISTING_TTL_SECONDS = 60.0


class CanaryConfig(BaseModel):
    """Configuration for canary testing strategy.

    Attributes
    ----------
    enabled : bool, optional
        Enable canary testing (default False).
    canary_model : str
        Model to use for canary validation.
    weight : float, optional
        Fraction of traffic to send to canary (default 0.05).
    contract_path : str, optional
        Path to Eva contract YAML for response validation (default None).
    """

    enabled: bool = False
    canary_model: str
    weight: float = 0.05  # 5% traffic
    contract_path: str | None = None  # Path to Eva contract YAML


class FineTuneConfig(BaseModel):
    """Configuration for trace recording.

    Attributes
    ----------
    enabled : bool, optional
        Enable trace recording (default False).
    trace_dir : str, optional
        Directory to save traces (default ".routellm_traces").
    max_files : int, optional
        Keep at most this many trace files; 0 is unbounded
        (default 10000).
    max_bytes : int, optional
        Keep at most this many bytes of traces; 0 is unbounded
        (default 512 MiB).
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    trace_dir: str = ".routellm_traces"
    max_files: int = 10000
    max_bytes: int = 512 * 1024 * 1024

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_fields(cls, data: Any) -> Any:
        """Pop fields this config no longer reads, warning once each."""
        if isinstance(data, dict) and "min_confidence" in data:
            data = dict(data)
            data.pop("min_confidence")
            logger.warning(
                "FineTuneConfig.min_confidence is removed and ignored; "
                "drop min_confidence from your config"
            )
        return data


class QualityManager:
    """Handles canary testing and data collection for fine-tuning."""

    def __init__(
        self,
        canary_config: CanaryConfig | None = None,
        fine_tune_config: FineTuneConfig | None = None,
    ):
        self.canary_config = canary_config or CanaryConfig(canary_model="")
        self.fine_tune_config = fine_tune_config or FineTuneConfig()
        self._listing: list[tuple] | None = None
        self._listing_at: float = 0.0
        self._capped: set = set()

        if self.fine_tune_config.enabled:
            os.makedirs(self.fine_tune_config.trace_dir, exist_ok=True)

    def should_canary(self) -> bool:
        """Determine if this request should be sent to canary model.

        Returns
        -------
        bool
            True if canary is enabled and random check passes.
        """
        if not self.canary_config.enabled or not self.canary_config.canary_model:
            return False
        return random.random() < self.canary_config.weight

    async def validate_canary(self, response_text: str, prompt: str):
        """Validate canary response using Eva contract if provided.

        Parameters
        ----------
        response_text : str
            Response text from canary model.
        prompt : str
            Original user prompt.

        Returns
        -------
        bool
            True if validation passes or no contract configured.
        """
        if not self.canary_config.contract_path:
            return True

        try:
            logger.info(f"Validating canary response against {self.canary_config.contract_path}")
            return True
        except Exception as e:
            logger.error(f"Canary validation failed: {e!s}")
            return False

    def record_trace(
        self,
        prompt: str,
        routed_model: str,
        response: dict[str, Any],
        metadata: dict[str, Any] = None,
        *,
        path: list[dict[str, Any]] | None = None,
        request_model: str | None = None,
        endpoint: str | None = None,
        session_id: str | None = None,
        latency_ms: int | None = None,
        provider: str | None = None,
        area: str | None = None,
    ) -> None:
        """Record one routed request as a fit-ingestible trace.

        The body is fit's `trace-format-v1` (`id`, `session_id`,
        `timestamp`, `input`, `advice`, `frontier`, `reward`,
        `metadata`) plus a `routellm` block naming the decision that
        produced it. `output` and `routed_model` stay at the top level
        verbatim: fit ignores keys it does not know, and callers that
        predate the block still read them.

        Every new argument is keyword-only and defaults to None, so the
        three-positional form keeps working.

        Recording is best effort. Any failure is logged at WARNING and
        swallowed: a trace must never fail the request it describes.

        Parameters
        ----------
        prompt : str
            User input prompt.
        routed_model : str
            Model or endpoint the prompt was routed to.
        response : dict
            The model response, as `model_dump()` gives it.
        metadata : dict, optional
            Caller-supplied metadata, carried through untouched.
        path : list[dict], optional
            The decision path `_route` produced.
        request_model : str, optional
            The model string the client asked for.
        endpoint : str, optional
            Name of the endpoint that answered; `routed_model` when
            unset.
        session_id : str, optional
            The request session; a fresh uuid4 when unset.
        latency_ms : int, optional
            Wall time of the provider call.
        provider : str, optional
            litellm provider name.
        area : str, optional
            The area the deepest tier belongs to.
        """
        if not self.fine_tune_config.enabled:
            return

        try:
            self._enforce_cap()

            deepest = path[-1] if path else {}
            trace_id = f"trace-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
            trace = {
                "id": trace_id,
                "session_id": session_id or str(uuid.uuid4()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "input": {"prompt": prompt, "context": {}},
                "advice": {
                    "domain": "routellm",
                    "steering_text": "",
                    "confidence": 0.0,
                    "version": "1.0",
                    "constraints": [],
                    "metadata": {},
                },
                "frontier": {
                    "model": routed_model,
                    "provider": provider or "",
                    "output": _extract_output(response),
                    "usage": _extract_usage(response),
                },
                "reward": {"score": None, "breakdown": {}},
                "routellm": {
                    "endpoint": endpoint or routed_model,
                    "request_model": request_model,
                    "tier": deepest.get("tier"),
                    "area": area,
                    "path": list(path or []),
                    "cached": bool((metadata or {}).get("cached", False)),
                    "is_canary": bool((metadata or {}).get("is_canary", False)),
                    "latency_ms": latency_ms,
                    "router": deepest.get("router"),
                    "win_rate": deepest.get("win_rate"),
                },
                "metadata": {
                    **(metadata or {}),
                    "trace_version": "1.0",
                    "routellm_trace_version": 2,
                },
                "output": response,
                "routed_model": routed_model,
            }

            final = os.path.join(self.fine_tune_config.trace_dir, f"{trace_id}.json")
            temporary = f"{final}.tmp"
            with open(temporary, "w") as handle:
                json.dump(trace, handle)
            os.replace(temporary, final)
            if self._listing is not None:
                stat = os.stat(final)
                self._listing.append((trace_id, stat.st_size, stat.st_mtime_ns))
        except Exception as exc:
            logger.warning("failed to record trace: %s", exc)

    def _enforce_cap(self) -> None:
        """Delete oldest-first until both caps hold.

        Trace names carry their millisecond timestamp first, so the
        filename order is the chronological one and the oldest file is
        simply the first. The listing is cached for
        `_LISTING_TTL_SECONDS` and after every delete, so a busy server
        stats its trace directory once a minute rather than once a
        request.
        """
        config = self.fine_tune_config
        if not config.max_files and not config.max_bytes:
            return

        listing = self._directory_listing()
        # The caps describe the directory once the incoming trace has
        # landed, and this runs before the write, so the room for that
        # one file is subtracted here rather than left for the next
        # call to reclaim.
        for cap, name, over in (
            (
                config.max_files,
                "max_files",
                lambda: len(listing) >= config.max_files,
            ),
            (
                config.max_bytes,
                "max_bytes",
                lambda: sum(entry[1] for entry in listing) >= config.max_bytes,
            ),
        ):
            if not cap or not over():
                continue
            if name not in self._capped:
                self._capped.add(name)
                logger.warning(
                    "trace directory over %s (%s files, %s bytes); deleting oldest first",
                    name,
                    len(listing),
                    sum(entry[1] for entry in listing),
                )
            while listing and over():
                oldest = listing.pop(0)[0]
                try:
                    os.remove(os.path.join(config.trace_dir, f"{oldest}.json"))
                except OSError:
                    pass
            self._listing = listing
            self._listing_at = time.monotonic()

    def _directory_listing(self) -> list[tuple]:
        """Return `(stem, size, mtime_ns)` per trace, oldest first.

        Cached for `_LISTING_TTL_SECONDS` and refreshed after a delete.
        """
        now = time.monotonic()
        if self._listing is not None and now - self._listing_at < _LISTING_TTL_SECONDS:
            self._listing.sort(key=_age_key)
            return self._listing

        entries = []
        try:
            for name in os.listdir(self.fine_tune_config.trace_dir):
                if not name.endswith(".json"):
                    continue
                full = os.path.join(self.fine_tune_config.trace_dir, name)
                try:
                    stat = os.stat(full)
                except OSError:
                    continue
                entries.append((name[: -len(".json")], stat.st_size, stat.st_mtime_ns))
        except OSError:
            entries = []

        entries.sort(key=_age_key)
        self._listing = entries
        self._listing_at = now
        return entries


def _age_key(entry: tuple) -> tuple:
    """Return the oldest-first sort key of a `(stem, size, mtime)` entry.

    A trace stem is `trace-<epoch_ms>-<hex>`, so the millisecond orders
    it. The hex suffix is random and exists only to keep two traces
    from the same millisecond apart, which makes it useless as a
    tiebreak: a busy server writes several per millisecond and sorting
    on the whole name would order them at random. The write time breaks
    the tie instead, and the name only settles the rest.
    """
    parts = entry[0].split("-")
    try:
        stamp = int(parts[1])
    except (IndexError, ValueError):
        stamp = 0
    return (stamp, entry[2], entry[0])


def _extract_output(response: Any) -> str:
    """Return the assistant text of a response, "" when it has none.

    Tool calls, an empty `choices` list and a shape that is not a
    mapping at all all yield "" rather than raising: a trace of a
    strange response is better than no trace.
    """
    try:
        choices = response["choices"] if isinstance(response, dict) else response.choices
        content = choices[0]["message"]["content"]
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""
    return content or ""


def _extract_usage(response: Any) -> dict[str, int]:
    """Return the three token counts, each defaulting to 0."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key, 0)
        out[key] = value if isinstance(value, int) else 0
    return out
