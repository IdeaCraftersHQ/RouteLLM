"""Resilience utilities for fault-tolerant model routing.

Implements circuit breaker pattern for handling transient failures and
fallback mechanisms for routing.
"""

import asyncio
import email.utils
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States in circuit breaker pattern.

    Attributes
    ----------
    CLOSED : str
        Normal operation; requests pass through.
    OPEN : str
        Failures exceeded; requests fail immediately.
    HALF_OPEN : str
        Recovery mode; testing if service is healthy.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Implements circuit breaker pattern for fault tolerance.

    Tracks failures and health status of downstream services, opening
    circuit when failure threshold exceeded to prevent cascading failures.
    """

    def __init__(
        self,
        fail_max: int = 5,
        fail_rate: float = 0.5,
        fail_wait_ms: int = 5000,
        fail_codes: list[int] | None = None,
        rate_interval_ms: int = 60000,
        rate_minimum: int = 10,
    ):
        """Initialize circuit breaker.

        Parameters
        ----------
        fail_max : int, optional
            Consecutive failures to trigger OPEN state (default 5).
        fail_rate : float, optional
            Failure rate threshold in [0, 1] (default 0.5).
        fail_wait_ms : int, optional
            Milliseconds to wait before trying recovery (default 5000).
        fail_codes : list[int], optional
            HTTP status codes to treat as failures (default [500, 502, 503, 504]).
        rate_interval_ms : int, optional
            Time window for computing failure rate (default 60000).
        rate_minimum : int, optional
            Minimum requests in window to compute rate (default 10).
        """
        self.fail_max = fail_max
        self.fail_rate = fail_rate
        self.fail_wait_ms = fail_wait_ms
        self.fail_codes = fail_codes or [500, 502, 503, 504]
        self.rate_interval_ms = rate_interval_ms
        self.rate_minimum = rate_minimum

        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0
        self.history: list[tuple[float, bool]] = []  # (timestamp, success)

    def _clean_history(self):
        """Remove stale entries from failure history."""
        now = time.time()
        interval_s = self.rate_interval_ms / 1000
        self.history = [h for h in self.history if now - h[0] <= interval_s]

    def _get_fail_rate(self) -> float:
        """Compute failure rate within time window.

        Returns
        -------
        float
            Failure rate in [0, 1]. Returns 0 if insufficient history.
        """
        self._clean_history()
        if len(self.history) < self.rate_minimum:
            return 0.0
        fails = sum(1 for h in self.history if not h[1])
        return fails / len(self.history)

    def can_execute(self) -> bool:
        """Check if request can proceed given circuit state.

        Returns
        -------
        bool
            True if circuit allows execution.
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            now_ms = time.time() * 1000
            if now_ms - self.last_failure_time >= self.fail_wait_ms:
                logger.info("Circuit Breaker: state changed to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return True  # Allow one trial

        return False

    def record_success(self):
        self.history.append((time.time(), True))
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit Breaker: state changed to CLOSED")
            self.state = CircuitState.CLOSED
            self.failures = 0
        elif self.state == CircuitState.CLOSED:
            self.failures = 0

    def record_failure(self, error: Any):
        # Check if error has status_code and if it's in fail_codes
        status_code = getattr(error, "status_code", None)
        if status_code and status_code not in self.fail_codes:
            return

        self.history.append((time.time(), False))
        self.failures += 1
        self.last_failure_time = time.time() * 1000

        if self.state == CircuitState.CLOSED:
            if self.failures >= self.fail_max or self._get_fail_rate() >= self.fail_rate:
                logger.warning(
                    f"Circuit Breaker: state changed to OPEN (failures: {self.failures}, rate: {self._get_fail_rate():.2f})"
                )
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit Breaker: trial failed, state changed back to OPEN")
            self.state = CircuitState.OPEN


class ResilienceConfig(BaseModel):
    """Retry, timeout, breaker and rate-limit-cooldown settings.

    Rate-limit cooldown is off by default. When `rate_limit_cooldown`
    is true a 429 is not retried against the same endpoint: the endpoint
    cools until the reset the upstream stated (a header, or its error
    body), else `rate_limit_default_cooldown_s`, and the router's walk
    moves to the next candidate. A cooling endpoint is skipped without a
    call. A 429 under cooldown is not a fault, so it does not count
    toward the circuit breaker.

    A timeout is retried like any failure unless `retry_on_timeout` is
    false. The retried request may already be running upstream, so a
    paid endpoint can bill the abandoned attempt as well as the retry;
    set it false where that matters.
    """

    max_retries: int = 3
    retry_on_codes: list[int] = [429, 500, 502, 503, 504]
    backoff_factor: float = 2.0
    initial_backoff_ms: int = 500
    timeout_ms: int = 30000
    retry_on_timeout: bool = True

    # Rate-limit cooldown
    rate_limit_cooldown: bool = False
    rate_limit_default_cooldown_s: float = 60.0
    rate_limit_daily_cooldown_s: float = 3600.0
    rate_limit_max_cooldown_s: float = 86400.0
    rate_limit_reset_headers: list[str] = ["retry-after", "x-ratelimit-reset"]
    rate_limit_daily_markers: list[str] = ["per-day", "per day", "daily"]

    # Circuit Breaker
    cb_enabled: bool = True
    cb_fail_max: int = 5
    cb_fail_rate: float = 0.5
    cb_fail_wait_ms: int = 5000
    cb_rate_interval_ms: int = 60000
    cb_rate_minimum: int = 10


@dataclass
class RateLimitInfo:
    """What a rate-limit error says about when to come back.

    Attributes
    ----------
    until : float
        Epoch seconds before which the endpoint should not be called.
    daily : bool
        Whether the error names a daily limit.
    stated : bool
        Whether `until` came from the upstream rather than a default.
    """

    until: float
    daily: bool = False
    stated: bool = False


class EndpointsCooling(Exception):
    """Every endpoint a request could use is cooling after a rate limit.

    Attributes
    ----------
    endpoints : dict[str, float]
        Endpoint name to the epoch second it is next tried.
    """

    def __init__(self, endpoints: dict[str, float], now: float | None = None):
        self.endpoints = dict(endpoints)
        now = time.time() if now is None else now
        self.retry_after = max(0.0, min(self.endpoints.values(), default=now) - now)
        listed = ", ".join(
            f"{name} (in {max(0, until - now):.0f}s)"
            for name, until in sorted(self.endpoints.items())
        )
        super().__init__(f"every candidate is rate-limited and cooling: {listed}")


def _chain(error: BaseException | None):
    """Yield `error`, then each exception it wraps, once."""
    seen: list[int] = []
    while error is not None and id(error) not in seen:
        seen.append(id(error))
        yield error
        error = getattr(error, "original_exception", None) or error.__cause__


def _status_of(error: BaseException) -> int | None:
    for item in _chain(error):
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def is_rate_limited(error: BaseException) -> bool:
    """Return whether `error` is a 429, however wrapped."""
    return _status_of(error) == 429


def _header_map(item) -> dict[str, str]:
    found: dict[str, str] = {}
    sources = [
        getattr(item, "litellm_response_headers", None),
        getattr(item, "headers", None),
        getattr(getattr(item, "response", None), "headers", None),
    ]
    body = _body_of(item)
    if isinstance(body, dict):
        meta = body.get("metadata") or (body.get("error") or {}).get("metadata")
        if isinstance(meta, dict):
            sources.append(meta.get("headers"))
    for source in sources:
        if not source:
            continue
        try:
            for key, value in dict(source).items():
                found.setdefault(str(key).lower(), str(value))
        except (TypeError, ValueError):
            continue
    return found


def _body_of(item):
    body = getattr(item, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    return body if isinstance(body, dict) else None


def _reset_seconds(name: str, value: str, now: float) -> float | None:
    """Return epoch seconds a reset header states, or None if unreadable.

    `retry-after` is a delay in seconds or an HTTP date. Any other
    reset header is read by magnitude: epoch milliseconds, epoch
    seconds, otherwise a delay in seconds.
    """
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        if name != "retry-after":
            return None
        parsed = email.utils.parsedate_to_datetime(value) if value else None
        return parsed.timestamp() if parsed else None
    if name == "retry-after":
        return now + number
    if number >= 1e11:
        return number / 1000
    if number >= 1e9:
        return number
    return now + number


def rate_limit_info(error: BaseException, config: ResilienceConfig, now: float) -> RateLimitInfo:
    """Read when a rate-limited endpoint may be called again.

    Looks for a stated reset in the headers of `error` and of anything
    it wraps (litellm keeps the upstream's headers on
    `litellm_response_headers`, and some upstreams repeat them under
    the body's `metadata.headers`), in the order of
    `config.rate_limit_reset_headers`. Without one the default applies:
    `rate_limit_daily_cooldown_s` when the message names a daily limit
    (any of `rate_limit_daily_markers`), else
    `rate_limit_default_cooldown_s`. A reset already in the past counts
    as unstated. The result never exceeds `rate_limit_max_cooldown_s`.
    """
    headers: dict[str, str] = {}
    text = ""
    for item in _chain(error):
        for key, value in _header_map(item).items():
            headers.setdefault(key, value)
        body = _body_of(item)
        text += f" {getattr(item, 'message', '')} {item!s} {json.dumps(body) if body else ''}"

    lowered = text.lower()
    daily = any(marker.lower() in lowered for marker in config.rate_limit_daily_markers)

    until = None
    for name in config.rate_limit_reset_headers:
        value = headers.get(name.lower())
        if value is None:
            continue
        candidate = _reset_seconds(name.lower(), value, now)
        if candidate is not None and candidate > now:
            until = candidate
            break

    stated = until is not None
    if until is None:
        wait = config.rate_limit_daily_cooldown_s if daily else config.rate_limit_default_cooldown_s
        until = now + wait
    until = min(until, now + config.rate_limit_max_cooldown_s)
    return RateLimitInfo(until=until, daily=daily, stated=stated)


class Resilience:
    def __init__(self, config: ResilienceConfig = None):
        self.config = config or ResilienceConfig()
        self.cb_map: dict[str, CircuitBreaker] = {}
        #: Endpoint name to the epoch second its rate-limit cooldown ends.
        self.cooldowns: dict[str, float] = {}
        #: Wall clock, replaceable so tests can move time.
        self.clock: Callable[[], float] = time.time

    def cooling(self, name: str) -> float | None:
        """Return when `name` is next tried, or None when it is not cooling."""
        until = self.cooldowns.get(name)
        if until is None:
            return None
        if self.clock() >= until:
            del self.cooldowns[name]
            return None
        return until

    def note_rate_limit(self, name: str, error: BaseException) -> bool:
        """Start `name` cooling when `error` is a rate limit and cooling is on.

        Returns
        -------
        bool
            Whether the error was a rate limit handled as one.
        """
        if not self.config.rate_limit_cooldown or not is_rate_limited(error):
            return False
        info = rate_limit_info(error, self.config, self.clock())
        self.cooldowns[name] = info.until
        logger.warning(
            "Endpoint %s rate-limited%s; cooling until %.0f (%s)",
            name,
            " (daily)" if info.daily else "",
            info.until,
            "stated by the upstream" if info.stated else "default",
        )
        return True

    def record_stream_end(self, name: str, error: BaseException | None) -> None:
        """Record how a stream that had already started ended.

        A stream's success is only known at its end, so the circuit
        breaker hears nothing at the first chunk: a clean end is a
        success, a failure after the first chunk is a breaker failure,
        and a rate-limit error event starts the cooldown instead.
        """
        if not self.config.cb_enabled and error is None:
            return
        cb = self.get_cb(name)
        if error is None:
            if self.config.cb_enabled:
                cb.record_success()
            return
        if self.note_rate_limit(name, error):
            return
        if self.config.cb_enabled:
            cb.record_failure(error)

    def get_cb(self, key: str) -> CircuitBreaker:
        if key not in self.cb_map:
            self.cb_map[key] = CircuitBreaker(
                fail_max=self.config.cb_fail_max,
                fail_rate=self.config.cb_fail_rate,
                fail_wait_ms=self.config.cb_fail_wait_ms,
                fail_codes=self.config.retry_on_codes,
                rate_interval_ms=self.config.cb_rate_interval_ms,
                rate_minimum=self.config.cb_rate_minimum,
            )
        return self.cb_map[key]

    async def wrap_acompletion(
        self,
        model_name: str,
        fn: Callable[..., Awaitable[Any]],
        *args,
        timeout_ms: int | None = None,
        defer_success: bool = False,
        **kwargs,
    ):
        """Call `fn` with retries, a breaker and a per-attempt timeout.

        `timeout_ms` overrides the configured one for this call, which is
        how an endpoint that is slow by nature gets a longer budget.

        With rate-limit cooldown on, a 429 starts the endpoint cooling
        and is raised at once, never retried in place. `defer_success`
        leaves the breaker's success to `record_stream_end`, for a
        result that is a stream whose outcome is not known yet.
        """
        cb = self.get_cb(model_name)
        timeout_s = (self.config.timeout_ms if timeout_ms is None else timeout_ms) / 1000

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            if self.config.cb_enabled and not cb.can_execute():
                raise Exception(f"Circuit breaker OPEN for {model_name}")

            if attempt > 0:
                wait = (self.config.initial_backoff_ms / 1000) * (
                    self.config.backoff_factor ** (attempt - 1)
                )
                logger.info(
                    f"Retrying {model_name} (attempt {attempt}/{self.config.max_retries}) in {wait:.2f}s..."
                )
                await asyncio.sleep(wait)

            try:
                res = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_s)
                if self.config.cb_enabled and not defer_success:
                    cb.record_success()
                return res
            except TimeoutError as e:
                logger.error(f"Timeout calling {model_name}: {e!s}")
                if self.config.cb_enabled:
                    cb.record_failure(e)
                last_error = e
                if attempt == self.config.max_retries or not self.config.retry_on_timeout:
                    raise
            except Exception as e:
                logger.error(f"Error calling {model_name}: {e!s}")
                if self.note_rate_limit(model_name, e):
                    raise
                if self.config.cb_enabled:
                    cb.record_failure(e)
                last_error = e
                status_code = getattr(e, "status_code", None)
                if attempt == self.config.max_retries:
                    raise

                # Check if retryable
                if status_code and status_code not in self.config.retry_on_codes:
                    raise

        if last_error:
            raise last_error

    def wrap_completion(self, model_name: str, fn: Callable[..., Any], *args, **kwargs):
        cb = self.get_cb(model_name)

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            if self.config.cb_enabled and not cb.can_execute():
                raise Exception(f"Circuit breaker OPEN for {model_name}")

            if attempt > 0:
                wait = (self.config.initial_backoff_ms / 1000) * (
                    self.config.backoff_factor ** (attempt - 1)
                )
                logger.info(
                    f"Retrying {model_name} (attempt {attempt}/{self.config.max_retries}) in {wait:.2f}s..."
                )
                time.sleep(wait)

            try:
                # Synchronous call doesn't easily support timeout without threads,
                # but we'll try to match the interface.
                # LiteLLM completion usually has its own timeout param.
                res = fn(*args, **kwargs)
                if self.config.cb_enabled:
                    cb.record_success()
                return res
            except Exception as e:
                logger.error(f"Error calling {model_name}: {e!s}")
                if self.note_rate_limit(model_name, e):
                    raise
                if self.config.cb_enabled:
                    cb.record_failure(e)
                last_error = e
                status_code = getattr(e, "status_code", None)
                if attempt == self.config.max_retries:
                    raise

                # Check if retryable
                if status_code and status_code not in self.config.retry_on_codes:
                    raise

        if last_error:
            raise last_error
