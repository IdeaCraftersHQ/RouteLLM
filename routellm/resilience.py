"""Resilience utilities for fault-tolerant model routing.

Implements circuit breaker pattern for handling transient failures and
fallback mechanisms for routing.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Callable, List, Optional, Any, Dict, Awaitable
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
        fail_codes: List[int] = None,
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
        self.history: List[tuple[float, bool]] = [] # (timestamp, success)

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
            return True # Allow one trial
            
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
                logger.warning(f"Circuit Breaker: state changed to OPEN (failures: {self.failures}, rate: {self._get_fail_rate():.2f})")
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit Breaker: trial failed, state changed back to OPEN")
            self.state = CircuitState.OPEN

class ResilienceConfig(BaseModel):
    max_retries: int = 3
    retry_on_codes: List[int] = [429, 500, 502, 503, 504]
    backoff_factor: float = 2.0
    initial_backoff_ms: int = 500
    timeout_ms: int = 30000
    
    # Circuit Breaker
    cb_enabled: bool = True
    cb_fail_max: int = 5
    cb_fail_rate: float = 0.5
    cb_fail_wait_ms: int = 5000
    cb_rate_interval_ms: int = 60000
    cb_rate_minimum: int = 10

class Resilience:
    def __init__(self, config: ResilienceConfig = None):
        self.config = config or ResilienceConfig()
        self.cb_map: Dict[str, CircuitBreaker] = {}

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

    async def wrap_acompletion(self, model_name: str, fn: Callable[..., Awaitable[Any]], *args, **kwargs):
        cb = self.get_cb(model_name)
        
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            if self.config.cb_enabled and not cb.can_execute():
                raise Exception(f"Circuit breaker OPEN for {model_name}")

            if attempt > 0:
                wait = (self.config.initial_backoff_ms / 1000) * (self.config.backoff_factor ** (attempt - 1))
                logger.info(f"Retrying {model_name} (attempt {attempt}/{self.config.max_retries}) in {wait:.2f}s...")
                await asyncio.sleep(wait)

            try:
                res = await asyncio.wait_for(fn(*args, **kwargs), timeout=self.config.timeout_ms / 1000)
                if self.config.cb_enabled:
                    cb.record_success()
                return res
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout calling {model_name}: {str(e)}")
                if self.config.cb_enabled:
                    cb.record_failure(e)
                last_error = e
                if attempt == self.config.max_retries:
                    raise
            except Exception as e:
                logger.error(f"Error calling {model_name}: {str(e)}")
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
                wait = (self.config.initial_backoff_ms / 1000) * (self.config.backoff_factor ** (attempt - 1))
                logger.info(f"Retrying {model_name} (attempt {attempt}/{self.config.max_retries}) in {wait:.2f}s...")
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
                logger.error(f"Error calling {model_name}: {str(e)}")
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
