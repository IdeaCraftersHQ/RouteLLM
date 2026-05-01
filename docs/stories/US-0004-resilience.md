# US-0004: Resilience — retry, circuit breaker, fallback, timeout

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want transient upstream failures retried, sustained
failures circuit-broken, and per-call timeouts enforced so my
service stays available when providers degrade.

## Context

`ResilienceConfig` combines four mechanisms:

- **Retry** — exponential backoff on 429/5xx, configurable max
  attempts.
- **Circuit breaker** — opens when success rate drops; HALF_OPEN
  state probes recovery; CLOSED on confirmed pass.
- **Fallback** — on exhausted retries, automatically calls the
  alternate model in the pair.
- **Timeout** — strict per-call deadline; raises on overrun.

## Acceptance criteria

- [ ] Transient 429 → retried with exponential backoff up to
      `max_retries`; succeeds on retry → no failure surfaced
- [ ] All retries exhausted → fallback model invoked
- [ ] Sustained failure rate above threshold → circuit OPEN; new
      calls short-circuit immediately
- [ ] After `cooldown` → circuit HALF_OPEN; next call probes
- [ ] Probe success → CLOSED; probe failure → OPEN again
- [ ] Per-call wall time > `timeout_ms` → timeout error raised
      regardless of retry state
- [ ] Resilience disabled by default; opt-in via ResilienceConfig

## Implementation

- Module: `routellm/resilience.py`
- Config: `ResilienceConfig(max_retries, timeout_ms, ...)`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_circuit_breaker_transitions`
- `routellm/tests/test_advanced_features_deep.py::test_circuit_breaker_rate_threshold`
- `routellm/tests/test_e2e_scenarios.py::test_scenario_resilience_exhaustion`

## Related

- US-0005 — load balancing (per-endpoint resilience tracked
  separately)
