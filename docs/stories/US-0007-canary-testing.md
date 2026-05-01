# US-0007: Canary testing

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want a small percentage of traffic split to a
candidate model so I can validate quality on live traffic before
fully cutting over.

## Context

`CanaryConfig` enables traffic splitting: with `weight` probability
each request is sent to `canary_model` instead of the routed
production model. Canary outputs are recorded for later comparison.
Canary failures do not affect main traffic — fallback to production
on error is the responsibility of the caller (or wired via
ResilienceConfig).

## Acceptance criteria

- [ ] With weight=0.05, ~5% of N requests routed to canary
- [ ] weight=0 → no canary traffic
- [ ] weight=1 → all canary traffic (effectively full cutover)
- [ ] Canary disabled by default
- [ ] Canary selection is per-request, deterministic given seed
      (for reproducibility in tests)
- [ ] Canary errors do not break the request — fall through to
      production model

## Implementation

- Module: `routellm/quality.py`
- Config: `CanaryConfig(enabled, canary_model, weight)`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_canary_selection`
- `routellm/tests/test_e2e_scenarios.py::test_scenario_canary_flow`

## Related

- US-0008 — trace collection (canary outputs recorded for compare)
