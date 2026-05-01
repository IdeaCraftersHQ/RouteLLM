# US-0001: Strong/weak routing via cost threshold

**Status:** shipped, no-e2e (e2e exists but not previously linked)
**Persona:** API consumer

## User goal

As a developer, I want simple LLM requests routed to a cheap model
and complex ones to a strong model so cost drops while quality holds
on hard queries.

## Context

Each request carries a cost threshold (encoded in the model field
as `router-<name>-<threshold>`). The configured router scores the
request; score above threshold → strong model; else weak. Supported
routers: `mf` (recommended), `bert`, `sw_ranking`, `causal_llm`,
`random`. Threshold calibrated via
`routellm.calibrate_threshold` against the user's traffic.

## Acceptance criteria

- [ ] Request with `model=router-mf-0.5` invokes mf router
- [ ] Score ≥ threshold → strong model called
- [ ] Score < threshold → weak model called
- [ ] Unknown router name → 4xx with descriptive error
- [ ] Threshold outside [0, 1] → 4xx
- [ ] Calibration command produces threshold for target strong-pct
- [ ] All 5 documented routers selectable

## Implementation

- Router selection: `routellm/routers/routers.py`
- Controller: `routellm/controller.py`
- Calibration: `routellm/calibrate_threshold.py`

## E2E tests

- `routellm/tests/test_controller_integration.py::test_controller_full_flow`

## Related

- US-0002 — exact-match cache (intercepts before routing)
- US-0009 — intent-based routing (alternative routing layer)
