# US-0005: Load balancing across endpoints

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want multiple endpoints (different keys, regions,
or providers) for the same logical model so I can spread load and
sidestep per-key quotas.

## Context

`LoadBalancer` rotates calls across configured `LoadBalancerEndpoint`
entries using a strategy: `round-robin` (uniform rotation) or
`weighted` (proportional to endpoint weights). Each endpoint
specifies its own `model`, `api_key`, and optional `api_base`.

## Acceptance criteria

- [ ] Round-robin: N requests across K endpoints distribute
      uniformly (±1)
- [ ] Weighted: distribution matches configured weights within
      tolerance
- [ ] Single endpoint configured → all requests go to it
- [ ] Endpoint with distinct `api_base` → request routed to that
      base URL
- [ ] Endpoint failure does not corrupt rotation state for the
      others
- [ ] Strategy is per-`LoadBalancerConfig`, not global

## Implementation

- Module: `routellm/traffic.py`
- Config: `LoadBalancer`, `LoadBalancerConfig`,
  `LoadBalancerEndpoint`

## E2E tests

- `routellm/tests/test_resilience_caching_traffic.py::test_load_balancer`

## Related

- US-0006 — traffic rules can route a logical model to a load
  balancer
- US-0004 — resilience runs per-endpoint failure tracking
