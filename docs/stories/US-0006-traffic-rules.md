# US-0006: Traffic management rules

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want pattern-based rules that override the
default model pair for specific request shapes (e.g., code prompts
go to a code-tuned pair) so different request classes get the right
specialist.

## Context

`TrafficManager` evaluates ordered `TrafficRule` entries against the
incoming request. First match wins. Each rule specifies a regex
`pattern` and a target `strong_model` / `weak_model` pair. No match
→ falls through to the controller's default pair. Rules compose
with `LoadBalancer` — a rule's target model name can resolve to a
load balancer.

## Acceptance criteria

- [ ] Request matching rule pattern → routed to rule's
      strong/weak pair (not default)
- [ ] First-match wins among rules
- [ ] No-match request → default controller pair
- [ ] Rule target model name registered in LoadBalancer → balancer
      handles dispatch
- [ ] Empty TrafficManager → behaves identically to no
      TrafficManager
- [ ] Invalid regex in rule → fails fast at config time

## Implementation

- Module: `routellm/traffic.py`
- Config: `TrafficManager(rules, load_balancers)`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_traffic_manager_rules`
- `routellm/tests/test_resilience_caching_traffic.py::test_traffic_manager_conditional_routing`
- `routellm/tests/test_e2e_scenarios.py::test_scenario_traffic_rule_to_load_balancer`

## Related

- US-0005 — load balancer often used as a rule target
- US-0009 — intent-based routing (alternative routing mechanism;
  uses model classifier instead of regex)
