# US-0102: Pluggable cache policy hooks

**Status:** paper
**Persona:** Platform operator / API consumer
**Slug:** RLM-NEW-CACHE-HOOKS

## User goal

As a platform operator, I want cache decisions (skip / lookup /
store / TTL / namespace) driven by per-request classification so
PII-bearing prompts are never cached, code prompts use exact-match
only, and cross-tenant reuse happens only when classification says
it's safe.

## Context

Caching today (US-0002 exact, US-0003 semantic) is global and
binary: enabled or not. Showcase scenario 3 needs differentiated
policy per request class. c12n classifies on PII / Domain /
Complexity / CodeContent / Toxicity / Jailbreak / CostEstimate /
OutputFormat — these labels should drive cache behavior.

Hook point: a callable `cache_policy: Callable[[Request,
ClassificationResult], CachePolicy]` invoked between request
intake and cache lookup. Returns:

```
CachePolicy:
  action: skip | lookup_only | lookup_and_store | store_only
  ttl: seconds (overrides config default)
  namespace: str (default: tenant_id; can override for global-safe)
  match: exact | semantic | both
```

Default policy = today's behavior (lookup_and_store, default TTL,
tenant namespace, both match strategies). Custom policy fully
overrides.

## Acceptance criteria

- [ ] Policy hook called before cache lookup
- [ ] `action=skip` → no lookup, no store; route as if cache
      disabled
- [ ] `action=lookup_only` → may hit but never stores miss
- [ ] `action=store_only` → never lookup, always store after route
- [ ] `action=lookup_and_store` (default) → existing US-0002/3
      behavior
- [ ] Policy `ttl` overrides config TTL on store
- [ ] Policy `namespace` overrides tenant default (e.g.,
      `namespace="global:domain:math"` → cross-tenant reuse)
- [ ] Policy `match=exact` → semantic disabled for this request
      even if globally enabled
- [ ] Policy hook receives request + classification result;
      classification injection is responsibility of upstream
      middleware (not this story)
- [ ] No hook configured → default policy applied

## Implementation notes

- Touches `routellm/caching.py` + `routellm/controller.py`
- Type: `routellm/types.py` adds `CachePolicy` dataclass
- Reference impl in `extensions/policy_routing/` (showcase scenario
  3 task T-0104) — wires c12n classifier + maps labels per the
  table in scenario 3 spec
- Backwards compat: no hook → identical to current behavior

## E2E tests

- planned: `routellm/tests/test_cache_policy.py::test_default_policy_unchanged_behavior`
- planned: `routellm/tests/test_cache_policy.py::test_skip_action`
- planned: `routellm/tests/test_cache_policy.py::test_lookup_only_no_store_on_miss`
- planned: `routellm/tests/test_cache_policy.py::test_store_only_no_lookup`
- planned: `routellm/tests/test_cache_policy.py::test_ttl_override`
- planned: `routellm/tests/test_cache_policy.py::test_namespace_override`
- planned: `routellm/tests/test_cache_policy.py::test_match_strategy_override`

## Related

- US-0002 — exact-match cache (consumer of policy)
- US-0003 — semantic cache (consumer of policy)
- US-0100 — auth (provides tenant_id for default namespace)
- US-0101 — audit (records `cache_namespace` per row)
- c12n integration — classification produces `ClassificationResult`
  consumed by policy
- showcase scenario 3 — class-aware LLM gateway (consumer)
