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
binary: enabled or not. A class-aware LLM gateway needs differentiated
policy per request class. c12n classifies on PII / Domain /
Complexity / CodeContent / Toxicity / Jailbreak / CostEstimate /
OutputFormat — these labels should drive cache behavior.

Primary namespace key = `workspace_id` (resolved from request auth
token → caller profile → workspace). Tenant-scoped is too coarse:
one tenant can hold multiple workspaces (client-A vs client-B);
shared `tenant_id` namespace leaks `Domain: legal` / `PII` rows
across client engagements within a single employee profile.
`tenant_id` remains a valid override when sibling workspaces under
one tenant want shared cache.

Hook point: a callable `cache_policy: Callable[[Request,
ClassificationResult], CachePolicy]` invoked between request
intake and cache lookup. Returns:

```
CachePolicy:
  action: skip | lookup_only | lookup_and_store | store_only
  ttl: seconds (overrides config default)
  namespace: str (default: workspace_id; can override — e.g.
              tenant_id for sibling-workspace shared cache, or
              "global:domain:math" for cross-tenant reuse)
  match: exact | semantic | both
```

Default policy = today's behavior except namespace defaults to
`workspace_id` (lookup_and_store, default TTL, both match
strategies). Custom policy fully overrides.

## Acceptance criteria

- [ ] Policy hook called before cache lookup
- [ ] `action=skip` → no lookup, no store; route as if cache
      disabled
- [ ] `action=lookup_only` → may hit but never stores miss
- [ ] `action=store_only` → never lookup, always store after route
- [ ] `action=lookup_and_store` (default) → existing US-0002/3
      behavior
- [ ] Policy `ttl` overrides config TTL on store
- [ ] Default cache namespace is `workspace_id` resolved from
      request's auth token → caller profile → workspace name
- [ ] Policy `namespace` overrides workspace default (e.g.,
      `tenant_id` for sibling-workspace shared cache;
      `"global:domain:math"` for cross-tenant reuse)
- [ ] Policy `match=exact` → semantic disabled for this request
      even if globally enabled
- [ ] Policy hook receives request + classification result;
      classification injection is responsibility of upstream
      middleware (not this story)
- [ ] No hook configured → default policy applied

## Implementation notes

- Touches `routellm/caching.py` + `routellm/controller.py`
- Type: `routellm/types.py` adds `CachePolicy` dataclass
- Reference impl in `extensions/policy_routing/` — wires c12n
  classifier + maps labels per its cache policy table
- Backwards compat: no hook → identical to current behavior

## E2E tests

- planned: `routellm/tests/test_cache_policy.py::test_default_policy_unchanged_behavior`
- planned: `routellm/tests/test_cache_policy.py::test_skip_action`
- planned: `routellm/tests/test_cache_policy.py::test_lookup_only_no_store_on_miss`
- planned: `routellm/tests/test_cache_policy.py::test_store_only_no_lookup`
- planned: `routellm/tests/test_cache_policy.py::test_ttl_override`
- planned: `routellm/tests/test_cache_policy.py::test_namespace_override`
- planned: `routellm/tests/test_cache_policy.py::test_cross_workspace_namespace_isolation`
- planned: `routellm/tests/test_cache_policy.py::test_match_strategy_override`

## Related

- US-0002 — exact-match cache (consumer of policy)
- US-0003 — semantic cache (consumer of policy)
- US-0100 — auth (provides profile + workspace → workspace_id)
- US-0101 — audit (records `cache_namespace` + `workspace_id` per row)
- c12n integration — classification produces `ClassificationResult`
  consumed by policy
- class-aware LLM gateway deployment (consumer)
