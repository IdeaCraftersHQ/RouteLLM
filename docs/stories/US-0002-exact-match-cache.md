# US-0002: Exact-match SQLite cache

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want repeat identical prompts answered from a
local SQLite cache so I don't pay for the same call twice.

## Context

`CacheConfig` enables a persistent SQLite-backed cache keyed on the
canonical prompt hash. TTL configurable. Hits return without
calling any model; misses store after the routed call returns.

## Acceptance criteria

- [ ] Identical prompt within TTL → cache hit; no model call
- [ ] Identical prompt after TTL → miss; new model call; cache
      refreshed
- [ ] Different prompt → miss; cache populated
- [ ] Cache disabled by default (must be explicitly enabled)
- [ ] Cache backend uses local SQLite path; no remote dependency
- [ ] Cache survives process restart

## Implementation

- Module: `routellm/caching.py`
- Config: `CacheConfig`
- Wired in: `routellm/controller.py`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_cache_ttl`

## Related

- US-0003 — semantic cache (alternative match strategy)
- US-0008 — trace collection (cache misses fed to traces)
