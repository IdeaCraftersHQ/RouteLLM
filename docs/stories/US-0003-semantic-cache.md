# US-0003: Semantic cache via embeddings

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want semantically similar prompts to share cache
hits ("hello" / "hi there") so I avoid paying for paraphrases.

## Context

Semantic cache embeds the prompt and matches against cached entries
above a similarity threshold (default 0.95). Below threshold →
miss. Operates alongside exact-match cache; semantic checked when
exact misses.

## Acceptance criteria

- [ ] Two prompts with cosine similarity ≥ threshold → semantic hit
- [ ] Similarity < threshold → miss; routed call; cache stored
- [ ] Threshold configurable per CacheConfig
- [ ] Semantic disabled by default
- [ ] Embedding provider injectable (mockable for tests)
- [ ] Semantic miss does not corrupt or evict exact-match entries

## Implementation

- Module: `routellm/caching.py`
- Config: `CacheConfig(semantic_enabled, semantic_threshold)`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_semantic_cache`

## Related

- US-0002 — exact-match cache (checked first)
- (future) RLM-NEW-CACHE-HOOKS — class-aware cache policy via c12n
