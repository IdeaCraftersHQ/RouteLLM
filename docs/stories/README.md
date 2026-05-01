# RouteLLM stories

User stories backing the routellm runtime. Each story has acceptance
criteria + a `## E2E Tests` section linking concrete test files +
function names.

Convention: every story (existing + new) MUST link e2e tests by
file::function. Failing tests are acceptable; "not yet implemented"
prose without a planned test path is non-conforming. Pattern adopted
from aps story 004.

## Index

### Routing
- [US-0001](US-0001-strong-weak-routing.md) — strong/weak routing via cost threshold
- [US-0009](US-0009-intent-based-routing.md) — intent-based routing middleware

### Caching
- [US-0002](US-0002-exact-match-cache.md) — exact-match SQLite cache
- [US-0003](US-0003-semantic-cache.md) — semantic cache via embeddings

### Resilience
- [US-0004](US-0004-resilience.md) — retry, circuit breaker, fallback, timeout

### Traffic
- [US-0005](US-0005-load-balancing.md) — load balancing (round-robin, weighted)
- [US-0006](US-0006-traffic-rules.md) — traffic management rules

### Quality
- [US-0007](US-0007-canary-testing.md) — canary testing
- [US-0008](US-0008-trace-collection.md) — trace collection for fine-tuning

### Server
- [US-0010](US-0010-openai-compatible-server.md) — OpenAI-compatible server

### Auth + observability (gap series — paper)
- [US-0100](US-0100-multi-tenant-auth.md) — multi-tenant API auth
- [US-0101](US-0101-audit-log.md) — audit log of routed requests
- [US-0102](US-0102-cache-policy-hooks.md) — pluggable cache policy hooks

## Status legend

- **shipped** — code present + e2e green
- **shipped, no-e2e** — code present, e2e missing or broken
- **partial** — code present, only some acceptance criteria met
- **paper** — story written, no impl

When advancing a story to `shipped`, update both this index and the
story's frontmatter.
