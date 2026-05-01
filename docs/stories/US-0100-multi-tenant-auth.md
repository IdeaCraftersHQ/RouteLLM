# US-0100: Multi-tenant API auth

**Status:** paper
**Persona:** Platform operator / API consumer
**Slug:** RLM-NEW-AUTH

## User goal

As a platform operator, I want each employee to call routellm with
their own API key and have requests scoped to their tenant so I
can attribute usage, enforce per-tenant cache isolation, and revoke
access without rotating shared secrets.

## Context

Today the OpenAI-compatible server (US-0010) accepts any caller
holding the upstream provider key. For LAN + VPN business
deployment (showcase scenario 3), we need:

- Per-employee bearer tokens issued by the gateway, not by upstream
- Tenant scoping (a "tenant" = a company / squad; multiple
  employees per tenant)
- Cache isolation: tenant A's cached prompt MUST NOT serve tenant
  B's request unless explicitly marked global-safe (e.g.,
  `Domain: math` per c12n)
- Token issuance + revocation via CLI (no web admin v1)

Decouple from upstream provider keys: routellm holds upstream
secrets in config; tenants never see them.

## Acceptance criteria

- [ ] Request without `Authorization: Bearer <token>` → 401
- [ ] Token issued via `routellm token issue --tenant <id>
      --user <id>` → opaque string, persisted to token store
- [ ] Token revoked via `routellm token revoke <token>` → next
      request 401
- [ ] Token expiry supported (`--expires <duration>`); expired
      token → 401
- [ ] Cache lookup keyed by `(tenant, prompt_hash)` — cross-tenant
      hits denied by default
- [ ] Trace rows (US-0008) carry `tenant_id` + `user_id`
- [ ] Token store backend pluggable (default: SQLite)
- [ ] Token store survives restart; in-flight requests not
      affected by issuance/revocation race

## Implementation notes

- New module: `routellm/auth.py`
- New CLI: `routellm token {issue|revoke|list}`
- Server middleware: enforce on every routed call; bypass for
  `/health`
- Cache key change: prepend tenant scope (touches US-0002 +
  US-0003 — backward compat: opt-in flag for v1)

## E2E tests

- planned: `routellm/tests/test_auth.py::test_request_without_token_401`
- planned: `routellm/tests/test_auth.py::test_token_issue_and_call`
- planned: `routellm/tests/test_auth.py::test_token_revoke_blocks_next_call`
- planned: `routellm/tests/test_auth.py::test_token_expiry_enforced`
- planned: `routellm/tests/test_auth.py::test_cache_isolated_per_tenant`
- planned: `routellm/tests/test_auth.py::test_cross_tenant_cache_hit_denied`

## Open questions

- Cache global-safe opt-in: per-tenant config flag, or per-prompt
  classification (deferred to RLM-NEW-CACHE-HOOKS — c12n labels
  drive it). **Decision: defer to cache-hooks story.** Auth story
  ships with strict per-tenant isolation only.

## Related

- US-0010 — OpenAI-compatible server (auth middleware mounts here)
- US-0002 / US-0003 — caching (key change touches both)
- US-0008 — trace collection (rows tagged with tenant)
- US-0101 — audit log (consumes auth identity)
- US-0102 — cache policy hooks (refines cross-tenant rules)
- showcase scenario 3 — class-aware LLM gateway
