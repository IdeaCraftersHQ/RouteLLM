# US-0101: Audit log of routed requests

**Status:** paper
**Persona:** Platform operator / Finance
**Slug:** RLM-NEW-AUDIT

## User goal

As a platform operator, I want every request recorded with who
made it, which model answered, whether the cache hit, and what it
cost so I can produce per-tenant spend reports, debug routing
behavior, and enforce budgets.

## Context

Trace collection (US-0008) records prompt + response for fine-tuning.
Audit logging is a separate concern: who-when-which-cost
attribution that survives independently of model output payloads
(which may be redacted under PII / privacy policy). Audit and
trace can share storage but must be queryable separately.

Audit row per request, regardless of cache hit / miss / failure.
Failed requests (4xx, 5xx, timeout) audited too — important for
incident debugging.

## Acceptance criteria

- [ ] Every successful routed request → 1 audit row
- [ ] Cache hit → audit row with `cache_status=hit`, no upstream
      cost
- [ ] Cache miss → audit row with `cache_status=miss`, upstream
      cost
- [ ] Failed request → audit row with `outcome=failed`, error code
- [ ] Row fields: `(ts, tenant_id, user_id, prompt_hash, model,
      route_decision, cache_status, tokens_in, tokens_out, cost,
      latency_ms, outcome, error_class)`
- [ ] No raw prompt or response stored in audit (PII-safe by
      default; pointer to trace row optional)
- [ ] CLI: `routellm audit query --tenant <id> --since <ts>` →
      paginated rows
- [ ] CLI: `routellm audit spend --tenant <id> --since <ts>` →
      aggregate cost report
- [ ] Storage backend pluggable (default: SQLite); rows append-only
- [ ] Disabled by default; opt-in via config

## Implementation notes

- New module: `routellm/audit.py`
- New CLI: `routellm audit {query|spend}`
- Server middleware: emits row from response post-processing
  (covers cache hit and miss paths uniformly)
- Independent of US-0008 trace collection — different table,
  different retention

## E2E tests

- planned: `routellm/tests/test_audit.py::test_audit_row_per_request`
- planned: `routellm/tests/test_audit.py::test_cache_hit_audited_no_cost`
- planned: `routellm/tests/test_audit.py::test_failed_request_audited`
- planned: `routellm/tests/test_audit.py::test_audit_query_filters`
- planned: `routellm/tests/test_audit.py::test_audit_spend_aggregation`
- planned: `routellm/tests/test_audit.py::test_audit_no_raw_prompt_stored`

## Related

- US-0008 — trace collection (sister concern, distinct storage)
- US-0100 — auth (provides tenant_id + user_id to row)
- showcase scenario 3 — class-aware LLM gateway
