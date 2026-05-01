# US-0008: Trace collection for fine-tuning

**Status:** shipped
**Persona:** API consumer / model trainer

## User goal

As a model trainer, I want every routed request/response recorded
in a standardized format so I can build training datasets for fit
or run offline evaluations without instrumenting per-call.

## Context

`FineTuneConfig` enables trace collection. Each completion produces
a trace row capturing: prompt, response, model used, route decision,
latency, tokens, cost. Traces persisted via `QualityManager` to a
configurable backend (default: SQLite). Format aligned with fit's
expected ingest schema.

## Acceptance criteria

- [ ] Trace recorded per successful completion
- [ ] Trace includes prompt + response + model + ts + tokens
- [ ] Failed calls not traced (or traced separately as failures)
- [ ] Trace backend configurable
- [ ] Disabled by default
- [ ] Output format consumable by fit's training pipeline

## Implementation

- Module: `routellm/quality.py`
- Config: `QualityManager(fine_tune_config=FineTuneConfig(...))`

## E2E tests

- `routellm/tests/test_advanced_features_deep.py::test_quality_manager_traces`

## Related

- US-0007 — canary outputs recorded as traces (compare A/B)
- (downstream) fit `FIT-NEWSLETTER-ADVISOR` — consumes traces for
  advisor training
