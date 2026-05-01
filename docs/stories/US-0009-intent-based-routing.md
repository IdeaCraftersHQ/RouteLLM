# US-0009: Intent-based routing middleware

**Status:** shipped
**Persona:** API consumer

## User goal

As a developer, I want detected request intent (marketing copy,
technical Q&A, code generation, etc.) to drive model selection so
specialist models handle the requests they're best at.

## Context

Intent middleware classifies the prompt's intent and selects the
strong/weak model pair from a configured intent → pair mapping.
Mappings persist to disk; can be edited via the web UI or CLI.
Intent middleware composes with traffic rules (US-0006) — intent
detection runs after rules but before the cost-threshold router
(US-0001).

## Acceptance criteria

- [ ] Marketing-intent prompt → marketing-pair selected
- [ ] Technical-intent prompt → technical-pair selected
- [ ] Unknown / ambiguous intent → default-pair fallback
- [ ] Mappings load from persisted file at controller init
- [ ] Saving updated mappings persists across restart
- [ ] Web UI accessible (Phase 2 — UI polish; CLI parity for v1)

## Implementation

- Module: `routellm/middleware/intent_model_selector.py`

## E2E tests

- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelector::test_get_model_pair_marketing`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelector::test_get_model_pair_technical`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelector::test_get_model_pair_default`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelector::test_save_load_mappings`

## Related

- US-0001 — cost-threshold router (composes after intent decision)
- US-0006 — traffic rules (composes before intent middleware)
- (future) c12n integration — intent detection could be replaced
  by c12n classification signals for richer per-class routing
