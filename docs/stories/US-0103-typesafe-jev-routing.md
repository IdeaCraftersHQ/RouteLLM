# US-0103: TypeSafe Jev routing

**Status:** shipped
**Persona:** API consumer
**Slug:** RLM-NEW-JEV-ROUTING

## User goal

As a developer, I want a router backed by TypeSafe's Jev model so
routing decisions come from a hosted judgment call instead of a
locally-trained classifier, and I want the same Jev-backed
detection available to intent-based model selection (US-0009) so
both routing and intent middleware can share one provider.

## Context

Existing routers (`mf`, `causal_llm`, `bert`, `sw_ranking`,
`random`) all run local models or heuristics. TypeSafe's Jev is a
hosted System One model that turns a prompt into a typed judgment;
a `jev` router calls it to score prompt difficulty and pick
strong/weak accordingly, at a documented confidence threshold.

Intent-based routing (US-0009) currently detects intent via a
configured LLM call inline in `IntentModelSelector`. This story
extracts that into a pluggable `intent_detector` slot so a
`JevIntentDetector` (calling the same TypeSafe SDK) and the
existing `DomainIntentDetector`-style detection both fit the same
interface. Below a confidence floor, `JevIntentDetector` returns
`"general"` rather than guessing.

Calibration run against model `jev-1.13.0` over 200 arena prompts
sets the documented threshold at 0.33 (50% strong-call rate).

The TypeSafe SDK is an optional dependency: importing `routellm`
must not require it. The `ImportError` naming the missing extra
surfaces only when a `Controller` is actually constructed with
`jev` selected, not at module import time.

## Acceptance criteria

- [x] `routers=["jev"]` routes a hard prompt to strong and a
      trivial prompt to weak at a documented threshold (0.33)
- [x] Router config accepts `model`, `timeout`, `max_prompt_chars`;
      API key comes only from `TYPESAFE_API_KEY`
- [x] `IntentModelSelector` accepts an `intent_detector` and
      delegates `detect_intent` to it; `JevIntentDetector` and
      `DomainIntentDetector` both fit that slot
- [x] Below the confidence floor, `JevIntentDetector` returns
      `"general"`
- [x] SDK missing → `ImportError` naming the extra at `Controller`
      construction, not at `import routellm`
- [x] Response model id is logged so thresholds can be pinned to a
      versioned model

## Implementation notes

Modules present:

- `routellm/routers/typesafe/router.py` — `jev` router, registered
  in `ROUTER_CLS` (`routellm/routers/routers.py`)
- `routellm/routers/base.py` — shared router base the `jev` router
  extends
- `routellm/middleware/jev_intent_detector.py` — `JevIntentDetector`
- `routellm/middleware/intent_model_selector.py` — `IntentModelSelector`
  with the `intent_detector` kwarg

## E2E tests

- `routellm/tests/test_jev_router.py::test_win_rate_is_noul_probability`
- `routellm/tests/test_jev_router.py::test_route_threshold`
- `routellm/tests/test_jev_router.py::test_prompt_truncated`
- `routellm/tests/test_jev_router.py::test_model_from_config`
- `routellm/tests/test_jev_router.py::test_custom_criteria_in_request`
- `routellm/tests/test_jev_router.py::test_api_error_propagates`
- `routellm/tests/test_jev_router.py::test_missing_sdk_error`
- `routellm/tests/test_jev_router.py::test_registered`
- `routellm/tests/test_jev_router.py::test_str_is_jev`
- `routellm/tests/test_jev_intent_detector.py::test_detect_intent_returns_choice`
- `routellm/tests/test_jev_intent_detector.py::test_confidence_floor_returns_general`
- `routellm/tests/test_jev_intent_detector.py::test_probabilities_map_matches_intents`
- `routellm/tests/test_jev_intent_detector.py::test_criteria_built_from_mappings`
- `routellm/tests/test_jev_intent_detector.py::test_selector_integration`
- `routellm/tests/test_jev_intent_detector.py::test_close_delegates_to_client_close`
- `routellm/tests/test_jev_intent_detector.py::test_get_intent_confidence_fills_missing_intent_with_zero`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelectorPluggableDetector::test_pluggable_detector_known_intent_routes_to_pair`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelectorPluggableDetector::test_pluggable_detector_unknown_intent_falls_back_to_default`
- `routellm/tests/test_typesafe_guard.py::test_require_typesafe_sdk_returns_module_when_importable`
- `routellm/tests/test_typesafe_guard.py::test_require_typesafe_sdk_raises_with_extra_hint_when_missing`

## Related

- US-0001 — strong/weak routing via cost threshold (sibling router
  family; `jev` joins `ROUTER_CLS`)
- US-0009 — intent-based routing middleware (`IntentModelSelector`
  gains the `intent_detector` slot this story defines)
