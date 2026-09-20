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

The TypeSafe SDK and the `jev` router live in the
`extensions/typesafe` package, not core: importing
`routellm` must not require either. Without that package installed,
`jev` never appears in `ROUTER_CLS`, and selecting it raises a
`KeyError` listing the routers that are actually registered.

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
- [x] Without the extension installed, `jev` is not in `ROUTER_CLS`
      and requesting it raises `KeyError` listing the available
      routers
- [x] Response model id is logged so thresholds can be pinned to a
      versioned model

## Implementation notes

Modules present:

- `extensions/typesafe/routellm_typesafe/router.py` — `jev`
  router, registered through the `routellm.routers` entry point the
  extension's `pyproject.toml` declares
- `routellm/routers/base.py` — shared router base the `jev` router
  extends
- `extensions/typesafe/routellm_typesafe/intent_detector.py` —
  `JevIntentDetector`
- `routellm/middleware/intent_model_selector.py` — `IntentModelSelector`
  with the `intent_detector` kwarg

## E2E tests

- `extensions/typesafe/tests/test_router.py::test_win_rate_is_noul_probability`
- `extensions/typesafe/tests/test_router.py::test_route_threshold`
- `extensions/typesafe/tests/test_router.py::test_prompt_truncated`
- `extensions/typesafe/tests/test_router.py::test_model_from_config`
- `extensions/typesafe/tests/test_router.py::test_custom_criteria_in_request`
- `extensions/typesafe/tests/test_router.py::test_default_criteria_not_shared_between_routers`
- `extensions/typesafe/tests/test_router.py::test_debug_log_records_response_model_id`
- `extensions/typesafe/tests/test_router.py::test_api_error_propagates`
- `extensions/typesafe/tests/test_router.py::test_registered_through_entry_point`
- `extensions/typesafe/tests/test_router.py::test_str_is_jev`
- `extensions/typesafe/tests/test_intent_detector.py::test_detect_intent_returns_choice`
- `extensions/typesafe/tests/test_intent_detector.py::test_confidence_floor_returns_general`
- `extensions/typesafe/tests/test_intent_detector.py::test_probabilities_map_matches_intents`
- `extensions/typesafe/tests/test_intent_detector.py::test_criteria_built_from_mappings`
- `extensions/typesafe/tests/test_intent_detector.py::test_selector_integration`
- `extensions/typesafe/tests/test_intent_detector.py::test_close_delegates_to_client_close`
- `extensions/typesafe/tests/test_intent_detector.py::test_get_intent_confidence_fills_missing_intent_with_zero`
- `extensions/typesafe/tests/test_prompts.py::test_router_file_values_reach_request`
- `extensions/typesafe/tests/test_prompts.py::test_router_kwarg_beats_file`
- `extensions/typesafe/tests/test_prompts.py::test_router_without_file_uses_defaults`
- `extensions/typesafe/tests/test_prompts.py::test_router_file_criteria_only_keeps_default_instructions`
- `extensions/typesafe/tests/test_prompts.py::test_router_rejects_unknown_criteria_key`
- `extensions/typesafe/tests/test_prompts.py::test_router_rejects_non_string_criteria_value`
- `extensions/typesafe/tests/test_prompts.py::test_router_empty_prompt_file_path_raises`
- `extensions/typesafe/tests/test_prompts.py::test_detector_file_values_reach_request`
- `extensions/typesafe/tests/test_prompts.py::test_detector_instructions_kwarg_beats_file`
- `extensions/typesafe/tests/test_prompts.py::test_detector_descriptions_general_beats_file`
- `extensions/typesafe/tests/test_prompts.py::test_detector_without_file_uses_defaults`
- `extensions/typesafe/tests/test_prompts.py::test_detector_empty_prompt_file_path_raises`
- `extensions/typesafe/tests/test_prompts.py::test_example_prompt_file_matches_built_in_defaults`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelectorPluggableDetector::test_pluggable_detector_known_intent_routes_to_pair`
- `routellm/tests/test_intent_model_selector.py::TestIntentModelSelectorPluggableDetector::test_pluggable_detector_unknown_intent_falls_back_to_default`

## Related

- US-0001 — strong/weak routing via cost threshold (sibling router
  family; `jev` joins `ROUTER_CLS`)
- US-0009 — intent-based routing middleware (`IntentModelSelector`
  gains the `intent_detector` slot this story defines)
