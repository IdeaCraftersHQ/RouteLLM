# US-0106: Traces to measured quality

**Status:** shipped
**Persona:** operator running a routed server

## User goal

As an operator, I want the `quality` number my selectors order on to
come from what my endpoints actually produced, not from a number
somebody typed into YAML once, so the router's ordering stays true as
the models behind the endpoints change.

## Context

`Endpoint.quality` is hand-written and every `quality_desc` selector
orders on it. Nobody retypes it when a model changes. Meanwhile the
server already recorded a trace of every routed request, in a private
format nothing read, missing the one fact that would make it useful:
which routing decision produced it.

This story closes the loop. Traffic becomes traces in fit's
`trace-format-v1` with the routing path attached; fit scores them
offline; the scores aggregate into a sidecar; the config points at the
sidecar and the same selectors now order on something measured. The
dead `trigger_fit` hook, which shelled out to a `fit train` command
that does not exist, was removed on the way past.

Scoring and aggregation are offline. Nothing in this story runs on the
serving path.

## Acceptance criteria

- [x] Each routed request records a fit-ingestible trace carrying its
      endpoint, request model, tier, area, decision path, cache and
      canary flags, latency, router and win rate
- [x] The pre-existing top-level `output` and `routed_model` keys are
      unchanged, and `record_trace`'s new arguments are keyword-only
      with None defaults, so existing callers are untouched
- [x] Trace ids do not collide within a millisecond, and writes are
      atomic
- [x] The trace directory is capped by file count and total bytes,
      oldest deleted first, with the listing cached for 60s
- [x] Recording never fails the request it describes
- [x] Recorded traces score offline against a fit `RewardScorer` with
      no LLM call, through four scorer specs
- [x] A spec that calls an LLM is refused without `--allow-llm`
- [x] Missing fit names the `routellm[fit]` extra
- [x] Scores aggregate into a versioned sidecar; below `--min-samples`
      an endpoint is omitted rather than written null
- [x] Aggregation imports no fit
- [x] `quality_from:` merges the sidecar into the registry at startup,
      before selectors resolve, with the sidecar winning by default and
      one INFO line per override
- [x] `quality_from_override: false` lets an explicit YAML quality win
- [x] A stale sidecar warns but still loads
- [x] `areas:` groups tiers, and a selector inside an area's tier
      orders on that area's measured number, falling back to overall
- [x] `trigger_fit` is gone and a config passing `min_confidence` still
      loads, with a deprecation warning

## Implementation

- Trace body and caps: `routellm/quality.py`
- Call sites and the area lookup: `routellm/controller.py`
- Scoring, aggregation, sidecar: `routellm/quality_scores.py`
- Areas and `area_quality` on the registry: `routellm/endpoints.py`
- Per-area ordering: `routellm/pairing.py`
- `quality_from` wiring: `routellm/openai_server.py`
- Config: `quality_from`, `quality_from_override`, `areas:`
- Extra: `routellm[fit]`
- Runbook: [docs/runbooks/traces-to-quality.md](../runbooks/traces-to-quality.md)

## E2E tests

- `routellm/tests/test_quality_traces.py::test_trace_has_every_fit_required_field`
- `routellm/tests/test_quality_traces.py::test_routellm_block_carries_the_path_and_endpoint`
- `routellm/tests/test_quality_traces.py::test_old_top_level_fields_are_unchanged`
- `routellm/tests/test_quality_traces.py::test_file_cap_deletes_oldest_first`
- `routellm/tests/test_quality_traces.py::test_byte_cap_trips_once_and_warns`
- `routellm/tests/test_quality_traces.py::test_ids_do_not_collide_within_a_millisecond`
- `routellm/tests/test_quality_traces.py::test_a_broken_response_still_records`
- `routellm/tests/test_quality_traces.py::test_controller_passes_the_path_into_the_trace`
- `routellm/tests/test_quality_traces.py::test_trigger_fit_is_gone`
- `routellm/tests/test_quality_traces.py::test_min_confidence_is_accepted_and_warned`
- `routellm/tests/test_quality_scores_score.py::test_composite_spec_scores_every_trace`
- `routellm/tests/test_quality_scores_score.py::test_missing_fit_names_the_extra`
- `routellm/tests/test_quality_scores_score.py::test_judge_spec_refused_without_allow_llm`
- `routellm/tests/test_quality_scores_score.py::test_rubric_scorer_is_deterministic`
- `routellm/tests/test_quality_scores_score.py::test_unknown_spec_lists_the_four_forms`
- `routellm/tests/test_quality_scores_score.py::test_rescore_off_skips_already_scored_ids`
- `routellm/tests/test_quality_scores_score.py::test_null_score_is_written_as_null`
- `routellm/tests/test_quality_scores_aggregate.py::test_mean_maps_to_the_zero_hundred_scale`
- `routellm/tests/test_quality_scores_aggregate.py::test_below_min_samples_is_left_out_entirely`
- `routellm/tests/test_quality_scores_aggregate.py::test_cached_traces_are_dropped`
- `routellm/tests/test_quality_scores_aggregate.py::test_by_area_splits_the_same_endpoint`
- `routellm/tests/test_quality_scores_aggregate.py::test_percentile_transform_spreads_clustered_scores`
- `routellm/tests/test_quality_scores_aggregate.py::test_aggregate_needs_no_fit`
- `routellm/tests/test_quality_from.py::test_no_quality_from_leaves_every_endpoint_untouched`
- `routellm/tests/test_quality_from.py::test_sidecar_overrides_the_yaml_quality_and_logs_it`
- `routellm/tests/test_quality_from.py::test_override_false_keeps_an_explicit_yaml_quality`
- `routellm/tests/test_quality_from.py::test_stale_sidecar_warns_but_loads`
- `routellm/tests/test_quality_from.py::test_selector_ordering_uses_the_merged_quality`
- `routellm/tests/test_quality_from.py::test_server_pops_quality_keys_out_of_router_config`
- `routellm/tests/test_quality_areas.py::test_areas_invert_into_tier_to_area`
- `routellm/tests/test_quality_areas.py::test_a_tier_in_two_areas_names_both`
- `routellm/tests/test_quality_areas.py::test_selector_in_an_area_tier_orders_on_the_area_quality`
- `routellm/tests/test_quality_areas.py::test_an_endpoint_without_an_area_number_falls_back_to_overall`
- `routellm/tests/test_quality_areas.py::test_the_recorded_trace_carries_the_area`
- `routellm/tests/test_examples.py::test_every_area_names_existing_tiers`

## Related

- US-0007 — canary testing; canary traces are counted, and marked
- US-0008 — trace collection; this story replaces its private format
  with fit's, and removes the `trigger_fit` hook it described
- US-0105 — endpoint registry and tiers; areas group those tiers and
  the sidecar merges into that registry
