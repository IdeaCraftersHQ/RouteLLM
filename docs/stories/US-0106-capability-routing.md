# US-0106: Capability tracking and capability-aware routing

**Status:** shipped
**Persona:** API consumer
**Slug:** RLM-CAPABILITY-ROUTING

## User goal

As an operator, I want each endpoint to carry a typed record of what it
can actually do, and each request to be checked against it before a
classifier runs, so a request carrying an image is never routed to a
text-only model and the first sign of trouble is not a provider error
after the tokens were spent.

## Context

Routing picked on *difficulty* and on *policy* but never on *fitness*.
The capability facts that existed were free-form tags -- `tools`,
`vision`, `long_context` -- an operator wrote by hand: invisible to the
router, drifting from the catalog for every model the catalog knows,
and unable to express "200k context" or "takes images but not audio".

`routellm/capabilities.py` introduces `Capabilities`, a typed record
merged per endpoint from three sources, highest precedence first: the
explicit `capabilities:` block, the deprecated tag aliases, then the
models.dev record the pairing path already fetches. `None` is the only
unknown marker, so "known false" and "not known" are different values
rather than a convention.

`routellm/requirements.py` reads what a request needs off the request:
`vision` from image parts, `tools` from `tools` or `functions`,
`structured_output` from `response_format`, and `context_needed` from
litellm's token counter plus `max_tokens`. Nothing else is read.

`routellm/routing.py` checks both sides of each level before running
its router. Both pass: unchanged. One passes: it is taken with no
router call, and the path records `capability_forced` and
`capability_requirement`. Neither: a `RoutingError`, which the server
already returns as a 400, before any upstream request.

Two opposite defaults for unknown, both deliberate. At *selection*
time an unknown capability fails its term, because startup is where an
operator can see the gap and `--capabilities` prints it. At *request*
time an unknown capability serves, because a config that declares
nothing must keep routing exactly as it did; `strict: true` per
endpoint flips it.

## Acceptance criteria

- [x] `capabilities:` on an endpoint states `vision`, `tools`,
      `structured_output`, `reasoning`, `open_weights`, `context`,
      `max_output`, `modalities_in`; an omitted field is unknown, which
      is distinct from `false`
- [x] Precedence is the explicit block, then the deprecated tag
      aliases, then the models.dev record; the aliases warn once per
      alias per process and tags stay selectable with `tag:`
- [x] Selectors take `vision:`, `tools:`, `structured_output:`,
      `reasoning:`, `open_weights:`, `context:>=N`, `context:<=N`,
      `max_output:>=N`, `input:image`, plus a `max_output_desc` order;
      `tools` is an alias of `tool_call`; a bare `context:N` is an
      error naming the term and showing the `>=` spelling
- [x] An unknown capability fails its term at selection time, and a
      failed catalog fetch is tolerated when every endpoint answers the
      query from its own block
- [x] Requirements are derived with no router and no network, and never
      raise on malformed message content
- [x] A level whose two sides both pass is byte-identical to what it
      was; exactly one passing takes it with no router call; neither
      passing raises a `RoutingError` naming the tier and both failures
- [x] A tier-valued side passes when any reachable leaf passes, folded
      once at construction and cached on the controller
- [x] Flat pairs -- the legacy pair, a traffic rule's, a middleware's
      -- get the same check, and an incapable fallback sibling is
      dropped
- [x] At request time an unknown capability serves and logs once per
      endpoint and requirement; `strict: true` refuses instead
- [x] `context` is judged only when both the count and the window are
      known
- [x] A request with no images, no tools and no `response_format`
      keeps the exact path key set it had
- [x] `/v1/models` entries carry an additive `routellm` object; the
      standard OpenAI keys keep their values and order
- [x] `python -m routellm.pairing --config c.yaml --capabilities`
      prints the matrix, `--capabilities --explain` prints both, and
      `--explain` alone is byte-identical to the previous output
- [x] `examples/multitier.yaml` carries no deprecated capability tag,
      every local endpoint declares its block, one endpoint sets
      `strict: true`, and one tier selects on a capability term

## E2E Tests

- `routellm/tests/test_capabilities.py::test_explicit_block_beats_the_catalog_record`
- `routellm/tests/test_capabilities.py::test_catalog_record_fills_what_the_config_omits`
- `routellm/tests/test_capabilities.py::test_unknown_stays_none_when_neither_source_knows`
- `routellm/tests/test_capabilities.py::test_known_false_is_distinct_from_unknown`
- `routellm/tests/test_capabilities.py::test_tools_tag_populates_the_typed_block_with_one_warning`
- `routellm/tests/test_capabilities.py::test_long_context_tag_sets_a_context_floor`
- `routellm/tests/test_capabilities.py::test_tags_survive_the_alias_and_stay_selectable`
- `routellm/tests/test_capabilities.py::test_anonymous_endpoint_carries_capabilities_none`
- `routellm/tests/test_capabilities.py::test_old_snapshot_without_the_new_fields_still_reads`
- `routellm/tests/test_capabilities.py::test_explicit_block_beats_a_tag_alias`
- `routellm/tests/test_capabilities.py::test_a_tag_alias_beats_the_catalog_record`
- `routellm/tests/test_capabilities.py::test_record_modalities_and_max_output_reach_the_block`
- `routellm/tests/test_capability_selectors.py::test_vision_true_selects_only_vision_endpoints`
- `routellm/tests/test_capability_selectors.py::test_tools_is_an_alias_of_tool_call`
- `routellm/tests/test_capability_selectors.py::test_context_ge_filters_on_the_merged_context`
- `routellm/tests/test_capability_selectors.py::test_context_le_filters_the_other_way`
- `routellm/tests/test_capability_selectors.py::test_bare_context_term_is_rejected_with_the_ge_spelling`
- `routellm/tests/test_capability_selectors.py::test_input_image_matches_modalities_in`
- `routellm/tests/test_capability_selectors.py::test_unknown_capability_fails_the_term`
- `routellm/tests/test_capability_selectors.py::test_local_endpoint_with_an_explicit_block_wins_without_tags`
- `routellm/tests/test_capability_selectors.py::test_max_output_desc_orders_and_puts_unknown_last`
- `routellm/tests/test_capability_selectors.py::test_structured_output_term_no_longer_raises_unsupported`
- `routellm/tests/test_capability_selectors.py::test_family_term_still_raises_unsupported`
- `routellm/tests/test_capability_selectors.py::test_catalog_failure_is_tolerated_when_blocks_answer_everything`
- `routellm/tests/test_capability_selectors.py::test_catalog_failure_raises_when_a_term_needs_the_record`
- `routellm/tests/test_capability_selectors.py::test_capability_key_sets_are_what_the_plan_fixes`
- `routellm/tests/test_capability_selectors.py::test_split_terms_routes_each_key_to_its_bucket`
- `routellm/tests/test_requirements.py::test_plain_text_request_has_no_requirements`
- `routellm/tests/test_requirements.py::test_image_url_part_sets_vision`
- `routellm/tests/test_requirements.py::test_input_image_part_sets_vision`
- `routellm/tests/test_requirements.py::test_string_content_never_sets_vision`
- `routellm/tests/test_requirements.py::test_malformed_content_parts_do_not_raise`
- `routellm/tests/test_requirements.py::test_tools_key_sets_tools`
- `routellm/tests/test_requirements.py::test_functions_key_sets_tools`
- `routellm/tests/test_requirements.py::test_empty_tools_list_does_not_set_tools`
- `routellm/tests/test_requirements.py::test_json_schema_response_format_sets_structured_output`
- `routellm/tests/test_requirements.py::test_json_object_response_format_sets_structured_output`
- `routellm/tests/test_requirements.py::test_text_response_format_sets_nothing`
- `routellm/tests/test_requirements.py::test_context_needed_counts_prompt_plus_max_tokens`
- `routellm/tests/test_requirements.py::test_token_counter_failure_falls_back_to_chars_over_four`
- `routellm/tests/test_requirements.py::test_streaming_and_sampling_fields_change_nothing`
- `routellm/tests/test_requirements.py::test_is_empty_is_true_only_for_a_plain_request`
- `routellm/tests/test_requirements.py::test_a_plain_request_needs_no_capability`
- `routellm/tests/test_requirements.py::test_prompt_text_joins_list_content_and_passes_a_string_through`
- `routellm/tests/test_requirements.py::test_unknown_model_name_still_counts`
- `routellm/tests/test_capability_routing.py::test_a_text_request_routes_exactly_as_before`
- `routellm/tests/test_capability_routing.py::test_a_vision_request_is_forced_to_the_only_capable_side`
- `routellm/tests/test_capability_routing.py::test_capability_forced_names_the_side_and_the_requirement`
- `routellm/tests/test_capability_routing.py::test_no_capable_side_raises_routing_error_naming_the_tier`
- `routellm/tests/test_capability_routing.py::test_a_tier_side_passes_when_any_reachable_leaf_passes`
- `routellm/tests/test_capability_routing.py::test_a_tier_side_fails_when_no_reachable_leaf_passes`
- `routellm/tests/test_capability_routing.py::test_build_tier_index_unions_the_reachable_leaves`
- `routellm/tests/test_capability_routing.py::test_flat_pair_from_a_traffic_rule_is_checked_too`
- `routellm/tests/test_capability_routing.py::test_middleware_pair_is_checked_too`
- `routellm/tests/test_capability_routing.py::test_legacy_flat_pair_is_checked_too`
- `routellm/tests/test_capability_routing.py::test_an_incapable_fallback_sibling_is_skipped`
- `routellm/tests/test_capability_routing.py::test_unknown_capability_serves_by_default_and_logs`
- `routellm/tests/test_capability_routing.py::test_unknown_capability_refuses_under_strict`
- `routellm/tests/test_capability_routing.py::test_strict_endpoint_refuses_a_request_it_cannot_prove`
- `routellm/tests/test_capability_routing.py::test_context_requirement_uses_the_merged_context_window`
- `routellm/tests/test_capability_routing.py::test_context_unknown_on_both_sides_never_refuses`
- `routellm/tests/test_capability_routing.py::test_vision_message_reaches_the_router_as_text`
- `routellm/tests/test_capability_routing.py::test_server_returns_400_with_a_json_body_for_a_refusal`
- `routellm/tests/test_capability_compat.py::test_config_without_capabilities_loads_and_routes`
- `routellm/tests/test_capability_compat.py::test_plain_request_takes_the_same_leaf_as_before`
- `routellm/tests/test_capability_compat.py::test_path_keys_are_unchanged_for_a_plain_request`
- `routellm/tests/test_capability_compat.py::test_router_is_still_called_once_per_level_for_a_plain_request`
- `routellm/tests/test_capability_compat.py::test_no_capability_block_anywhere_means_no_refusal_ever`
- `routellm/tests/test_capability_compat.py::test_selectors_without_capability_terms_pick_what_they_picked`
- `routellm/tests/test_capability_compat.py::test_examples_multitier_still_resolves_every_selector`
- `routellm/tests/test_capability_compat.py::test_token_counting_imports_no_torch`
- `routellm/tests/test_capability_surfaces.py::test_models_lists_tier_capabilities`
- `routellm/tests/test_capability_surfaces.py::test_models_entry_keeps_the_openai_keys_untouched`
- `routellm/tests/test_capability_surfaces.py::test_models_marks_unknown_capabilities`
- `routellm/tests/test_capability_surfaces.py::test_router_entry_has_no_tier_capabilities`
- `routellm/tests/test_capability_surfaces.py::test_capabilities_flag_prints_the_matrix`
- `routellm/tests/test_capability_surfaces.py::test_capabilities_flag_marks_unknown_with_a_question_mark`
- `routellm/tests/test_capability_surfaces.py::test_capabilities_and_explain_print_both`
- `routellm/tests/test_capability_surfaces.py::test_matrix_names_the_leaf_behind_a_tier_capability`
- `routellm/tests/test_capability_surfaces.py::test_matrix_flags_terms_used_by_selectors_with_unknowns`
- `routellm/tests/test_capability_surfaces.py::test_explain_without_the_flag_is_byte_identical_to_today`
- `routellm/tests/test_capability_surfaces.py::test_matrix_renders_a_known_false_as_no`
- `routellm/tests/test_capability_surfaces.py::test_matrix_reports_no_unknowns_when_every_block_is_complete`
- `routellm/tests/test_capability_surfaces.py::test_record_without_modalities_leaves_vision_unknown`

## Breaking changes

- **`open_weights` and `structured_output` in a selector.** These
  previously raised "not supported by routellm pairing". They are now
  answered from the merged capabilities. A config that relied on the
  error to catch a typo no longer gets one.
- **The capability tags are deprecated.** `tools`, `vision` and
  `long_context` still populate the typed block for one release, each
  logging one warning per process. They stop populating it in the next
  release; write a `capabilities:` block instead. The tags themselves
  remain selectable with `tag:`.

## Related

- US-0105 -- endpoint registry and tiers (the registry, tiers, and the
  decision path this builds on)
- US-0010 -- OpenAI-compatible server (the `/v1/models` surface the
  `routellm` object extends, and the 400 a refusal reuses)
- US-0001 -- strong/weak routing via cost threshold (the flat pair the
  capability check also covers)
