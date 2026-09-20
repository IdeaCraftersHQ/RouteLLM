# US-0105: Endpoint registry and tiers

**Status:** shipped
**Persona:** API consumer
**Slug:** RLM-ENDPOINT-REGISTRY-TIERS

## User goal

As an operator, I want to name the models my router can reach and
compose them into nested strong/weak pairs, so one server can span a
cloud provider and two local runtimes, each with its own base URL and
credential, and so a routing decision is explainable after the fact.

## Context

A model was a raw litellm string carrying no base URL and no
credential of its own: one `--base-url` and one `--api-key` applied to
every call, so a deployment could reach exactly one provider. Routing
was also flat — one strong/weak pair, one router, one threshold.

`routellm/endpoints.py` introduces `Endpoint`, `Tier`, `Selector`, and
`EndpointRegistry`, built from the `endpoints:` and `tiers:` keys of
the `--config` file. An endpoint gives a reachable model a stable name
that routing, caching, and traces key on, plus its own `api_base` and
an `api_key_env` naming the variable read at call time — so a config
loads on a machine holding none of the keys. A tier is a named
strong/weak pair whose sides may themselves be tiers, so a request
addressed to a tier walks a tree, one router call per level, each
level resolving its router and threshold first-hit-wins over its own
value, the parent's, the request's, and the controller defaults.

`routellm/routing.py` holds the four model-name forms and the
recursive walk; `routellm/pairing.py` resolves a side that selects
over endpoint tags and the models.dev catalog into one endpoint name
at controller construction, so nothing downstream sees a policy. The
server exposes the tier and router names it answers to under
`GET /v1/models` and returns the decision path on a non-streamed
response.

## Acceptance criteria

- [x] `endpoints:` names a model with its own `api_base`, `tags`,
      `quality`, and `extra`; a name the registry does not know stays
      a raw litellm model name with the controller defaults
- [x] Credentials precedence is balancer, then the endpoint's own
      `api_base`/`api_key_env`, then `--base-url`/`--api-key`;
      `api_key_env` is read at call time and a missing variable is an
      error naming both the endpoint and the variable
- [x] Routers that embed build their OpenAI client on first use, not
      at import; an endpoint named `embedding` supplies its base URL
      and credential, and each router keeps its own `embedding_model`
- [x] `tiers:` composes endpoints and other tiers; router and
      threshold are inherited when a level names none, and the graph
      is validated for unknown references, namespace collisions,
      cycles, and depth
- [x] A non-streamed response carries the decision path under a
      top-level `routellm.path`, one entry per level, recording where
      each level's router and threshold came from
- [x] A tier side may be a `Selector` over endpoint tags plus the
      models.dev catalog, resolved to one endpoint name before the
      graph is re-validated, and explainable via
      `python -m routellm.pairing --config <file>`
- [x] `GET /v1/models` lists every tier name plus
      `router-<name>-<default_threshold>` per loaded router, and never
      a tier that was not configured or derived
- [x] A config `default` tier always wins; with none, both model flags
      together derive an implicit one and neither flag derives nothing
      at all, leaving the legacy flat form on the historic pair with a
      startup warning. One flag alone is an argument error

## Implementation notes

- `routellm/endpoints.py` — `Endpoint`, `Tier`, `Selector`,
  `EndpointRegistry` (`from_config`, `resolve`, `get_tier`,
  `revalidate`), `MAX_TIER_DEPTH`
- `routellm/routing.py` — `parse_model_name`, `resolve_level`,
  `resolve_tier`, `sibling_of`
- `routellm/pairing.py` — `rank_candidates`, `resolve_pairing`,
  `resolve_registry_pairings`, snapshot under `ROUTELLM_CATALOG_CACHE`
- `routellm/routers/embeddings.py` — `configure_embeddings`,
  `get_embedding_client`, `reset_embedding_client`
- `routellm/openai_server.py` — `build_registry`, `legacy_pair`,
  `GET /v1/models`, `--default-threshold`, the `routellm` key on the
  response
- `config.example.yaml` — worked `endpoints:`/`tiers:` pair spanning a
  cloud model, Ollama, and colibri

## E2E tests

- `routellm/tests/test_endpoints.py::test_from_config_builds_named_endpoints`
- `routellm/tests/test_endpoints.py::test_from_config_defaults_fill_omitted_fields`
- `routellm/tests/test_endpoints.py::test_from_config_without_endpoints_key_is_empty`
- `routellm/tests/test_endpoints.py::test_from_config_ignores_unknown_top_level_keys`
- `routellm/tests/test_endpoints.py::test_endpoint_name_charset_rejected`
- `routellm/tests/test_endpoints.py::test_endpoint_empty_model_rejected`
- `routellm/tests/test_endpoints.py::test_resolve_empty_name_raises`
- `routellm/tests/test_endpoints.py::test_get_unknown_name_lists_known_names`
- `routellm/tests/test_endpoints.py::test_resolve_known_name_returns_endpoint`
- `routellm/tests/test_endpoints.py::test_resolve_raw_model_warns_once`
- `routellm/tests/test_endpoints.py::test_resolve_warns_once_per_distinct_name`
- `routellm/tests/test_endpoints.py::test_credentials_reads_env_at_call_time`
- `routellm/tests/test_endpoints.py::test_credentials_missing_env_names_endpoint_and_variable`
- `routellm/tests/test_endpoints.py::test_credentials_fall_back_to_defaults`
- `routellm/tests/test_endpoints.py::test_credentials_of_anonymous_endpoint_are_the_defaults`
- `routellm/tests/test_endpoints.py::test_controller_defaults_to_empty_registry`
- `routellm/tests/test_endpoints.py::test_controller_passes_endpoint_call_params`
- `routellm/tests/test_endpoints.py::test_controller_request_kwargs_win_over_extra`
- `routellm/tests/test_endpoints.py::test_controller_extra_applied_when_request_is_silent`
- `routellm/tests/test_endpoints.py::test_controller_balancer_overrides_endpoint`
- `routellm/tests/test_endpoints.py::test_controller_balancer_target_model_equal_to_endpoint_name`
- `routellm/tests/test_endpoints.py::test_controller_raw_model_still_works`
- `routellm/tests/test_endpoints.py::test_acompletion_passes_endpoint_call_params`
- `routellm/tests/test_endpoints.py::test_acompletion_applies_endpoint_extra`
- `routellm/tests/test_router_embeddings.py::test_client_built_from_embedding_endpoint`
- `routellm/tests/test_router_embeddings.py::test_endpoint_wins_over_environment`
- `routellm/tests/test_router_embeddings.py::test_endpoint_missing_env_variable_raises`
- `routellm/tests/test_router_embeddings.py::test_registry_without_embedding_endpoint_falls_back_to_env`
- `routellm/tests/test_router_embeddings.py::test_environment_fallback_uses_base_url`
- `routellm/tests/test_router_embeddings.py::test_no_endpoint_and_no_env_raises_naming_both`
- `routellm/tests/test_router_embeddings.py::test_client_is_cached_across_calls`
- `routellm/tests/test_router_embeddings.py::test_reset_clears_client_and_registry`
- `routellm/tests/test_router_embeddings.py::test_configure_with_none_clears_registry`
- `routellm/tests/test_router_embeddings.py::test_router_and_server_import_without_openai_key`
- `routellm/tests/test_tiers.py::test_from_config_builds_tiers`
- `routellm/tests/test_tiers.py::test_tier_router_and_threshold_default_to_none`
- `routellm/tests/test_tiers.py::test_from_config_without_tiers_key_is_empty`
- `routellm/tests/test_tiers.py::test_tier_name_charset_rejected`
- `routellm/tests/test_tiers.py::test_tier_threshold_out_of_range_rejected`
- `routellm/tests/test_tiers.py::test_unknown_reference_rejected`
- `routellm/tests/test_tiers.py::test_cycle_rejected_naming_the_cycle`
- `routellm/tests/test_tiers.py::test_self_reference_rejected`
- `routellm/tests/test_tiers.py::test_depth_four_accepted`
- `routellm/tests/test_tiers.py::test_depth_five_rejected`
- `routellm/tests/test_tiers.py::test_tier_endpoint_name_collision_rejected`
- `routellm/tests/test_tiers.py::test_tier_router_name_collision_rejected_at_construction`
- `routellm/tests/test_tiers.py::test_strong_model_naming_a_tier_rejected`
- `routellm/tests/test_tiers.py::test_weak_model_naming_a_tier_rejected`
- `routellm/tests/test_tiers.py::test_parse_tier_with_request_router_and_threshold`
- `routellm/tests/test_tiers.py::test_parse_legacy_form_uses_default_tier_when_one_exists`
- `routellm/tests/test_tiers.py::test_parse_legacy_form_is_flat_without_a_default_tier`
- `routellm/tests/test_tiers.py::test_parse_router_prefixed_tier_name`
- `routellm/tests/test_tiers.py::test_parse_bare_tier_name`
- `routellm/tests/test_tiers.py::test_parse_unknown_tier_lists_tiers`
- `routellm/tests/test_tiers.py::test_parse_hyphenated_tier_name_rejected`
- `routellm/tests/test_tiers.py::test_parse_qualified_remainder_must_be_router_form`
- `routellm/tests/test_tiers.py::test_high_score_cascades_to_the_deepest_strong_leaf`
- `routellm/tests/test_tiers.py::test_low_score_stops_at_the_root_weak_leaf`
- `routellm/tests/test_tiers.py::test_path_records_one_entry_per_level_with_win_rate`
- `routellm/tests/test_tiers.py::test_router_runs_on_the_original_prompt_once_per_level`
- `routellm/tests/test_tiers.py::test_path_is_logged_at_info`
- `routellm/tests/test_tiers.py::test_model_counts_key_is_the_request_string_and_final_endpoint`
- `routellm/tests/test_tiers.py::test_child_without_values_inherits_the_parent_level`
- `routellm/tests/test_tiers.py::test_child_own_values_beat_the_parent`
- `routellm/tests/test_tiers.py::test_root_without_values_takes_the_request_values`
- `routellm/tests/test_tiers.py::test_root_without_request_values_takes_the_controller_defaults`
- `routellm/tests/test_tiers.py::test_default_router_falls_back_to_the_first_configured_router`
- `routellm/tests/test_tiers.py::test_traffic_rule_bypasses_the_tree`
- `routellm/tests/test_tiers.py::test_middleware_bypasses_the_tree`
- `routellm/tests/test_tiers.py::test_bypassed_pair_uses_the_root_level_threshold`
- `routellm/tests/test_tiers.py::test_fallback_descends_a_tier_sibling_by_weak_without_routers`
- `routellm/tests/test_tiers.py::test_canary_model_resolves_through_the_registry`
- `routellm/tests/test_tiers.py::test_both_models_none_without_a_default_tier_rejected`
- `routellm/tests/test_tiers.py::test_model_pair_returns_the_flat_pair_when_set`
- `routellm/tests/test_tiers.py::test_model_pair_raises_when_only_tiers_exist`
- `routellm/tests/test_tiers.py::test_flat_pair_still_routes_without_tiers`
- `routellm/tests/test_tiers.py::test_acompletion_walks_the_tree_and_attaches_the_path`
- `routellm/tests/test_tiers.py::test_server_returns_the_routing_path`
- `routellm/tests/test_tiers.py::test_server_exposes_a_default_threshold_flag`
- `routellm/tests/test_tiers.py::test_route_only_router_reports_null_win_rate`
- `routellm/tests/test_tiers.py::test_scorer_called_once_per_level`
- `routellm/tests/test_tiers.py::test_routing_leaves_the_router_instance_untouched`
- `routellm/tests/test_tiers.py::test_middleware_pair_falls_back_to_its_own_other_side`
- `routellm/tests/test_tiers.py::test_traffic_rule_pair_falls_back_within_that_pair`
- `routellm/tests/test_tiers.py::test_successful_canary_is_not_labelled_a_fallback`
- `routellm/tests/test_tiers.py::test_real_fallback_is_still_labelled`
- `routellm/tests/test_pairing.py::test_selector_defaults_to_quality_desc`
- `routellm/tests/test_pairing.py::test_selector_rejects_unknown_order`
- `routellm/tests/test_pairing.py::test_selector_rejects_empty_select`
- `routellm/tests/test_pairing.py::test_tier_side_accepts_a_selector_mapping`
- `routellm/tests/test_pairing.py::test_validation_skips_selector_sides`
- `routellm/tests/test_pairing.py::test_tag_only_selection_needs_no_catalog`
- `routellm/tests/test_pairing.py::test_catalog_term_selection`
- `routellm/tests/test_pairing.py::test_terms_are_anded`
- `routellm/tests/test_pairing.py::test_candidate_without_a_catalog_record_fails_a_catalog_term`
- `routellm/tests/test_pairing.py::test_cost_ordering_puts_none_last`
- `routellm/tests/test_pairing.py::test_cost_desc_still_puts_none_last`
- `routellm/tests/test_pairing.py::test_context_desc_ordering`
- `routellm/tests/test_pairing.py::test_quality_tiebreaks_on_release_date`
- `routellm/tests/test_pairing.py::test_unrated_endpoints_sort_after_rated_ones`
- `routellm/tests/test_pairing.py::test_no_candidate_raises_naming_the_select`
- `routellm/tests/test_pairing.py::test_unknown_query_key_surfaces_the_term`
- `routellm/tests/test_pairing.py::test_bare_free_text_token_rejected`
- `routellm/tests/test_pairing.py::test_tag_terms_are_stripped_before_the_catalog_parser`
- `routellm/tests/test_pairing.py::test_resolve_registry_pairings_rewrites_both_sides`
- `routellm/tests/test_pairing.py::test_identical_winners_raise`
- `routellm/tests/test_pairing.py::test_resolved_registry_still_validates`
- `routellm/tests/test_pairing.py::test_controller_resolves_selectors_at_construction`
- `routellm/tests/test_pairing.py::test_candidate_table_logged_at_info`
- `routellm/tests/test_pairing.py::test_catalog_unavailable_with_a_tag_only_policy_passes`
- `routellm/tests/test_pairing.py::test_catalog_unavailable_with_a_catalog_term_raises`
- `routellm/tests/test_pairing.py::test_fresh_snapshot_is_used_without_fetching`
- `routellm/tests/test_pairing.py::test_stale_snapshot_is_used_with_a_warning`
- `routellm/tests/test_pairing.py::test_successful_fetch_writes_the_snapshot`
- `routellm/tests/test_pairing.py::test_provider_mapping_for_a_bare_openai_name`
- `routellm/tests/test_pairing.py::test_provider_mapping_for_a_local_ollama_name`
- `routellm/tests/test_pairing.py::test_provider_mapping_for_an_unresolvable_name`
- `routellm/tests/test_pairing.py::test_provider_mapping_aliases_gemini_to_google`
- `routellm/tests/test_pairing.py::test_provider_mapping_logged_at_debug`
- `routellm/tests/test_pairing.py::test_explain_output_contains_the_pick`
- `routellm/tests/test_pairing.py::test_explain_exits_one_on_a_resolution_error`
- `routellm/tests/test_openai_server_models.py::test_models_lists_tiers_and_routers`
- `routellm/tests/test_openai_server_models.py::test_models_ids_are_routable`
- `routellm/tests/test_openai_server_models.py::test_flags_build_an_implicit_default_tier`
- `routellm/tests/test_openai_server_models.py::test_config_default_tier_wins_over_the_flags`
- `routellm/tests/test_openai_server_models.py::test_implicit_tier_reuses_a_configured_endpoint`
- `routellm/tests/test_openai_server_models.py::test_no_flags_invents_no_default_tier`
- `routellm/tests/test_openai_server_models.py::test_one_model_flag_alone_is_an_argparse_error`
- `routellm/tests/test_openai_server_models.py::test_config_example_loads_into_the_registry`

## Related

- US-0001 — strong/weak routing via cost threshold (the flat pair a
  tier generalises)
- US-0010 — OpenAI-compatible server (the surface `/v1/models` and the
  `routellm` response key extend)
- US-0104 — router registry (a tier's `router:` is resolved through it)
