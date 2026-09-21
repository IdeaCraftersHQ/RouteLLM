# Test Index: tests

| Test File | What It Tests | Key Functions |
|-----------|---------------|----------------|
| `test_gateway.py` | Gateway | test_payment_challenge_fields, test_payment_receipt_fields |
| `test_x402_adapter.py` | X402 Adapter | test_x402_adapter_name, test_x402_adapter_networks, test_x402_adapter_custom_networks (+ 1 more) |
| `test_advanced_features_deep.py` | Advanced Features Deep | test_circuit_breaker_transitions, test_circuit_breaker_rate_threshold, test_cache_ttl (+ 4 more) |
| `test_controller_integration.py` | Controller Integration | test_controller_full_flow |
| `test_controller_payment.py` | Controller Payment | test_controller_no_gateway_reraises_402 |
| `test_e2e_scenarios.py` | E2E Scenarios |  |
| `test_intent_model_selector.py` | Intent Model Selector | test_get_model_pair_marketing, test_get_model_pair_technical, test_get_model_pair_default (+ 1 more) |
| `test_openai_client.py` | Openai Client |  |
| `test_openai_server.py` | Openai Server |  |
| `test_resilience_caching_traffic.py` | Resilience Caching Traffic | test_resilience_retry, test_resilience_fallback, test_traffic_manager_conditional_routing (+ 1 more) |
| `test_x402_integration.py` | X402 Integration | test_full_402_flow |
| `test_xrr_integration.py` | Xrr Integration | test_controller_with_xrr |
| `test_prompt_file.py` | Generic Prompt File Loader | test_load_full_file, test_unrequested_section_ignored, test_section_wrong_type_raises (+ 13 more) |
| `test_router_registry.py` | Router Registry (register + entry-point discovery) | test_register_get_names_round_trip, test_duplicate_name_raises, test_decorator_form_registers (+ 11 more) |
| `test_endpoints.py` | Endpoint Registry (named endpoints + per-endpoint credentials) | test_from_config_builds_named_endpoints, test_resolve_raw_model_warns_once, test_credentials_reads_env_at_call_time (+ 21 more) |
| `test_router_embeddings.py` | Lazy Embedding Client (endpoint/env sources, caching, import without a key) | test_client_built_from_embedding_endpoint, test_no_endpoint_and_no_env_raises_naming_both, test_router_and_server_import_without_openai_key (+ 7 more) |
| `test_tiers.py` | Tiers (nested strong/weak pairs, recursive walk, inherited router/threshold, decision path) | test_high_score_cascades_to_the_deepest_strong_leaf, test_child_without_values_inherits_the_parent_level, test_fallback_descends_a_tier_sibling_by_weak_without_routers (+ 42 more) |
| `test_pairing.py` | Policy-based pairing (selectors over endpoint tags + the models.dev catalog, ordering, snapshot cache, explain surface) | test_tag_only_selection_needs_no_catalog, test_catalog_term_selection, test_explain_without_a_flag_reads_the_discovered_config (+ 32 more) |
| `test_intent_tiers.py` | Intents choosing a tier (`intent_routing`, the middleware `get_tier` hook, `intent_tiers` on the selector, the `intents:` config section, the explain line) | test_middleware_tier_is_entered_from_the_default_tier, test_an_explicit_tier_ignores_the_middleware_tier, test_a_mapping_to_a_missing_tier_is_rejected (+ 18 more) |
| `test_openai_server_models.py` | Server model listing (`GET /v1/models`) + the implicit `default` tier derived from the model flags, the config example's `endpoints:`/`tiers:` load, and config discovery with no `--config` | test_models_lists_tiers_and_routers, test_flags_build_an_implicit_default_tier, test_server_discovers_the_user_config_without_a_flag (+ 10 more) |
| `test_openai_server_params.py` | What the server forwards to litellm (only client-set sampling parameters, never a Pydantic default), the `--host` flag and its loopback default, and the middleware import cycle that broke entry-point discovery | test_unset_sampling_params_are_not_forwarded, test_client_set_sampling_params_are_forwarded_verbatim, test_stream_is_forwarded_only_when_the_client_set_it (+ 7 more) |
| `test_examples.py` | The shipped `examples/multitier.yaml`: load, selector resolution against a fake catalog, governance restrictions, and the intent-to-tier mapping | test_example_loads_into_the_registry, test_every_selector_resolves_to_an_endpoint, test_private_never_leaves_our_own_hardware (+ 6 more) |
| `test_config_discovery.py` | Layered config discovery: the precedence chain (system → user → project walk-up → env → flag), deep merge with null-deletes, per-key origins, and the `explain` surface | test_chain_order_lowest_first, test_walk_up_stops_at_home, test_null_deletes_a_key_from_a_lower_layer (+ 14 more) |
| `test_config_cli.py` | The `python -m routellm.config` inspection surface: `path` picks the winner, `paths` marks every searched location used or absent, `show` carries origin comments, `--format json` for both, and `--config` honoured before or after the subcommand | test_path_prints_the_winner, test_paths_marks_used_and_absent, test_show_carries_origin_comments (+ 10 more) |
| `test_evaluate_config_source.py` | `evals/evaluate.py` reads config through `load_config`, not a raw `yaml.safe_load(open(...))`, checked by source/AST scan since the module imports heavy optional deps | test_evaluate_py_does_not_raw_load_the_config_file, test_evaluate_py_uses_load_config_for_the_explicit_layer, test_evaluate_py_imports_load_config_from_routellm_config (+ 1 more) |

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.