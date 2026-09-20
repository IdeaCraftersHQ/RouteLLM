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

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.