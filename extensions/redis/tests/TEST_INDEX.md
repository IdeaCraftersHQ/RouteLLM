# Test Index: tests

| Test File | What It Tests | Key Functions |
|-----------|---------------|----------------|
| `test_config.py` | Config | test_parse_redis_uri_with_auth, test_parse_uri_with_query_parameters, test_parse_uri_with_ssl_parameters (+ 8 more) |
| `test_connection.py` | Connection | test_get_connection_cluster_mode, test_get_connection_with_decode_responses_false, test_get_connection_includes_version_in_lib_name (+ 2 more) |
| `test_entraid_auth.py` | Entraid Auth | test_raises_error_when_package_not_available, test_creates_service_principal_provider, test_creates_default_credential_provider (+ 3 more) |
| `test_integration.py` | Integration | test_server_starts_successfully, test_server_responds_to_initialize_request, test_server_tool_count_and_names |
| `test_logging_utils.py` | Logging Utils |  |
| `test_main.py` | Main | test_run_propagates_exceptions, test_cli_with_individual_parameters, test_cli_with_cluster_mode (+ 3 more) |
| `test_server.py` | Server | test_mcp_server_name, test_mcp_server_initialization, test_mcp_server_run_method (+ 4 more) |
| `test_hash.py` | Hash | test_hset_with_expiration, test_hset_float_value, test_hget_success (+ 10 more) |
| `test_json.py` | Json | test_json_set_with_expiration, test_json_set_redis_error, test_json_get_specific_field (+ 7 more) |
| `test_list.py` | List | test_lpush_with_expiration, test_rpush_success, test_rpush_redis_error (+ 8 more) |
| `test_misc.py` | Misc | test_search_redis_documents_url_not_configured, test_search_redis_documents_success_json_response, test_search_redis_documents_http_client_error |
| `test_pub_sub.py` | Pub Sub | test_publish_no_subscribers, test_publish_connection_error, test_publish_numeric_message (+ 7 more) |
| `test_redis_query_engine.py` | Redis Query Engine | test_get_indexes_empty, test_create_vector_index_hash_success, test_create_vector_index_hash_redis_error (+ 5 more) |
| `test_server_management.py` | Server Management | test_dbsize_zero_keys, test_info_success_default_section, test_info_all_sections (+ 5 more) |
| `test_set.py` | Set | test_sadd_with_expiration, test_sadd_redis_error, test_srem_success (+ 7 more) |
| `test_sorted_set.py` | Sorted Set | test_zadd_with_expiration, test_zadd_redis_error, test_zrange_success_without_scores (+ 7 more) |
| `test_stream.py` | Stream | test_xadd_with_expiration, test_xadd_redis_error, test_xrange_success (+ 7 more) |
| `test_string.py` | String | test_set_with_expiration, test_set_connection_error, test_get_success (+ 5 more) |

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.