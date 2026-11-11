"""Privacy suite, redact world: no canary reaches an off-host fake, the client reads the original.

See `privacy_cases_redact` for the cases and `privacy_fixture.redact_files`
for the world.
"""

import importlib

import pytest

from routellm.tests import privacy_fixture as pf
from routellm.tests.privacy_fixture import CASE_MODULES, CASES, check

pytestmark = pytest.mark.slow

for _module in CASE_MODULES:
    importlib.import_module(_module)

BATCH = [n for n in CASES if CASES[n].redact]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return pf.run_cases(tmp_path_factory.mktemp("privacy-redact"), BATCH, redact=True)


@pytest.mark.parametrize("name", BATCH)
def test_the_redact_cases_hold_the_privacy_claim(name, results):
    check(name, results[name])


def obs(results, name):
    return results[name]["obs"]


def test_the_cloud_gets_placeholders_and_no_canary(results):
    got = obs(results, "redact_main_call_sends_no_canary_and_the_client_reads_the_original")
    assert got["off_host_hits"] >= 1 and got["leaks"] == []
    assert got["placeholders_sent"] and got["mapping_free"]


def test_the_client_reads_the_original_text(results):
    assert obs(results, "redact_main_call_sends_no_canary_and_the_client_reads_the_original")[
        "restored"
    ]


def test_a_stream_is_restored_from_pieces_that_split_placeholders(results):
    got = obs(results, "redact_stream_split_over_chunks_is_restored")
    assert got["leaks"] == [] and got["restored"] and not got["placeholder_in_client_stream"]
    assert got["pieces"] >= 3


def test_the_fallback_leaf_gets_no_canary_either(results):
    got = obs(results, "redact_sibling_fallback_is_redacted_too")
    assert got["cloud_hits"] >= 1 and got["gated_hits"] >= 1 and got["leaks"] == []


def test_compare_targets_get_no_canary(results):
    got = obs(results, "redact_compare_targets_are_redacted")
    assert got["hit_both"] and got["leaks"] == []


def test_a_redactor_failure_is_5xx_for_both_paths_with_no_call(results):
    got = obs(results, "redact_redactor_failure_is_a_5xx_with_zero_provider_calls")
    assert got["status"] == 500 and got["stream_status"] == 500
    assert got["code"] == "transform_failed" and got["off_host_hits"] == 0


def test_a_local_leaf_is_left_raw(results):
    got = obs(results, "redact_local_leaf_stays_raw")
    assert got["local_saw_raw"] and got["off_host_hits"] == 0


@pytest.mark.parametrize(
    "name",
    [
        "redact_aux_detector_never_sees_the_class_off_host",
        "redact_aux_embedding_router_is_skipped",
        "redact_aux_semantic_cache_never_embeds",
    ],
)
def test_no_auxiliary_call_leaves_the_host(name, results):
    got = obs(results, name)
    assert got["off_host_hits"] == 0 and got["leaks"] == []


def test_fields_outside_messages_carry_no_canary(results):
    got = obs(results, "redact_request_fields_carry_no_canary")
    assert got["off_host_hits"] >= 1 and got["leaks"] == []
    assert got["allowlisted_kept"] and got["placeholders_sent"]


def test_fields_the_http_body_drops_carry_no_canary_from_a_library_call(results):
    got = obs(results, "redact_direct_fields_carry_no_canary")
    assert got["status"] == 200 and got["off_host_hits"] >= 1 and got["leaks"] == []


@pytest.mark.parametrize(
    "name",
    [
        "redact_unknown_field_is_refused_with_zero_calls",
        "redact_unredactable_name_is_refused_with_zero_calls",
    ],
)
def test_a_field_that_cannot_be_redacted_is_refused_not_sent(name, results):
    got = obs(results, name)
    assert got["kind"] == "TransformError" and got["off_host_hits"] == 0


def test_a_non_local_transcriber_is_refused_before_audio_leaves(results):
    got = obs(results, "redact_non_local_transcriber_is_refused_before_transcription")
    assert got["status"] == 403 and got["off_host_hits"] == 0
