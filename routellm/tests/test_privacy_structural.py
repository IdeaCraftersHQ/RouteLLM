"""Privacy suite, structural and client-path cases (see `privacy_fixture`).

One subprocess runs a batch of cases against one app; each case then
reports as its own test. The claim under test: whatever happens to the
local engines, a request addressed to the private tier never reaches a
cloud or gated leaf.
"""

import importlib

import pytest

from routellm.tests import privacy_fixture as pf
from routellm.tests.privacy_fixture import CASE_MODULES, CASES, check

pytestmark = pytest.mark.slow

# Importing registers the cases.
for _module in CASE_MODULES:
    importlib.import_module(_module)

BREAKER = ["breaker_open_on_every_local_leaf"]
BATCH = [n for n in CASES if n not in BREAKER and not CASES[n].policy]
POLICY_BATCH = [n for n in CASES if CASES[n].policy and not CASES[n].gate and not CASES[n].aux]
GATE_BATCH = [n for n in CASES if CASES[n].gate]
AUX_BATCH = [n for n in CASES if CASES[n].aux]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return pf.run_cases(tmp_path_factory.mktemp("privacy"), BATCH)


@pytest.fixture(scope="module")
def breaker_results(tmp_path_factory):
    return pf.run_cases(
        tmp_path_factory.mktemp("privacy-breaker"),
        BREAKER,
        resilience={"cb_fail_max": 1, "cb_fail_wait_ms": 600000},
    )


@pytest.fixture(scope="module")
def policy_results(tmp_path_factory):
    return pf.run_cases(tmp_path_factory.mktemp("privacy-policy"), POLICY_BATCH, class_policy=True)


@pytest.fixture(scope="module")
def gate_results(tmp_path_factory):
    return pf.run_cases(
        tmp_path_factory.mktemp("privacy-gate"),
        GATE_BATCH,
        extra_config=pf.GATE_CONFIG,
        class_policy=True,
    )


@pytest.fixture(scope="module")
def aux_results(tmp_path_factory):
    return pf.run_cases(
        tmp_path_factory.mktemp("privacy-aux"),
        AUX_BATCH,
        extra_config=pf.AUX_CONFIG,
        class_policy=True,
    )


@pytest.mark.parametrize("name", AUX_BATCH)
def test_an_auxiliary_call_takes_zero_hits_for_a_restricted_class(name, aux_results):
    check(name, aux_results[name])


@pytest.mark.parametrize("name", GATE_BATCH)
def test_a_gated_leaf_takes_zero_hits_for_a_denied_class(name, gate_results):
    check(name, gate_results[name])


def test_gate_denials_name_the_gate_and_the_class(gate_results):
    for name in (
        "gate_forced_side_onto_the_gated_leaf_is_403",
        "gate_sibling_is_never_called",
    ):
        assert "gate:gated:class:leaky" in gate_results[name]["obs"]["message"], name


def test_the_gate_controls_reach_the_gated_leaf(gate_results):
    assert (
        gate_results["gate_control_allowed_class_reaches_the_gated_primary"]["hits"]["gated_main"]
        >= 1
    )
    assert (
        gate_results["gate_control_allowed_class_falls_back_to_the_gated_sibling"]["hits"][
            "gated_main"
        ]
        >= 1
    )
    assert (
        gate_results["gate_control_allowed_class_is_classified_by_the_gated_detector"]["hits"][
            "gated_main"
        ]
        >= 1
    )


@pytest.mark.parametrize("name", POLICY_BATCH)
def test_a_class_policy_deny_is_a_403_and_never_a_downgrade(name, policy_results):
    check(name, policy_results[name])


def test_class_policy_denials_name_the_class_and_the_reason(policy_results):
    for name in (
        "policy_deny_is_403_and_never_downgrades_to_local",
        "policy_forced_side_onto_a_cloud_leaf_is_403",
        "policy_cloud_sibling_is_never_called",
        "policy_denied_pool_members_are_never_called",
    ):
        assert "policy:private:reach" in policy_results[name]["obs"]["message"], name


@pytest.mark.parametrize("name", BATCH)
def test_private_never_leaves_the_host(name, results):
    check(name, results[name])


def test_breaker_open_never_falls_through_to_off_host(breaker_results):
    got = breaker_results[BREAKER[0]]
    assert got["obs"]["breakers"] == {"local_main": "open", "local_small": "open"}
    assert got["obs"]["first"] == 502
    check(BREAKER[0], got)


def test_the_control_reaches_off_host_leaves(results):
    # The zeros above mean something only if the fakes can be hit.
    assert results["general_reaches_cloud"]["hits"]["cloud_main"] == 1
    assert results["general_falls_back_to_gated"]["obs"]["gated"] == 1


def test_selector_tier_resolves_to_local_leaves_only(results):
    # `tag:local !tag:cloud` on both sides; the cloud leaf out-ranks every
    # local one, so only the `!` keeps it from winning the strong side.
    sides = results["selector_resolves_to_local_leaves_only"]["obs"]["sides"]
    assert sides == ["local_main", "local_small"]
    assert set(sides) <= set(pf.LOCAL)
