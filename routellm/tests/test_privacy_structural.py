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
POLICY_BATCH = [n for n in CASES if CASES[n].policy and not CASES[n].gate and not CASES[n].route]
ROUTE_BATCH = [n for n in CASES if CASES[n].route]
GATE_BATCH = [n for n in CASES if CASES[n].gate]


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
def route_results(tmp_path_factory):
    return pf.run_cases(tmp_path_factory.mktemp("privacy-route"), ROUTE_BATCH, route=True)


@pytest.mark.parametrize("name", ROUTE_BATCH)
def test_a_route_policy_only_narrows_and_private_never_leaves_the_host(name, route_results):
    check(name, route_results[name])


def test_a_missing_unknown_or_invalid_project_takes_the_default_policy(route_results):
    froms = {
        "route_no_project_header_takes_the_default_policy_and_stays_local": "absent",
        "route_unknown_project_takes_the_default_policy_and_stays_local": "unknown",
        "route_invalid_project_takes_the_default_policy_and_stays_local": "invalid",
    }
    for name, why in froms.items():
        obs = route_results[name]["obs"]
        assert (obs["policy"], obs["from"]) == ("default", why), name


def test_a_policy_that_would_widen_a_class_is_refused_naming_class_and_tier(route_results):
    message = route_results["route_a_policy_cannot_widen_a_class"]["obs"]["message"]
    assert "'private'" in message and "'general'" in message


def test_attribute_headers_are_not_forwarded(route_results):
    obs = route_results["route_attribute_headers_never_reach_an_upstream"]["obs"]
    assert obs["forwarded"] == []


def test_the_route_control_reaches_the_cloud(route_results):
    got = route_results["route_control_open_project_general_reaches_the_cloud"]
    assert got["hits"]["cloud_main"] >= 1 and got["obs"]["policy"] == "open-project"


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
