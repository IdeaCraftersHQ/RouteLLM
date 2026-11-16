"""Privacy suite, decision step: a restrictive class never leaves the host.

Two worlds (see `privacy_decision_world`): direct tiers under a class
policy, and an entry tier under policy routing. The classifier's model is
a loopback fake; every leaf is a fake, so nothing leaves the host.
"""

import importlib

import pytest

from routellm.tests import privacy_fixture as pf
from routellm.tests.privacy_fixture import CASE_MODULES, CASES, check

pytestmark = pytest.mark.slow

for _module in CASE_MODULES:
    importlib.import_module(_module)

DIRECT = [n for n in CASES if CASES[n].decision and not CASES[n].route]
ROUTED = [n for n in CASES if CASES[n].decision and CASES[n].route]


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    return pf.run_cases(tmp_path_factory.mktemp("privacy-decision"), DIRECT, decision=True)


@pytest.fixture(scope="module")
def routed(tmp_path_factory):
    return pf.run_cases(
        tmp_path_factory.mktemp("privacy-decision-route"), ROUTED, route=True, decision=True
    )


@pytest.mark.parametrize("name", DIRECT)
def test_a_restrictive_class_takes_zero_off_host_hits(name, direct):
    check(name, direct[name])


@pytest.mark.parametrize("name", ROUTED)
def test_a_restrictive_class_takes_zero_off_host_hits_under_policy_routing(name, routed):
    check(name, routed[name])


def _by_source(results, source):
    return {n: r["obs"] for n, r in results.items() if n.startswith(f"decision_{source}_on_")}


@pytest.mark.parametrize(
    "source, calls", [("declaration", 0), ("recognizer", 0), ("classifier", 1)]
)
def test_the_class_comes_from_its_source_and_the_model_is_asked_only_when_needed(
    source, calls, direct
):
    served = {n: o for n, o in _by_source(direct, source).items() if o["status"] == 200}
    assert len(served) == 5
    for name, obs in served.items():
        want = "model" if source == "classifier" else source.removesuffix("ation")
        assert obs["class"] == "private", name
        assert obs["source"] in (want, "declared"), name
        assert obs["decider"] == calls, name


def test_a_model_answering_a_declarable_class_is_an_error_not_the_class(direct):
    obs = direct["decision_a_model_answering_a_declarable_class_is_an_error_to_the_strictest"][
        "obs"
    ]
    assert (obs["class"], obs["source"], obs["error"]) == ("private", "error", "invalid_answer")
    assert (
        direct["decision_control_a_declared_declarable_class_reaches_its_leaf"]["obs"]["class"]
        == "leaky"
    )


def test_a_recognizer_hit_below_the_strictest_class_skips_the_model(direct):
    obs = direct["decision_a_recognizer_hit_settles_the_class_without_the_model"]["obs"]
    assert (obs["class"], obs["source"], obs["decider"]) == ("internal", "recognizer", 0)


def test_a_refused_tier_names_nothing_it_was_not_given(direct):
    for source in ("declaration", "recognizer", "classifier"):
        obs = direct[f"decision_{source}_on_general_is_403"]["obs"]
        assert obs["status"] == 403 and "'private'" in obs["message"], source


@pytest.mark.parametrize(
    "label, kind",
    [
        ("returns_5xx", "transport"),
        ("times_out", "timeout"),
        ("answers_with_no_json", "invalid_answer"),
        ("answers_with_an_unknown_label", "invalid_answer"),
        ("is_unsure", "low_confidence"),
    ],
)
def test_a_classifier_failure_falls_to_the_strictest_class_and_is_recorded(label, kind, direct):
    obs = direct[f"decision_classifier_{label}_falls_to_the_strictest_class"]["obs"]
    assert (obs["class"], obs["source"], obs["error"], obs["kind"]) == (
        "private",
        "error",
        kind,
        kind,
    )


def test_a_failed_classifier_does_not_lower_a_declared_class(direct):
    obs = direct["decision_classifier_down_does_not_lower_a_declared_class"]["obs"]
    assert obs["class"] == "private" and obs["source"] == "error"


def test_the_decision_model_is_called_on_a_refuse(direct):
    obs = direct["decision_classifier_returns_5xx_under_refuse_is_400"]["obs"]
    assert obs["decider"] == 1 and "refuse" in obs["message"]


def test_sticky_keeps_a_restrictive_conversation_restrictive(direct):
    one = direct["decision_a_restrictive_turn_keeps_later_turns_restrictive"]["obs"]
    assert one["first"]["class"] == "private" and one["first"]["source"] == "model"
    # The floor is the strictest class: the model is not asked again.
    assert (one["second"]["class"], one["second"]["source"]) == ("private", "sticky")
    assert one["second"]["decider"] == 1


def test_sticky_holds_a_mid_class_floor_against_a_lower_label(direct):
    two = direct["decision_a_mid_class_turn_keeps_later_turns_at_that_class"]["obs"]
    assert two["second"]["class"] == "internal" and two["second"]["decider"] == 2


def test_sticky_does_not_taint_a_benign_conversation(direct):
    obs = direct["decision_a_benign_first_turn_does_not_make_the_conversation_restrictive"]["obs"]
    assert obs["second"]["class"] == "general"


def test_the_classifier_never_lowers_a_declared_class_and_can_raise_it(direct):
    kept = direct["decision_the_classifier_cannot_lower_a_declared_class"]["obs"]
    assert (kept["class"], kept["decider"]) == ("internal", 1)
    raised = direct["decision_the_classifier_can_raise_a_declared_class"]["obs"]
    assert (raised["class"], raised["source"]) == ("private", "model")


def test_the_direct_control_reaches_the_cloud(direct):
    got = direct["decision_control_a_general_label_reaches_the_cloud"]
    assert got["obs"]["class"] == "general"
    assert got["hits"]["cloud_main"] + got["hits"]["gated_main"] >= 1


def test_the_routed_control_reaches_the_cloud(routed):
    got = routed["decision_route_control_a_general_label_reaches_the_cloud"]
    assert got["hits"]["cloud_main"] + got["hits"]["gated_main"] >= 1


def test_the_routed_sticky_conversation_stays_local(routed):
    obs = routed["decision_route_a_restrictive_turn_keeps_later_turns_restrictive"]["obs"]
    assert obs["second"]["class"] == "private" and obs["second"]["source"] == "sticky"
