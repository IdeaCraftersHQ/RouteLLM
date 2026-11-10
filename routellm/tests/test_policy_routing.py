"""Policy routing: attributes and a policy select the tier, in process, fakes only.

Every upstream is a counting fake. The same suite runs over two label
sets (`A`, `B`) that share no name: header, attribute, classes, intents,
tiers and project values are all config.
"""

import pytest

from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import PolicyDenied
from routellm.tests.policy_routing_world import (
    GATE,
    IMAGE,
    A,
    B,
    Decider,
    close,
    make_world,
    routing_section,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


@pytest.fixture(params=[A, B], ids=["names-a", "names-b"])
def L(request):
    return request.param


@pytest.fixture
def world(L):
    w = make_world(L)
    yield w
    close(w)


def served(w, res):
    """The tier the request entered, as the path records it."""
    return w.path(res)[0]["tier"]


# --- policy selection ----------------------------------------------------------------


def test_no_project_header_takes_the_default_policy_and_the_path_says_so(world):
    res = world.ask()
    rec = world.record(res)
    assert (rec["policy"], rec["policy_from"]) == ("default", "absent")
    assert served(world, res) == world.L.tiers["code"]


def test_an_unknown_project_takes_the_default_policy(world):
    res = world.ask(project="nobody")
    rec = world.record(res)
    assert (rec["policy"], rec["policy_from"]) == ("default", "unknown")


@pytest.mark.parametrize("value", ["Bad Value", "../x", "a" * 80, "UPPER", "café"])
def test_an_invalid_project_value_takes_the_default_policy(world, value):
    res = world.ask(project=value)
    rec = world.record(res)
    assert (rec["policy"], rec["policy_from"]) == ("default", "invalid")
    assert rec["attributes"] == {}


def test_a_known_project_takes_its_own_policy(world):
    res = world.ask(project=world.L.projects["open"])
    rec = world.record(res)
    assert (rec["policy"], rec["policy_from"]) == (world.L.projects["open"], "attribute")


def test_an_unknown_project_never_gets_the_privileges_of_another_policy(world):
    # `hard` is allowed for the open project only; an unknown value must
    # fall to the default policy, whose fallback replaces it.
    res = world.ask(project="nobody", decision=Decider(world.L.low, world.L.hard))
    assert served(world, res) == world.L.tiers["code"]
    assert world.record(res)["refused"] == {world.L.tiers["hard"]: world.L.tiers["code"]}


# --- rules ---------------------------------------------------------------------------


def test_first_matching_rule_wins(world):
    # Intent `hard` (rule 1) and the client rule (rule 2) both match.
    res = world.ask(
        project=world.L.projects["open"],
        client=world.L.client_value,
        decision=Decider(world.L.low, world.L.hard),
    )
    assert served(world, res) == world.L.tiers["hard"]
    assert world.record(res)["rule"] == 1


def test_the_else_tier_is_taken_when_no_rule_matches(world):
    res = world.ask(project=world.L.projects["open"])
    assert (served(world, res), world.record(res)["rule"]) == (world.L.tiers["code"], "else")


def test_an_attribute_rule_matches_on_the_attribute_value(world):
    res = world.ask(client=world.L.client_value)
    assert served(world, res) == world.L.tiers["light"]
    assert world.hits() == {"loc2": 1}


def test_a_modality_rule_matches_an_image(world):
    res = world.ask(messages=IMAGE)
    assert served(world, res) == world.L.tiers["vision"]


def test_without_a_decision_step_intent_rules_never_match(world):
    res = world.ask(project=world.L.projects["open"])
    assert world.record(res)["intent"] is None
    assert served(world, res) == world.L.tiers["code"]


def test_an_explicitly_addressed_tier_is_not_replaced_by_a_rule(world):
    res = world.ask(model=world.L.tiers["light"], decision=Decider(world.L.low, world.L.hard))
    assert served(world, res) == world.L.tiers["light"]
    assert world.record(res)["rule"] is None


# --- effective class -----------------------------------------------------------------


def test_a_declared_class_stricter_than_the_policy_default_wins(world):
    res = world.ask(declared=world.L.high)
    rec = world.record(res)
    assert rec["effective_class"] == world.L.high
    assert served(world, res) == world.L.tiers["private"]
    assert world.hits() == {"loc": 1}


def test_a_decision_stricter_than_policy_and_declared_wins(world):
    res = world.ask(decision=Decider(world.L.high, world.L.hard))
    rec = world.record(res)
    assert rec["effective_class"] == world.L.high
    assert rec["classes"] == {
        "policy": world.L.low,
        "declared": world.L.low,
        "decision": world.L.high,
    }
    assert world.hits() == {"loc": 1}


def test_a_policy_default_stricter_than_declared_and_decision_wins(world):
    res = world.ask(project=world.L.projects["floor"], decision=Decider(world.L.low, None))
    assert world.record(res)["effective_class"] == world.L.high
    assert world.hits() == {"loc": 1}


def test_nothing_lowers_the_class(world):
    # Declared high; policy floor low; the decision step says low.
    res = world.ask(declared=world.L.high, decision=Decider(world.L.low, world.L.hard))
    assert world.record(res)["effective_class"] == world.L.high
    assert world.hits() == {"loc": 1}


def test_the_path_records_every_source_attribute_and_the_matched_rule(world):
    res = world.ask(
        project=world.L.projects["open"], client=world.L.client_value, decision=Decider(world.L.low)
    )
    rec = world.record(res)
    assert rec["attributes"] == {
        world.L.key: world.L.projects["open"],
        world.L.client: world.L.client_value,
    }
    assert {"policy", "declared", "decision"} <= set(rec["classes"])
    assert (rec["rule"], rec["tier"], rec["addressed"]) == (
        2,
        world.L.tiers["light"],
        world.L.entry,
    )


# --- allowed tiers ---------------------------------------------------------------------


def test_a_target_outside_the_allowed_tiers_takes_the_fallback(world):
    res = world.ask(decision=Decider(world.L.low, world.L.hard))
    assert served(world, res) == world.L.tiers["code"]
    assert world.record(res)["refused"] == {world.L.tiers["hard"]: world.L.tiers["code"]}
    assert "cld2" not in world.hits()


def test_no_fallback_is_403_naming_the_policy_and_the_tier(world):
    with pytest.raises(PolicyDenied) as exc:
        world.ask(project=world.L.projects["nofb"], decision=Decider(world.L.low, world.L.hard))
    assert exc.value.status == 403
    assert world.L.projects["nofb"] in str(exc.value)
    assert world.L.tiers["hard"] in str(exc.value)
    assert world.hits() == {}


def test_an_addressed_tier_outside_the_allowed_tiers_is_403_never_moved(world):
    with pytest.raises(PolicyDenied, match=world.L.tiers["hard"]):
        world.ask(model=world.L.tiers["hard"])
    assert world.hits() == {}


# --- a policy never widens a class -------------------------------------------------------


def _widening(L):
    """A section that starts, yet can pick a cloud tier for the strict class.

    Class `high` lists only the private tier. The rule below sends intent
    `hard` to the cloud tier whatever the class; the else tier keeps the
    startup check satisfied.
    """
    t = L.tiers
    return routing_section(
        L,
        rules=[{"when": {"intent": L.hard}, "tier": t["hard"]}],
        policies={
            "default": {"data_class": L.low, "tiers": [t["hard"], t["private"]]},
        },
        **{"else": t["private"]},
    )


def test_a_policy_cannot_widen_a_class(L):
    w = make_world(L, section=_widening(L))
    try:
        with pytest.raises(PolicyDenied) as exc:
            w.ask(declared=L.high, decision=Decider(L.high, L.hard))
        assert L.high in str(exc.value) and L.tiers["hard"] in str(exc.value)
        assert w.hits() == {}
    finally:
        close(w)


def test_a_policy_cannot_widen_a_class_by_the_leaf_either(L, monkeypatch):
    # Even with the tier check out of the way, every leaf is still held to the class.
    w = make_world(L, section=_widening(L))
    try:
        monkeypatch.setattr("routellm.controller.check_entry_tier", lambda *a, **k: None)
        with pytest.raises(PolicyDenied, match="policy:"):
            w.ask(declared=L.high, decision=Decider(L.high, L.hard))
        assert w.hits() == {}
    finally:
        close(w)


# --- tags and gates -----------------------------------------------------------------------


def test_a_denied_tag_makes_an_endpoint_no_candidate(world):
    res = world.ask(project=world.L.projects["deny"], decision=Decider(world.L.low, world.L.hard))
    assert served(world, res) == world.L.tiers["hard"]
    # The tier's strong side carries the denied tag: its weak side serves.
    assert world.hits() == {"cld": 1}


def test_a_gate_requiring_policy_allow_refuses_a_policy_that_does_not_list_the_tag(L):
    w = make_world(L, gates=[GATE])
    try:
        res = w.ask(model=L.tiers["gated"])
        assert w.hits() == {"loc2": 1}
        assert w.path(res)[0]["capability_requirement"] == "gate:metered:policy"
    finally:
        close(w)


def test_a_gate_requiring_policy_allow_admits_a_policy_that_lists_the_tag(L):
    w = make_world(L, gates=[GATE])
    try:
        w.ask(model=L.tiers["gated"], project=L.projects["gate"])
        assert w.hits() == {"gat": 1}
    finally:
        close(w)


def test_a_denied_tag_beats_an_allowed_one(L):
    section = routing_section(L)
    section["policies"][L.projects["gate"]]["deny_tags"] = ["gated"]
    w = make_world(L, section=section, gates=[GATE])
    try:
        w.ask(model=L.tiers["gated"], project=L.projects["gate"])
        assert w.hits() == {"loc2": 1}
    finally:
        close(w)


# --- config absent: today's behaviour ------------------------------------------------------


def test_without_the_section_attributes_are_ignored_and_nothing_is_recorded(L):
    import asyncio

    from routellm.caching import CacheConfig
    from routellm.controller import Controller
    from routellm.endpoints import EndpointRegistry
    from routellm.policy_routing import RequestAttributes
    from routellm.tests.fake_upstream import FakeUpstream
    from routellm.tests.policy_routing_world import NAMES, TAGS, class_policy, tier_config
    from routellm.tests.test_class_tiers import MESSAGES, endpoint

    fakes = {n: FakeUpstream(n) for n in NAMES}
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {n: endpoint(TAGS[n], fakes[n].base) for n in NAMES},
            "tiers": tier_config(L),
            "strict_registry": True,
        }
    )
    c = Controller(
        routers=["random"],
        endpoints=registry,
        policy=class_policy(L),
        strict_registry=True,
        cache_config=CacheConfig(enabled=False),
    )
    try:
        res = asyncio.run(
            c.acompletion(
                model=L.tiers["code"],
                messages=MESSAGES,
                declared_class=L.low,
                attributes=RequestAttributes({L.key: "x"}),
            )
        )
        assert "policy_routing" not in res._hidden_params["routellm_path"][0]
    finally:
        for f in fakes.values():
            f.server.shutdown()
            f.server.server_close()


# --- the policy is bound to the caller -----------------------------------------------------


def test_a_caller_may_use_a_policy_the_policy_file_lists_for_it(world):
    res = world.ask(project=world.L.projects["open"], principal="limited")
    assert world.record(res)["policy"] == world.L.projects["open"]


def test_a_header_naming_a_policy_the_caller_lacks_is_403_naming_it(world):
    with pytest.raises(PolicyDenied) as exc:
        world.ask(project=world.L.projects["nofb"], principal="limited")
    assert exc.value.status == 403 and world.L.projects["nofb"] in str(exc.value)
    assert world.hits() == {}


def test_a_caller_with_no_project_list_may_use_only_the_default_policy(world):
    assert world.record(world.ask(principal="plain"))["policy"] == "default"
    with pytest.raises(PolicyDenied):
        world.ask(project=world.L.projects["open"], principal="plain")


def test_without_auth_a_non_default_policy_is_403(world):
    with pytest.raises(PolicyDenied):
        world.ask(project=world.L.projects["open"], principal=None)
    assert world.hits() == {}


def test_without_auth_the_default_policy_and_unknown_values_still_apply(world):
    assert world.record(world.ask(principal=None))["policy"] == "default"
    rec = world.record(world.ask(project="nobody", principal=None))
    assert (rec["policy"], rec["policy_from"]) == ("default", "unknown")


def test_a_spoofed_project_cannot_unlock_a_gated_endpoint(L):
    w = make_world(L, gates=[GATE])
    try:
        with pytest.raises(PolicyDenied, match=L.projects["gate"]):
            w.ask(model=L.tiers["gated"], project=L.projects["gate"], principal="limited")
        assert w.hits() == {}
    finally:
        close(w)


def test_an_authorised_policy_unlocks_the_gated_endpoint(L):
    w = make_world(L, gates=[GATE])
    try:
        w.ask(model=L.tiers["gated"], project=L.projects["gate"], principal="tester")
        assert w.hits() == {"gat": 1}
    finally:
        close(w)
