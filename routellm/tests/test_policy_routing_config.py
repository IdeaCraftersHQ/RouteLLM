"""Policy routing at load: the section's shape and its startup checks."""

import pytest

from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy_routing import (
    PolicyRoutingError,
    RequestAttributes,
    build_policy_routing,
)
from routellm.tests.policy_routing_world import (
    NAMES,
    TAGS,
    A,
    class_policy,
    routing_section,
    tier_config,
)
from routellm.tests.test_class_tiers import endpoint


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


def registry():
    return EndpointRegistry.from_config(
        {
            "endpoints": {n: endpoint(TAGS[n]) for n in NAMES},
            "tiers": tier_config(A),
            "strict_registry": True,
        }
    )


def load(section=None, **over):
    raw = {**(section or routing_section(A)), **over}
    pol = class_policy(A, only=set(raw["policies"]))
    router = build_policy_routing({"policy_routing": raw}, pol)
    router.check(registry(), pol)
    return router


def refused(**over) -> str:
    with pytest.raises(PolicyRoutingError) as exc:
        load(**over)
    return str(exc.value)


def test_no_section_builds_nothing():
    assert build_policy_routing({}) is None
    assert build_policy_routing(None) is None


def test_a_clean_section_loads():
    load()


def test_it_needs_a_class_policy():
    router = build_policy_routing({"policy_routing": routing_section(A)}, None)
    with pytest.raises(PolicyRoutingError, match="policy:"):
        router.check(registry(), None)


def test_a_default_policy_is_mandatory():
    section = routing_section(A)
    del section["policies"]["default"]
    assert "default" in refused(section=section)


def test_the_key_must_be_an_attribute():
    assert "key" in refused(key="nope")


def test_an_attribute_may_not_take_a_reserved_matcher_name():
    section = routing_section(A, attributes={"class": {"header": "X-C"}}, key="class")
    assert "reserved" in refused(section=section)


def test_a_rule_on_an_unknown_matcher_is_refused():
    msg = refused(rules=[{"when": {"colour": "red"}, "tier": "code"}])
    assert "colour" in msg


def test_a_rule_naming_an_unknown_tier_is_refused():
    msg = refused(rules=[{"when": {"intent": "x"}, "tier": "nowhere"}])
    assert "'nowhere'" in msg and "not a tier" in msg


def test_a_rule_whose_tier_its_class_does_not_list_is_a_widening_and_refused():
    msg = refused(rules=[{"when": {"class": "private"}, "tier": "hard"}])
    assert "class 'private'" in msg and "widen" in msg


def test_a_policy_fallback_must_be_one_of_its_tiers():
    section = routing_section(A)
    section["policies"]["default"]["fallback"] = "hard"
    assert "fallback" in refused(section=section)


def test_a_class_no_selectable_tier_can_serve_is_refused():
    section = routing_section(A, rules=[], **{"else": "code"})
    section["policies"] = {"default": {"data_class": "general", "tiers": ["code"]}}
    msg = refused(section=section)
    assert "class 'private'" in msg and "no tier" in msg


def test_an_unknown_policy_class_is_refused():
    section = routing_section(A)
    section["policies"]["default"]["data_class"] = "mystery"
    assert "mystery" in refused(section=section)


def test_the_section_rejects_unknown_keys():
    with pytest.raises(PolicyRoutingError, match="surprise"):
        build_policy_routing({"policy_routing": {**routing_section(A), "surprise": 1}}, None)


def test_the_header_defaults_from_the_attribute_name_and_the_pattern_is_safe():
    section = routing_section(A, attributes={"project": {}, "client": {}})
    router = load(section=section)
    got = router.attributes({"x-route-project": "alpha", "X-Route-Client": "Bad Value"})
    assert got == RequestAttributes({"project": "alpha"}, ("client",))


def test_an_attribute_pattern_is_configurable():
    section = routing_section(A, attributes={"project": {"pattern": "^[A-Z]{3}$"}, "client": {}})
    router = load(section=section)
    assert router.attributes({"x-route-project": "ABC"}).values == {"project": "ABC"}
    assert router.attributes({"x-route-project": "abc"}).invalid == ("project",)


def test_headers_are_read_case_insensitively_and_empty_is_absent():
    router = load()
    assert router.attributes({"x-route-project": "alpha"}).values == {"project": "alpha"}
    got = router.attributes({"X-ROUTE-PROJECT": ""})
    assert got.values == {} and got.invalid == ()


def test_the_decision_classes_must_run_in_the_policy_files_order():
    from types import SimpleNamespace

    router = build_policy_routing({"policy_routing": routing_section(A)}, class_policy(A))
    ok = SimpleNamespace(data_classes={"general": "", "private": ""}, intents={"hard": ""})
    router.check(registry(), class_policy(A), ok)
    flipped = SimpleNamespace(data_classes={"private": "", "general": ""}, intents={"hard": ""})
    with pytest.raises(PolicyRoutingError, match="order"):
        router.check(registry(), class_policy(A), flipped)


def test_a_rule_intent_must_be_a_decision_intent():
    from types import SimpleNamespace

    router = build_policy_routing({"policy_routing": routing_section(A)}, class_policy(A))
    cfg = SimpleNamespace(data_classes={"general": "", "private": ""}, intents={"chat": ""})
    with pytest.raises(PolicyRoutingError, match="'hard'"):
        router.check(registry(), class_policy(A), cfg)


def test_a_principal_listing_an_unknown_policy_is_a_startup_error():
    from routellm.policy import parse_policy

    raw = {
        "classes": {
            "general": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["code"]}
        },
        "principals": {"ops": {"classes": ["general"], "projects": ["ghost"]}},
    }
    pol = parse_policy(raw, "p.yaml")
    router = build_policy_routing({"policy_routing": routing_section(A)}, pol)
    with pytest.raises(PolicyRoutingError, match="ghost"):
        router.check(registry(), pol)
