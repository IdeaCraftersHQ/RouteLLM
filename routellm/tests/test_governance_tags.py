"""Tests for governance tag namespaces: reach set, single jur, known namespaces."""

import pytest

from routellm.endpoints import EndpointRegistry


def _load(tags_by_name, **top):
    endpoints = {n: {"model": f"m_{n}", "tags": t} for n, t in tags_by_name.items()}
    return EndpointRegistry.from_config({"endpoints": endpoints, **top})


# --- parsed view ---------------------------------------------------------


def test_reach_and_jur_are_parsed_once_on_the_endpoint():
    reg = _load({"a": ["local", "vpn", "jur:ca", "fast"]})
    ep = reg.get("a")
    assert ep.reach == frozenset({"local", "vpn"})
    assert ep.jur == "ca"


def test_endpoint_without_governance_tags_has_empty_view():
    ep = _load({"a": ["fast"]}).get("a")
    assert ep.reach == frozenset()
    assert ep.jur is None


# --- jurisdiction --------------------------------------------------------


def test_two_jur_tags_rejected():
    with pytest.raises(ValueError, match=r"a.*jur"):
        _load({"a": ["jur:us", "jur:ca"]})


def test_uppercase_jur_rejected():
    with pytest.raises(ValueError, match=r"a.*jur:CA"):
        _load({"a": ["jur:CA"]})


@pytest.mark.parametrize("bad", ["jur:", "jur:usa", "jur:u1"])
def test_malformed_jur_value_rejected(bad):
    with pytest.raises(ValueError, match=r"jur"):
        _load({"a": [bad]})


# --- namespaces ----------------------------------------------------------


def test_unknown_namespace_rejected_naming_endpoint_and_tag():
    with pytest.raises(ValueError, match=r"endpoint 'a'.*jurr:ca"):
        _load({"a": ["jurr:ca"]})


def test_extra_namespace_declared_in_config_is_accepted():
    reg = _load({"a": ["team:search", "team:infra"]}, tag_namespaces=["team"])
    assert "team:search" in reg.get("a").tags


def test_extra_namespace_still_rejects_others():
    with pytest.raises(ValueError, match=r"unit:x"):
        _load({"a": ["unit:x"]}, tag_namespaces=["team"])


@pytest.mark.parametrize("bad", ["Team", "a b", "local", "", 3])
def test_bad_tag_namespaces_entry_rejected(bad):
    with pytest.raises(ValueError, match="tag_namespaces"):
        _load({"a": ["fast"]}, tag_namespaces=[bad])


# --- governance mode -----------------------------------------------------


def test_governance_requires_reach():
    with pytest.raises(ValueError, match=r"endpoint 'a'.*reach"):
        _load({"a": ["jur:us"]}, governance=True)


def test_governance_requires_exactly_one_jur():
    with pytest.raises(ValueError, match=r"endpoint 'a'.*jur"):
        _load({"a": ["cloud"]}, governance=True)


def test_governance_accepts_complete_endpoints():
    reg = _load({"a": ["cloud", "jur:us"], "b": ["local", "vpn", "jur:ca"]}, governance=True)
    assert reg.governance is True


def test_governance_off_by_default_leaves_untagged_endpoints_alone():
    reg = _load({"a": ["fast"]})
    assert reg.governance is False


def test_governance_must_be_boolean():
    with pytest.raises(ValueError, match="governance"):
        _load({"a": ["cloud", "jur:us"]}, governance="yes")
