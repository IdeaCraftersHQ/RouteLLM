"""Selector negation (`!tag:`) and any-of (`tag:a|b`), against a fake catalog."""

import pytest

from routellm import pairing
from routellm.endpoints import EndpointRegistry, Selector
from routellm.pairing import CATALOG_CACHE_ENV, rank_candidates

CONFIG = {
    "tag_namespaces": ["team"],
    "endpoints": {
        "cloud_ca": {"model": "openai/a", "tags": ["cloud", "jur:ca"], "quality": 10},
        "cloud_us": {"model": "openai/b", "tags": ["cloud", "jur:us"], "quality": 20},
        "vpn_fr": {"model": "openai/c", "tags": ["vpn", "jur:fr"], "quality": 30},
        "vpn_plain": {"model": "openai/d", "tags": ["vpn", "team:red"], "quality": 40},
        "loc": {
            "model": "ollama_chat/e",
            "api_base": "http://127.0.0.1:11434",
            "tags": ["local", "team:blue"],
            "quality": 50,
        },
    },
}


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(tmp_path / "models_dev.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])
    return EndpointRegistry.from_config(CONFIG)


def _names(registry, select):
    return sorted(n for n, _ in rank_candidates(registry, Selector(select=select)))


def test_negation_excludes_every_cloud_endpoint(registry):
    assert _names(registry, "!tag:cloud") == ["loc", "vpn_fr", "vpn_plain"]


def test_negation_composes_with_positive_terms(registry):
    assert _names(registry, "tag:vpn !tag:jur:fr") == ["vpn_plain"]


def test_negated_namespaced_tag(registry):
    assert _names(registry, "!tag:jur:ca") == ["cloud_us", "loc", "vpn_fr", "vpn_plain"]


def test_any_of_within_namespace_picks_both(registry):
    assert _names(registry, "tag:jur:ca|jur:fr") == ["cloud_ca", "vpn_fr"]


def test_any_of_is_not_all_of(registry):
    # No endpoint carries two jur tags; all-of would match nothing.
    assert _names(registry, "tag:jur:ca|jur:us|jur:fr") == ["cloud_ca", "cloud_us", "vpn_fr"]


def test_any_of_reach_labels(registry):
    assert _names(registry, "tag:local|vpn") == ["loc", "vpn_fr", "vpn_plain"]


def test_negated_any_of_is_none_of(registry):
    assert _names(registry, "!tag:jur:ca|jur:fr") == ["cloud_us", "loc", "vpn_plain"]


def test_any_of_composes_with_negation(registry):
    assert _names(registry, "tag:jur:ca|jur:us !tag:jur:us") == ["cloud_ca"]


def test_plain_tag_unchanged(registry):
    assert _names(registry, "tag:cloud") == ["cloud_ca", "cloud_us"]


@pytest.mark.parametrize(
    "select",
    [
        "tag:jur:ca|team:red",  # mixed namespace
        "tag:local|jur:ca",  # reach mixed with namespaced
        "tag:fast|slow",  # plain non-reach labels share no namespace
        "tag:jur:ca|",  # empty alternative
        "tag:|jur:ca",
        "tag:",  # empty label
        "!tag:",
        "!tag:|",
        "!tag:nope:x",  # negated unknown namespace
        "tag:nope:x|nope:y",  # or'd unknown namespace
        "tag:nope:x",  # positive unknown namespace (likely a typo)
        "tag:jur:",  # empty value
        "!!tag:cloud",
        "!",
        "!free:true",  # negation only applies to tag:
        "!tool_call:true",
        "! tag:cloud",
    ],
)
def test_malformed_selector_is_a_load_error(registry, select):
    with pytest.raises(ValueError):
        rank_candidates(registry, Selector(select=select))
