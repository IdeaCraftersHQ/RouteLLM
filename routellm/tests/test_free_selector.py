"""The `free:` selector term and `pool:` resolution, against a fake catalog."""

import pytest

from routellm import pairing
from routellm.endpoints import EndpointRegistry, Selector
from routellm.pairing import (
    CATALOG_CACHE_ENV,
    ModelRecord,
    rank_candidates,
    resolve_registry_pairings,
)

CATALOG = [
    ModelRecord("openrouter", "vendor/free-m:free", cost_input=0.0, cost_output=0.0),
    ModelRecord("openrouter", "vendor/paid-m", cost_input=0.3, cost_output=0.9),
    ModelRecord("openrouter", "vendor/half-m", cost_input=0.0, cost_output=0.4),
    ModelRecord("openrouter", "vendor/unpriced-m"),
]

CONFIG = {
    "endpoints": {
        "or_free": {"model": "openrouter/vendor/free-m:free", "tags": ["cloud", "openrouter"]},
        "or_paid": {"model": "openrouter/vendor/paid-m", "tags": ["cloud", "openrouter"]},
        "or_half": {"model": "openrouter/vendor/half-m", "tags": ["cloud", "openrouter"]},
        "or_unpriced": {"model": "openrouter/vendor/unpriced-m", "tags": ["cloud", "openrouter"]},
        "or_uncatalogued": {
            "model": "openrouter/vendor/new-m:free",
            "tags": ["cloud", "openrouter", "free"],
        },
        "local_a": {
            "model": "ollama_chat/a",
            "api_base": "http://127.0.0.1:11434",
            "tags": ["local"],
            "quality": 50,
        },
    }
}


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(tmp_path / "models_dev.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: list(CATALOG))
    return EndpointRegistry.from_config(CONFIG)


def _names(registry, select):
    return [name for name, _ in rank_candidates(registry, Selector(select=select))]


def test_free_true_keeps_only_catalog_priced_zero(registry):
    assert _names(registry, "tag:openrouter free:true") == ["or_free"]


def test_free_true_excludes_paid_half_priced_unpriced_and_uncatalogued(registry):
    names = _names(registry, "tag:openrouter free:true")
    for excluded in ("or_paid", "or_half", "or_unpriced", "or_uncatalogued"):
        assert excluded not in names


def test_free_false_is_everything_not_known_free(registry):
    assert sorted(_names(registry, "tag:openrouter free:false")) == [
        "or_half",
        "or_paid",
        "or_uncatalogued",
        "or_unpriced",
    ]


def test_free_composes_with_other_terms(registry):
    assert _names(registry, "tag:cloud free:true tag:openrouter") == ["or_free"]


def test_free_rejects_a_non_boolean_value(registry):
    with pytest.raises(ValueError, match="free"):
        _names(registry, "free:maybe")


def test_pool_resolves_to_an_ordered_list_with_the_head_as_the_side(registry):
    config = {
        "endpoints": {
            "x1": {"model": "ollama_chat/a", "tags": ["pool"], "quality": 90},
            "x2": {"model": "ollama_chat/b", "tags": ["pool"], "quality": 80},
            "x3": {"model": "ollama_chat/c", "tags": ["pool"], "quality": 70},
            "x4": {"model": "ollama_chat/d", "tags": ["pool"], "quality": 60},
            "y": {"model": "ollama_chat/e", "tags": ["other"]},
        },
        "tiers": {"default": {"strong": {"select": "tag:pool", "pool": 3}, "weak": "y"}},
    }
    reg = EndpointRegistry.from_config(config)
    resolve_registry_pairings(reg)
    reg.revalidate()
    assert reg.get_tier("default").strong == "x1"
    assert reg.pool_for("default", "strong") == ["x1", "x2", "x3"]
    assert reg.pool_for("default", "weak") == ["y"]


def test_pool_must_be_positive():
    with pytest.raises(ValueError):
        Selector(select="tag:a", pool=0)
