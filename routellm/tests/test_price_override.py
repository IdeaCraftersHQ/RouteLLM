"""Endpoint `price` / `free`: an explicit price beats the catalog everywhere."""

import pytest

from routellm import pairing
from routellm.endpoints import EndpointRegistry, Selector
from routellm.pairing import CATALOG_CACHE_ENV, ModelRecord, rank_candidates

CATALOG = [
    ModelRecord("openrouter", "vendor/paid-m", cost_input=3.0, cost_output=9.0),
    ModelRecord("openrouter", "vendor/free-m:free", cost_input=0.0, cost_output=0.0),
]


def _cfg(**per_endpoint):
    base = {
        "or_paid": {"model": "openrouter/vendor/paid-m", "tags": ["t"]},
        "or_free": {"model": "openrouter/vendor/free-m:free", "tags": ["t"]},
        "local": {"model": "ollama_chat/a", "api_base": "http://127.0.0.1:11434", "tags": ["t"]},
    }
    for name, extra in per_endpoint.items():
        base[name] = {**base[name], **extra}
    return {"endpoints": base}


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(tmp_path / "models_dev.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: list(CATALOG))
    return lambda **kw: EndpointRegistry.from_config(_cfg(**kw))


def _names(registry, select, order="quality_desc"):
    return [n for n, _ in rank_candidates(registry, Selector(select=select, order=order))]


def test_free_shorthand_makes_uncatalogued_endpoint_free(make):
    assert set(_names(make(local={"free": True}), "tag:t free:true")) == {"local", "or_free"}


def test_price_zero_makes_uncatalogued_endpoint_free(make):
    reg = make(local={"price": {"input": 0, "output": 0}})
    assert "local" in _names(reg, "tag:t free:true")


def test_nonzero_override_is_not_free(make):
    reg = make(local={"price": {"input": 0, "output": 0.2}})
    assert "local" not in _names(reg, "tag:t free:true")
    assert "local" in _names(reg, "tag:t free:false")


def test_free_override_beats_catalog_price(make):
    reg = make(or_paid={"free": True})
    assert "or_paid" in _names(reg, "tag:t free:true")


def test_price_override_beats_catalog_free(make):
    reg = make(or_free={"price": {"input": 1, "output": 1}})
    assert "or_free" not in _names(reg, "tag:t free:true")


def test_cost_asc_orders_on_override(make):
    reg = make(local={"price": {"input": 0.1, "output": 0.1}})
    # catalog prices: or_free 0, or_paid 12; local override 0.2 sits between.
    assert _names(reg, "tag:t", "cost_asc") == ["or_free", "local", "or_paid"]


def test_cost_ordering_override_replaces_catalog_price(make):
    reg = make(or_paid={"price": {"input": 0, "output": 0.01}})
    assert _names(reg, "tag:t", "cost_asc")[:2] == ["or_free", "or_paid"]


@pytest.mark.parametrize(
    "bad",
    [
        {"price": {"input": -1, "output": 0}},
        {"price": {"input": 0, "output": -0.1}},
        {"price": {"input": "cheap", "output": 0}},
        {"price": {"input": True, "output": 0}},
        {"price": {"input": float("nan"), "output": 0}},
        {"price": {"input": 0}},
        {"price": {"input": 0, "output": 0, "currency": "usd"}},
        {"price": 0},
        {"free": False},
        {"free": "yes"},
        {"free": True, "price": {"input": 0, "output": 0}},
    ],
)
def test_malformed_override_rejected_at_load(make, bad):
    with pytest.raises(Exception, match=r"price|free|input|output|Input|Extra|greater"):
        make(local=bad)
