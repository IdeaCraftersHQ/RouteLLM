"""Tests for the capability terms a startup selector may carry.

Covers the boolean terms (`vision:`, `tools:`, `structured_output:`,
`reasoning:`, `open_weights:`), the range terms (`context:>=N`,
`max_output:<=N`), `input:image` over `modalities_in`, the
`max_output_desc` order, the terms `_record_matches` no longer rejects,
and when a failed catalog fetch is tolerable.

The models.dev catalog is never fetched: the `catalog` fixture patches
both the fetch and the snapshot path, as test_examples.py does, because
`load_catalog` reads the snapshot before it ever fetches.
"""

import pytest

import routellm.pairing as pairing
from routellm.capabilities import (
    CAPABILITY_KEYS,
    RANGE_KEYS,
    Capabilities,
    parse_capability_terms,
)
from routellm.capabilities import matches as caps_match
from routellm.endpoints import Endpoint, EndpointRegistry, Selector
from routellm.pairing import (
    CatalogUnavailable,
    ModelRecord,
    rank_candidates,
    resolve_pairing,
)

FAKE_CATALOG = [
    ModelRecord(
        provider="openai",
        id="gpt-4o",
        cost_input=2.5,
        cost_output=10.0,
        tool_call=True,
        reasoning=True,
        context=128000,
        release_date="2024-05-13",
        structured_output=True,
        open_weights=False,
        modalities_in=["text", "image"],
        max_output=16384,
    ),
    ModelRecord(
        provider="openai",
        id="gpt-4o-mini",
        cost_input=0.15,
        cost_output=0.6,
        tool_call=True,
        reasoning=False,
        context=128000,
        release_date="2024-07-18",
        structured_output=True,
        open_weights=False,
        modalities_in=["text"],
        max_output=4096,
    ),
    ModelRecord(
        provider="anthropic",
        id="claude-sonnet-4",
        cost_input=3.0,
        cost_output=15.0,
        tool_call=True,
        reasoning=True,
        context=300000,
        release_date="2025-02-24",
        structured_output=False,
        open_weights=False,
        modalities_in=["text", "image"],
        max_output=64000,
    ),
]


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """Patch the catalog fetch, and keep the snapshot out of $HOME."""
    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: list(FAKE_CATALOG))
    return FAKE_CATALOG


@pytest.fixture
def no_catalog(monkeypatch, tmp_path):
    """Make every catalog fetch fail, with no snapshot to fall back on."""
    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "missing.json"))

    def _boom():
        raise CatalogUnavailable("no network in this test")

    monkeypatch.setattr(pairing, "_fetch_catalog", _boom)


def _registry(**endpoints):
    """Build a registry from `name=Endpoint` keyword pairs."""
    return EndpointRegistry(endpoints=endpoints)


def _names(registry, select, order="quality_desc"):
    """Return the ranked endpoint names a selector picks."""
    return [
        name
        for name, _ in rank_candidates(registry, Selector(select=select, order=order))
    ]


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def test_vision_true_selects_only_vision_endpoints(catalog):
    registry = _registry(
        seeing=Endpoint(name="seeing", model="gpt-4o", quality=90),
        blind=Endpoint(name="blind", model="gpt-4o-mini", quality=80),
    )

    assert _names(registry, "vision:true") == ["seeing"]


def test_tools_is_an_alias_of_tool_call(catalog):
    registry = _registry(
        cloud=Endpoint(name="cloud", model="gpt-4o", quality=90),
        local=Endpoint(
            name="local",
            model="ollama_chat/qwen3:8b",
            quality=70,
            capabilities=Capabilities(tools=False),
        ),
    )

    assert _names(registry, "tools:true") == ["cloud"]
    assert _names(registry, "tool_call:true") == ["cloud"]


def test_context_ge_filters_on_the_merged_context(catalog):
    registry = _registry(
        wide=Endpoint(name="wide", model="anthropic/claude-sonnet-4", quality=95),
        narrow=Endpoint(name="narrow", model="gpt-4o", quality=90),
        local=Endpoint(
            name="local",
            model="ollama_chat/qwen3:8b",
            quality=70,
            capabilities=Capabilities(context=262144),
        ),
    )

    assert sorted(_names(registry, "context:>=200000")) == ["local", "wide"]


def test_context_le_filters_the_other_way(catalog):
    registry = _registry(
        wide=Endpoint(name="wide", model="anthropic/claude-sonnet-4", quality=95),
        narrow=Endpoint(name="narrow", model="gpt-4o", quality=90),
    )

    assert _names(registry, "context:<=200000") == ["narrow"]


def test_bare_context_term_is_rejected_with_the_ge_spelling():
    with pytest.raises(ValueError) as excinfo:
        parse_capability_terms(["context:200000"])

    message = str(excinfo.value)
    assert "context" in message
    assert "context:>=200000" in message


def test_input_image_matches_modalities_in(catalog):
    registry = _registry(
        seeing=Endpoint(name="seeing", model="gpt-4o", quality=90),
        blind=Endpoint(name="blind", model="gpt-4o-mini", quality=80),
    )

    assert _names(registry, "input:image") == ["seeing"]


def test_unknown_capability_fails_the_term():
    caps = Capabilities()
    query = parse_capability_terms(["vision:true"])

    assert caps_match(caps, query) is False
    assert caps_match(Capabilities(vision=True), query) is True


def test_local_endpoint_with_an_explicit_block_wins_without_tags(catalog):
    registry = _registry(
        cloud=Endpoint(name="cloud", model="gpt-4o-mini", quality=80),
        local=Endpoint(
            name="local",
            model="ollama_chat/qwen3:8b",
            quality=95,
            capabilities=Capabilities(vision=True, tools=True, context=262144),
        ),
    )

    assert _names(registry, "vision:true tools:true context:>=200000") == ["local"]


def test_max_output_desc_orders_and_puts_unknown_last(catalog):
    registry = _registry(
        big=Endpoint(name="big", model="anthropic/claude-sonnet-4", tags=["pool"]),
        mid=Endpoint(name="mid", model="gpt-4o", tags=["pool"]),
        small=Endpoint(name="small", model="gpt-4o-mini", tags=["pool"]),
        mystery=Endpoint(
            name="mystery", model="ollama_chat/qwen3:8b", tags=["pool"]
        ),
    )

    ranked = _names(registry, "tag:pool", order="max_output_desc")
    assert ranked == ["big", "mid", "small", "mystery"]


def test_structured_output_term_no_longer_raises_unsupported(catalog):
    registry = _registry(
        structured=Endpoint(name="structured", model="gpt-4o", quality=90),
        plain=Endpoint(name="plain", model="anthropic/claude-sonnet-4", quality=95),
    )

    assert _names(registry, "structured_output:true") == ["structured"]
    assert _names(registry, "open_weights:false") == ["plain", "structured"]


def test_family_term_still_raises_unsupported(catalog):
    registry = _registry(cloud=Endpoint(name="cloud", model="gpt-4o"))

    with pytest.raises(ValueError, match="family"):
        rank_candidates(registry, Selector(select="family:gpt"))


# ---------------------------------------------------------------------------
# Catalog necessity
# ---------------------------------------------------------------------------


def test_catalog_failure_is_tolerated_when_blocks_answer_everything(no_catalog, caplog):
    registry = _registry(
        local=Endpoint(
            name="local",
            model="ollama_chat/qwen3:8b",
            quality=90,
            capabilities=Capabilities(vision=True, tools=True, context=262144),
        ),
        other=Endpoint(
            name="other",
            model="ollama_chat/llama3:8b",
            quality=80,
            capabilities=Capabilities(vision=False, tools=True, context=8192),
        ),
    )

    assert resolve_pairing(registry, Selector(select="vision:true")) == "local"


def test_catalog_failure_raises_when_a_term_needs_the_record(no_catalog):
    registry = _registry(
        local=Endpoint(
            name="local",
            model="ollama_chat/qwen3:8b",
            capabilities=Capabilities(vision=True),
        ),
        cloud=Endpoint(name="cloud", model="gpt-4o"),
    )

    with pytest.raises(CatalogUnavailable):
        rank_candidates(registry, Selector(select="vision:true"))


# ---------------------------------------------------------------------------
# Grammar surface
# ---------------------------------------------------------------------------


def test_capability_key_sets_are_what_the_plan_fixes():
    assert CAPABILITY_KEYS == {
        "vision",
        "tools",
        "structured_output",
        "reasoning",
        "open_weights",
    }
    assert RANGE_KEYS == {"context", "max_output"}


def test_split_terms_routes_each_key_to_its_bucket():
    tags, capability_terms, query = pairing._split_terms(
        "tag:local vision:true context:>=1000 provider:openai"
    )

    assert tags == ["local"]
    assert capability_terms == ["vision:true", "context:>=1000"]
    assert query == "provider:openai"
