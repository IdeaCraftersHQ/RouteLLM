"""Tests for policy-based pairing: a tier side that selects, not names.

Covers the `Selector` model, the `select` grammar split into local tag
terms and catalog terms, candidate ordering, the resolution that
rewrites a tier's sides into endpoint names, the snapshot cache and its
staleness behaviour, the litellm-provider to models.dev-provider
mapping, and the `python -m routellm.pairing` explain surface.

The models.dev catalog is never fetched: every test patches
`routellm.pairing._fetch_catalog` with a fake list of records.
"""
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import routellm.pairing as pairing
from routellm.endpoints import EndpointRegistry
from routellm.pairing import (
    CATALOG_CACHE_ENV,
    CatalogUnavailable,
    ModelRecord,
    Selector,
    catalog_provider_for,
    resolve_pairing,
    resolve_registry_pairings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fake catalog
# ---------------------------------------------------------------------------


def _record(provider, model_id, **kw):
    """Build one catalog record with sane defaults."""
    return ModelRecord(
        provider=provider,
        id=model_id,
        cost_input=kw.get("cost_input"),
        cost_output=kw.get("cost_output"),
        tool_call=kw.get("tool_call", False),
        reasoning=kw.get("reasoning", False),
        context=kw.get("context"),
        release_date=kw.get("release_date"),
    )


FAKE_CATALOG = [
    _record(
        "openai",
        "gpt-4o",
        cost_input=2.5,
        cost_output=10.0,
        tool_call=True,
        reasoning=True,
        context=128000,
        release_date="2024-05-13",
    ),
    _record(
        "openai",
        "gpt-4o-mini",
        cost_input=0.15,
        cost_output=0.6,
        tool_call=True,
        reasoning=False,
        context=128000,
        release_date="2024-07-18",
    ),
    _record(
        "anthropic",
        "claude-sonnet-4",
        cost_input=3.0,
        cost_output=15.0,
        tool_call=True,
        reasoning=True,
        context=200000,
        release_date="2025-05-22",
    ),
    # models.dev namespaces every openrouter id by its vendor.
    _record(
        "openrouter",
        "deepseek/deepseek-chat",
        cost_input=0.32,
        cost_output=0.89,
        tool_call=True,
        reasoning=False,
        context=64000,
        release_date="2025-03-24",
    ),
    # cohere ids are bare, and litellm's `cohere_chat` maps onto them.
    _record(
        "cohere",
        "command-a-03-2025",
        cost_input=2.5,
        cost_output=10.0,
        tool_call=True,
        reasoning=False,
        context=256000,
        release_date="2025-03-13",
    ),
    # A foil for the alias tests: listed, but no tool calls.
    _record(
        "openai",
        "gpt-4o-mini-no-tools",
        cost_input=0.15,
        cost_output=0.6,
        tool_call=False,
        reasoning=False,
        context=128000,
        release_date="2024-07-18",
    ),
]

CONFIG = {
    "endpoints": {
        "cloud_strong": {"model": "gpt-4o", "tags": ["cloud"], "quality": 90},
        "cloud_cheap": {"model": "gpt-4o-mini", "tags": ["cloud"], "quality": 60},
        "sonnet": {"model": "anthropic/claude-sonnet-4", "tags": ["cloud"]},
        "local_fast": {
            "model": "ollama_chat/qwen3:8b",
            "tags": ["local"],
            "quality": 40,
        },
        "local_big": {
            "model": "ollama_chat/qwen3:32b",
            "tags": ["local"],
            "quality": 55,
        },
    }
}


@pytest.fixture
def catalog(monkeypatch):
    """Patch the catalog fetch with the fake records."""
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: list(FAKE_CATALOG))
    return FAKE_CATALOG


@pytest.fixture
def no_catalog(monkeypatch):
    """Patch the catalog fetch to fail as an unreachable source would."""

    def _boom():
        raise CatalogUnavailable("models.dev unreachable")

    monkeypatch.setattr(pairing, "_fetch_catalog", _boom)


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    """Point the snapshot cache at a temporary path."""
    path = tmp_path / "models_dev.json"
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(path))
    return path


@pytest.fixture
def registry():
    """Registry with no tiers; tiers are added per test."""
    return EndpointRegistry.from_config(CONFIG)


def _tiered(strong, weak, **kw):
    """Build a registry carrying one `default` tier with these sides."""
    config = dict(CONFIG)
    config["tiers"] = {
        "default": {
            "router": "hi",
            "threshold": 0.12,
            "strong": strong,
            "weak": weak,
            **kw,
        }
    }
    return EndpointRegistry.from_config(config)


# ---------------------------------------------------------------------------
# Selector model
# ---------------------------------------------------------------------------


def test_selector_defaults_to_quality_desc():
    selector = Selector(select="tag:local")
    assert selector.order == "quality_desc"


def test_selector_rejects_unknown_order():
    with pytest.raises(ValidationError):
        Selector(select="tag:local", order="vibes")


def test_selector_rejects_empty_select():
    with pytest.raises(ValidationError):
        Selector(select="   ")


def test_tier_side_accepts_a_selector_mapping():
    registry = _tiered({"select": "tag:local", "order": "cost_asc"}, "cloud_cheap")
    tier = registry.get_tier("default")
    assert isinstance(tier.strong, Selector)
    assert tier.strong.select == "tag:local"
    assert tier.strong.order == "cost_asc"


def test_validation_skips_selector_sides():
    # A Selector side names no endpoint, so reference validation must
    # not reject it before pairing has resolved it.
    tier = _tiered({"select": "tag:local"}, {"select": "tag:cloud"}).get_tier("default")

    assert isinstance(tier.strong, Selector)
    assert isinstance(tier.weak, Selector)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_tag_only_selection_needs_no_catalog(registry, no_catalog, cache_dir):
    selector = Selector(select="tag:local", order="quality_desc")
    assert resolve_pairing(registry, selector) == "local_big"


def test_catalog_term_selection(registry, catalog, cache_dir):
    selector = Selector(select="tool_call:true reasoning:true", order="quality_desc")
    assert resolve_pairing(registry, selector) == "cloud_strong"


def test_terms_are_anded(registry, catalog, cache_dir):
    selector = Selector(select="tag:cloud reasoning:false", order="quality_desc")
    assert resolve_pairing(registry, selector) == "cloud_cheap"


def test_candidate_without_a_catalog_record_fails_a_catalog_term(
    registry, catalog, cache_dir
):
    # local_fast/local_big map to no models.dev provider, so they drop
    # out the moment a catalog term is present even though they carry
    # the tag.
    selector = Selector(select="tool_call:true", order="quality_desc")
    assert resolve_pairing(registry, selector) == "cloud_strong"


def test_cost_ordering_puts_none_last(registry, catalog, cache_dir):
    # local_fast maps to no catalog record and so carries no cost; it
    # must sort after every priced endpoint, not before the cheapest.
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "cloud_strong": {"model": "gpt-4o", "tags": ["mixed"]},
                "cloud_cheap": {"model": "gpt-4o-mini", "tags": ["mixed"]},
                "local_fast": {"model": "ollama_chat/qwen3:8b", "tags": ["mixed"]},
            }
        }
    )
    selector = Selector(select="tag:mixed", order="cost_asc")

    assert resolve_pairing(registry, selector) == "cloud_cheap"

    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered == ["cloud_cheap", "cloud_strong", "local_fast"]


def test_cost_desc_still_puts_none_last(registry, catalog, cache_dir):
    # "Unknown last" holds in both directions: an unpriced endpoint is
    # never promoted to the front of a descending order either.
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "cloud_strong": {"model": "gpt-4o", "tags": ["mixed"]},
                "cloud_cheap": {"model": "gpt-4o-mini", "tags": ["mixed"]},
                "local_fast": {"model": "ollama_chat/qwen3:8b", "tags": ["mixed"]},
            }
        }
    )
    selector = Selector(select="tag:mixed", order="cost_desc")
    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered == ["cloud_strong", "cloud_cheap", "local_fast"]


def test_context_desc_ordering(registry, catalog, cache_dir):
    selector = Selector(select="tag:cloud", order="context_desc")
    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered[0] == "sonnet"


def test_quality_tiebreaks_on_release_date(catalog, cache_dir):
    # Two endpoints share a manual quality, so the newer release wins
    # even though its name sorts later.
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "a_older": {"model": "gpt-4o", "tags": ["cloud"], "quality": 70},
                "z_newer": {
                    "model": "anthropic/claude-sonnet-4",
                    "tags": ["cloud"],
                    "quality": 70,
                },
            }
        }
    )
    selector = Selector(select="tag:cloud", order="quality_desc")
    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered == ["z_newer", "a_older"]


def test_undated_sorts_after_dated_among_the_unrated(catalog, cache_dir):
    # Neither endpoint carries a manual quality, so the release-date
    # tiebreak decides. An endpoint with no catalog record has no date
    # at all and must sort last, not first.
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "a_undated": {"model": "ollama_chat/qwen3:8b", "tags": ["mixed"]},
                "z_dated": {"model": "gpt-4o", "tags": ["mixed"]},
            }
        }
    )
    selector = Selector(select="tag:mixed", order="quality_desc")
    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered == ["z_dated", "a_undated"]


def test_unrated_endpoints_sort_after_rated_ones(registry, catalog, cache_dir):
    # sonnet carries no manual quality, so it loses to every rated
    # endpoint however recent its release.
    selector = Selector(select="tag:cloud", order="quality_desc")
    ordered = [name for name, _ in pairing.rank_candidates(registry, selector)]
    assert ordered == ["cloud_strong", "cloud_cheap", "sonnet"]


def test_no_candidate_raises_naming_the_select(registry, catalog, cache_dir):
    selector = Selector(select="tag:nonexistent")
    with pytest.raises(ValueError, match="tag:nonexistent"):
        resolve_pairing(registry, selector)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------


def test_unknown_query_key_surfaces_the_term(registry, catalog, cache_dir):
    selector = Selector(select="wizardry:true")
    with pytest.raises(ValueError, match="wizardry:true"):
        resolve_pairing(registry, selector)


def test_bare_free_text_token_rejected(registry, catalog, cache_dir):
    selector = Selector(select="tag:cloud sonnet")
    with pytest.raises(ValueError, match="sonnet"):
        resolve_pairing(registry, selector)


@pytest.mark.parametrize(
    "key", ["open_weights", "structured_output", "temperature", "family"]
)
@pytest.mark.parametrize("value", ["true", "false"])
def test_unsupported_catalog_term_rejected_whatever_its_value(
    registry, catalog, cache_dir, key, value
):
    # A false-valued tri-state is still a term pairing cannot judge; it
    # must be rejected rather than silently matching every candidate.
    selector = Selector(select=f"{key}:{value}")
    with pytest.raises(ValueError, match=key):
        resolve_pairing(registry, selector)


def test_tag_terms_are_stripped_before_the_catalog_parser(
    registry, no_catalog, cache_dir
):
    # `tag` is not a key hop.aim knows; it must never reach parse_query.
    selector = Selector(select="tag:local tag:local")
    assert resolve_pairing(registry, selector) == "local_big"


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------


def test_resolve_registry_pairings_rewrites_both_sides(catalog, cache_dir):
    registry = _tiered(
        {"select": "tool_call:true reasoning:true", "order": "quality_desc"},
        {"select": "tag:local", "order": "quality_asc"},
    )
    resolve_registry_pairings(registry)

    tier = registry.get_tier("default")
    assert tier.strong == "cloud_strong"
    assert tier.weak == "local_fast"


def test_identical_winners_raise(catalog, cache_dir):
    registry = _tiered({"select": "tag:local"}, {"select": "tag:local"})
    with pytest.raises(ValueError, match="both sides"):
        resolve_registry_pairings(registry)


def test_resolved_registry_still_validates(catalog, cache_dir):
    registry = _tiered({"select": "tag:cloud"}, {"select": "tag:local"})
    resolve_registry_pairings(registry)
    registry.revalidate()


def test_controller_resolves_selectors_at_construction(
    catalog, cache_dir, monkeypatch
):
    import routellm.controller
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    monkeypatch.setitem(
        routellm.controller.ROUTER_CLS, "hi", lambda **kw: _StubRouter()
    )
    registry = _tiered({"select": "tag:cloud"}, {"select": "tag:local"})

    Controller(
        routers=["hi"],
        config={},
        endpoints=registry,
        cache_config=CacheConfig(enabled=False),
        default_router="hi",
    )

    tier = registry.get_tier("default")
    assert tier.strong == "cloud_strong"
    assert tier.weak == "local_big"


class _StubRouter:
    """Minimal router double: the controller only instantiates it here."""

    def route(self, prompt, threshold, model_pair):
        return model_pair.weak


def test_candidate_table_logged_at_info(registry, catalog, cache_dir, caplog):
    selector = Selector(select="tag:local", order="quality_desc")
    with caplog.at_level(logging.INFO, logger="routellm.pairing"):
        resolve_pairing(registry, selector)
    text = caplog.text
    assert "local_big" in text
    assert "local_fast" in text


# ---------------------------------------------------------------------------
# Catalog availability and the snapshot
# ---------------------------------------------------------------------------


def test_catalog_unavailable_with_a_tag_only_policy_passes(
    registry, no_catalog, cache_dir
):
    assert resolve_pairing(registry, Selector(select="tag:cloud")) == "cloud_strong"


def test_catalog_unavailable_with_a_catalog_term_raises(
    registry, no_catalog, cache_dir
):
    with pytest.raises(CatalogUnavailable):
        resolve_pairing(registry, Selector(select="tool_call:true"))


def test_fresh_snapshot_is_used_without_fetching(registry, no_catalog, cache_dir):
    pairing.write_snapshot(cache_dir, FAKE_CATALOG, fetched_at=time.time())
    assert resolve_pairing(registry, Selector(select="tool_call:true")) == (
        "cloud_strong"
    )


def test_stale_snapshot_is_used_with_a_warning(
    registry, no_catalog, cache_dir, caplog
):
    stale = time.time() - (pairing.CATALOG_TTL_SECONDS + 60)
    pairing.write_snapshot(cache_dir, FAKE_CATALOG, fetched_at=stale)

    with caplog.at_level(logging.WARNING, logger="routellm.pairing"):
        picked = resolve_pairing(registry, Selector(select="tool_call:true"))

    assert picked == "cloud_strong"
    assert "stale" in caplog.text.lower()


def test_successful_fetch_writes_the_snapshot(registry, catalog, cache_dir):
    resolve_pairing(registry, Selector(select="tool_call:true"))

    payload = json.loads(cache_dir.read_text())
    assert "fetched_at" in payload
    ids = {record["id"] for record in payload["models"]}
    assert "gpt-4o" in ids


# ---------------------------------------------------------------------------
# Provider mapping
# ---------------------------------------------------------------------------


def test_fetch_gives_up_at_the_timeout_without_waiting_for_the_worker(
    monkeypatch, cache_dir
):
    """A hung fetch must not hold startup past the timeout.

    The worker sleeps far longer than the timeout; the call has to
    surface `CatalogUnavailable` promptly rather than blocking until
    the thread finishes, which is what a `with` block around the pool
    would do through `shutdown(wait=True)`.
    """
    import hop.aim as aim

    started = threading.Event()

    class _HangingRegistry:
        async def models(self, filter=None):
            started.set()
            time.sleep(10)
            return []

    monkeypatch.setattr(aim, "Registry", _HangingRegistry)
    monkeypatch.setattr(pairing, "CATALOG_FETCH_TIMEOUT", 0.2)

    start = time.monotonic()
    with pytest.raises(CatalogUnavailable):
        pairing._fetch_catalog()
    elapsed = time.monotonic() - start

    assert started.is_set(), "the worker never ran, so nothing was timed"
    assert elapsed < 1.0, f"gave up after {elapsed:.2f}s, expected well under 1s"


def test_provider_mapping_for_a_bare_openai_name():
    assert catalog_provider_for("gpt-4o") == ("openai", "gpt-4o")


def test_provider_mapping_for_a_local_ollama_name():
    assert catalog_provider_for("ollama_chat/qwen3:8b") is None


def test_provider_mapping_for_an_unresolvable_name():
    assert catalog_provider_for("anyscale/mistralai/Mixtral-8x7B") is None


def test_provider_mapping_aliases_gemini_to_google():
    assert catalog_provider_for("gemini/gemini-2.0-flash") == (
        "google",
        "gemini-2.0-flash",
    )


def test_provider_mapping_keeps_the_openrouter_namespace():
    """openrouter catalog ids carry the vendor, and so must the mapping.

    Every models.dev id under the `openrouter` provider is namespaced
    `<vendor>/<model>`, and litellm strips only its own `openrouter/`
    prefix, so the remainder is already the catalog id.
    """
    assert catalog_provider_for("openrouter/deepseek/deepseek-chat") == (
        "openrouter",
        "deepseek/deepseek-chat",
    )


def test_provider_mapping_aliases_cohere_chat_to_cohere():
    """litellm's `cohere_chat` is models.dev's `cohere`, id unchanged."""
    assert catalog_provider_for("cohere_chat/command-a-03-2025") == (
        "cohere",
        "command-a-03-2025",
    )


def test_provider_mapping_aliases_the_bare_cohere_route():
    """litellm also emits a bare `cohere` provider; it is the same catalog."""
    assert catalog_provider_for("cohere/embed-english-v3.0") == (
        "cohere",
        "embed-english-v3.0",
    )


def test_an_openrouter_endpoint_satisfies_a_catalog_term(catalog, cache_dir):
    """An openrouter model reaches the catalog, so it can match on facts.

    Before openrouter was aliased it was a tag-only candidate: it
    failed every catalog term, and a config had to carry a hand-written
    capability tag to keep it selectable. Its record now answers for it.
    """
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "via_openrouter": {
                    "model": "openrouter/deepseek/deepseek-chat",
                    "tags": ["cloud"],
                },
                "no_tools": {"model": "gpt-4o-mini-no-tools", "tags": ["cloud"]},
            }
        }
    )

    assert resolve_pairing(registry, Selector(select="tool_call:true")) == (
        "via_openrouter"
    )


def test_a_cohere_endpoint_satisfies_a_catalog_term(catalog, cache_dir):
    """The same for cohere, which models.dev lists under bare ids."""
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "via_cohere": {
                    "model": "cohere_chat/command-a-03-2025",
                    "tags": ["cloud"],
                },
                "no_tools": {"model": "gpt-4o-mini-no-tools", "tags": ["cloud"]},
            }
        }
    )

    assert resolve_pairing(registry, Selector(select="tool_call:true")) == "via_cohere"


def test_provider_mapping_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="routellm.pairing"):
        catalog_provider_for("gpt-4o")
    assert "openai" in caplog.text


# ---------------------------------------------------------------------------
# Explain surface
# ---------------------------------------------------------------------------


def test_explain_output_contains_the_pick(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  cloud_strong: {model: gpt-4o, tags: [cloud], quality: 90}\n"
        "  local_fast: {model: 'ollama_chat/qwen3:8b', tags: [local], quality: 40}\n"
        "  local_big: {model: 'ollama_chat/qwen3:32b', tags: [local], quality: 55}\n"
        "tiers:\n"
        "  default:\n"
        "    router: mf\n"
        "    threshold: 0.12\n"
        "    strong: {select: 'tag:cloud', order: quality_desc}\n"
        "    weak: {select: 'tag:local', order: quality_asc}\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "routellm.pairing", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_subprocess_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "default" in result.stdout
    assert "cloud_strong" in result.stdout
    assert "local_fast" in result.stdout


def test_explain_exits_one_on_a_resolution_error(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  local_fast: {model: 'ollama_chat/qwen3:8b', tags: [local]}\n"
        "tiers:\n"
        "  default:\n"
        "    strong: {select: 'tag:nowhere'}\n"
        "    weak: local_fast\n"
    )

    result = subprocess.run(
        [sys.executable, "-m", "routellm.pairing", "--config", str(config)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_subprocess_env(tmp_path),
    )

    assert result.returncode == 1
    assert "tag:nowhere" in result.stderr


def _subprocess_env(tmp_path):
    """Environment pointing the snapshot cache inside `tmp_path`."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env[CATALOG_CACHE_ENV] = str(tmp_path / "subprocess_cache.json")
    return env
