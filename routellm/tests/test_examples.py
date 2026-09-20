"""Tests for the shipped example configs under `examples/`.

`examples/multitier.yaml` is the one config in the repo whose tier
sides are almost all selectors, so it is the only place where a typo in
a tag, a term, or a tier name stays invisible until a server starts.
These tests load it, resolve every selector against a fake catalog, and
state that each of its documented entry points reaches an endpoint.

The models.dev catalog is never fetched: `_fetch_catalog` is patched
with records for the cloud models the example names, so the result does
not move when models.dev does.
"""
from pathlib import Path

import pytest
import yaml

import routellm.pairing as pairing
from routellm.endpoints import EndpointRegistry, Selector
from routellm.pairing import ModelRecord, resolve_registry_pairings

REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLE = REPO_ROOT / "examples" / "multitier.yaml"

#: The tiers an app is told to address by name, plus `default`.
ENTRY_TIERS = (
    "default",
    "coding",
    "copywriting",
    "design",
    "private",
    "vpn",
    "fast",
)

#: One record per `(provider, id)` the example's cloud endpoints map
#: onto. Endpoints whose litellm provider has no models.dev counterpart
#: -- Ollama and the OpenAI-compatible local server -- deliberately get
#: none: they are tag-only candidates, and their absence here is part of
#: what the example is written against.
FAKE_CATALOG = [
    ModelRecord(
        provider="anthropic",
        id="claude-sonnet-5",
        cost_input=2.0,
        cost_output=10.0,
        tool_call=True,
        reasoning=True,
        context=1000000,
        release_date="2026-06-29",
    ),
    ModelRecord(
        provider="anthropic",
        id="claude-haiku-4-5",
        cost_input=1.0,
        cost_output=5.0,
        tool_call=True,
        reasoning=True,
        context=200000,
        release_date="2025-10-15",
    ),
    ModelRecord(
        provider="openai",
        id="gpt-5.4",
        cost_input=2.5,
        cost_output=15.0,
        tool_call=True,
        reasoning=True,
        context=1050000,
        release_date="2026-03-05",
    ),
    ModelRecord(
        provider="openai",
        id="gpt-5.4-mini",
        cost_input=0.75,
        cost_output=4.5,
        tool_call=True,
        reasoning=True,
        context=400000,
        release_date="2026-03-17",
    ),
    ModelRecord(
        provider="google",
        id="gemini-3.1-pro-preview",
        cost_input=2.0,
        cost_output=12.0,
        tool_call=True,
        reasoning=True,
        context=1048576,
        release_date="2026-02-19",
    ),
    ModelRecord(
        provider="google",
        id="gemini-3.5-flash",
        cost_input=1.5,
        cost_output=9.0,
        tool_call=True,
        reasoning=True,
        context=1048576,
        release_date="2026-05-19",
    ),
    # models.dev namespaces openrouter ids by vendor, exactly as the
    # litellm model name does after its own prefix is stripped.
    ModelRecord(
        provider="openrouter",
        id="deepseek/deepseek-v4.1-flash",
        cost_input=0.15,
        cost_output=0.6,
        tool_call=True,
        reasoning=True,
        context=1048576,
        release_date="2026-09-10",
    ),
    # cohere ids are bare; litellm's `cohere_chat` maps onto them.
    ModelRecord(
        provider="cohere",
        id="command-a-03-2025",
        cost_input=2.5,
        cost_output=10.0,
        tool_call=True,
        reasoning=False,
        context=256000,
        release_date="2025-03-13",
    ),
    ModelRecord(
        provider="openai",
        id="text-embedding-3-small",
        cost_input=0.02,
        cost_output=0.0,
        tool_call=False,
        reasoning=False,
        context=8191,
        release_date="2024-01-25",
    ),
]


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    """Patch the catalog fetch, and keep the snapshot out of $HOME."""
    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: list(FAKE_CATALOG))
    return FAKE_CATALOG


@pytest.fixture
def config():
    """The example, loaded as the server loads it."""
    with open(EXAMPLE) as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def resolved(config, catalog):
    """The example's registry with every selector resolved to a name."""
    registry = EndpointRegistry.from_config(config)
    resolve_registry_pairings(registry)
    registry.revalidate()
    return registry


def _endpoint_for(registry, name):
    """Walk from a tier or endpoint name down to one endpoint name.

    Parameters
    ----------
    registry : EndpointRegistry
        A registry whose selectors are already resolved.
    name : str
        Tier or endpoint name to start from.

    Returns
    -------
    str
        The endpoint reached by taking `strong` at every tier level.
    """
    seen = set()
    while registry.has_tier(name):
        assert name not in seen, f"cycle through {name}"
        seen.add(name)
        name = registry.get_tier(name).strong
    return name


def _reachable_endpoints(registry, name):
    """Return every endpoint reachable from a tier, by either side.

    Taking only `strong` proves one path; a governance restriction has
    to hold on all of them, because a failed level falls back to its
    sibling and a nested tier is descended by `weak`.

    Parameters
    ----------
    registry : EndpointRegistry
        A registry whose selectors are already resolved.
    name : str
        Tier or endpoint name to start from.

    Returns
    -------
    set[str]
        Every endpoint name reachable through any strong/weak descent.
    """
    found: set[str] = set()
    seen: set[str] = set()
    pending = [name]

    while pending:
        current = pending.pop()
        if not registry.has_tier(current):
            found.add(current)
            continue
        if current in seen:
            continue
        seen.add(current)
        tier = registry.get_tier(current)
        pending.extend([tier.strong, tier.weak])

    return found


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_example_loads_into_the_registry(config):
    """The example's `endpoints:` and `tiers:` survive a real load."""
    registry = EndpointRegistry.from_config(config)

    # Construction validates the tier graph: references, the shared
    # namespace, cycles, and depth.
    registry.revalidate()

    assert registry.names()
    assert set(ENTRY_TIERS).issubset(registry.tier_names())


def test_every_tier_side_names_a_tier_an_endpoint_or_selects(config):
    """No side is a stray string that happens to look like a name."""
    registry = EndpointRegistry.from_config(config)
    known = set(registry.tier_names()) | set(registry.names())

    for name in registry.tier_names():
        tier = registry.get_tier(name)
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            if isinstance(value, Selector):
                continue
            assert value in known, f"{name}.{side} names {value!r}"


def test_local_endpoints_never_use_the_default_ollama_port(config):
    """Every local base URL is a tunnel, never Ollama's own 11434.

    11434 is whatever Ollama happens to run on the machine the server
    is started on; these endpoints mean the host at the far end of a
    tunnel, so pointing them there would silently answer from the
    wrong box.
    """
    registry = EndpointRegistry.from_config(config)

    for name in registry.names():
        api_base = registry.get(name).api_base or ""
        assert "11434" not in api_base, f"{name} points at the default port"


# ---------------------------------------------------------------------------
# Selector resolution
# ---------------------------------------------------------------------------


def test_every_selector_resolves_to_an_endpoint(resolved):
    """After resolution no side is a selector, and each names an endpoint."""
    endpoints = set(resolved.names())
    tiers = set(resolved.tier_names())

    for name in resolved.tier_names():
        tier = resolved.get_tier(name)
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            assert not isinstance(value, Selector), f"{name}.{side} unresolved"
            assert value in endpoints | tiers, f"{name}.{side} is {value!r}"


@pytest.mark.parametrize("tier", ENTRY_TIERS)
def test_each_entry_tier_reaches_an_endpoint(resolved, tier):
    """Every tier an app may address walks down to a real endpoint."""
    assert _endpoint_for(resolved, tier) in set(resolved.names())


def test_private_never_leaves_our_own_hardware(resolved):
    """Nothing reachable from `private` is anything but our own hardware.

    Stronger than `vpn`: every reachable endpoint must be tagged
    `local` AND must not be tagged `cloud`, so no descent can reach a
    model that runs somewhere else.
    """
    reachable = _reachable_endpoints(resolved, "private")

    assert reachable, "private reaches no endpoint at all"
    for name in sorted(reachable):
        tags = resolved.get(name).tags
        assert "local" in tags, f"private reaches {name}, which is not local"
        assert "cloud" not in tags, f"private reaches {name}, which is cloud"


def test_vpn_only_reaches_the_private_network(resolved):
    """Nothing reachable from `vpn`, by any descent, lacks the tag.

    `vpn` constrains the path, not the hardware: a vendor whose public
    API is also exposed privately carries `cloud` and `vpn` both, so a
    vendor-hosted endpoint winning here is correct. What would not be
    correct is reaching an endpoint carrying no `vpn` tag at all, and
    checking only the two immediate sides would miss one that a nested
    tier or a fallback descent brings into range.
    """
    reachable = _reachable_endpoints(resolved, "vpn")

    assert reachable, "vpn reaches no endpoint at all"
    for name in sorted(reachable):
        assert "vpn" in resolved.get(name).tags, (
            f"vpn reaches {name}, which is not on the private network"
        )


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


def test_every_intent_maps_to_a_configured_tier(config):
    """No intent points at a tier the example does not define."""
    registry = EndpointRegistry.from_config(config)
    intent_tiers = config["intents"]["tiers"]

    assert intent_tiers
    for intent, tier in intent_tiers.items():
        assert registry.has_tier(tier), f"intent {intent!r} -> {tier!r}"


def test_every_intent_carries_a_description(config):
    """Both detectors read `descriptions`, so each intent needs one."""
    intents = config["intents"]
    descriptions = intents.get("descriptions") or {}

    for intent in intents["tiers"]:
        assert descriptions.get(intent), f"intent {intent!r} has no description"
