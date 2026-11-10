"""Privacy suite, config-shaped cases: load errors and the tag lint.

The lint is the offline form of the preflight an operator runs before a
start: every leaf reachable from a private tier is local, none is cloud,
its base is on the host, and no Ollama model id names a hosted variant.
It runs here over the fixture world, and over mutated copies that it
must reject.
"""

import copy
from urllib.parse import urlparse

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.policy import PolicyError, parse_policy
from routellm.tests.privacy_fixture import LEAVES, LOCAL, build_config

BASES = {name: f"http://127.0.0.1:{9000 + i}/v1" for i, name in enumerate(LEAVES)}
PRIVATE_TIERS = ("private", "private_audio", "default")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _config():
    return build_config(BASES)


def _leaves(registry, name):
    """Every endpoint reachable from `name`, through nested tiers."""
    if not registry.has_tier(name):
        return {name}
    tier = registry.get_tier(name)
    return _leaves(registry, tier.strong) | _leaves(registry, tier.weak)


def lint_private_reach(config, tiers=PRIVATE_TIERS) -> list[str]:
    """Return the problems with what `tiers` can reach; empty when clean."""
    registry = EndpointRegistry.from_config(copy.deepcopy(config))
    problems = []
    for tier in tiers:
        for leaf in sorted(_leaves(registry, tier)):
            ep = registry.get(leaf)
            if "local" not in ep.tags:
                problems.append(f"{tier}: {leaf} is not tagged local")
            if "cloud" in ep.tags:
                problems.append(f"{tier}: {leaf} is tagged cloud")
            host = urlparse(ep.api_base or "").hostname
            if host not in LOOPBACK:
                problems.append(f"{tier}: {leaf} base host {host!r} is not on the host")
            provider, _, model = ep.model.partition("/")
            if provider.startswith("ollama") and (":cloud" in model or "-cloud" in model):
                problems.append(f"{tier}: {leaf} names a hosted Ollama model {model!r}")
    return problems


def test_the_fixture_world_passes_the_lint():
    assert lint_private_reach(_config()) == []


def test_the_lint_covers_every_local_leaf():
    registry = EndpointRegistry.from_config(_config())
    reached = set().union(*(_leaves(registry, t) for t in PRIVATE_TIERS))
    assert reached == set(LOCAL)


def test_a_cloud_leaf_under_a_private_tier_fails_the_lint():
    cfg = _config()
    cfg["tiers"]["private"]["weak"] = "cloud_main"
    problems = lint_private_reach(cfg)
    assert any("cloud_main is not tagged local" in p for p in problems)
    assert any("cloud_main is tagged cloud" in p for p in problems)


def test_a_cloud_leaf_behind_a_nested_tier_fails_the_lint():
    cfg = _config()
    cfg["tiers"]["default"]["strong"] = "general"
    assert any("gated_main" in p for p in lint_private_reach(cfg))


def test_a_local_tag_on_a_hosted_ollama_model_fails_the_lint():
    cfg = _config()
    cfg["endpoints"]["local_small"]["model"] = "ollama_chat/some-model:cloud"
    problems = lint_private_reach(cfg)
    assert any("hosted Ollama model" in p and "local_small" in p for p in problems)


def test_a_dash_cloud_ollama_model_fails_the_lint_too():
    cfg = _config()
    cfg["endpoints"]["local_small"]["model"] = "ollama/some-model-cloud"
    assert any("hosted Ollama model" in p for p in lint_private_reach(cfg))


def test_a_local_leaf_on_a_remote_host_fails_the_lint():
    cfg = _config()
    cfg["endpoints"]["local_small"]["api_base"] = "https://models.example.com/v1"
    # The load now refuses this unless opted out; the lint backs the opt-out.
    cfg["endpoints"]["local_small"]["allow_public_base"] = True
    assert any("not on the host" in p for p in lint_private_reach(cfg))


def test_a_local_ollama_model_is_not_flagged():
    cfg = _config()
    cfg["endpoints"]["local_small"]["model"] = "ollama_chat/some-model:8b"
    assert lint_private_reach(cfg) == []


# -- load errors ------------------------------------------------------------


def test_a_leaf_with_an_unknown_key_does_not_load():
    cfg = _config()
    cfg["endpoints"]["local_main"]["api_base_env"] = "SOMEWHERE"
    with pytest.raises(ValueError, match="local_main.*api_base_env|api_base_env.*local_main"):
        EndpointRegistry.from_config(cfg)


def test_a_local_leaf_without_a_base_does_not_load():
    cfg = _config()
    del cfg["endpoints"]["local_main"]["api_base"]
    with pytest.raises(ValueError, match="local_main"):
        EndpointRegistry.from_config(cfg)


def test_a_tier_side_naming_a_raw_model_does_not_load():
    # A side that is not a configured endpoint or tier would be a raw
    # provider model, reachable with ambient credentials.
    cfg = _config()
    cfg["tiers"]["private"]["weak"] = "gpt-4o"
    with pytest.raises(ValueError, match="gpt-4o"):
        EndpointRegistry.from_config(cfg)


# -- class to tiers: the private class's tiers reach local leaves only -------

PRIVATE_POLICY = {
    "classes": {
        "private": {"treatment": "block", "reach": ["local"], "tiers": list(PRIVATE_TIERS)},
        "general": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["general"]},
    }
}


def _start(config):
    return Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(copy.deepcopy(config)),
        policy=parse_policy(PRIVATE_POLICY, "policy.yaml"),
        cache_config=CacheConfig(enabled=False),
    )


def test_the_fixture_world_starts_under_the_class_policy():
    _start(_config())


def test_a_cloud_leaf_under_a_private_tier_stops_startup():
    cfg = _config()
    cfg["tiers"]["private"]["weak"] = "cloud_main"
    with pytest.raises(PolicyError, match=r"class 'private', tier 'private'.*leaf 'cloud_main'"):
        _start(cfg)


def test_a_cloud_leaf_behind_a_nested_tier_stops_startup():
    cfg = _config()
    cfg["tiers"]["default"]["strong"] = "general"
    with pytest.raises(PolicyError, match="tier 'default'.*leaf 'gated_main'"):
        _start(cfg)
