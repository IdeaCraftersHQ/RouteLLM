"""Privacy suite, layered config: a second layer cannot open the private tier.

The fixture world is the system layer (with a `policy:`); a project
layer then tries to put a cloud leaf under, or retag a leaf of, the
local-only tier. Startup must fail, before any request is served.
"""

import copy
from pathlib import Path

import pytest
import yaml

from routellm import config as cfg
from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.policy import PolicyError, parse_policy
from routellm.tests.test_privacy_config import BASES, PRIVATE_POLICY, _config


def _write(path: Path, mapping: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping))
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    project = tmp_path / "home" / "repo"
    project.mkdir(parents=True)
    monkeypatch.chdir(project)
    _write(Path(cfg.SYSTEM_PATH), {**_config(), "policy": "policy.yaml"})
    return project / ".routellm.yaml"


def _start(loaded):
    data = copy.deepcopy(loaded.data)
    data.pop("policy")
    return Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(data),
        policy=parse_policy(PRIVATE_POLICY, "policy.yaml"),
        cache_config=CacheConfig(enabled=False),
    )


def test_the_system_layer_alone_starts(world):
    _start(cfg.load_config(interpolate=False))


def test_a_second_layer_adding_a_tagged_cloud_leaf_under_the_private_tier_stops_startup(world):
    _write(
        world,
        {
            "endpoints": {
                "extra_cloud": {
                    "model": "openai/x",
                    "api_base": BASES["cloud_main"],
                    "api_key_env": "FAKE_KEY",
                    "tags": ["cloud", "jur:us"],
                }
            },
            "tiers": {"private": {"weak": "extra_cloud"}},
        },
    )
    with pytest.raises(cfg.ConfigError) as exc:
        _start(cfg.load_config(interpolate=False))
    assert str(world) in str(exc.value) and "extra_cloud" in str(exc.value)


def test_a_second_layer_retagging_a_cloud_leaf_local_stops_startup(world):
    _write(
        world,
        {
            "endpoints": {"cloud_main": {"tags": ["local", "jur:ca"]}},
            "tiers": {"private": {"weak": "cloud_main"}},
        },
    )
    with pytest.raises(cfg.ConfigError, match="cloud_main"):
        cfg.load_config(interpolate=False)


def test_a_second_layer_adding_an_untagged_leaf_under_the_private_tier_stops_startup(world):
    # Passes the pin (no governance tag), so the class walk must refuse it.
    _write(
        world,
        {
            "endpoints": {
                "extra_cloud": {
                    "model": "openai/x",
                    "api_base": BASES["cloud_main"],
                    "api_key_env": "FAKE_KEY",
                }
            },
            "tiers": {"private": {"weak": "extra_cloud"}},
        },
    )
    loaded = cfg.load_config(interpolate=False)
    with pytest.raises(PolicyError, match=r"class 'private', tier 'private'.*leaf 'extra_cloud'"):
        _start(loaded)
