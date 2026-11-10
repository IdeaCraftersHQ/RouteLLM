"""Privacy suite, strict registry: raw cloud model names are refused at load.

With `strict_registry: true` a name that is not a registry endpoint
never reaches litellm. The paths that take a name from outside the
registry (traffic rule, canary target, load-balancer key and target)
must each refuse a raw cloud model when the controller is built, before
a single request, so a private deployment cannot be pointed at a hosted
provider by ambient credentials.
"""

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.quality import CanaryConfig, QualityManager
from routellm.tests.privacy_fixture import LEAVES, build_config
from routellm.traffic import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancerEndpoint,
    TrafficManager,
    TrafficRule,
)

BASES = {name: f"http://127.0.0.1:{9000 + i}/v1" for i, name in enumerate(LEAVES)}
RAW_CLOUD = "openai/gpt-4o"


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch, tmp_path):
    # The fixture's selector tier reads the catalog; an empty snapshot
    # keeps that off the network.
    snapshot = tmp_path / "models_dev.json"
    write_snapshot(snapshot, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(snapshot))


def build(strict=True, **kwargs):
    registry = EndpointRegistry.from_config({**build_config(BASES), "strict_registry": strict})
    return Controller(
        routers=["random"],
        endpoints=registry,
        strict_registry=strict,
        cache_config=CacheConfig(enabled=False),
        **kwargs,
    )


def _balancer(key, *targets):
    cfg = LoadBalancerConfig(endpoints=[LoadBalancerEndpoint(model=t) for t in targets])
    return {key: LoadBalancer(cfg)}


def test_the_fixture_world_loads_strict():
    build()


def test_traffic_rule_naming_a_raw_cloud_model_is_refused():
    rule = TrafficRule(pattern="x", strong_model="local_main", weak_model=RAW_CLOUD)
    with pytest.raises(ValueError, match=r"traffic rule 0 weak model.*gpt-4o"):
        build(traffic_manager=TrafficManager(rules=[rule]))


def test_traffic_rule_naming_a_raw_cloud_strong_model_is_refused():
    rule = TrafficRule(pattern="x", strong_model=RAW_CLOUD, weak_model="local_small")
    with pytest.raises(ValueError, match=r"traffic rule 0 strong model.*gpt-4o"):
        build(traffic_manager=TrafficManager(rules=[rule]))


def test_canary_naming_a_raw_cloud_model_is_refused():
    canary = CanaryConfig(enabled=True, canary_model=RAW_CLOUD, weight=1.0)
    with pytest.raises(ValueError, match=r"canary_model.*gpt-4o"):
        build(quality_manager=QualityManager(canary_config=canary))


def test_load_balancer_key_naming_a_raw_cloud_model_is_refused():
    balancers = _balancer(RAW_CLOUD, "local_main", "local_small")
    with pytest.raises(ValueError, match=r"load balancer key.*gpt-4o"):
        build(traffic_manager=TrafficManager(load_balancers=balancers))


def test_load_balancer_target_naming_a_raw_cloud_model_is_refused():
    balancers = _balancer("local_main", "local_small", RAW_CLOUD)
    with pytest.raises(ValueError, match=r"load balancer 'local_main' target.*gpt-4o"):
        build(traffic_manager=TrafficManager(load_balancers=balancers))


def test_the_same_names_load_when_every_target_is_an_endpoint():
    rule = TrafficRule(pattern="x", strong_model="local_main", weak_model="local_small")
    canary = CanaryConfig(enabled=True, canary_model="local_small", weight=1.0)
    build(
        traffic_manager=TrafficManager(
            rules=[rule], load_balancers=_balancer("local_main", "local_small")
        ),
        quality_manager=QualityManager(canary_config=canary),
    )


def test_the_raw_cloud_names_load_when_strict_is_off():
    # The control: refusal comes from strict mode, not from the names.
    rule = TrafficRule(pattern="x", strong_model="local_main", weak_model=RAW_CLOUD)
    canary = CanaryConfig(enabled=True, canary_model=RAW_CLOUD, weight=1.0)
    build(
        strict=False,
        traffic_manager=TrafficManager(
            rules=[rule], load_balancers=_balancer(RAW_CLOUD, "local_main", RAW_CLOUD)
        ),
        quality_manager=QualityManager(canary_config=canary),
    )
