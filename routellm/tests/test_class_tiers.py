"""Class to allowed tiers: the entry check and the startup reachability walk.

Startup cases build a `Controller` over a registry and a policy and assert
the `PolicyError` names class, tier, leaf and tag. Request cases run the
controller in process against fake upstreams that count hits: a refused
request leaves every count at zero.
"""

import asyncio

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.decision import Decision
from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import PolicyDenied, PolicyError, parse_policy
from routellm.quality import CanaryConfig, QualityManager
from routellm.tests.fake_upstream import FakeUpstream
from routellm.traffic import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancerEndpoint,
    TrafficManager,
    TrafficRule,
)

MESSAGES = [{"role": "user", "content": "hello"}]
PAIR = {"router": "random", "threshold": 0.0}


def endpoint(tags, base="http://127.0.0.1:1/v1", quality=None):
    spec = {"model": "openai/m", "api_base": base, "api_key_env": "FAKE_KEY", "tags": tags}
    return {**spec, "quality": quality} if quality is not None else spec


def endpoints(bases=None):
    bases = bases or {}

    def ep(name, tags, **kw):
        return endpoint(tags, bases.get(name, "http://127.0.0.1:1/v1"), **kw)

    return {
        "loc": ep("loc", ["local", "jur:ca"], quality=80),
        "loc2": ep("loc2", ["local", "jur:ca"], quality=40),
        "loc_us": ep("loc_us", ["local", "jur:us"], quality=30),
        "lan": ep("lan", ["local", "vpn", "jur:ca"], quality=35),
        "cld": ep("cld", ["cloud", "jur:us"], quality=90),
        "cld_ca": ep("cld_ca", ["cloud", "jur:ca"], quality=50),
    }


def tiers(**over):
    base = {
        "private": {**PAIR, "strong": "loc", "weak": "loc2"},
        "general": {**PAIR, "strong": "cld", "weak": "loc"},
        "default": {**PAIR, "strong": "private", "weak": "loc2"},
    }
    base.update(over)
    return base


def policy(**classes):
    spec = {
        "secret": {"treatment": "block", "reach": ["local"], "jur": ["ca"], "tiers": ["private"]},
        "open": {
            "treatment": "raw",
            "reach": ["local", "vpn", "cloud"],
            "tiers": ["default", "general"],
        },
    }
    spec.update(classes)
    return parse_policy({"classes": spec}, "p.yaml")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


def build(tier_cfg=None, pol=None, eps=None, bases=None, **kwargs):
    registry = EndpointRegistry.from_config(
        {
            "endpoints": eps or endpoints(bases),
            "tiers": tier_cfg or tiers(),
            "strict_registry": True,
        }
    )
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    return Controller(
        routers=["random"],
        endpoints=registry,
        policy=pol or policy(),
        strict_registry=True,
        **kwargs,
    )


def refused(**kwargs) -> str:
    with pytest.raises(PolicyError) as exc:
        build(**kwargs)
    return str(exc.value)


# --- startup: the walk --------------------------------------------------------


def test_a_clean_config_starts():
    build()


def test_a_cloud_sibling_fails_startup_naming_class_tier_leaf_and_tag():
    msg = refused(tier_cfg=tiers(private={**PAIR, "strong": "loc", "weak": "cld"}))
    assert "class 'secret'" in msg
    assert "tier 'private'" in msg
    assert "leaf 'cld'" in msg
    assert "'cloud'" in msg


def test_a_sibling_the_fallback_rule_names_is_walked(monkeypatch):
    # Today every sibling is also a side of some level. The walk still asks
    # the fallback rule, so a rule that reached further is held to the class.
    monkeypatch.setattr("routellm.class_tiers.sibling_of", lambda path, registry: ("cld", "cld"))
    msg = refused()
    assert "fallback sibling" in msg and "leaf 'cld'" in msg and "class 'secret'" in msg


def test_a_nested_tier_is_walked():
    inner = {"inner": {**PAIR, "strong": "cld", "weak": "loc2"}}
    msg = refused(tier_cfg=tiers(**inner, private={**PAIR, "strong": "loc", "weak": "inner"}))
    assert "leaf 'cld'" in msg and "tier 'private'" in msg


def test_a_selector_side_is_walked_after_pairing():
    sel = {"select": "tag:cloud", "order": "quality_desc"}
    msg = refused(tier_cfg=tiers(private={**PAIR, "strong": "loc", "weak": sel}))
    assert "leaf 'cld'" in msg


def test_a_pool_member_is_walked():
    # Head is local, the second pool member is a cloud leaf in `ca`.
    strong = {"select": "tag:jur:ca", "order": "quality_desc", "pool": 3}
    cfg = tiers(private={**PAIR, "strong": strong, "weak": "loc2"})
    msg = refused(tier_cfg=cfg)
    assert "leaf 'cld_ca'" in msg and "pool" in msg


def test_a_jurisdiction_outside_the_class_list_fails_naming_the_tag():
    msg = refused(tier_cfg=tiers(private={**PAIR, "strong": "loc", "weak": "loc_us"}))
    assert "class 'secret'" in msg and "leaf 'loc_us'" in msg and "'jur:us'" in msg


def test_a_reach_wider_than_the_class_fails_naming_the_extra_label():
    msg = refused(tier_cfg=tiers(private={**PAIR, "strong": "loc", "weak": "lan"}))
    assert "leaf 'lan'" in msg and "'vpn'" in msg and "'local'" not in msg.split("carries")[1][:12]


def test_an_empty_class_jur_allows_any_jurisdiction():
    pol = policy(secret={"treatment": "block", "reach": ["local"], "tiers": ["private"]})
    build(tier_cfg=tiers(private={**PAIR, "strong": "loc", "weak": "loc_us"}), pol=pol)


def test_every_violation_is_listed():
    msg = refused(tier_cfg=tiers(private={**PAIR, "strong": "cld", "weak": "loc_us"}))
    assert "leaf 'cld'" in msg and "leaf 'loc_us'" in msg


def test_a_class_is_held_only_to_its_own_tiers():
    # `open` may reach the cloud; only `secret` is held to `private`.
    build(tier_cfg=tiers(general={**PAIR, "strong": "cld", "weak": "cld_ca"}))


def test_an_unknown_tier_in_a_class_fails_startup():
    pol = policy(open={"treatment": "raw", "reach": ["cloud"], "tiers": ["nope"]})
    msg = refused(pol=pol)
    assert "class 'open'" in msg and "'nope'" in msg


def test_a_vault_class_is_skipped():
    build(pol=policy(sealed={"treatment": "vault"}))


# --- startup: names that bypass the tree --------------------------------------


def test_a_traffic_rule_pair_is_walked():
    rule = TrafficRule(pattern="x", strong_model="loc", weak_model="cld")
    msg = refused(traffic_manager=TrafficManager(rules=[rule]))
    assert "class 'secret'" in msg and "traffic rule 0" in msg and "leaf 'cld'" in msg


def test_the_canary_model_is_walked():
    qm = QualityManager(canary_config=CanaryConfig(enabled=True, canary_model="cld", percentage=10))
    msg = refused(quality_manager=qm)
    assert "canary_model" in msg and "leaf 'cld'" in msg


def test_a_load_balancer_target_of_a_reachable_leaf_is_walked():
    balancer = LoadBalancer(
        LoadBalancerConfig(
            endpoints=[LoadBalancerEndpoint(model="loc"), LoadBalancerEndpoint(model="cld")]
        )
    )
    msg = refused(traffic_manager=TrafficManager(load_balancers={"loc": balancer}))
    assert "load balancer 'loc' target" in msg and "leaf 'cld'" in msg


def test_a_load_balancer_on_an_unreached_leaf_is_left_alone():
    balancer = LoadBalancer(
        LoadBalancerConfig(
            endpoints=[LoadBalancerEndpoint(model="cld"), LoadBalancerEndpoint(model="cld_ca")]
        )
    )
    build(traffic_manager=TrafficManager(load_balancers={"cld": balancer}))


# --- request: the entry tier ----------------------------------------------------


@pytest.fixture
def world():
    fakes = {name: FakeUpstream(name) for name in ("loc", "loc2", "cld")}
    bases = {n: f.base for n, f in fakes.items()}
    eps = endpoints(bases)
    controller = build(tier_cfg=tiers(general={**PAIR, "strong": "cld", "weak": "cld"}), eps=eps)
    controller.resilience.config.max_retries = 0
    yield controller, fakes
    for fake in fakes.values():
        fake.server.shutdown()
        fake.server.server_close()


def ask(controller, model, declared="secret"):
    return asyncio.run(
        controller.acompletion(model=model, messages=MESSAGES, declared_class=declared)
    )


def total(fakes):
    return sum(len(f.hits) for f in fakes.values())


def test_a_tier_outside_the_class_is_refused_naming_class_and_tier(world):
    controller, fakes = world
    with pytest.raises(PolicyDenied) as exc:
        ask(controller, "general")
    assert "'secret'" in str(exc.value) and "'general'" in str(exc.value)
    assert exc.value.status == 403
    assert total(fakes) == 0


def test_an_allowed_tier_is_served(world):
    controller, fakes = world
    ask(controller, "private")
    assert total(fakes) == 1


def test_the_legacy_router_form_lands_on_default_and_is_held_to_it(world):
    controller, fakes = world
    with pytest.raises(PolicyDenied, match="'default'"):
        ask(controller, "router-random-0.5")
    assert total(fakes) == 0


def test_a_request_level_router_does_not_change_the_tier_checked(world):
    controller, fakes = world
    with pytest.raises(PolicyDenied, match="'general'"):
        ask(controller, "general:router-random-0.5")
    assert total(fakes) == 0


def test_the_other_class_may_use_its_own_tiers(world):
    controller, fakes = world
    ask(controller, "general", declared="open")
    assert len(fakes["cld"].hits) == 1


def test_a_missing_class_is_refused(world):
    controller, fakes = world
    with pytest.raises(PolicyDenied):
        ask(controller, "private", declared=None)
    assert total(fakes) == 0


def test_a_flat_pair_has_no_tier_and_is_refused():
    fakes = {name: FakeUpstream(name) for name in ("loc", "loc2")}
    try:
        controller = build(
            tier_cfg={"private": {**PAIR, "strong": "loc", "weak": "loc2"}},
            pol=policy(open={"treatment": "raw", "reach": ["local"], "tiers": ["private"]}),
            eps={n: endpoint(["local", "jur:ca"], f.base) for n, f in fakes.items()},
            strong_model="loc",
            weak_model="loc2",
        )
        with pytest.raises(PolicyDenied, match="flat pair"):
            ask(controller, "router-random-0.5", declared="open")
        assert total(fakes) == 0
    finally:
        for fake in fakes.values():
            fake.server.shutdown()
            fake.server.server_close()


class Decider:
    """A decision step that answers one class, whatever was declared."""

    def __init__(self, data_class):
        self.data_class = data_class

    def bind(self, transport):
        pass

    async def classify(self, messages, **kwargs):
        return Decision(self.data_class, source="model")


def test_the_decided_class_is_the_one_checked(world):
    # Declared `open` may enter `general`; the decision raised it to
    # `secret`, which may not.
    controller, fakes = world
    controller.decision = Decider("secret")
    with pytest.raises(PolicyDenied, match=r"'secret'.*'general'"):
        ask(controller, "general", declared="open")
    assert total(fakes) == 0


def test_a_decision_that_keeps_the_class_changes_nothing(world):
    controller, fakes = world
    controller.decision = Decider("open")
    ask(controller, "general", declared="open")
    assert len(fakes["cld"].hits) == 1


def test_the_sync_entry_point_is_checked_too(world):
    controller, fakes = world
    with pytest.raises(PolicyDenied):
        controller.completion(model="general", messages=MESSAGES, declared_class="secret")
    assert total(fakes) == 0


def test_no_policy_no_check():
    registry = EndpointRegistry.from_config({"endpoints": endpoints(), "tiers": tiers()})
    controller = Controller(
        routers=["random"], endpoints=registry, cache_config=CacheConfig(enabled=False)
    )
    assert controller.policy is None
    controller._check_entry_tier("general", None, None)
