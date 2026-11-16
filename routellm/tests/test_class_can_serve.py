"""Class policy at every candidate: the one check behind tier, forced side,
sibling, pool, load balancer, canary, traffic rule, middleware and comparison.

Startup already refuses a config whose allowed tiers reach a forbidden
leaf, so these cases start under a permissive policy and swap in the
strict one afterwards: the request-time check is what stands between a
request and a leaf that startup could not know about (a retag, a hot
reload, a selector that resolved late). Every upstream is a counting
fake; a denied leaf takes zero hits and the request fails 403 rather
than moving to another class or tier.
"""

import asyncio

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.policy import PolicyDenied
from routellm.quality import CanaryConfig, QualityManager
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import build_comparison_config
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.test_class_tiers import (  # noqa: F401  (autouse env fixture)
    MESSAGES,
    _env,
    endpoint,
    policy,
)
from routellm.traffic import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancerEndpoint,
    TrafficManager,
    TrafficRule,
)
from routellm.types import ModelPair


class _PinStrong:
    """Always the strong side, so every case routes the same way."""

    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()
PAIR = {"router": "pin_strong", "threshold": 0.0}

NAMES = ("loc", "loc2", "loc_us", "cld", "cld_ca")
TAGS = {
    "loc": ["local", "jur:ca"],
    "loc2": ["local", "jur:ca"],
    "loc_us": ["local", "jur:us"],
    "cld": ["cloud", "jur:us"],
    "cld_ca": ["cloud", "jur:ca"],
}
QUALITY = {"loc": 80, "loc2": 40, "loc_us": 30, "cld": 90, "cld_ca": 50}

#: Startup-permissive twin of the strict `secret` class: same tiers, any reach.
LOOSE = policy(
    secret={"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["private", "default"]}
)
STRICT = policy(
    secret={
        "treatment": "block",
        "reach": ["local"],
        "jur": ["ca"],
        "tiers": ["private", "default"],
    }
)


@pytest.fixture
def fakes():
    made = {name: FakeUpstream(name) for name in NAMES}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def make(fakes, tiers, *, strict=True, extra_eps=None, **kwargs):
    """A controller started under the loose policy, then held to the strict one."""
    eps = {
        n: {**endpoint(TAGS[n], fakes[n].base, quality=QUALITY[n]), **(extra_eps or {}).get(n, {})}
        for n in NAMES
    }
    registry = EndpointRegistry.from_config(
        {"endpoints": eps, "tiers": tiers, "strict_registry": True}
    )
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    controller = Controller(
        routers=["random"],
        endpoints=registry,
        policy=LOOSE,
        strict_registry=True,
        **kwargs,
    )
    controller.resilience.config.max_retries = 0
    if strict:
        controller.policy = STRICT
    return controller


def tier_cfg(strong, weak):
    return {
        "private": {**PAIR, "strong": strong, "weak": weak},
        "default": {**PAIR, "strong": "private", "weak": "loc2"},
        "general": {**PAIR, "strong": "cld", "weak": "cld_ca"},
    }


def ask(controller, model="private", declared="secret", **kwargs):
    return asyncio.run(
        controller.acompletion(model=model, messages=MESSAGES, declared_class=declared, **kwargs)
    )


def hits(fakes):
    return {n: len(f.hits) for n, f in fakes.items() if f.hits}


# --- forced side -------------------------------------------------------------


def test_a_denied_side_is_never_chosen_the_allowed_one_is_forced(fakes):
    c = make(fakes, tier_cfg("cld", "loc"))
    res = ask(c)
    assert hits(fakes) == {"loc": 1}
    assert res._hidden_params["routellm_path"][0]["capability_requirement"].startswith("policy:")


def test_forced_side_onto_a_cloud_leaf_is_403_when_the_other_cannot_serve(fakes):
    # loc2 cannot take tools; the only side that can is the cloud leaf.
    caps = {"loc2": {"strict": True, "capabilities": {"tools": False}}}
    c = make(fakes, tier_cfg("cld", "loc2"), extra_eps=caps)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    with pytest.raises(PolicyDenied, match=r"policy:secret:reach") as exc:
        ask(c, tools=tools)
    assert exc.value.status == 403
    assert hits(fakes) == {}


def test_both_sides_denied_is_403_naming_reach(fakes):
    c = make(
        fakes, {**tier_cfg("loc", "loc2"), "private": {**PAIR, "strong": "cld", "weak": "cld_ca"}}
    )
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert hits(fakes) == {}


def test_a_jurisdiction_violation_names_jur(fakes):
    c = make(fakes, tier_cfg("loc_us", "loc_us"))
    with pytest.raises(PolicyDenied, match="policy:secret:jur"):
        ask(c)
    assert hits(fakes) == {}


def test_the_sync_entry_point_is_held_to_it(fakes):
    c = make(fakes, tier_cfg("cld", "cld_ca"))
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        c.completion(model="private", messages=MESSAGES, declared_class="secret")
    assert hits(fakes) == {}


def test_a_nested_tier_side_needs_every_leaf_to_pass(fakes):
    # `private.weak` is the `general` tier: its leaves are cloud.
    c = make(fakes, {**tier_cfg("loc", "general")})
    fakes["loc"].mode = "500"
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert len(fakes["loc"].hits) >= 1
    assert "cld" not in hits(fakes) and "cld_ca" not in hits(fakes)


# --- sibling fallback ----------------------------------------------------------


def test_a_cloud_sibling_is_never_called_and_the_request_fails_403(fakes):
    c = make(fakes, tier_cfg("loc", "cld"))
    fakes["loc"].mode = "500"
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert len(fakes["loc"].hits) >= 1
    assert len(fakes["cld"].hits) == 0


def test_the_sync_sibling_is_held_to_it_too(fakes):
    c = make(fakes, tier_cfg("loc", "cld"))
    fakes["loc"].mode = "500"
    with pytest.raises(PolicyDenied):
        c.completion(model="private", messages=MESSAGES, declared_class="secret")
    assert len(fakes["cld"].hits) == 0


def test_an_allowed_sibling_still_takes_over(fakes):
    c = make(fakes, tier_cfg("loc", "loc2"))
    fakes["loc"].mode = "500"
    ask(c)
    assert len(fakes["loc2"].hits) == 1


def test_a_class_that_may_reach_the_cloud_falls_back_to_it(fakes):
    c = make(fakes, tier_cfg("loc", "cld"), strict=False)
    c.policy = policy()
    fakes["loc"].mode = "500"
    ask(c, model="default", declared="open")  # default -> private -> loc, then cld
    assert len(fakes["cld"].hits) == 1


# --- candidate pool --------------------------------------------------------------


def test_a_denied_pool_member_is_skipped_the_next_allowed_one_serves(fakes):
    pool = {"select": "tag:jur:ca", "order": "quality_desc", "pool": 3}
    c = make(fakes, tier_cfg(pool, "loc2"))
    fakes["loc"].mode = "500"
    ask(c)
    assert len(fakes["cld_ca"].hits) == 0
    assert len(fakes["loc2"].hits) == 1


# --- load balancer ------------------------------------------------------------------


def test_a_balancer_never_selects_a_denied_target(fakes):
    balancer = LoadBalancer(
        LoadBalancerConfig(
            strategy="round-robin",
            endpoints=[LoadBalancerEndpoint(model="loc"), LoadBalancerEndpoint(model="cld")],
        )
    )
    c = make(
        fakes,
        tier_cfg("loc", "loc2"),
        traffic_manager=TrafficManager(load_balancers={"loc": balancer}),
    )
    for _ in range(4):
        ask(c)
    assert len(fakes["cld"].hits) == 0
    assert len(fakes["loc"].hits) == 4


def test_a_balancer_with_no_allowed_target_is_403(fakes):
    balancer = LoadBalancer(
        LoadBalancerConfig(endpoints=[LoadBalancerEndpoint(model="cld")]),
    )
    c = make(
        fakes,
        tier_cfg("loc", "loc2"),
        traffic_manager=TrafficManager(load_balancers={"loc": balancer}),
    )
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert hits(fakes) == {}


# --- names that bypass the tree --------------------------------------------------------


def test_a_traffic_rule_pair_with_a_denied_side_is_forced_to_the_other(fakes):
    rule = TrafficRule(pattern="hello", strong_model="cld", weak_model="loc")
    c = make(fakes, tier_cfg("loc", "loc2"), traffic_manager=TrafficManager(rules=[rule]))
    ask(c)
    assert hits(fakes) == {"loc": 1}


def test_a_traffic_rule_pair_with_both_sides_denied_is_403(fakes):
    rule = TrafficRule(pattern="hello", strong_model="cld", weak_model="cld_ca")
    c = make(fakes, tier_cfg("loc", "loc2"), traffic_manager=TrafficManager(rules=[rule]))
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert hits(fakes) == {}


class PairWare:
    """A middleware that answers every prompt with one pair (and no tier)."""

    def __init__(self, pair=None, tier=None):
        self.pair, self.tier = pair, tier

    def get_model_pair(self, prompt):
        return self.pair

    def get_tier(self, prompt):
        return self.tier


def test_a_middleware_pair_with_both_sides_denied_is_403(fakes):
    ware = PairWare(pair=ModelPair(strong="cld", weak="cld_ca"))
    c = make(fakes, tier_cfg("loc", "loc2"), middleware=[ware])
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        ask(c)
    assert hits(fakes) == {}


def test_a_denied_canary_is_not_used_the_picked_leaf_serves(fakes):
    qm = QualityManager(canary_config=CanaryConfig(enabled=True, canary_model="cld", weight=1.0))
    c = make(fakes, tier_cfg("loc", "loc2"), quality_manager=qm)
    res = ask(c)
    assert len(fakes["cld"].hits) == 0
    assert len(fakes["loc"].hits) + len(fakes["loc2"].hits) == 1
    assert res is not None


# --- intent tier switch ------------------------------------------------------------------


def test_an_intent_tier_outside_the_class_is_ignored_and_recorded(fakes):
    c = make(fakes, tier_cfg("loc", "loc2"), middleware=[PairWare(tier="general")])
    res = ask(c, model="default")
    path = res._hidden_params["routellm_path"]
    assert path[0]["intent_ignored"] == "general"
    assert "tier_from" not in path[0]
    assert len(fakes["cld"].hits) == 0 and len(fakes["cld_ca"].hits) == 0


def test_an_intent_tier_inside_the_class_is_honoured(fakes):
    c = make(fakes, tier_cfg("loc", "loc2"), middleware=[PairWare(tier="private")])
    res = ask(c, model="default")
    assert res._hidden_params["routellm_path"][0]["tier_from"] == "intent"


def test_a_class_that_may_enter_the_intent_tier_is_switched(fakes):
    c = make(fakes, tier_cfg("loc", "loc2"), middleware=[PairWare(tier="general")])
    c.policy = policy()
    res = ask(c, model="default", declared="open")
    assert res._hidden_params["routellm_path"][0]["tier_from"] == "intent"
    assert len(fakes["cld"].hits) + len(fakes["cld_ca"].hits) == 1


def test_an_intent_tier_walked_at_startup_is_held_to_the_class(fakes):
    # `general` is an allowed tier of the class here, so startup walks it.
    from routellm.policy import PolicyError

    pol = policy(
        secret={
            "treatment": "block",
            "reach": ["local"],
            "jur": ["ca"],
            "tiers": ["private", "general"],
        }
    )
    eps = {n: endpoint(TAGS[n], fakes[n].base) for n in NAMES}
    registry = EndpointRegistry.from_config(
        {"endpoints": eps, "tiers": tier_cfg("loc", "loc2"), "strict_registry": True}
    )
    with pytest.raises(PolicyError, match=r"class 'secret'.*tier 'general'.*leaf 'cld'"):
        Controller(
            routers=["random"],
            endpoints=registry,
            policy=pol,
            strict_registry=True,
            middleware=[PairWare(tier="general")],
            cache_config=CacheConfig(enabled=False),
        )


# --- comparison ----------------------------------------------------------------------------


def with_comparison(fakes, comparison=None):
    return make(
        fakes,
        tier_cfg("loc", "loc2"),
        comparison_config=build_comparison_config(
            {"comparison": {"enabled": True, **(comparison or {})}}
        ),
    )


def ask_shadowed(c, header, declared="secret", model="private"):
    async def go():
        res = await c.acompletion(
            model=model, messages=MESSAGES, declared_class=declared, compare=header
        )
        await c.comparison.drain()
        return res

    return asyncio.run(go())


def test_a_denied_shadow_target_is_dropped_recorded_and_never_called(fakes):
    c = with_comparison(fakes)
    res = ask_shadowed(c, "cld,loc2")
    assert len(fakes["cld"].hits) == 0
    assert len(fakes["loc2"].hits) == 1  # allowed target still shadows
    denied = res._hidden_params["routellm_path"][0]["comparison_denied"]
    assert denied == [{"endpoint": "cld", "reason": "policy:secret:reach"}]


def test_a_shadow_with_only_denied_targets_runs_nothing(fakes):
    c = with_comparison(fakes)
    res = ask_shadowed(c, "cld,cld_ca")
    assert hits(fakes) == {"loc": 1}
    assert [d["endpoint"] for d in res._hidden_params["routellm_path"][0]["comparison_denied"]] == [
        "cld",
        "cld_ca",
    ]


def test_the_shadow_control_class_that_may_reach_the_cloud_shadows_there(fakes):
    c = with_comparison(fakes)
    c.policy = policy()
    ask_shadowed(c, "cld_ca", declared="open", model="general")
    assert len(fakes["cld_ca"].hits) == 1


def compare(c, models, operator=True, **kw):
    kwargs = {"messages": MESSAGES}
    return asyncio.run(c.comparison.compare(models, kwargs, operator=operator, **kw))


def test_compare_with_a_denied_target_is_403_and_calls_nothing(fakes):
    c = with_comparison(fakes)
    with pytest.raises(PolicyDenied, match="policy:secret:reach"):
        compare(c, ["loc", "cld"], data_class="secret")
    assert hits(fakes) == {}


def test_compare_naming_a_tier_the_class_may_not_enter_is_403(fakes):
    c = with_comparison(fakes)
    with pytest.raises(PolicyDenied, match="'general'"):
        compare(c, ["loc", "loc2"], tier="general", data_class="secret")
    assert hits(fakes) == {}


def test_compare_across_allowed_targets_runs(fakes):
    c = with_comparison(fakes)
    out = compare(c, ["loc", "loc2"], data_class="secret")
    assert [r["error"] for r in out["results"]] == [None, None]


def test_a_client_compare_with_no_known_class_under_a_policy_is_refused(fakes):
    c = with_comparison(fakes, {"allow_untiered_compare": True})
    with pytest.raises(PolicyDenied):
        compare(c, ["loc", "loc2"], operator=False, data_class=None)
    assert hits(fakes) == {}
