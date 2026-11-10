"""Egress gate at request time: one more reason at the one choke point.

A gated endpoint whose gate refuses the request's class is not a
candidate: not as a tier side, a forced side, a sibling, a pool member,
a balancer target, a shadow or a comparison target, and not as the
endpoint the intent detector sends the prompt to. Every upstream is a
counting fake; a refused endpoint takes zero hits and the request goes
to the tier's other side or fails 403, never to another class.
"""

import asyncio

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.decision import Decision
from routellm.egress_gate import build_egress_gates
from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import PolicyDenied, parse_policy
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import build_comparison_config
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.test_class_tiers import MESSAGES, endpoint
from routellm.traffic import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancerEndpoint,
    TrafficManager,
)


class _PinStrong:
    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()
PAIR = {"router": "pin_strong", "threshold": 0.0}

NAMES = ("loc", "loc2", "gat", "gat2")
TAGS = {
    "loc": ["local", "jur:ca"],
    "loc2": ["local", "jur:ca"],
    "gat": ["cloud", "gated", "jur:us"],
    "gat2": ["cloud", "gated", "jur:ca"],
}
QUALITY = {"loc": 40, "loc2": 80, "gat": 90, "gat2": 95}

TIERS = ["gfirst", "gsib", "gpool", "gforced", "gboth", "default"]
POLICY = parse_policy(
    {
        "classes": {
            c: {"treatment": "raw", "reach": ["local", "cloud"], "tiers": TIERS}
            for c in ("general", "internal")
        }
    },
    "p.yaml",
)
GATE = {
    "name": "gated",
    "select": "tag:gated",
    "deny_classes": ["internal"],
    "models": {"allow": ["openai/vendor-*"]},
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


@pytest.fixture
def fakes():
    made = {name: FakeUpstream(name) for name in NAMES}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def tier_cfg(**over):
    pool = {"select": "tag:local|cloud", "order": "quality_asc", "pool": 4}
    base = {
        "gfirst": {**PAIR, "strong": "gat", "weak": "loc"},
        "gsib": {**PAIR, "strong": "loc", "weak": "gat"},
        "gpool": {**PAIR, "strong": pool, "weak": "loc2"},
        "gforced": {**PAIR, "strong": "gat", "weak": "loc2"},
        "gboth": {**PAIR, "strong": "gat", "weak": "gat2"},
        "default": {**PAIR, "strong": "gfirst", "weak": "loc"},
    }
    base.update(over)
    return base


def make(fakes, gates=(GATE,), *, extra_eps=None, policy=POLICY, **kwargs):
    eps = {}
    for n in NAMES:
        spec = endpoint(TAGS[n], fakes[n].base, quality=QUALITY[n])
        if n.startswith("gat"):
            spec["model"] = f"openai/vendor-{n}"
        eps[n] = {**spec, **(extra_eps or {}).get(n, {})}
    registry = EndpointRegistry.from_config(
        {
            "endpoints": eps,
            "tiers": kwargs.pop("tiers", None) or tier_cfg(),
            "strict_registry": True,
        }
    )
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    controller = Controller(
        routers=["random"],
        endpoints=registry,
        policy=policy,
        strict_registry=True,
        egress_gates=build_egress_gates({"egress_gates": list(gates)}, registry) if gates else None,
        **kwargs,
    )
    controller.resilience.config.max_retries = 0
    return controller


def ask(controller, model="gfirst", declared="internal", **kwargs):
    return asyncio.run(
        controller.acompletion(model=model, messages=MESSAGES, declared_class=declared, **kwargs)
    )


def hits(fakes):
    return {n: len(f.hits) for n, f in fakes.items() if f.hits}


# --- primary and forced side ---------------------------------------------------------------


def test_a_restricted_class_never_reaches_a_gated_primary_the_other_side_is_forced(fakes):
    c = make(fakes)
    res = ask(c)
    assert hits(fakes) == {"loc": 1}
    why = res._hidden_params["routellm_path"][0]["capability_requirement"]
    assert why == "gate:gated:class:internal"


def test_an_allowed_class_hits_the_gated_leaf_once_and_the_path_names_the_gate(fakes):
    c = make(fakes)
    res = ask(c, declared="general")
    assert hits(fakes) == {"gat": 1}
    assert res._hidden_params["routellm_path"][-1]["egress_gate"] == {"passed": ["gated"]}


def test_a_leaf_no_gate_matches_carries_no_gate_note(fakes):
    c = make(fakes)
    res = ask(c, model="gsib")
    assert "egress_gate" not in res._hidden_params["routellm_path"][-1]


def test_forced_side_onto_a_gated_leaf_is_403(fakes):
    # loc2 takes no tools, so the gated leaf is the only side that could serve.
    caps = {"loc2": {"strict": True, "capabilities": {"tools": False}}}
    c = make(fakes, extra_eps=caps)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    with pytest.raises(PolicyDenied, match=r"gate:gated:class:internal") as exc:
        ask(c, model="gforced", tools=tools)
    assert exc.value.status == 403
    assert hits(fakes) == {}


def test_both_sides_gated_is_403_with_zero_hits(fakes):
    c = make(fakes)
    with pytest.raises(PolicyDenied, match="gate:gated:class:internal"):
        ask(c, model="gboth")
    assert hits(fakes) == {}


def test_the_sync_entry_point_is_held_to_it(fakes):
    c = make(fakes)
    with pytest.raises(PolicyDenied, match="gate:gated"):
        c.completion(model="gboth", messages=MESSAGES, declared_class="internal")
    assert hits(fakes) == {}


# --- sibling fallback and pool ---------------------------------------------------------------


def test_a_gated_sibling_is_never_called_and_the_request_fails_403(fakes):
    c = make(fakes)
    fakes["loc"].mode = "500"
    with pytest.raises(PolicyDenied, match="gate:gated:class:internal"):
        ask(c, model="gsib")
    assert len(fakes["loc"].hits) >= 1
    assert len(fakes["gat"].hits) == 0


def test_the_sync_gated_sibling_is_held_to_it_too(fakes):
    c = make(fakes)
    fakes["loc"].mode = "500"
    with pytest.raises(PolicyDenied):
        c.completion(model="gsib", messages=MESSAGES, declared_class="internal")
    assert len(fakes["gat"].hits) == 0


def test_an_allowed_class_falls_back_to_the_gated_sibling(fakes):
    c = make(fakes)
    fakes["loc"].mode = "500"
    ask(c, model="gsib", declared="general")
    assert len(fakes["gat"].hits) == 1


def test_a_gated_pool_member_is_skipped_for_the_ungated_ones(fakes):
    # quality_asc over local|cloud: loc, loc2, gat, gat2; the weak side is loc2.
    c = make(fakes)
    fakes["loc"].mode = "500"
    fakes["loc2"].mode = "500"
    with pytest.raises(PolicyDenied, match="gate:gated"):
        ask(c, model="gpool")
    assert len(fakes["gat"].hits) == 0 and len(fakes["gat2"].hits) == 0


def test_a_gated_pool_member_is_walked_for_an_allowed_class(fakes):
    c = make(fakes)
    fakes["loc"].mode = "500"
    fakes["loc2"].mode = "500"
    ask(c, model="gpool", declared="general")
    assert len(fakes["gat"].hits) == 1


def test_the_denied_reason_is_in_the_path(fakes):
    c = make(fakes)
    res = ask(c, model="gpool")  # loc serves; the gated pool members are dropped
    assert hits(fakes) == {"loc": 1}
    assert res._hidden_params["routellm_path"][-1]["egress_gate_denied"] == [
        "gate:gated:class:internal"
    ]


# --- order: the class is final before the gate -------------------------------------------------


class Decider:
    """A decision step that answers one class, whatever was declared."""

    def __init__(self, data_class=None, error=None):
        self.data_class, self.error = data_class, error

    def bind(self, transport):
        pass

    async def classify(self, messages, **kwargs):
        if self.error:
            raise self.error
        return Decision(self.data_class, source="model")


def test_the_gate_reads_the_class_the_decision_concluded_not_the_declared_one(fakes):
    c = make(fakes)
    c.decision = Decider("internal")
    ask(c, declared="general")  # declared class would pass the gate
    assert hits(fakes) == {"loc": 1}


def test_a_decision_that_keeps_the_class_lets_the_gated_leaf_serve(fakes):
    c = make(fakes)
    c.decision = Decider("general")
    ask(c, declared="general")
    assert hits(fakes) == {"gat": 1}


def test_a_failing_decision_falls_to_the_restrictive_class_gated_leaves_get_zero(fakes):
    # on_error resolves to the most restrictive class, here `internal`.
    c = make(fakes)
    c.decision = Decider("internal")
    for model in ("gfirst", "gsib"):
        ask(c, model=model, declared="general")
    assert len(fakes["gat"].hits) == 0


# --- load balancer ---------------------------------------------------------------------------------


def test_a_balancer_never_selects_a_gated_target(fakes):
    balancer = LoadBalancer(
        LoadBalancerConfig(
            strategy="round-robin",
            endpoints=[LoadBalancerEndpoint(model="loc"), LoadBalancerEndpoint(model="gat")],
        )
    )
    c = make(fakes, traffic_manager=TrafficManager(load_balancers={"loc": balancer}))
    for _ in range(4):
        ask(c, model="gsib")
    assert len(fakes["gat"].hits) == 0 and len(fakes["loc"].hits) == 4


# --- middleware: the intent detector -------------------------------------------------------------------


class Detector:
    """Middleware whose detector sends the prompt to one registry endpoint."""

    def __init__(self, endpoint_name, tier=None):
        self.egress_endpoint = endpoint_name
        self.tier = tier
        self.calls = 0

    def get_model_pair(self, prompt):
        self.calls += 1

    def get_tier(self, prompt):
        self.calls += 1
        return self.tier


def test_the_detector_gets_zero_calls_for_a_restricted_class_when_its_endpoint_is_gated(fakes):
    ware = Detector("gat")
    c = make(fakes, middleware=[ware])
    ask(c, model="default")
    assert ware.calls == 0
    assert hits(fakes) == {"loc": 1}


def test_the_detector_runs_for_an_allowed_class(fakes):
    ware = Detector("gat")
    c = make(fakes, middleware=[ware])
    ask(c, model="default", declared="general")
    assert ware.calls > 0


def test_a_detector_on_an_ungated_endpoint_runs_for_every_class(fakes):
    ware = Detector("loc")
    c = make(fakes, middleware=[ware])
    ask(c, model="default")
    assert ware.calls > 0


def test_an_in_process_detector_is_not_gated(fakes):
    ware = Detector(None)
    c = make(fakes, middleware=[ware])
    ask(c, model="default")
    assert ware.calls > 0


# --- comparison ----------------------------------------------------------------------------------------------


def with_comparison(fakes, comparison=None):
    return make(
        fakes,
        comparison_config=build_comparison_config(
            {"comparison": {"enabled": True, **(comparison or {})}}
        ),
    )


def ask_shadowed(c, header, declared="internal", model="gsib"):
    async def go():
        res = await c.acompletion(
            model=model, messages=MESSAGES, declared_class=declared, compare=header
        )
        await c.comparison.drain()
        return res

    return asyncio.run(go())


def test_a_gated_shadow_target_is_dropped_recorded_and_never_called(fakes):
    c = with_comparison(fakes)
    res = ask_shadowed(c, "gat,loc2")
    assert len(fakes["gat"].hits) == 0
    assert len(fakes["loc2"].hits) == 1
    denied = res._hidden_params["routellm_path"][0]["comparison_denied"]
    assert denied == [{"endpoint": "gat", "reason": "gate:gated:class:internal"}]


def test_an_allowed_class_shadows_on_the_gated_leaf(fakes):
    c = with_comparison(fakes)
    ask_shadowed(c, "gat", declared="general")
    assert len(fakes["gat"].hits) == 1  # the shadow; the primary is `loc`


def compare(c, models, operator=True, **kw):
    return asyncio.run(
        c.comparison.compare(models, {"messages": MESSAGES}, operator=operator, **kw)
    )


def test_compare_with_a_gated_target_for_a_restricted_class_is_403(fakes):
    c = with_comparison(fakes)
    with pytest.raises(PolicyDenied, match="gate:gated:class:internal"):
        compare(c, ["loc", "gat"], operator=False, tier="default", data_class="internal")
    assert hits(fakes) == {}


def test_compare_with_a_gated_target_for_an_allowed_class_runs(fakes):
    c = with_comparison(fakes)
    out = compare(c, ["loc", "gat"], operator=False, tier="default", data_class="general")
    assert [r["error"] for r in out["results"]] == [None, None]
    assert len(fakes["gat"].hits) == 1


def test_operator_replay_with_no_class_is_still_held_to_the_gate(fakes):
    c = with_comparison(fakes)
    with pytest.raises(PolicyDenied, match="gate:gated:class:undecided"):
        compare(c, ["loc", "gat"])
    assert hits(fakes) == {}


def test_operator_replay_with_no_class_may_compare_ungated_endpoints(fakes):
    c = with_comparison(fakes)
    out = compare(c, ["loc", "loc2"])
    assert [r["error"] for r in out["results"]] == [None, None]


# --- require_policy_allow: fails closed until policy routing exists ------------------------------------------------


def test_a_gate_requiring_a_policy_allow_denies_every_request(fakes):
    gate = {**GATE, "deny_classes": [], "require_policy_allow": True}
    c = make(fakes, gates=(gate,))
    res = ask(c, declared="general")
    assert hits(fakes) == {"loc": 1}
    assert res._hidden_params["routellm_path"][0]["capability_requirement"] == "gate:gated:policy"


# --- nothing configured -----------------------------------------------------------------------------------------------


def test_no_gates_is_todays_behaviour(fakes):
    c = make(fakes, gates=())
    ask(c, declared="internal")
    assert hits(fakes) == {"gat": 1}


def test_a_gate_with_no_request_time_condition_changes_no_request(fakes):
    gate = {"name": "gated", "select": "tag:gated", "models": {"allow": ["openai/vendor-*"]}}
    c = make(fakes, gates=(gate,))
    ask(c, declared="internal")
    assert hits(fakes) == {"gat": 1}


# --- a flat pair that bypasses the tree: its forced side ------------------------------------------------------------


def test_a_traffic_rule_pair_with_a_gated_side_is_forced_to_the_other(fakes):
    from routellm.traffic import TrafficRule

    rule = TrafficRule(pattern="hello", strong_model="gat", weak_model="loc")
    c = make(fakes, traffic_manager=TrafficManager(rules=[rule]))
    ask(c)
    assert hits(fakes) == {"loc": 1}


def test_a_traffic_rule_pair_with_both_sides_gated_is_403(fakes):
    from routellm.traffic import TrafficRule

    rule = TrafficRule(pattern="hello", strong_model="gat", weak_model="gat2")
    c = make(fakes, traffic_manager=TrafficManager(rules=[rule]))
    with pytest.raises(PolicyDenied, match="gate:gated"):
        ask(c)
    assert hits(fakes) == {}
