"""A policy-routing world for in-process tests: fakes, labels, config.

Every name a test sees (header, attribute, classes, intents, tiers,
project values) comes from a `Labels`, so a second set with different
names has to pass the same suite: nothing in `routellm` may depend on one.
"""

import asyncio
from dataclasses import dataclass, field

from routellm.auth import Principal
from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.decision import Decision
from routellm.egress_gate import build_egress_gates
from routellm.endpoints import EndpointRegistry
from routellm.policy import parse_policy
from routellm.policy_routing import build_policy_routing
from routellm.routers.routers import ROUTER_CLS
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.test_class_tiers import MESSAGES, endpoint


class _PinStrong:
    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()
PAIR = {"router": "pin_strong", "threshold": 0.0}

NAMES = ("loc", "loc2", "cld", "cld2", "gat")
TAGS = {
    "loc": ["local", "jur:ca"],
    "loc2": ["local", "jur:ca"],
    "cld": ["cloud", "jur:us"],
    "cld2": ["cloud", "jur:us", "premium"],
    "gat": ["cloud", "gated", "jur:us"],
}
IMAGE = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "what is this"},
        ],
    }
]


@dataclass(frozen=True)
class Labels:
    low: str
    high: str
    key: str
    header: str
    client: str
    client_header: str
    client_value: str
    hard: str
    entry: str
    tiers: dict = field(default_factory=dict)
    projects: dict = field(default_factory=dict)


A = Labels(
    low="general",
    high="private",
    key="project",
    header="X-Route-Project",
    client="client",
    client_header="X-Route-Client",
    client_value="codex",
    hard="hard",
    entry="auto",
    tiers={
        "private": "private",
        "code": "code",
        "hard": "hard",
        "light": "light",
        "vision": "vision",
        "gated": "gtier",
    },
    projects={
        "open": "alpha",
        "nofb": "beta",
        "floor": "gamma",
        "gate": "delta",
        "deny": "epsilon",
    },
)
B = Labels(
    low="open",
    high="sealed",
    key="tenant",
    header="X-Tenant",
    client="agent",
    client_header="X-Agent",
    client_value="cli7",
    hard="deep",
    entry="smart",
    tiers={
        "private": "vault_t",
        "code": "std",
        "hard": "big",
        "light": "tiny",
        "vision": "eyes",
        "gated": "metered",
    },
    projects={
        "open": "acme.one",
        "nofb": "acme.two",
        "floor": "acme-three",
        "gate": "acme_four",
        "deny": "acme5",
    },
)


def tier_config(L: Labels) -> dict:
    t = L.tiers
    return {
        t["private"]: {**PAIR, "strong": "loc", "weak": "loc2"},
        t["code"]: {**PAIR, "strong": "cld", "weak": "loc"},
        t["hard"]: {**PAIR, "strong": "cld2", "weak": "cld"},
        t["light"]: {**PAIR, "strong": "loc2", "weak": "loc"},
        t["vision"]: {**PAIR, "strong": "cld", "weak": "loc"},
        t["gated"]: {**PAIR, "strong": "gat", "weak": "loc2"},
        L.entry: {**PAIR, "strong": "cld", "weak": "loc"},
        "default": {**PAIR, "strong": "cld", "weak": "loc"},
    }


def class_policy(L: Labels, only=None):
    """The class policy; `only` limits the principals' projects to those names."""
    t = L.tiers
    keep = [p for p in L.projects.values() if only is None or p in only]
    open_tiers = [t["code"], t["hard"], t["light"], t["vision"], t["gated"]]
    return parse_policy(
        {
            "classes": {
                L.low: {"treatment": "raw", "reach": ["local", "cloud"], "tiers": open_tiers},
                L.high: {
                    "treatment": "block",
                    "reach": ["local"],
                    "jur": ["ca"],
                    "tiers": [t["private"]],
                },
            },
            "principals": {
                "tester": {"classes": [L.low, L.high], "projects": keep},
                "limited": {
                    "classes": [L.low, L.high],
                    "projects": [p for p in [L.projects["open"]] if p in keep],
                },
                "plain": {"classes": [L.low, L.high]},
            },
        },
        "classes.yaml",
    )


def routing_section(L: Labels, **over) -> dict:
    t, p = L.tiers, L.projects
    section = {
        "attributes": {
            L.key: {"header": L.header},
            L.client: {"header": L.client_header},
        },
        "key": L.key,
        "entry": [L.entry],
        "rules": [
            {"when": {"class": L.high}, "tier": t["private"]},
            {"when": {"intent": L.hard}, "tier": t["hard"]},
            {"when": {L.client: L.client_value}, "tier": t["light"]},
            {"when": {"modality": "image"}, "tier": t["vision"]},
        ],
        "else": t["code"],
        "policies": {
            "default": {
                "data_class": L.low,
                "tiers": [t["code"], t["light"], t["private"], t["vision"], t["gated"]],
                "fallback": t["code"],
            },
            p["open"]: {
                "data_class": L.low,
                "tiers": [t["code"], t["hard"], t["light"], t["private"], t["vision"]],
            },
            p["nofb"]: {"tiers": [t["code"], t["private"]]},
            p["floor"]: {"data_class": L.high, "tiers": [t["private"]], "fallback": t["private"]},
            p["gate"]: {
                "data_class": L.low,
                "tiers": [t["code"], t["gated"], t["private"]],
                "allow_tags": ["gated"],
            },
            p["deny"]: {
                "data_class": L.low,
                "tiers": [t["code"], t["hard"], t["private"]],
                "deny_tags": ["premium"],
            },
        },
    }
    return {**section, **over}


GATE = {
    "name": "metered",
    "select": "tag:gated",
    "require_policy_allow": True,
}


class Decider:
    """A decision step with a fixed answer."""

    def __init__(self, data_class, intent=None):
        self.data_class, self.intent = data_class, intent

    async def classify(self, messages, **_):
        return Decision(self.data_class, intent=self.intent, source="model")


class World:
    def __init__(self, L: Labels, fakes, controller, router):
        self.L, self.fakes, self.controller, self.router = L, fakes, controller, router

    def attrs(self, project=None, client=None, **headers):
        sent = dict(headers)
        if project is not None:
            sent[self.L.header] = project
        if client is not None:
            sent[self.L.client_header] = client
        return self.router.attributes(sent)

    def ask(
        self,
        *,
        project=None,
        client=None,
        declared=None,
        decision=None,
        model=None,
        messages=MESSAGES,
        principal="tester",
        **kw,
    ):
        self.controller.decision = decision
        return asyncio.run(
            self.controller.acompletion(
                model=model or self.L.entry,
                messages=messages,
                declared_class=declared or self.L.low,
                attributes=self.attrs(project, client),
                principal=Principal(principal) if principal else None,
                **kw,
            )
        )

    def path(self, res):
        return res._hidden_params["routellm_path"]

    def record(self, res):
        return self.path(res)[0]["policy_routing"]

    def hits(self):
        return {n: len(f.hits) for n, f in self.fakes.items() if f.hits}


def make_world(L: Labels, *, section=None, gates=(), policy=None, **kwargs) -> World:
    fakes = {name: FakeUpstream(name) for name in NAMES}
    eps = {n: endpoint(TAGS[n], fakes[n].base, quality=50 + i) for i, n in enumerate(NAMES)}
    registry = EndpointRegistry.from_config(
        {"endpoints": eps, "tiers": tier_config(L), "strict_registry": True}
    )
    section = section or routing_section(L)
    pol = policy or class_policy(L, only=set(section["policies"]))
    router = build_policy_routing({"policy_routing": section}, pol)
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    controller = Controller(
        routers=["random"],
        endpoints=registry,
        policy=pol,
        strict_registry=True,
        policy_routing=router,
        egress_gates=(
            build_egress_gates({"egress_gates": list(gates)}, registry) if gates else None
        ),
        **kwargs,
    )
    controller.resilience.config.max_retries = 0
    return World(L, fakes, controller, router)


def close(world: World) -> None:
    for fake in world.fakes.values():
        fake.server.shutdown()
        fake.server.server_close()
