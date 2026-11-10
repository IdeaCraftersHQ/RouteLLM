"""Per-class egress for the calls that read a prompt before any leaf does.

Routers that embed, the semantic cache, the intent detector, the
comparison judge and the decision classifier each send request content
somewhere that is not a routed leaf. Each is held to the request's
effective class: a call whose endpoint fails the class is not made.
Every remote is a counting loopback fake; a refused step takes zero hits
and the request is still answered by an allowed leaf.
"""

import asyncio
import sqlite3

import pytest

from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.decision import build_decision
from routellm.endpoints import EndpointRegistry
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector
from routellm.policy import PolicyError, parse_policy
from routellm.routers.base import Router
from routellm.routers.embeddings import embed
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import build_comparison_config
from routellm.tests.decision_fakes import CLASSES
from routellm.tests.fake_embedding_server import FakeEmbeddingServer
from routellm.tests.fake_upstream import FakeUpstream

MESSAGES = [{"role": "user", "content": "hello there"}]


class _EmbeddingRouter(Router):
    """Picks strong, but only after embedding the prompt."""

    requires_embedding = True

    def calculate_strong_win_rate(self, prompt):
        return 1.0 if embed(prompt) else 0.0


ROUTER_CLS["emb"] = _EmbeddingRouter

TIERS = ["default", "emb_tier"]
POLICY = parse_policy(
    {
        "classes": {
            "open": {
                "treatment": "raw",
                "reach": ["local", "cloud"],
                "tiers": [*TIERS, "mixed"],
            },
            "site": {
                "treatment": "raw",
                "reach": ["local", "vpn"],
                "tiers": [*TIERS, "site_tier"],
            },
            "lan": {"treatment": "raw", "reach": ["local"], "tiers": TIERS},
            "redacted": {"treatment": "redact", "reach": ["local"], "tiers": TIERS},
            "sealed": {"treatment": "block", "reach": ["local"], "tiers": TIERS},
        }
    },
    "p.yaml",
)
RESTRICTED = ("redacted", "sealed")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")


@pytest.fixture
def world():
    chat = {n: FakeUpstream(n) for n in ("loc", "loc2", "cld")}
    emb = FakeEmbeddingServer()
    yield chat, emb
    for fake in chat.values():
        fake.server.shutdown()
        fake.server.server_close()
    emb.close()


def build(world, *, embed_reach="cloud", tmp_path=None, semantic=False, **kw):
    """A controller whose embedding endpoint carries `embed_reach` and every remote is a fake."""
    chat, emb = world

    def ep(base, reach, jur="ca", **extra):
        return {
            "model": "openai/m",
            "api_base": base,
            "api_key_env": "FAKE_KEY",
            "tags": [reach, f"jur:{jur}"],
            **extra,
        }

    registry = EndpointRegistry.from_config(
        {
            "strict_registry": True,
            "endpoints": {
                "loc": ep(chat["loc"].base, "local", quality=40),
                "loc2": ep(chat["loc2"].base, "local", quality=80),
                "cld": ep(chat["cld"].base, "cloud", jur="us"),
                "vpn": ep(chat["loc2"].base, "vpn"),
                "embedding": ep(emb.base, embed_reach, model="openai/embed-model"),
            },
            "tiers": {
                "default": {"router": "pin", "threshold": 0.0, "strong": "loc", "weak": "loc2"},
                "emb_tier": {"router": "emb", "threshold": 0.5, "strong": "loc", "weak": "loc2"},
                "mixed": {"router": "pin", "threshold": 0.0, "strong": "loc", "weak": "cld"},
                "site_tier": {"router": "pin", "threshold": 0.0, "strong": "loc", "weak": "vpn"},
            },
        }
    )
    if semantic:
        kw["cache_config"] = CacheConfig(
            enabled=True, db_path=str(tmp_path / "cache.db"), semantic_enabled=True
        )
    controller = Controller(
        routers=["random", "emb"],
        endpoints=registry,
        policy=POLICY,
        strict_registry=True,
        **kw,
    )
    controller.resilience.config.max_retries = 0
    return controller


class _Pin:
    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin"] = lambda **_: _Pin()


def ask(c, model="default", declared="sealed", **kwargs):
    return asyncio.run(
        c.acompletion(model=model, messages=MESSAGES, declared_class=declared, **kwargs)
    )


def chat_hits(world):
    return {n: len(f.hits) for n, f in world[0].items() if f.hits}


# --- routers that embed ------------------------------------------------------------------------


@pytest.mark.parametrize("declared", RESTRICTED)
def test_a_restricted_class_never_embeds_through_a_cloud_endpoint(world, declared):
    res = ask(build(world), model="emb_tier", declared=declared)

    assert world[1].hits == []
    assert chat_hits(world) == {"loc2": 1}
    entry = res._hidden_params["routellm_path"][0]
    assert entry["gate"] == "gate:router"
    assert entry["win_rate"] is None
    assert entry["picked"] == "loc2"


def test_a_class_whose_reach_the_endpoint_fails_skips_the_router_too(world):
    ask(build(world), model="emb_tier", declared="lan")

    assert world[1].hits == []
    assert chat_hits(world) == {"loc2": 1}


def test_the_router_runs_when_the_embedding_endpoint_passes_the_class(world):
    res = ask(build(world, embed_reach="local"), model="emb_tier", declared="sealed")

    assert len(world[1].hits) == 1
    assert chat_hits(world) == {"loc": 1}
    assert "gate" not in res._hidden_params["routellm_path"][0]


def test_the_open_class_is_the_control_it_embeds_through_the_cloud_endpoint(world):
    ask(build(world), model="emb_tier", declared="open")

    assert len(world[1].hits) == 1
    assert chat_hits(world) == {"loc": 1}


def test_the_sync_entry_point_is_held_to_it(world):
    build(world).completion(model="emb_tier", messages=MESSAGES, declared_class="sealed")

    assert world[1].hits == []


# --- semantic cache --------------------------------------------------------------------------------


@pytest.mark.parametrize("declared", RESTRICTED)
def test_a_restricted_class_never_looks_up_or_stores_semantically(world, tmp_path, declared):
    c = build(world, embed_reach="local", tmp_path=tmp_path, semantic=True)

    ask(c, declared=declared)
    ask(c, declared=declared)

    assert world[1].hits == []


def test_a_raw_class_whose_reach_the_embedding_endpoint_fails_skips_the_cache(world, tmp_path):
    ask(build(world, tmp_path=tmp_path, semantic=True), declared="lan")

    assert world[1].hits == []


def test_the_semantic_cache_runs_for_a_class_it_may_serve(world, tmp_path):
    ask(build(world, tmp_path=tmp_path, semantic=True), declared="open")

    assert len(world[1].hits) >= 1


def test_a_restricted_response_is_stored_without_an_embedding(world, tmp_path):
    c = build(world, embed_reach="local", tmp_path=tmp_path, semantic=True)
    ask(c, declared="sealed")

    rows = (
        sqlite3.connect(c.cache.config.db_path)
        .execute("SELECT embedding FROM completion_cache")
        .fetchall()
    )
    assert rows == [(None,)]


# --- intent detector -------------------------------------------------------------------------------


class Detector:
    """Middleware whose detector sends the prompt to one registry endpoint."""

    def __init__(self, endpoint_name):
        self.egress_endpoint = endpoint_name
        self.calls = 0

    def get_model_pair(self, prompt):
        self.calls += 1

    def get_tier(self, prompt):
        self.calls += 1


@pytest.mark.parametrize("declared", ("lan", *RESTRICTED))
def test_the_detector_gets_zero_calls_when_its_endpoint_fails_the_class(world, declared):
    ware = Detector("cld")
    ask(build(world, middleware=[ware]), declared=declared)

    assert ware.calls == 0


def test_the_detector_runs_for_a_class_its_endpoint_passes(world):
    ware = Detector("cld")
    ask(build(world, middleware=[ware]), declared="open")

    assert ware.calls > 0


@pytest.mark.parametrize("declared", RESTRICTED)
def test_a_configured_intent_detector_sees_no_restricted_prompt(world, declared):
    c = build(world)
    c.middleware = [_selector(c)]

    res = ask(c, declared=declared)

    assert world[0]["cld"].hits == []
    assert res.choices[0].message.content


def test_the_intent_detector_is_called_for_the_open_class(world):
    c = build(world)
    c.middleware = [_selector(c)]

    ask(c, declared="open")

    assert len(world[0]["cld"].hits) >= 1


def _selector(c):
    return IntentModelSelector(
        intent_mappings=[IntentModelMapping(intent="code")],
        intent_endpoint=c.endpoints.get("cld"),
        intent_tiers={"code": "default"},
    )


# --- comparison judge ------------------------------------------------------------------------------


def judge_calls(world, monkeypatch, declared, model):
    built = []

    def fake_build(spec, allow_llm=False, trace_count=0):
        built.append(spec)
        from routellm.tests import fake_scorer

        return fake_scorer.make()

    monkeypatch.setattr("routellm.quality_scores.build_scorer", fake_build)
    c = build(
        world,
        comparison_config=build_comparison_config(
            {"comparison": {"enabled": True, "judge": {"scorer": "judge:cld"}}}
        ),
    )

    async def go():
        await c.acompletion(model=model, messages=MESSAGES, declared_class=declared, compare="loc2")
        await c.comparison.drain()

    asyncio.run(go())
    return built


def test_the_judge_is_not_built_or_called_for_a_class_whose_reach_it_fails(world, monkeypatch):
    # The tier holds a vpn leaf, so it is not local-only: only the class can refuse the judge.
    assert judge_calls(world, monkeypatch, "site", "site_tier") == []


def test_the_judge_runs_for_a_class_it_passes(world, monkeypatch):
    assert judge_calls(world, monkeypatch, "open", "mixed") == ["judge:cld"]


# --- decision classifier ---------------------------------------------------------------------------


def decision_for(world, endpoint):
    chat, _ = world
    registry = build(world).endpoints
    return build_decision({"decision": {"endpoint": endpoint, "data_classes": CLASSES}}, registry)


def test_a_cloud_decision_endpoint_is_refused_under_a_class_that_may_not_reach_it(world):
    # The classifier reads every request before its class is known.
    with pytest.raises(PolicyError, match="decision endpoint 'cld'.*sealed"):
        build(world, decision=decision_for(world, "cld"))


def test_a_decision_endpoint_every_class_allows_starts(world):
    build(world, decision=decision_for(world, "loc"))
