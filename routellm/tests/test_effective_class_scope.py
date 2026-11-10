"""The cache, semantic gate and traces key off the floor-raised class.

A request declared with a low class, sent under a project whose routing
policy floors it to a stricter class, is held to the stricter class
everywhere it reads one. Fakes only; an upstream hit is a call that
crossed the wire.
"""

import asyncio
import json
import os

import pytest

from routellm.caching import CacheConfig
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import PolicyDenied, parse_policy
from routellm.quality import FineTuneConfig, QualityManager
from routellm.server_config import build_comparison_config
from routellm.tests.policy_routing_world import (
    A,
    B,
    close,
    make_world,
    routing_section,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))


@pytest.fixture(params=[A, B], ids=["names-a", "names-b"])
def L(request):
    return request.param


def policy_with(L, high_treatment="raw", high_cache="exact", low_cache="exact"):
    t = L.tiers
    open_tiers = [t["code"], t["hard"], t["light"], t["vision"], t["gated"], t["private"]]
    high = {"treatment": high_treatment, "reach": ["local"], "jur": ["ca"], "tiers": [t["private"]]}
    if high_cache:
        high["cache"] = high_cache
    return parse_policy(
        {
            "classes": {
                L.low: {
                    "treatment": "raw",
                    "reach": ["local", "cloud"],
                    "tiers": open_tiers,
                    "cache": low_cache,
                },
                L.high: high,
            },
            "principals": {
                "tester": {"classes": [L.low, L.high], "projects": list(L.projects.values())}
            },
        },
        "classes.yaml",
    )


def world(L, tmp_path, **kw):
    cache = CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"))
    return make_world(L, policy=policy_with(L, **kw.pop("policy", {})), cache_config=cache, **kw)


def ask(w, project):
    """The same prompt, declared low, to the private tier, under `project`."""
    return w.ask(model=w.L.tiers["private"], project=project, declared=w.L.low)


def test_a_floor_raised_request_is_cached_under_the_floor_class_not_the_declared_one(L, tmp_path):
    w = world(L, tmp_path)
    try:
        ask(w, L.projects["open"])
        assert w.hits() == {"loc": 1}
        # Same prompt and model, floor project: a hit on the open project's
        # entry would mean the declared class's namespace served it.
        ask(w, L.projects["floor"])
        assert w.hits() == {"loc": 2}
        ask(w, L.projects["floor"])
        ask(w, L.projects["open"])
        assert w.hits() == {"loc": 2}
    finally:
        close(w)


def test_a_floor_class_with_cache_none_stores_and_serves_nothing(L, tmp_path):
    w = world(L, tmp_path, policy={"high_cache": "none"})
    try:
        ask(w, L.projects["floor"])
        ask(w, L.projects["floor"])
        assert w.hits() == {"loc": 2}
    finally:
        close(w)


def test_a_floor_class_that_is_not_raw_is_never_cached(L, tmp_path):
    w = world(L, tmp_path, policy={"high_treatment": "block", "high_cache": "exact"})
    try:
        ask(w, L.projects["floor"])
        ask(w, L.projects["floor"])
        assert w.hits() == {"loc": 2}
    finally:
        close(w)


def test_the_semantic_gate_reads_the_floor_class(L, tmp_path):
    w = world(L, tmp_path, policy={"high_treatment": "block"})
    try:
        seen = []
        orig = w.controller._semantic_cache_allowed
        w.controller._semantic_cache_allowed = lambda cls, *a: (seen.append(cls), orig(cls, *a))[1]
        w.controller._cache_scope = lambda cls, _o=w.controller._cache_scope: (
            _o(cls)[0],
            "both",
        )
        ask(w, L.projects["floor"])
        assert seen == [L.high]
    finally:
        close(w)


def test_traces_keep_digests_only_for_a_non_raw_floor_class(L, tmp_path):
    traces = tmp_path / "traces"
    qm = QualityManager(
        fine_tune_config=FineTuneConfig(enabled=True, trace_dir=str(traces), capture_content=True)
    )
    w = world(L, tmp_path, policy={"high_treatment": "block"}, quality_manager=qm)
    try:
        ask(w, L.projects["floor"])
        ask(w, L.projects["open"])
        stored = [json.loads((traces / n).read_text()) for n in sorted(os.listdir(traces))]
        by_capture = sorted(t["metadata"]["content_captured"] for t in stored)
        assert by_capture == [False, True]
    finally:
        close(w)


# --- compare: the route policy always rides along ---------------------------------------


def deny_section(L, tag):
    section = routing_section(L)
    section["policies"][L.projects["open"]]["deny_tags"] = [tag]
    return section


def test_an_operator_replay_is_held_to_the_selected_policys_denied_tags(L, tmp_path):
    section = deny_section(L, "premium")
    w = make_world(
        L,
        section=section,
        comparison_config=build_comparison_config({"comparison": {"enabled": True}}),
    )
    try:
        with pytest.raises(PolicyDenied, match="tag:premium"):
            asyncio.run(
                w.controller.comparison.compare(
                    ["cld2", "loc"],
                    {"messages": [{"role": "user", "content": "hi"}]},
                    operator=True,
                    route_policy=L.projects["open"],
                )
            )
        assert w.hits() == {}
    finally:
        close(w)


def test_an_operator_replay_without_a_route_policy_is_not_tag_gated(L, tmp_path):
    w = make_world(
        L,
        section=deny_section(L, "premium"),
        comparison_config=build_comparison_config({"comparison": {"enabled": True}}),
    )
    try:
        asyncio.run(
            w.controller.comparison.compare(
                ["cld2", "loc2"], {"messages": [{"role": "user", "content": "hi"}]}, operator=True
            )
        )
        assert w.hits() == {"cld2": 1, "loc2": 1}
    finally:
        close(w)
