"""Comparison routing: shadow mode, the compare runner, and their limits.

Every upstream is a thread on 127.0.0.1 that records what reaches it,
so each assertion is about calls that actually crossed the wire.
Nothing leaves the machine.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from routellm.caching import CacheConfig
from routellm.config import ConfigError
from routellm.controller import HOP_HEADER, Controller
from routellm.endpoints import EndpointRegistry
from routellm.pairing import ModelRecord
from routellm.quality import FineTuneConfig, QualityManager
from routellm.routers.routers import ROUTER_CLS
from routellm.server_config import SERVER_SECTIONS, build_comparison_config
from routellm.tests import fake_scorer
from routellm.tests.fake_upstream import FakeUpstream


class _PinStrong:
    """Always the strong side, so the primary is the same on every run."""

    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()

MESSAGES = [{"role": "user", "content": "hello there"}]
SCORER = "module:routellm.tests.fake_scorer:make"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")


@pytest.fixture
def ups():
    made = {name: FakeUpstream(name) for name in "abc"}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def build(ups, comparison, tmp_path, *, tags=None, tiers=None, cache=False, traces=True):
    tags = tags or {}
    endpoints = {
        name: {
            "model": f"openai/model-{name}",
            "api_base": ups[name].base,
            "api_key_env": "FAKE_KEY",
            **({"tags": tags[name]} if name in tags else {}),
        }
        for name in ups
    }
    config = {
        "endpoints": endpoints,
        "tiers": tiers
        or {"default": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "b"}},
    }
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(config),
        cache_config=CacheConfig(enabled=cache, db_path=str(tmp_path / "cache.db")),
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(enabled=traces, trace_dir=str(tmp_path / "traces"))
        ),
        comparison_config=build_comparison_config({"comparison": comparison}),
    )
    controller.resilience.config.max_retries = 0
    return controller


def ask(controller, model="default", compare=None, **kwargs):
    async def go():
        res = await controller.acompletion(
            model=model, messages=MESSAGES, compare=compare, **kwargs
        )
        if controller.comparison is not None:
            await controller.comparison.drain()
        return res

    return asyncio.run(go())


def traces(tmp_path):
    folder = Path(tmp_path / "traces")
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def shadows(tmp_path):
    return [
        t
        for t in traces(tmp_path)
        if (t["routellm"].get("comparison") or {}).get("role") == "shadow"
    ]


# ------------------------------------------------------------------ config


def test_comparison_is_a_known_server_section():
    assert "comparison" in SERVER_SECTIONS


def test_off_by_default_and_without_a_section():
    assert build_comparison_config({}).enabled is False
    assert build_comparison_config(None).enabled is False


def test_unknown_key_is_refused():
    with pytest.raises(ConfigError, match="comparison"):
        build_comparison_config({"comparison": {"enabled": True, "sampel": 1}})


def test_disabled_ignores_the_header(ups, tmp_path):
    controller = build(ups, {"enabled": False}, tmp_path)
    assert controller.comparison is None
    ask(controller, compare="b")
    assert len(ups["b"].hits) == 0


def test_unknown_endpoint_in_config_is_refused_at_load(ups, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        build(
            ups,
            {"enabled": True, "profiles": {"p": ["a", "nope"]}},
            tmp_path,
        )


def test_unknown_tier_in_sampling_is_refused_at_load(ups, tmp_path):
    with pytest.raises(ValueError, match="ghost"):
        build(
            ups,
            {"enabled": True, "sampling": {"ghost": {"rate": 0.1, "models": ["b"]}}},
            tmp_path,
        )


def test_pay_endpoint_cannot_be_a_shadow(ups, tmp_path):
    endpoints = {"a": {"model": "openai/x", "api_base": ups["a"].base}}
    endpoints["p"] = {"model": "openai/y", "api_base": ups["b"].base, "pay": True}
    config = {
        "endpoints": endpoints,
        "tiers": {
            "default": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "p"}
        },
    }
    with pytest.raises(ValueError, match="pay"):
        Controller(
            routers=["random"],
            endpoints=EndpointRegistry.from_config(config),
            comparison_config=build_comparison_config(
                {"comparison": {"enabled": True, "profiles": {"x": ["p"]}}}
            ),
        )


def test_local_only_tier_sampled_to_a_cloud_model_is_refused_at_load(ups, tmp_path):
    with pytest.raises(ValueError, match="local"):
        build(
            ups,
            {"enabled": True, "sampling": {"default": {"rate": 1.0, "models": ["c"]}}},
            tmp_path,
            tags={"a": ["local"], "b": ["local"]},
        )


def test_local_only_tier_sampled_to_a_local_model_is_fine(ups, tmp_path):
    build(
        ups,
        {"enabled": True, "sampling": {"default": {"rate": 1.0, "models": ["c"]}}},
        tmp_path,
        tags={"a": ["local"], "b": ["local"], "c": ["local"]},
    )


def test_experiment_naming_a_local_only_tier_with_cloud_models_is_refused(ups, tmp_path):
    with pytest.raises(ValueError, match="local"):
        build(
            ups,
            {
                "enabled": True,
                "experiments": {"x": {"models": ["c"], "rate": 1.0, "tiers": ["default"]}},
            },
            tmp_path,
            tags={"a": ["local"], "b": ["local"]},
        )


def test_remote_judge_with_a_local_only_tier_in_scope_is_refused_at_load(ups, tmp_path):
    with pytest.raises(ValueError, match="judge"):
        build(
            ups,
            {
                "enabled": True,
                "sampling": {"default": {"rate": 1.0, "models": ["c"]}},
                "judge": {"scorer": "judge:gpt-4o"},
            },
            tmp_path,
            tags={"a": ["local"], "b": ["local"], "c": ["local"]},
        )


def test_local_judge_with_a_local_only_tier_is_fine(ups, tmp_path):
    build(
        ups,
        {
            "enabled": True,
            "sampling": {"default": {"rate": 1.0, "models": ["c"]}},
            "judge": {"scorer": "judge:c"},
        },
        tmp_path,
        tags={"a": ["local"], "b": ["local"], "c": ["local"]},
    )


# ------------------------------------------------------------- shadow mode


def test_header_shadow_records_metrics_and_leaves_the_answer_alone(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    res = ask(controller, compare="b,c")

    assert res.choices[0].message.content == "hello from a"
    assert len(ups["b"].hits) == 1 and len(ups["c"].hits) == 1

    rows = shadows(tmp_path)
    assert sorted(r["routellm"]["endpoint"] for r in rows) == ["b", "c"]
    metrics = rows[0]["routellm"]["comparison"]["metrics"]
    assert metrics["latency_ms"] >= 0
    assert metrics["prompt_tokens"] == 3 and metrics["completion_tokens"] == 3
    assert metrics["tokens_per_second"] > 0
    assert metrics["ttft_ms"] is None
    assert rows[0]["frontier"]["output"].startswith("hello from")

    primary = [t for t in traces(tmp_path) if t["routellm"]["comparison"]["role"] == "primary"]
    assert len(primary) == 1
    ids = {t["routellm"]["comparison"]["id"] for t in traces(tmp_path)}
    assert len(ids) == 1


def test_trace_without_comparison_has_no_comparison_block(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    ask(controller)
    (only,) = traces(tmp_path)
    assert only["routellm"].get("comparison") is None


def test_shadow_does_not_delay_the_primary(ups, tmp_path):
    ups["b"].delay = 1.0
    controller = build(ups, {"enabled": True}, tmp_path)

    async def go():
        started = time.monotonic()
        res = await controller.acompletion(model="default", messages=MESSAGES, compare="b")
        took = time.monotonic() - started
        await controller.comparison.drain()
        return res, took

    res, took = asyncio.run(go())
    assert res.choices[0].message.content == "hello from a"
    assert took < 0.8


def test_shadow_failure_is_invisible_and_recorded(ups, tmp_path):
    ups["b"].mode = "500"
    controller = build(ups, {"enabled": True}, tmp_path)
    res = ask(controller, compare="b")
    assert res.choices[0].message.content == "hello from a"
    (row,) = shadows(tmp_path)
    assert row["routellm"]["comparison"]["error"]
    assert row["frontier"]["output"] == ""


def test_shadow_calls_carry_the_hop_header_and_no_stream(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    ask(controller, compare="b", stream=False)
    _, headers, body = ups["b"].hits[0]
    assert headers.get(HOP_HEADER.lower()) == "1"
    assert not body.get("stream")


def test_streaming_primary_with_non_stream_shadow(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)

    async def go():
        res = await controller.acompletion(
            model="default", messages=MESSAGES, compare="b", stream=True
        )
        pieces = [chunk.choices[0].delta.content async for chunk in res]
        await controller.comparison.drain()
        return pieces

    pieces = asyncio.run(go())
    assert "hello from a" in pieces
    assert len(ups["b"].hits) == 1 and not ups["b"].hits[0][2].get("stream")
    assert [r["routellm"]["endpoint"] for r in shadows(tmp_path)] == ["b"]


def test_shadow_to_the_primarys_own_endpoint_is_skipped(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    ask(controller, compare="a,b")
    assert len(ups["a"].hits) == 1
    assert len(ups["b"].hits) == 1


def test_header_resolves_profiles_and_ignores_unknown_names(ups, tmp_path):
    controller = build(ups, {"enabled": True, "profiles": {"duo": ["b", "c"]}}, tmp_path)
    ask(controller, compare="duo")
    assert len(ups["b"].hits) == 1 and len(ups["c"].hits) == 1
    res = ask(controller, compare="nope")
    assert res.choices[0].message.content == "hello from a"
    assert len(ups["b"].hits) == 1


def test_header_can_be_switched_off(ups, tmp_path):
    controller = build(ups, {"enabled": True, "allow_header": False}, tmp_path)
    ask(controller, compare="b")
    assert len(ups["b"].hits) == 0


def test_shadow_does_not_touch_the_cache(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path, cache=True)
    ask(controller, compare="b")
    params = controller._cache_params({"messages": MESSAGES})
    assert controller.cache.get("hello there", "b", params) is None
    assert controller.cache.get("hello there", "a", params) is not None


def test_cached_primary_runs_no_shadow(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path, cache=True)
    ask(controller)
    ask(controller, compare="b")
    assert len(ups["b"].hits) == 0


# --------------------------------------------------------------- privacy


LOCAL_AB = {"a": ["local"], "b": ["local"], "c": ["cloud"]}


def test_local_only_tier_never_shadows_to_a_non_local_endpoint(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path, tags=LOCAL_AB)
    res = ask(controller, compare="c")
    assert res.choices[0].message.content == "hello from a"
    assert len(ups["c"].hits) == 0
    assert shadows(tmp_path) == []


def test_local_only_tier_may_shadow_to_a_local_endpoint(ups, tmp_path):
    tags = {"a": ["local"], "b": ["local"], "c": ["local"]}
    controller = build(ups, {"enabled": True}, tmp_path, tags=tags)
    ask(controller, compare="c")
    assert len(ups["c"].hits) == 1


def test_mixed_tier_may_shadow_to_a_cloud_endpoint(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path, tags={"a": ["local"], "c": ["cloud"]})
    ask(controller, compare="c")
    assert len(ups["c"].hits) == 1


# -------------------------------------------------------------- sampling


def test_sampling_rate_one_shadows_every_request_of_that_tier(ups, tmp_path):
    cfg = {"enabled": True, "sampling": {"default": {"rate": 1.0, "models": ["b"]}}}
    controller = build(ups, cfg, tmp_path)
    ask(controller)
    ask(controller)
    assert len(ups["b"].hits) == 2


def test_sampling_rate_zero_never_shadows(ups, tmp_path):
    cfg = {"enabled": True, "sampling": {"default": {"rate": 0.0, "models": ["b"]}}}
    controller = build(ups, cfg, tmp_path)
    ask(controller)
    assert len(ups["b"].hits) == 0


def test_sampling_is_per_tier(ups, tmp_path):
    tiers = {
        "default": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "b"},
        "other": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "b"},
    }
    cfg = {"enabled": True, "sampling": {"other": {"rate": 1.0, "models": ["c"]}}}
    controller = build(ups, cfg, tmp_path, tiers=tiers)
    ask(controller, model="default")
    assert len(ups["c"].hits) == 0
    ask(controller, model="other")
    assert len(ups["c"].hits) == 1


def test_named_experiment_samples_and_tags_its_traces(ups, tmp_path):
    cfg = {"enabled": True, "experiments": {"bench": {"models": ["b", "c"], "rate": 1.0}}}
    controller = build(ups, cfg, tmp_path)
    ask(controller)
    assert len(ups["b"].hits) == 1 and len(ups["c"].hits) == 1
    assert {r["routellm"]["comparison"]["experiment"] for r in shadows(tmp_path)} == {"bench"}


def test_experiment_by_header_ignores_its_rate(ups, tmp_path):
    cfg = {"enabled": True, "experiments": {"bench": {"models": ["c"], "rate": 0.0}}}
    controller = build(ups, cfg, tmp_path)
    ask(controller, compare="bench")
    assert len(ups["c"].hits) == 1


def test_experiment_tier_filter(ups, tmp_path):
    cfg = {
        "enabled": True,
        "experiments": {"bench": {"models": ["c"], "rate": 1.0, "tiers": ["nowhere"]}},
    }
    with pytest.raises(ValueError, match="nowhere"):
        build(ups, cfg, tmp_path)


def test_experiment_without_tiers_skips_a_local_only_tier_for_cloud_models(ups, tmp_path):
    cfg = {"enabled": True, "experiments": {"bench": {"models": ["c"], "rate": 1.0}}}
    controller = build(ups, cfg, tmp_path, tags=LOCAL_AB)
    ask(controller)
    assert len(ups["c"].hits) == 0


# ---------------------------------------------------------------- limits


def _priced(controller, cost):
    controller._catalog_records = {
        name: ModelRecord(provider="p", id=name, cost_input=cost, cost_output=cost)
        for name in "abc"
    }


def test_cost_comes_from_catalog_prices(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    _priced(controller, 1_000_000.0)
    ask(controller, compare="b")
    (row,) = shadows(tmp_path)
    assert row["routellm"]["comparison"]["metrics"]["cost_usd"] == pytest.approx(6.0)


def test_unpriced_endpoint_has_null_cost(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    controller._catalog_records = {}
    ask(controller, compare="b")
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["metrics"]["cost_usd"] is None


def test_global_budget_stops_shadows_once_spent(ups, tmp_path):
    controller = build(ups, {"enabled": True, "budget_usd": 1.0}, tmp_path)
    _priced(controller, 100_000.0)  # 0.6 USD per call
    for _ in range(4):
        ask(controller, compare="b")
    assert len(ups["b"].hits) == 2
    assert len(shadows(tmp_path)) == 2


def test_experiment_budget_is_its_own(ups, tmp_path):
    cfg = {
        "enabled": True,
        "experiments": {"bench": {"models": ["b"], "rate": 1.0, "budget_usd": 1.0}},
    }
    controller = build(ups, cfg, tmp_path)
    _priced(controller, 100_000.0)
    for _ in range(4):
        ask(controller)
    assert len(ups["b"].hits) == 2
    ask(controller, compare="c")
    assert len(ups["c"].hits) == 1


def test_concurrency_cap_drops_a_shadow_that_does_not_fit(ups, tmp_path):
    ups["b"].delay = 0.5
    controller = build(ups, {"enabled": True, "max_concurrency": 1}, tmp_path)

    async def go():
        await asyncio.gather(
            controller.acompletion(model="default", messages=MESSAGES, compare="b"),
            controller.acompletion(model="default", messages=MESSAGES, compare="b"),
        )
        await controller.comparison.drain()

    asyncio.run(go())
    assert len(ups["b"].hits) == 1


def test_experiment_concurrency_cap(ups, tmp_path):
    ups["b"].delay = 0.5
    cfg = {
        "enabled": True,
        "max_concurrency": 10,
        "experiments": {"bench": {"models": ["b"], "rate": 1.0, "max_concurrency": 1}},
    }
    controller = build(ups, cfg, tmp_path)

    async def go():
        await asyncio.gather(
            controller.acompletion(model="default", messages=MESSAGES),
            controller.acompletion(model="default", messages=MESSAGES),
        )
        await controller.comparison.drain()

    asyncio.run(go())
    assert len(ups["b"].hits) == 1


def test_slot_is_released_after_the_shadow_finishes(ups, tmp_path):
    controller = build(ups, {"enabled": True, "max_concurrency": 1}, tmp_path)
    ask(controller, compare="b")
    ask(controller, compare="b")
    assert len(ups["b"].hits) == 2


def test_slow_shadow_times_out_and_records_the_error(ups, tmp_path):
    ups["b"].delay = 1.0
    controller = build(ups, {"enabled": True, "timeout_ms": 100}, tmp_path)
    ask(controller, compare="b")
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["error"] == "timeout"


def test_shadow_that_cannot_serve_the_request_is_recorded_not_called(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    controller._catalog_records = {}
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    controller.endpoints.get("b").capabilities = None
    controller.endpoints.get("b").strict = True
    ask(controller, compare="b", tools=tools)
    assert len(ups["b"].hits) == 0
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["error"].startswith("unsupported")


# ------------------------------------------------------------------ judge


def test_judge_scores_shadow_output(ups, tmp_path):
    fake_scorer.CALLS.clear()
    controller = build(ups, {"enabled": True, "judge": {"scorer": SCORER}}, tmp_path)
    ask(controller, compare="b")
    (row,) = shadows(tmp_path)
    assert row["routellm"]["comparison"]["score"] == pytest.approx(len("hello from b") / 100)
    assert row["reward"]["score"] == pytest.approx(len("hello from b") / 100)
    assert fake_scorer.CALLS[0][1]["prompt"] == "hello there"


def test_no_judge_means_no_score(ups, tmp_path):
    controller = build(ups, {"enabled": True}, tmp_path)
    ask(controller, compare="b")
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["score"] is None


def test_remote_judge_never_sees_local_only_data(ups, tmp_path, monkeypatch):
    # The header may reach a local-only tier even though sampling cannot
    # be configured to; the judge is then skipped, not called.
    built = []

    def fake_build(spec, allow_llm=False, trace_count=0):
        built.append(spec)
        return fake_scorer.make()

    monkeypatch.setattr("routellm.quality_scores.build_scorer", fake_build)
    tags = {"a": ["local"], "b": ["local"], "c": ["local"]}
    controller = build(
        ups,
        {"enabled": True, "judge": {"scorer": "judge:gpt-4o"}},
        tmp_path,
        tags=tags,
    )
    ask(controller, compare="c")
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["score"] is None
    assert built == []


def test_remote_judge_does_see_data_from_a_mixed_tier(ups, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "routellm.quality_scores.build_scorer", lambda spec, **_: fake_scorer.make()
    )
    controller = build(
        ups, {"enabled": True, "judge": {"scorer": "judge:gpt-4o"}}, tmp_path, tags={"a": ["cloud"]}
    )
    ask(controller, compare="c")
    assert shadows(tmp_path)[0]["routellm"]["comparison"]["score"] is not None


# ---------------------------------------------------------- scoring pipeline


def test_offline_scoring_skips_failed_comparison_traces(ups, tmp_path):
    from routellm.quality_scores import score_traces

    ups["b"].mode = "500"
    controller = build(ups, {"enabled": True}, tmp_path)
    ask(controller, compare="b,c")
    assert len(traces(tmp_path)) == 3

    out = tmp_path / "scores.jsonl"
    written = score_traces(str(tmp_path / "traces"), SCORER, str(out))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == 2
    assert sorted(r["endpoint"] for r in rows) == ["a", "c"]


def test_untiered_compare_opt_out_still_refuses_a_pay_endpoint(ups, tmp_path):
    endpoints = {
        "a": {"model": "openai/x", "api_base": ups["a"].base},
        "p": {"model": "openai/y", "api_base": ups["b"].base, "pay": True},
        "q": {"model": "openai/z", "api_base": ups["c"].base},
    }
    config = {
        "endpoints": endpoints,
        "tiers": {
            "default": {"router": "pin_strong", "threshold": 0.0, "strong": "a", "weak": "q"}
        },
    }
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(config),
        comparison_config=build_comparison_config(
            {"comparison": {"enabled": True, "allow_untiered_compare": True}}
        ),
    )
    with pytest.raises(ValueError, match="pay"):
        asyncio.run(controller.comparison.compare(["a", "p"], {"messages": MESSAGES}))


def test_allow_list_must_name_configured_endpoints(ups, tmp_path):
    with pytest.raises(ValueError, match="ghost"):
        build(ups, {"enabled": True, "compare_allowed_models": ["ghost"]}, tmp_path)
