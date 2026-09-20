"""Tests for tiers: nested strong/weak pairs resolved recursively.

Covers `Tier` validation and the registry's `tiers:` load, the model-name
grammar the controller parses, the recursive walk with inherited router
and threshold, the decision path the walk returns, the composition with
traffic rules and canary, and the server field that carries the path.

Routers are stubbed with fixed win rates and injected into the dict the
controller imports: the root conftest replaces `routellm.routers.routers`
with a stub, so `routellm.controller.ROUTER_CLS` and the registry's own
dict are different objects under pytest.
"""
import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import routellm.controller
from routellm.caching import CacheConfig
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry, Tier
from routellm.quality import CanaryConfig, QualityManager
from routellm.resilience import ResilienceConfig
from routellm.routers.base import Router
from routellm.traffic import TrafficManager, TrafficRule
from routellm.types import ModelPair

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "endpoints": {
        "cloud_strong": {"model": "gpt-4o"},
        "frontier_local": {"model": "ollama_chat/qwen3:32b"},
        "local_fast": {"model": "ollama_chat/qwen3:8b"},
    },
    "tiers": {
        "premium": {
            "router": "hi",
            "threshold": 0.33,
            "strong": "cloud_strong",
            "weak": "frontier_local",
        },
        "default": {
            "router": "hi",
            "threshold": 0.12,
            "strong": "premium",
            "weak": "local_fast",
        },
    },
}


class _StubRouter(Router):
    """Router returning a fixed win rate and counting its calls.

    Only the scorer is implemented, so the base class supplies both
    `route` and `route_with_score` and the stub reports a real score.
    """

    win_rate = 0.5
    calls: list[str] = []

    def calculate_strong_win_rate(self, prompt):
        type(self).calls.append(prompt)
        return type(self).win_rate


def _make_router(win_rate):
    """Build a fresh stub router class with its own call log."""

    class Stub(_StubRouter):
        pass

    Stub.win_rate = win_rate
    Stub.calls = []
    return Stub


@pytest.fixture
def hi_router(monkeypatch):
    """Stub router scoring 0.9, registered under the name 'hi'."""
    cls = _make_router(0.9)
    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "hi", lambda **kw: cls())
    return cls


@pytest.fixture
def lo_router(monkeypatch):
    """Stub router scoring 0.05, registered under the name 'lo'."""
    cls = _make_router(0.05)
    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "lo", lambda **kw: cls())
    return cls


@pytest.fixture
def registry():
    """Registry built from the two-level sample config."""
    return EndpointRegistry.from_config(CONFIG)


def _controller(registry, tmp_path, **kwargs):
    """Controller on the sample registry with caching off."""
    kwargs.setdefault("routers", ["hi"])
    kwargs.setdefault("strong_model", None)
    kwargs.setdefault("weak_model", None)
    kwargs.setdefault("default_router", "hi")
    kwargs.setdefault("default_threshold", 0.5)
    return Controller(
        config={},
        endpoints=registry,
        cache_config=CacheConfig(enabled=False),
        **kwargs,
    )


@pytest.fixture
def mock_completion(monkeypatch):
    """Patch litellm completion; the mock records its kwargs."""
    res = MagicMock()
    res._hidden_params = {}
    res.model_dump.return_value = {"choices": []}
    mock = MagicMock(return_value=res)
    monkeypatch.setattr(routellm.controller, "completion", mock)
    return mock


# ---------------------------------------------------------------------------
# Tier model and registry loading
# ---------------------------------------------------------------------------


def test_from_config_builds_tiers(registry):
    assert registry.tier_names() == ["default", "premium"]

    premium = registry.get_tier("premium")
    assert premium.router == "hi"
    assert premium.threshold == 0.33
    assert premium.strong == "cloud_strong"
    assert premium.weak == "frontier_local"


def test_tier_router_and_threshold_default_to_none():
    tier = Tier(name="bare", strong="a", weak="b")

    assert tier.router is None
    assert tier.threshold is None


def test_from_config_without_tiers_key_is_empty():
    assert EndpointRegistry.from_config({"endpoints": {}}).tier_names() == []


def test_tier_name_charset_rejected():
    with pytest.raises(ValidationError):
        Tier(name="fast-tier", strong="a", weak="b")


def test_tier_threshold_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Tier(name="fast", strong="a", weak="b", threshold=1.5)


def test_unknown_reference_rejected():
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {"t": {"strong": "a", "weak": "nowhere"}},
    }

    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(config)

    assert "nowhere" in str(excinfo.value)


def test_cycle_rejected_naming_the_cycle():
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {
            "one": {"strong": "two", "weak": "a"},
            "two": {"strong": "one", "weak": "a"},
        },
    }

    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(config)

    message = str(excinfo.value)
    assert "cycle" in message.lower()
    assert "one" in message
    assert "two" in message


def test_self_reference_rejected():
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {"loop": {"strong": "loop", "weak": "a"}},
    }

    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(config)

    assert "cycle" in str(excinfo.value).lower()


def test_depth_four_accepted():
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {
            "t1": {"strong": "t2", "weak": "a"},
            "t2": {"strong": "t3", "weak": "a"},
            "t3": {"strong": "t4", "weak": "a"},
            "t4": {"strong": "a", "weak": "a"},
        },
    }

    assert len(EndpointRegistry.from_config(config).tier_names()) == 4


def test_depth_five_rejected():
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {
            "t1": {"strong": "t2", "weak": "a"},
            "t2": {"strong": "t3", "weak": "a"},
            "t3": {"strong": "t4", "weak": "a"},
            "t4": {"strong": "t5", "weak": "a"},
            "t5": {"strong": "a", "weak": "a"},
        },
    }

    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(config)

    assert "depth" in str(excinfo.value).lower()


def test_tier_endpoint_name_collision_rejected():
    config = {
        "endpoints": {"shared": {"model": "m"}},
        "tiers": {"shared": {"strong": "shared", "weak": "shared"}},
    }

    with pytest.raises(ValueError) as excinfo:
        EndpointRegistry.from_config(config)

    assert "shared" in str(excinfo.value)


def test_tier_router_name_collision_rejected_at_construction(
    registry, tmp_path, hi_router
):
    config = {
        "endpoints": {"a": {"model": "m"}},
        "tiers": {"hi": {"strong": "a", "weak": "a"}},
    }
    colliding = EndpointRegistry.from_config(config)

    with pytest.raises(ValueError) as excinfo:
        _controller(colliding, tmp_path, strong_model="a", weak_model="a")

    assert "hi" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Model-name grammar
# ---------------------------------------------------------------------------


def test_parse_tier_with_request_router_and_threshold(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    assert controller._parse_model_name("premium:router-hi-0.7") == (
        "premium",
        "hi",
        0.7,
    )


def test_parse_legacy_form_uses_default_tier_when_one_exists(
    registry, tmp_path, hi_router
):
    controller = _controller(registry, tmp_path)

    assert controller._parse_model_name("router-hi-0.5") == ("default", "hi", 0.5)


def test_parse_legacy_form_is_flat_without_a_default_tier(tmp_path, hi_router):
    flat = EndpointRegistry.from_config({"endpoints": {"a": {"model": "m"}}})
    controller = _controller(
        flat, tmp_path, strong_model="a", weak_model="a", default_router=None
    )

    assert controller._parse_model_name("router-hi-0.5") == (None, "hi", 0.5)


def test_parse_router_prefixed_tier_name(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    assert controller._parse_model_name("router-premium") == ("premium", None, None)


def test_parse_bare_tier_name(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    assert controller._parse_model_name("premium") == ("premium", None, None)


def test_parse_unknown_tier_lists_tiers(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    with pytest.raises(RoutingError) as excinfo:
        controller._parse_model_name("nope")

    message = str(excinfo.value)
    assert "premium" in message
    assert "default" in message


def test_parse_hyphenated_tier_name_rejected(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    with pytest.raises(RoutingError):
        controller._parse_model_name("pre-mium")


def test_parse_qualified_remainder_must_be_router_form(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    with pytest.raises(RoutingError):
        controller._parse_model_name("premium:hi-0.7")


# ---------------------------------------------------------------------------
# Recursive resolution and the decision path
# ---------------------------------------------------------------------------


def test_high_score_cascades_to_the_deepest_strong_leaf(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(registry, tmp_path)

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    path = res._hidden_params["routellm_path"]
    assert [entry["tier"] for entry in path] == ["default", "premium"]
    assert [entry["picked"] for entry in path] == ["premium", "cloud_strong"]
    assert mock_completion.call_args[1]["model"] == "gpt-4o"


def test_low_score_stops_at_the_root_weak_leaf(
    registry, tmp_path, lo_router, mock_completion
):
    low_config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "premium": {**CONFIG["tiers"]["premium"], "router": "lo"},
            "default": {**CONFIG["tiers"]["default"], "router": "lo"},
        },
    }
    controller = _controller(
        EndpointRegistry.from_config(low_config),
        tmp_path,
        routers=["lo"],
        default_router="lo",
    )

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "easy"}]
    )

    path = res._hidden_params["routellm_path"]
    assert [entry["tier"] for entry in path] == ["default"]
    assert path[0]["picked"] == "local_fast"
    assert mock_completion.call_args[1]["model"] == "ollama_chat/qwen3:8b"


def test_path_records_one_entry_per_level_with_win_rate(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(registry, tmp_path)

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    path = res._hidden_params["routellm_path"]
    assert len(path) == 2
    assert [entry["win_rate"] for entry in path] == [0.9, 0.9]
    assert [entry["threshold"] for entry in path] == [0.12, 0.33]
    assert [entry["router"] for entry in path] == ["hi", "hi"]


def test_router_runs_on_the_original_prompt_once_per_level(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(registry, tmp_path)

    controller.completion(
        model="default", messages=[{"role": "user", "content": "the prompt"}]
    )

    assert hi_router.calls == ["the prompt", "the prompt"]


def test_path_is_logged_at_info(registry, tmp_path, hi_router, mock_completion, caplog):
    caplog.set_level(logging.INFO, logger="routellm.controller")
    controller = _controller(registry, tmp_path)

    controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    assert any("cloud_strong" in record.message for record in caplog.records)


def test_model_counts_key_is_the_request_string_and_final_endpoint(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(registry, tmp_path)

    controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    assert controller.model_counts["default"]["cloud_strong"] == 1


# ---------------------------------------------------------------------------
# Inheritance of router and threshold
# ---------------------------------------------------------------------------


def _inheritance_registry(child_spec):
    """Registry whose root delegates to a child carrying `child_spec`."""
    return EndpointRegistry.from_config(
        {
            "endpoints": {
                "strong_leaf": {"model": "m_strong"},
                "weak_leaf": {"model": "m_weak"},
            },
            "tiers": {
                "root": {
                    "router": "hi",
                    "threshold": 0.1,
                    "strong": "child",
                    "weak": "weak_leaf",
                },
                "child": {
                    "strong": "strong_leaf",
                    "weak": "weak_leaf",
                    **child_spec,
                },
            },
        }
    )


def test_child_without_values_inherits_the_parent_level(
    tmp_path, hi_router, mock_completion
):
    controller = _controller(
        _inheritance_registry({}), tmp_path, strong_model="strong_leaf",
        weak_model="weak_leaf"
    )

    res = controller.completion(
        model="root", messages=[{"role": "user", "content": "hard"}]
    )

    child = res._hidden_params["routellm_path"][1]
    assert child["router"] == "hi"
    assert child["router_from"] == "parent"
    assert child["threshold"] == 0.1
    assert child["threshold_from"] == "parent"


def test_child_own_values_beat_the_parent(tmp_path, hi_router, lo_router, mock_completion):
    controller = _controller(
        _inheritance_registry({"router": "lo", "threshold": 0.9}),
        tmp_path,
        routers=["hi", "lo"],
        strong_model="strong_leaf",
        weak_model="weak_leaf",
    )

    res = controller.completion(
        model="root", messages=[{"role": "user", "content": "hard"}]
    )

    child = res._hidden_params["routellm_path"][1]
    assert child["router"] == "lo"
    assert child["router_from"] == "tier"
    assert child["threshold"] == 0.9
    assert child["threshold_from"] == "tier"
    assert child["picked"] == "weak_leaf"


def test_root_without_values_takes_the_request_values(
    tmp_path, hi_router, mock_completion
):
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m_a"}, "b": {"model": "m_b"}},
            "tiers": {"bare": {"strong": "a", "weak": "b"}},
        }
    )
    controller = _controller(registry, tmp_path, strong_model="a", weak_model="b")

    res = controller.completion(
        model="bare:router-hi-0.25", messages=[{"role": "user", "content": "hard"}]
    )

    root = res._hidden_params["routellm_path"][0]
    assert root["router"] == "hi"
    assert root["router_from"] == "request"
    assert root["threshold"] == 0.25
    assert root["threshold_from"] == "request"


def test_root_without_request_values_takes_the_controller_defaults(
    tmp_path, hi_router, mock_completion
):
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m_a"}, "b": {"model": "m_b"}},
            "tiers": {"bare": {"strong": "a", "weak": "b"}},
        }
    )
    controller = _controller(
        registry,
        tmp_path,
        default_router="hi",
        default_threshold=0.4,
        strong_model="a",
        weak_model="b",
    )

    res = controller.completion(
        model="bare", messages=[{"role": "user", "content": "hard"}]
    )

    root = res._hidden_params["routellm_path"][0]
    assert root["router"] == "hi"
    assert root["router_from"] == "default"
    assert root["threshold"] == 0.4
    assert root["threshold_from"] == "default"


def test_default_router_falls_back_to_the_first_configured_router(
    tmp_path, hi_router, mock_completion
):
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "m_a"}, "b": {"model": "m_b"}},
            "tiers": {"bare": {"strong": "a", "weak": "b"}},
        }
    )
    controller = _controller(
        registry, tmp_path, default_router=None, strong_model="a", weak_model="b"
    )

    res = controller.completion(
        model="bare", messages=[{"role": "user", "content": "hard"}]
    )

    assert res._hidden_params["routellm_path"][0]["router"] == "hi"


# ---------------------------------------------------------------------------
# Composition: traffic rules, middleware, canary, fallback
# ---------------------------------------------------------------------------


def test_traffic_rule_bypasses_the_tree(registry, tmp_path, hi_router, mock_completion):
    controller = _controller(
        registry,
        tmp_path,
        traffic_manager=TrafficManager(
            rules=[
                TrafficRule(
                    pattern="special",
                    strong_model="cloud_strong",
                    weak_model="local_fast",
                )
            ]
        ),
    )

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "this is special"}]
    )

    path = res._hidden_params["routellm_path"]
    assert len(path) == 1
    assert path[0]["tier"] is None
    assert path[0]["pair_from"] == "traffic_rule"
    assert path[0]["picked"] == "cloud_strong"
    assert hi_router.calls == ["this is special"]


def test_middleware_bypasses_the_tree(registry, tmp_path, hi_router, mock_completion):
    class _Middleware:
        def get_model_pair(self, prompt):
            return ModelPair(strong="cloud_strong", weak="local_fast")

    controller = _controller(registry, tmp_path, middleware=[_Middleware()])

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    path = res._hidden_params["routellm_path"]
    assert len(path) == 1
    assert path[0]["tier"] is None
    assert path[0]["pair_from"] == "middleware"


def test_bypassed_pair_uses_the_root_level_threshold(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(
        registry,
        tmp_path,
        traffic_manager=TrafficManager(
            rules=[
                TrafficRule(
                    pattern="special",
                    strong_model="cloud_strong",
                    weak_model="local_fast",
                )
            ]
        ),
    )

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "this is special"}]
    )

    entry = res._hidden_params["routellm_path"][0]
    assert entry["threshold"] == 0.12
    assert entry["threshold_from"] == "tier"


def test_fallback_descends_a_tier_sibling_by_weak_without_routers(
    registry, tmp_path, lo_router, monkeypatch
):
    config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "premium": {**CONFIG["tiers"]["premium"], "router": "lo"},
            "default": {
                "router": "lo",
                "threshold": 0.9,
                "strong": "premium",
                "weak": "local_fast",
            },
        },
    }
    controller = _controller(
        EndpointRegistry.from_config(config),
        tmp_path,
        routers=["lo"],
        default_router="lo",
        resilience_config=ResilienceConfig(max_retries=0),
    )

    res = MagicMock()
    res._hidden_params = {}
    res.model_dump.return_value = {"choices": []}
    seen = []

    def _completion(**kwargs):
        seen.append(kwargs["model"])
        if len(seen) == 1:
            raise RuntimeError("first endpoint down")
        return res

    monkeypatch.setattr(routellm.controller, "completion", _completion)

    out = controller.completion(
        model="default", messages=[{"role": "user", "content": "easy"}]
    )

    # Root scores 0.05 < 0.9, so it picks the weak leaf; the sibling is
    # the `premium` tier, descended by its weak side with no router run.
    assert seen == ["ollama_chat/qwen3:8b", "ollama_chat/qwen3:32b"]
    assert len(lo_router.calls) == 1
    assert out._hidden_params["routellm_path"][-1]["fallback_from"] == "premium"


def test_canary_model_resolves_through_the_registry(
    registry, tmp_path, hi_router, mock_completion
):
    controller = _controller(
        registry,
        tmp_path,
        quality_manager=QualityManager(
            canary_config=CanaryConfig(
                enabled=True, canary_model="local_fast", weight=1.0
            )
        ),
    )

    controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    assert mock_completion.call_args[1]["model"] == "ollama_chat/qwen3:8b"


# ---------------------------------------------------------------------------
# Flat pairs and evals
# ---------------------------------------------------------------------------


def test_both_models_none_without_a_default_tier_rejected(tmp_path, hi_router):
    flat = EndpointRegistry.from_config({"endpoints": {"a": {"model": "m"}}})

    with pytest.raises(ValueError):
        _controller(flat, tmp_path)


def test_model_pair_returns_the_flat_pair_when_set(tmp_path, hi_router):
    flat = EndpointRegistry.from_config({"endpoints": {"a": {"model": "m"}}})
    controller = _controller(flat, tmp_path, strong_model="a", weak_model="a")

    assert controller.model_pair == ModelPair(strong="a", weak="a")


def test_model_pair_raises_when_only_tiers_exist(registry, tmp_path, hi_router):
    controller = _controller(registry, tmp_path)

    with pytest.raises(RoutingError) as excinfo:
        controller.model_pair

    assert "--strong-model" in str(excinfo.value)


def test_flat_pair_still_routes_without_tiers(tmp_path, hi_router, mock_completion):
    flat = EndpointRegistry.from_config(
        {"endpoints": {"big": {"model": "m_big"}, "small": {"model": "m_small"}}}
    )
    controller = _controller(
        flat, tmp_path, strong_model="big", weak_model="small", default_router=None
    )

    res = controller.completion(
        model="router-hi-0.5", messages=[{"role": "user", "content": "hard"}]
    )

    assert mock_completion.call_args[1]["model"] == "m_big"
    assert res._hidden_params["routellm_path"][0]["tier"] is None


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_walks_the_tree_and_attaches_the_path(
    registry, tmp_path, hi_router, monkeypatch
):
    res = MagicMock()
    res._hidden_params = {}
    res.model_dump.return_value = {"choices": []}
    monkeypatch.setattr(
        routellm.controller, "acompletion", AsyncMock(return_value=res)
    )
    controller = _controller(registry, tmp_path)

    out = await controller.acompletion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    path = out._hidden_params["routellm_path"]
    assert [entry["picked"] for entry in path] == ["premium", "cloud_strong"]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_server_returns_the_routing_path(tmp_path):
    """The server copies the path into the response JSON.

    Run in a subprocess: `openai_server` parses argv at import, so a
    single process cannot import it twice with different arguments.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  default: {strong: big, weak: small}\n"
    )

    snippet = f"""
import json, sys
from unittest.mock import AsyncMock, MagicMock
sys.argv = ["x", "--config", {str(config)!r}, "--routers", "random",
            "--default-threshold", "0.5"]
import routellm.openai_server as server
from fastapi.testclient import TestClient

res = MagicMock()
res._hidden_params = {{}}
res.model_dump.return_value = {{"id": "c1", "choices": []}}
server.acompletion = AsyncMock(return_value=res)
import routellm.controller as controller
controller.acompletion = AsyncMock(return_value=res)

with TestClient(server.app) as client:
    reply = client.post(
        "/v1/chat/completions",
        json={{"model": "default", "messages": [{{"role": "user", "content": "hi"}}]}},
    )
print(json.dumps({{"status": reply.status_code, "body": reply.json()}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == 200
    assert payload["body"]["routellm"]["path"][0]["tier"] == "default"


def test_server_exposes_a_default_threshold_flag():
    source = (REPO_ROOT / "routellm" / "openai_server.py").read_text()

    assert "--default-threshold" in source


# ---------------------------------------------------------------------------
# The route_with_score hook
# ---------------------------------------------------------------------------


def test_route_only_router_reports_null_win_rate(
    registry, tmp_path, monkeypatch, mock_completion
):
    """A router implementing only `route` still routes, scoring None."""

    class RouteOnly(Router):
        def calculate_strong_win_rate(self, prompt):
            raise AssertionError("route-only router must not be scored")

        def route(self, prompt, threshold, routed_pair):
            return routed_pair.strong

    monkeypatch.setitem(
        routellm.controller.ROUTER_CLS, "hi", lambda **kw: RouteOnly()
    )
    controller = _controller(registry, tmp_path)

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    path = res._hidden_params["routellm_path"]
    assert [entry["picked"] for entry in path] == ["premium", "cloud_strong"]
    assert [entry["win_rate"] for entry in path] == [None, None]


def test_scorer_called_once_per_level(registry, tmp_path, monkeypatch, mock_completion):
    """A scoring router is scored exactly once at each tier level."""
    calls = []

    class Counting(Router):
        def calculate_strong_win_rate(self, prompt):
            calls.append(prompt)
            return 0.9

    monkeypatch.setitem(
        routellm.controller.ROUTER_CLS, "hi", lambda **kw: Counting()
    )
    controller = _controller(registry, tmp_path)

    res = controller.completion(
        model="default", messages=[{"role": "user", "content": "the prompt"}]
    )

    assert calls == ["the prompt", "the prompt"]
    assert [entry["win_rate"] for entry in res._hidden_params["routellm_path"]] == [
        0.9,
        0.9,
    ]


def test_routing_leaves_the_router_instance_untouched(
    registry, tmp_path, hi_router, mock_completion
):
    """Walking the tree must not instrument the shared router.

    One router instance serves every request, so a walk that swaps
    attributes on it in place races once two requests overlap. The
    scorer must still resolve to the class after a walk.
    """
    controller = _controller(registry, tmp_path)

    controller.completion(
        model="default", messages=[{"role": "user", "content": "hard"}]
    )

    instance = controller.routers["hi"]
    assert "calculate_strong_win_rate" not in vars(instance)
