"""Tests for the capability check the tier walk runs before each router.

Covers the three outcomes (both sides pass, exactly one passes, neither
passes), the tier union over reachable leaves, the flat pairs a traffic
rule / middleware / legacy config produces, the fallback sibling filter,
the two opposite defaults for an unknown capability, and the 400 body a
refusal produces on the server.

Routers are stubs registered through `routellm.controller.ROUTER_CLS`,
exactly as test_tiers.py does, and caching is off throughout.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import routellm.controller
from routellm.caching import CacheConfig
from routellm.capabilities import Capabilities, build_tier_index, satisfies
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.requirements import Requirements
from routellm.routers.base import Router
from routellm.traffic import TrafficManager, TrafficRule
from routellm.types import Middleware, ModelPair

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "endpoints": {
        "seeing": {
            "model": "ollama_chat/qwen3:32b",
            "capabilities": {"vision": True, "tools": True, "context": 200000},
        },
        "blind": {
            "model": "ollama_chat/qwen3:8b",
            "capabilities": {"vision": False, "tools": True, "context": 8192},
        },
        "also_blind": {
            "model": "ollama_chat/llama3:8b",
            "capabilities": {"vision": False, "tools": False},
        },
    },
    "tiers": {
        "default": {
            "router": "hi",
            "threshold": 0.12,
            "strong": "seeing",
            "weak": "blind",
        },
    },
}


class _StubRouter(Router):
    """Router returning a fixed win rate and counting its calls."""

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


@pytest.fixture(autouse=True)
def _no_catalog(monkeypatch, tmp_path):
    """Keep every test off the network and out of the real cache."""
    from routellm import pairing

    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])


def _controller(config, tmp_path, **kwargs):
    """Controller on a registry built from `config`, caching off."""
    kwargs.setdefault("routers", ["hi"])
    kwargs.setdefault("strong_model", None)
    kwargs.setdefault("weak_model", None)
    kwargs.setdefault("endpoints", EndpointRegistry.from_config(config))
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    return Controller(**kwargs)


def _text(content="explain recursion"):
    """One plain user message."""
    return [{"role": "user", "content": content}]


def _vision():
    """One user message carrying a text part and an image part."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this picture"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ],
        }
    ]


def _route(controller, messages, model="default", **kwargs):
    """Run the routing half of a request and return (picked, path)."""
    request = {"model": model, "messages": messages, **kwargs}
    tier, router, threshold = controller._parse_model_name(model)
    from routellm import requirements as requirements_module

    prompt = requirements_module._prompt_text(messages)
    reqs = requirements_module.derive(messages, request, model)
    picked, path, pair = controller._route(prompt, request, tier, router, threshold, reqs)
    return picked, path, pair


# ---------------------------------------------------------------------------
# The no-op path and the forced one
# ---------------------------------------------------------------------------


def test_a_text_request_routes_exactly_as_before(hi_router, tmp_path):
    controller = _controller(CONFIG, tmp_path)

    picked, path, _ = _route(controller, _text())

    assert picked == "seeing"
    assert len(hi_router.calls) == 1
    assert "capability_forced" not in path[0]
    assert "capability_requirement" not in path[0]


def test_a_vision_request_is_forced_to_the_only_capable_side(hi_router, tmp_path):
    controller = _controller(CONFIG, tmp_path)

    picked, path, _ = _route(controller, _vision())

    assert picked == "seeing"
    assert hi_router.calls == []


def test_capability_forced_names_the_side_and_the_requirement(hi_router, tmp_path):
    controller = _controller(CONFIG, tmp_path)

    _, path, _ = _route(controller, _vision())

    assert path[0]["capability_forced"] == "strong"
    assert path[0]["capability_requirement"] == "vision"
    assert path[0]["win_rate"] is None
    assert path[0]["router"] == "hi"
    assert path[0]["router_from"] == "tier"
    assert path[0]["threshold"] == 0.12


def test_no_capable_side_raises_routing_error_naming_the_tier(hi_router, tmp_path):
    config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "blind",
                "weak": "also_blind",
            }
        },
    }
    controller = _controller(config, tmp_path)

    with pytest.raises(RoutingError) as excinfo:
        _route(controller, _vision())

    message = str(excinfo.value)
    assert "default" in message
    assert "vision" in message
    assert "blind" in message
    assert "also_blind" in message


# ---------------------------------------------------------------------------
# Tier-valued sides
# ---------------------------------------------------------------------------


def test_a_tier_side_passes_when_any_reachable_leaf_passes(hi_router, tmp_path):
    config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "mixed": {
                "router": "hi",
                "threshold": 0.5,
                "strong": "seeing",
                "weak": "blind",
            },
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "mixed",
                "weak": "also_blind",
            },
        },
    }
    controller = _controller(config, tmp_path)

    picked, path, _ = _route(controller, _vision())

    assert path[0]["capability_forced"] == "strong"
    assert picked == "seeing"


def test_a_tier_side_fails_when_no_reachable_leaf_passes(hi_router, tmp_path):
    config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "dark": {
                "router": "hi",
                "threshold": 0.5,
                "strong": "blind",
                "weak": "also_blind",
            },
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "dark",
                "weak": "seeing",
            },
        },
    }
    controller = _controller(config, tmp_path)

    picked, path, _ = _route(controller, _vision())

    assert path[0]["capability_forced"] == "weak"
    assert path[0]["capability_requirement"] == "vision"
    assert picked == "seeing"


def test_build_tier_index_unions_the_reachable_leaves(tmp_path):
    registry = EndpointRegistry.from_config(CONFIG)

    index = build_tier_index(registry, {})

    assert index["default"].vision is True
    assert index["default"].context == 200000


# ---------------------------------------------------------------------------
# Flat pairs
# ---------------------------------------------------------------------------


def _flat_config():
    """Endpoints only, no tiers: a controller with a flat pair."""
    return {"endpoints": CONFIG["endpoints"]}


def test_flat_pair_from_a_traffic_rule_is_checked_too(hi_router, tmp_path):
    manager = TrafficManager(
        rules=[
            TrafficRule(
                condition=lambda prompt, kwargs: True,
                strong_model="blind",
                weak_model="seeing",
            )
        ]
    )
    controller = _controller(
        _flat_config(),
        tmp_path,
        strong_model="blind",
        weak_model="also_blind",
        traffic_manager=manager,
        default_router="hi",
    )

    picked, path, _ = _route(controller, _vision(), model="router-hi-0.5")

    assert picked == "seeing"
    assert path[0]["capability_forced"] == "weak"
    assert path[0]["capability_requirement"] == "vision"
    assert hi_router.calls == []


def test_middleware_pair_is_checked_too(hi_router, tmp_path):
    class _Pair(Middleware):
        def get_model_pair(self, prompt):
            return ModelPair(strong="blind", weak="seeing")

    controller = _controller(
        _flat_config(),
        tmp_path,
        strong_model="blind",
        weak_model="also_blind",
        middleware=[_Pair()],
        default_router="hi",
    )

    picked, path, _ = _route(controller, _vision(), model="router-hi-0.5")

    assert picked == "seeing"
    assert path[0]["capability_forced"] == "weak"


def test_legacy_flat_pair_is_checked_too(hi_router, tmp_path):
    controller = _controller(
        _flat_config(),
        tmp_path,
        strong_model="seeing",
        weak_model="blind",
        default_router="hi",
    )

    picked, path, _ = _route(controller, _vision(), model="router-hi-0.5")

    assert picked == "seeing"
    assert path[0]["capability_forced"] == "strong"
    assert hi_router.calls == []


# ---------------------------------------------------------------------------
# Fallback siblings
# ---------------------------------------------------------------------------


def test_an_incapable_fallback_sibling_is_skipped(hi_router, tmp_path):
    config = {
        "endpoints": CONFIG["endpoints"],
        "tiers": {
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "seeing",
                "weak": "blind",
            }
        },
    }
    controller = _controller(config, tmp_path)
    reqs = Requirements(vision=True)
    path = [
        {
            "tier": "default",
            "router": "hi",
            "router_from": "tier",
            "threshold": 0.12,
            "threshold_from": "tier",
            "win_rate": None,
            "picked": "seeing",
        }
    ]

    _, with_check, sibling, _ = controller._models_to_try("seeing", path, None, reqs)
    _, without_check, plain_sibling, _ = controller._models_to_try(
        "seeing", path, None, Requirements()
    )

    assert sibling is None
    assert with_check == ["seeing"]
    assert plain_sibling == "blind"
    assert without_check == ["seeing", "blind"]


# ---------------------------------------------------------------------------
# The two opposite defaults for unknown
# ---------------------------------------------------------------------------


def test_unknown_capability_serves_by_default_and_logs(caplog):
    import logging

    caps = Capabilities()
    with caplog.at_level(logging.INFO, logger="routellm.capabilities"):
        failed = satisfies(caps, Requirements(vision=True), False, "mystery")

    assert failed is None


def test_unknown_capability_refuses_under_strict():
    caps = Capabilities()

    assert satisfies(caps, Requirements(vision=True), True, "mystery") == "vision"


def test_strict_endpoint_refuses_a_request_it_cannot_prove(hi_router, tmp_path):
    config = {
        "endpoints": {
            "mystery": {"model": "ollama_chat/qwen3:32b", "strict": True},
            "blind": CONFIG["endpoints"]["blind"],
        },
        "tiers": {
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "mystery",
                "weak": "blind",
            }
        },
    }
    controller = _controller(config, tmp_path)

    with pytest.raises(RoutingError, match="vision"):
        _route(controller, _vision())


def test_context_requirement_uses_the_merged_context_window(hi_router, tmp_path):
    caps_wide = Capabilities(context=200000)
    caps_narrow = Capabilities(context=8192)
    reqs = Requirements(context_needed=100000)

    assert satisfies(caps_wide, reqs, False, "seeing") is None
    assert satisfies(caps_narrow, reqs, False, "blind") == "context"


def test_context_unknown_on_both_sides_never_refuses():
    reqs = Requirements(context_needed=10_000_000)

    assert satisfies(Capabilities(), reqs, False, "a") is None
    assert satisfies(Capabilities(), reqs, True, "a") is None


def test_vision_message_reaches_the_router_as_text(hi_router, tmp_path):
    config = {
        "endpoints": {
            "seeing": CONFIG["endpoints"]["seeing"],
            "also_seeing": {
                "model": "ollama_chat/llava:13b",
                "capabilities": {"vision": True},
            },
        },
        "tiers": {
            "default": {
                "router": "hi",
                "threshold": 0.12,
                "strong": "seeing",
                "weak": "also_seeing",
            }
        },
    }
    controller = _controller(config, tmp_path)

    _route(controller, _vision())

    assert hi_router.calls == ["what is in this picture"]
    assert all(isinstance(call, str) for call in hi_router.calls)


# ---------------------------------------------------------------------------
# Server surface
# ---------------------------------------------------------------------------


_SERVER_BODY = """
import json, sys
from unittest.mock import AsyncMock, MagicMock
sys.argv = {argv!r}
import routellm.openai_server as server
from fastapi.testclient import TestClient

res = MagicMock()
res._hidden_params = {{}}
res.model_dump.return_value = {{"id": "c1", "choices": []}}
server.acompletion = AsyncMock(return_value=res)
import routellm.controller as controller
controller.acompletion = AsyncMock(return_value=res)

with TestClient(server.app) as client:
    reply = client.post("/v1/chat/completions", json={{
        "model": "default",
        "messages": [{{"role": "user", "content": [
            {{"type": "text", "text": "what is this"}},
            {{"type": "image_url", "image_url": {{"url": "https://x/y.png"}}}},
        ]}}],
    }})
    print(json.dumps({{"status": reply.status_code, "body": reply.json()}}))
"""


def test_server_returns_400_with_a_json_body_for_a_refusal(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  blind:\n"
        "    model: ollama_chat/qwen3:8b\n"
        "    capabilities: {vision: false}\n"
        "  also_blind:\n"
        "    model: ollama_chat/llama3:8b\n"
        "    capabilities: {vision: false}\n"
        "tiers:\n"
        "  default:\n"
        "    router: random\n"
        "    threshold: 0.5\n"
        "    strong: blind\n"
        "    weak: also_blind\n"
    )
    argv = [
        "openai_server",
        "--config",
        str(config),
        "--routers",
        "random",
    ]

    result = subprocess.run(
        [sys.executable, "-c", _SERVER_BODY.format(argv=argv)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ROUTELLM_CATALOG_CACHE": str(tmp_path / "catalog.json"),
        },
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["status"] == 400
    assert payload["body"]["object"] == "error"
    assert "vision" in payload["body"]["message"]
