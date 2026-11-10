"""Strict registry mode: no raw model name reaches litellm on any path."""

import asyncio
import sys
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from routellm.caching import CacheConfig
from routellm.comparison import ComparisonConfig
from routellm.controller import Controller, RoutingError
from routellm.endpoints import EndpointRegistry
from routellm.quality import CanaryConfig, QualityManager
from routellm.resilience import ResilienceConfig
from routellm.server_config import build_comparison_config
from routellm.traffic import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancerEndpoint,
    TrafficManager,
    TrafficRule,
)
from routellm.types import ModelPair

CONFIG = {
    "endpoints": {
        "big": {"model": "openai/big-1", "api_base": "http://big.example"},
        "small": {"model": "openai/small-1", "api_base": "http://small.example"},
        "alt": {"model": "openai/alt-1", "api_base": "http://alt.example"},
    },
    "tiers": {"default": {"router": "random", "strong": "big", "weak": "small"}},
}
MESSAGES = [{"role": "user", "content": "hello"}]


def registry(strict=False, **extra):
    return EndpointRegistry.from_config({**CONFIG, **extra, "strict_registry": strict})


def controller(strict=True, **kwargs):
    kwargs.setdefault("endpoints", registry(strict))
    kwargs.setdefault("cache_config", CacheConfig(enabled=False))
    return Controller(routers=["random"], strict_registry=strict, **kwargs)


# --- resolve ----------------------------------------------------------------


def test_resolve_unknown_name_raises_when_strict():
    with pytest.raises(ValueError, match="gpt-4o.*strict_registry"):
        registry(True).resolve("gpt-4o")


def test_resolve_known_name_still_works_when_strict():
    assert registry(True).resolve("big").model == "openai/big-1"


def test_resolve_wraps_raw_name_when_off():
    assert registry(False).resolve("gpt-4o").model == "gpt-4o"


def test_config_key_must_be_boolean():
    with pytest.raises(ValueError, match="strict_registry"):
        EndpointRegistry.from_config({**CONFIG, "strict_registry": "yes"})


def test_default_is_off():
    assert EndpointRegistry.from_config(CONFIG).strict_registry is False
    assert controller(strict=False).endpoints.strict_registry is False


def test_controller_flag_turns_registry_strict():
    c = Controller(routers=["random"], endpoints=registry(False), strict_registry=True)
    with pytest.raises(ValueError):
        c.endpoints.resolve("gpt-4o")


# --- flat pair --------------------------------------------------------------


def test_raw_flat_pair_refused():
    with pytest.raises(ValueError, match="strong_model.*gpt-4o"):
        controller(strong_model="gpt-4o", weak_model="small")
    with pytest.raises(ValueError, match="weak_model.*gpt-3.5"):
        controller(strong_model="big", weak_model="gpt-3.5")


def test_endpoint_flat_pair_accepted():
    controller(strong_model="big", weak_model="small")


def test_raw_flat_pair_accepted_when_off():
    controller(strict=False, strong_model="gpt-4o", weak_model="gpt-3.5")


def test_legacy_pair_without_registry_refused():
    with pytest.raises(ValueError):
        controller(endpoints=EndpointRegistry(), strong_model="gpt-4", weak_model="mixtral")


# --- traffic rules, canary, load balancer -----------------------------------


def test_traffic_rule_raw_model_refused():
    tm = TrafficManager(rules=[TrafficRule(pattern="x", strong_model="big", weak_model="gpt-4o")])
    with pytest.raises(ValueError, match="traffic rule 0 weak model.*gpt-4o"):
        controller(traffic_manager=tm)


def test_traffic_rule_endpoints_accepted():
    tm = TrafficManager(rules=[TrafficRule(pattern="x", strong_model="alt", weak_model="small")])
    controller(traffic_manager=tm)


def test_canary_raw_model_refused():
    qm = QualityManager(canary_config=CanaryConfig(enabled=True, canary_model="gpt-4o", weight=1.0))
    with pytest.raises(ValueError, match="canary_model.*gpt-4o"):
        controller(quality_manager=qm)


def test_canary_endpoint_accepted():
    qm = QualityManager(canary_config=CanaryConfig(enabled=True, canary_model="alt", weight=1.0))
    controller(quality_manager=qm)


def _lb(key, *targets):
    return {
        key: LoadBalancer(
            LoadBalancerConfig(endpoints=[LoadBalancerEndpoint(model=t) for t in targets])
        )
    }


def test_load_balancer_raw_key_refused():
    with pytest.raises(ValueError, match="load balancer key.*gpt-4o"):
        controller(traffic_manager=TrafficManager(load_balancers=_lb("gpt-4o", "alt")))


def test_load_balancer_raw_target_refused():
    with pytest.raises(ValueError, match="load balancer 'big' target.*gpt-4o"):
        controller(traffic_manager=TrafficManager(load_balancers=_lb("big", "alt", "gpt-4o")))


def test_load_balancer_target_resolves_to_endpoint_params():
    c = controller(traffic_manager=TrafficManager(load_balancers=_lb("big", "alt")))
    model, base, _key, _extra = c._endpoint_call_params("big")
    assert (model, base) == ("openai/alt-1", "http://alt.example")


# --- middleware -------------------------------------------------------------


class _Pair:
    """Middleware answering a fixed pair per request."""

    def __init__(self, pair):
        self.pair = pair

    def get_model_pair(self, prompt):
        return self.pair


def test_middleware_default_pair_raw_refused():
    from routellm.middleware.intent_model_selector import IntentModelSelector

    selector = IntentModelSelector([], default_model_pair=ModelPair(strong="big", weak="gpt-4o"))
    with pytest.raises(ValueError, match="default pair weak model.*gpt-4o"):
        controller(middleware=[selector])


def test_middleware_intent_pair_raw_refused():
    from routellm.middleware.intent_model_selector import (
        IntentModelMapping,
        IntentModelSelector,
    )

    mapping = IntentModelMapping(
        intent="code", model_pair=ModelPair(strong="gpt-4o", weak="small"), description="c"
    )
    with pytest.raises(ValueError, match="intent 'code' pair strong model.*gpt-4o"):
        controller(middleware=[IntentModelSelector([mapping])])


def test_intent_detection_raw_model_refused():
    from routellm.middleware.intent_model_selector import IntentModelSelector

    selector = IntentModelSelector([], intent_detection_model="gpt-3.5-turbo")
    with pytest.raises(ValueError, match="detection model.*gpt-3.5-turbo"):
        controller(middleware=[selector])


def test_intent_detection_endpoint_accepted():
    from routellm.middleware.intent_model_selector import IntentModelSelector

    reg = registry(True)
    selector = IntentModelSelector([], intent_endpoint=reg.get("small"))
    controller(endpoints=reg, middleware=[selector])


def test_middleware_pair_raw_refused_per_request():
    c = controller(middleware=[_Pair(ModelPair(strong="big", weak="gpt-4o"))])
    with pytest.raises(RoutingError, match="gpt-4o"):
        asyncio.run(c.acompletion(router="random", threshold=0.5, messages=MESSAGES))


# --- comparison / shadow targets -------------------------------------------


def test_comparison_raw_target_refused_when_strict():
    cfg = build_comparison_config({"comparison": {"enabled": True, "profiles": {"x": ["gpt-4o"]}}})
    assert isinstance(cfg, ComparisonConfig)
    with pytest.raises(ValueError, match="gpt-4o"):
        controller(comparison_config=cfg)


# --- legacy router-- form, end to end ---------------------------------------


def _ok():
    res = MagicMock()
    res.choices = [MagicMock()]
    res.choices[0].message.content = "ok"
    res.model_dump.return_value = {"id": "t", "choices": [{"message": {"content": "ok"}}]}
    return res


@pytest.mark.asyncio
async def test_legacy_router_form_routes_to_endpoint_model(monkeypatch):
    mock = AsyncMock(return_value=_ok())
    monkeypatch.setattr("routellm.controller.acompletion", mock)
    c = controller(resilience_config=ResilienceConfig(max_retries=0))

    await c.acompletion(model="router-random-0.5", messages=MESSAGES)

    assert mock.call_args.kwargs["model"] in {"openai/big-1", "openai/small-1"}


@pytest.mark.asyncio
async def test_fallback_sibling_is_an_endpoint_model(monkeypatch):
    mock = AsyncMock(side_effect=[RuntimeError("boom"), _ok()])
    monkeypatch.setattr("routellm.controller.acompletion", mock)
    c = controller(resilience_config=ResilienceConfig(max_retries=0))

    await c.acompletion(model="default", messages=MESSAGES)

    models = [call.kwargs["model"] for call in mock.call_args_list]
    assert len(models) == 2 and set(models) == {"openai/big-1", "openai/small-1"}


@pytest.mark.asyncio
async def test_canary_and_traffic_rule_endpoints_route_per_request(monkeypatch):
    mock = AsyncMock(return_value=_ok())
    monkeypatch.setattr("routellm.controller.acompletion", mock)
    tm = TrafficManager(rules=[TrafficRule(pattern="hello", strong_model="alt", weak_model="alt")])
    qm = QualityManager(canary_config=CanaryConfig(enabled=True, canary_model="small", weight=1.0))
    c = controller(traffic_manager=tm, quality_manager=qm)

    await c.acompletion(model="default", messages=MESSAGES)

    assert mock.call_args.kwargs["model"] == "openai/small-1"


# --- server: --strict-registry ---------------------------------------------


@pytest.fixture(scope="module")
def server():
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def _args(**kw):
    return Namespace(
        **{
            "strong_model": None,
            "weak_model": None,
            "routers": ["random"],
            "default_threshold": 0.5,
            **kw,
        }
    )


def test_server_raw_flags_refused_when_strict(server, monkeypatch):
    monkeypatch.setattr(server, "args", _args(strong_model="gpt-4o", weak_model="small"))
    with pytest.raises(ValueError, match="--strong-model 'gpt-4o'"):
        server.build_registry({"endpoints": CONFIG["endpoints"], "strict_registry": True})


def test_server_raw_flags_wrapped_when_off(server, monkeypatch):
    monkeypatch.setattr(server, "args", _args(strong_model="gpt-4o", weak_model="small"))
    reg = server.build_registry({"endpoints": CONFIG["endpoints"]})
    assert reg.get(server.IMPLICIT_STRONG).model == "gpt-4o"


def test_server_endpoint_flags_keep_strict(server, monkeypatch):
    monkeypatch.setattr(server, "args", _args(strong_model="big", weak_model="small"))
    reg = server.build_registry({"endpoints": CONFIG["endpoints"], "strict_registry": True})
    assert reg.strict_registry is True
    assert reg.get_tier("default").strong == "big"


def test_server_cli_flag_exists(server):
    assert server.parser.parse_args(["--strict-registry"]).strict_registry is True
    assert server.parser.parse_args([]).strict_registry is False
