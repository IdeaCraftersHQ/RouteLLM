import pytest
import os
import json
import asyncio
import random
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from routellm.controller import Controller
from routellm.resilience import ResilienceConfig
from routellm.caching import CacheConfig
from routellm.traffic import TrafficManager, TrafficRule, LoadBalancer, LoadBalancerConfig, LoadBalancerEndpoint
from routellm.quality import QualityManager, FineTuneConfig, CanaryConfig
from routellm.types import ModelPair

class MockError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code

def create_mock_response(content):
    res = MagicMock()
    res.choices = [MagicMock()]
    res.choices[0].message.content = content
    res.model_dump.return_value = {"choices": [{"message": {"content": content}}]}
    return res

class SimpleRouter:
    """A minimal router that always returns the strong model for testing."""
    def route(self, prompt, threshold, routed_pair):
        return routed_pair.strong

@pytest.fixture(scope="function")
def e2e_controller(tmp_path):
    db_path = str(tmp_path / f"e2e_cache_{random.randint(0, 10000)}.db")
    trace_dir = str(tmp_path / f"e2e_traces_{random.randint(0, 10000)}")
    
    lb_config = LoadBalancerConfig(
        strategy="round-robin",
        endpoints=[
            LoadBalancerEndpoint(model="phys-1", api_key="key-1"),
            LoadBalancerEndpoint(model="phys-2", api_key="key-2")
        ]
    )
    
    tm = TrafficManager(
        rules=[
            TrafficRule(pattern="FORCE_LB", strong_model="lb-model", weak_model="gpt-3.5-turbo")
        ],
        load_balancers={"lb-model": LoadBalancer(lb_config)}
    )

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        resilience_config=ResilienceConfig(max_retries=1, initial_backoff_ms=0),
        cache_config=CacheConfig(db_path=db_path, enabled=True, semantic_enabled=True),
        traffic_manager=tm,
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(enabled=True, trace_dir=trace_dir),
            canary_config=CanaryConfig(enabled=False, canary_model="canary-v1", weight=0.0)
        )
    )
    # Ensure we use our predictable router
    controller.routers["random"] = SimpleRouter()
    return controller

@pytest.mark.asyncio
async def test_scenario_resilience_exhaustion(e2e_controller, monkeypatch):
    mock_acompletion = AsyncMock(side_effect=MockError("Permanent Fail", 500))
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)
    
    with pytest.raises(Exception):
        await e2e_controller.acompletion(
            router="random", threshold=0.5, 
            messages=[{"role": "user", "content": "fail me"}]
        )
    
    assert mock_acompletion.call_count == 4

@pytest.mark.asyncio
async def test_scenario_traffic_rule_to_load_balancer(e2e_controller, monkeypatch):
    # Disable cache to test LB round-robin
    e2e_controller.cache.config.enabled = False
    
    mock_acompletion = AsyncMock(return_value=create_mock_response("balanced success"))
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)
    
    await e2e_controller.acompletion(
        router="random", threshold=0.5,
        messages=[{"role": "user", "content": "FORCE_LB"}]
    )
    assert mock_acompletion.call_args[1]["model"] == "phys-1"

    await e2e_controller.acompletion(
        router="random", threshold=0.5,
        messages=[{"role": "user", "content": "FORCE_LB"}]
    )
    assert mock_acompletion.call_args[1]["model"] == "phys-2"

@pytest.mark.asyncio
async def test_scenario_semantic_cache_chain(e2e_controller, monkeypatch):
    mock_acompletion = AsyncMock(return_value=create_mock_response("fruit info"))
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)
    
    with patch("routellm.caching.Cache._get_embedding") as mock_emb:
        vec = np.array([1.0, 0.0], dtype=np.float32)
        mock_emb.return_value = vec
        
        await e2e_controller.acompletion(
            router="random", threshold=0.5,
            messages=[{"role": "user", "content": "apples"}]
        )
        assert mock_acompletion.call_count == 1
        
        mock_acompletion.reset_mock()
        res = await e2e_controller.acompletion(
            router="random", threshold=0.5,
            messages=[{"role": "user", "content": "pears"}]
        )
        assert mock_acompletion.call_count == 0
        assert res.choices[0].message.content == "fruit info"

@pytest.mark.asyncio
async def test_scenario_canary_flow(e2e_controller, monkeypatch):
    e2e_controller.quality_manager.canary_config.enabled = True
    e2e_controller.quality_manager.canary_config.weight = 1.0
    
    mock_acompletion = AsyncMock(return_value=create_mock_response("canary ok"))
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)
    
    res = await e2e_controller.acompletion(
        router="random", threshold=0.5,
        messages=[{"role": "user", "content": "hello"}]
    )
    
    assert mock_acompletion.call_args[1]["model"] == "canary-v1"
    assert res.choices[0].message.content == "canary ok"
    
    trace_dir = e2e_controller.quality_manager.fine_tune_config.trace_dir
    assert len(os.listdir(trace_dir)) == 1
