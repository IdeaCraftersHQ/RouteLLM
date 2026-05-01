import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from routellm.controller import Controller, ModelPair
from routellm.resilience import ResilienceConfig
from routellm.caching import CacheConfig
from routellm.traffic import TrafficManager, TrafficRule, LoadBalancer, LoadBalancerConfig, LoadBalancerEndpoint

@pytest.fixture
def controller():
    return Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        resilience_config=ResilienceConfig(max_retries=1, initial_backoff_ms=10),
        cache_config=CacheConfig(enabled=False), # Disable for main tests
    )

class MockError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code

@pytest.mark.asyncio
async def test_resilience_retry(controller, monkeypatch):
    # Mock litellm.acompletion to fail once then succeed
    mock_acompletion = AsyncMock()
    
    success_res = MagicMock()
    success_res.choices = [MagicMock()]
    success_res.choices[0].message.content = "success"
    success_res.model_dump.return_value = {"id": "test", "choices": [{"message": {"content": "success"}}]}
    
    mock_acompletion.side_effect = [
        MockError("Rate limit", 429),
        success_res
    ]
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)

    res = await controller.acompletion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}]
    )
    
    assert res.choices[0].message.content == "success"
    assert mock_acompletion.call_count == 2

@pytest.mark.asyncio
async def test_resilience_fallback(controller, monkeypatch):
    # Mock litellm.acompletion to fail completely for the first model
    mock_acompletion = AsyncMock()
    
    fallback_res = MagicMock()
    fallback_res.choices = [MagicMock()]
    fallback_res.choices[0].message.content = "fallback success"
    fallback_res.model_dump.return_value = {"id": "test", "choices": [{"message": {"content": "fallback success"}}]}

    # First model (routed) fails, second model (fallback) succeeds
    mock_acompletion.side_effect = [
        MockError("Routed model failed", 500), # Attempt 1
        MockError("Routed model failed", 500), # Retry 1
        fallback_res
    ]
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)

    res = await controller.acompletion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "hello"}]
    )
    
    assert res.choices[0].message.content == "fallback success"
    # 2 calls for first model (1 initial + 1 retry), 1 call for fallback model
    assert mock_acompletion.call_count == 3

def test_traffic_manager_conditional_routing():
    rules = [
        TrafficRule(
            pattern="code",
            strong_model="codellama",
            weak_model="gpt-3.5-turbo"
        )
    ]
    tm = TrafficManager(rules=rules)
    
    # Prompt with "code"
    pair = tm.get_model_pair("write some code", {})
    assert pair.strong == "codellama"
    
    # Prompt without "code"
    pair = tm.get_model_pair("hello", {})
    assert pair is None

def test_load_balancer():
    config = LoadBalancerConfig(
        strategy="round-robin",
        endpoints=[
            LoadBalancerEndpoint(model="m1"),
            LoadBalancerEndpoint(model="m2")
        ]
    )
    lb = LoadBalancer(config)
    
    assert lb.select().model == "m1"
    assert lb.select().model == "m2"
    assert lb.select().model == "m1"
