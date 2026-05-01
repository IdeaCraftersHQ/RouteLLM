import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from routellm.controller import Controller
from routellm.resilience import ResilienceConfig
from routellm.caching import CacheConfig
from routellm.traffic import TrafficManager, TrafficRule
from routellm.quality import QualityManager, FineTuneConfig, CanaryConfig

@pytest.fixture
def complex_controller(tmp_path):
    db_path = str(tmp_path / "integration_cache.db")
    trace_dir = str(tmp_path / "integration_traces")
    
    return Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        resilience_config=ResilienceConfig(max_retries=1),
        cache_config=CacheConfig(db_path=db_path),
        traffic_manager=TrafficManager(rules=[
            TrafficRule(pattern="special", strong_model="special-strong", weak_model="special-weak")
        ]),
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(enabled=True, trace_dir=trace_dir),
            canary_config=CanaryConfig(enabled=True, canary_model="canary-1", weight=1.0)
        )
    )

@pytest.mark.asyncio
async def test_controller_full_flow(complex_controller, monkeypatch):
    # Mock acompletion to return a success response
    mock_res = MagicMock()
    mock_res.choices = [MagicMock()]
    mock_res.choices[0].message.content = "integration success"
    mock_res.model_dump.return_value = {"choices": [{"message": {"content": "integration success"}}]}
    
    mock_acompletion = AsyncMock(return_value=mock_res)
    monkeypatch.setattr("routellm.controller.acompletion", mock_acompletion)

    # First call: triggers traffic rule -> triggers canary -> records trace -> populates cache
    res = await complex_controller.acompletion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "this is special"}]
    )
    
    assert res.choices[0].message.content == "integration success"
    # Should have called with canary model
    assert mock_acompletion.call_args[1]["model"] == "canary-1"
    
    # Check trace was recorded
    assert len(os.listdir(complex_controller.quality_manager.fine_tune_config.trace_dir)) == 1
    
    # Second call: should hit cache
    # Reset mock to ensure no new network calls
    mock_acompletion.reset_mock()
    
    res2 = await complex_controller.acompletion(
        router="random",
        threshold=0.5,
        messages=[{"role": "user", "content": "this is special"}]
    )
    
    assert res2.choices[0].message.content == "integration success"
    assert mock_acompletion.call_count == 0
