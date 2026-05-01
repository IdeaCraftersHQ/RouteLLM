import pytest
import time
import json
import os
import numpy as np
from unittest.mock import MagicMock, patch
from routellm.resilience import CircuitBreaker, CircuitState
from routellm.caching import Cache, CacheConfig

def test_circuit_breaker_transitions():
    cb = CircuitBreaker(fail_max=2, fail_wait_ms=100)
    
    # 1. Closed -> Open
    assert cb.state == CircuitState.CLOSED
    cb.record_failure(Exception())
    assert cb.state == CircuitState.CLOSED
    cb.record_failure(Exception())
    assert cb.state == CircuitState.OPEN
    
    # 2. Open: cannot execute
    assert cb.can_execute() is False
    
    # 3. Open -> Half-Open (after wait)
    time.sleep(0.15)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN
    
    # 4. Half-Open -> Closed (on success)
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failures == 0

def test_circuit_breaker_rate_threshold():
    # Fail rate 50% with minimum 4 requests
    cb = CircuitBreaker(fail_rate=0.5, rate_minimum=4)
    
    cb.record_success()
    cb.record_failure(Exception())
    cb.record_success()
    cb.record_failure(Exception())
    
    # 2/4 = 50%, should open
    assert cb.state == CircuitState.OPEN

def test_cache_ttl(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    config = CacheConfig(db_path=db_path, ttl_seconds=1)
    cache = Cache(config)
    
    prompt = "test prompt"
    model = "test-model"
    params = {"temp": 0.7}
    resp = {"choices": [{"text": "hello"}]}
    
    cache.put(prompt, model, params, resp)
    assert cache.get(prompt, model, params) == resp
    
    # Wait for expiry
    time.sleep(1.1)
    assert cache.get(prompt, model, params) is None

@patch("routellm.caching.Cache._get_embedding")
def test_semantic_cache(mock_emb, tmp_path):
    db_path = str(tmp_path / "test_semantic.db")
    config = CacheConfig(db_path=db_path, semantic_enabled=True, semantic_threshold=0.9)
    cache = Cache(config)
    
    # Mock embeddings
    # "apple" vector
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    # "pear" vector (similar)
    v2 = np.array([0.95, 0.1], dtype=np.float32)
    # "car" vector (different)
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    
    mock_emb.side_effect = [v1, v2, v3]
    
    prompt1 = "apple"
    model = "m"
    resp = {"text": "fruit"}
    
    # 1. Put prompt1
    cache.put(prompt1, model, {}, resp)
    
    # 2. Get prompt2 (similar to prompt1)
    # mock_emb returns v2
    res = cache.get("pear", model, {})
    assert res == resp
    
    # 3. Get prompt3 (different from prompt1)
    # mock_emb returns v3
    res = cache.get("car", model, {})
    assert res is None

def test_traffic_manager_rules():
    from routellm.traffic import TrafficManager, TrafficRule
    from routellm.types import ModelPair
    
    rules = [
        TrafficRule(pattern="urgent", strong_model="high-tier", weak_model="mid-tier"),
        TrafficRule(max_tokens=10, strong_model="small", weak_model="tiny")
    ]
    tm = TrafficManager(rules=rules)
    
    # Pattern match
    pair = tm.get_model_pair("this is urgent", {})
    assert pair.strong == "high-tier"
    
    # Token limit match
    pair = tm.get_model_pair("short", {"max_tokens": 5})
    assert pair.strong == "small"
    
    # No match
    pair = tm.get_model_pair("normal request", {"max_tokens": 100})
    assert pair is None

def test_quality_manager_traces(tmp_path):
    from routellm.quality import QualityManager, FineTuneConfig
    
    trace_dir = str(tmp_path / "traces")
    config = FineTuneConfig(enabled=True, trace_dir=trace_dir)
    qm = QualityManager(fine_tune_config=config)
    
    prompt = "hello"
    model = "m1"
    resp = {"choices": [{"message": {"content": "hi"}}]}
    
    qm.record_trace(prompt, model, resp)
    
    # Verify file exists and content
    files = os.listdir(trace_dir)
    assert len(files) == 1
    with open(os.path.join(trace_dir, files[0]), "r") as f:
        data = json.load(f)
        assert data["input"]["prompt"] == prompt
        assert data["routed_model"] == model
        assert data["output"] == resp

def test_canary_selection():
    from routellm.quality import QualityManager, CanaryConfig
    
    # 100% weight to ensure it always triggers for testing
    config = CanaryConfig(enabled=True, canary_model="canary-v1", weight=1.0)
    qm = QualityManager(canary_config=config)
    
    assert qm.should_canary() is True
    
    # 0% weight
    config.weight = 0.0
    qm = QualityManager(canary_config=config)
    assert qm.should_canary() is False
