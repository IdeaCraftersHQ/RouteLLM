"""Traces in fit's `trace-format-v1` shape, carrying the routing path.

Covers the trace body written by `QualityManager.record_trace`, the
`routellm` block that names the decision that produced it, rotation
against the file and byte caps.
"""

import json
import logging
import os

import pytest

from routellm.quality import FineTuneConfig, QualityManager


def _response(text: str = "hi") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _manager(tmp_path, **kwargs) -> QualityManager:
    config = FineTuneConfig(
        enabled=True, trace_dir=str(tmp_path / "traces"), **kwargs
    )
    return QualityManager(fine_tune_config=config)


def _traces(manager: QualityManager) -> list[dict]:
    directory = manager.fine_tune_config.trace_dir
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as handle:
            out.append(json.load(handle))
    return out


def test_trace_has_every_fit_required_field(tmp_path):
    """A written trace survives fit's own ingester with its text intact."""
    pytest.importorskip("fit")
    from fit.training import TraceIngester

    manager = _manager(tmp_path)
    manager.record_trace(
        "what is 2+2",
        "local_fast",
        _response("four"),
        session_id="sess-1",
    )

    records = TraceIngester().load_batch(manager.fine_tune_config.trace_dir)
    assert len(records) == 1
    record = records[0]
    assert record.prompt == "what is 2+2"
    assert record.frontier_output == "four"
    assert record.frontier_model == "local_fast"


def test_routellm_block_carries_the_path_and_endpoint(tmp_path):
    manager = _manager(tmp_path)
    path = [
        {"tier": "coding", "router": "jev", "win_rate": 0.4, "picked": "coding_fast"},
        {"tier": "coding_fast", "router": "jev", "win_rate": 0.81, "picked": "local_fast"},
    ]
    manager.record_trace(
        "hello",
        "local_fast",
        _response(),
        {"cached": False, "is_canary": True},
        path=path,
        request_model="coding",
        endpoint="local_fast",
        session_id="sess-9",
        latency_ms=1830,
        provider="ollama",
        area="coding",
    )

    block = _traces(manager)[0]["routellm"]
    assert block["path"] == path
    assert block["endpoint"] == "local_fast"
    assert block["request_model"] == "coding"
    assert block["tier"] == "coding_fast"
    assert block["router"] == "jev"
    assert block["win_rate"] == 0.81
    assert block["area"] == "coding"
    assert block["latency_ms"] == 1830
    assert block["cached"] is False
    assert block["is_canary"] is True


def test_old_top_level_fields_are_unchanged(tmp_path):
    """The two pre-existing tests read `output` and `routed_model`."""
    manager = _manager(tmp_path)
    response = _response()
    manager.record_trace("hello", "m1", response)

    trace = _traces(manager)[0]
    assert trace["output"] == response
    assert trace["routed_model"] == "m1"
    assert trace["input"]["prompt"] == "hello"


def test_disabled_writes_nothing(tmp_path):
    directory = tmp_path / "traces"
    manager = QualityManager(
        fine_tune_config=FineTuneConfig(trace_dir=str(directory))
    )
    manager.record_trace("hello", "m1", _response())
    assert not directory.exists()


def test_file_cap_deletes_oldest_first(tmp_path):
    manager = _manager(tmp_path, max_files=3)
    for index in range(5):
        manager.record_trace(f"prompt-{index}", "m1", _response())

    prompts = [trace["input"]["prompt"] for trace in _traces(manager)]
    assert sorted(prompts) == ["prompt-2", "prompt-3", "prompt-4"]


def test_byte_cap_trips_once_and_warns(tmp_path, caplog):
    manager = _manager(tmp_path, max_bytes=200)
    manager.record_trace("first", "m1", _response())

    with caplog.at_level(logging.WARNING, logger="routellm.quality"):
        manager.record_trace("second", "m1", _response())
    first = [record for record in caplog.records if "max_bytes" in record.message]
    assert len(first) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="routellm.quality"):
        manager.record_trace("third", "m1", _response())
    assert not [record for record in caplog.records if "max_bytes" in record.message]


def test_ids_do_not_collide_within_a_millisecond(tmp_path):
    manager = _manager(tmp_path)
    for _ in range(50):
        manager.record_trace("hello", "m1", _response())

    files = os.listdir(manager.fine_tune_config.trace_dir)
    assert len(files) == 50


def test_a_broken_response_still_records(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace("hello", "m1", {"choices": []})

    trace = _traces(manager)[0]
    assert trace["frontier"]["output"] == ""
    assert trace["frontier"]["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_controller_passes_the_path_into_the_trace(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    from litellm.utils import ModelResponse

    from routellm.controller import Controller

    response = ModelResponse(
        **{
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "created": 0,
            "model": "m",
            "object": "chat.completion",
        }
    )

    with patch("routellm.controller.completion", MagicMock(return_value=response)):
        controller = Controller(
            routers=["random"],
            strong_model="strong-1",
            weak_model="weak-1",
            progress_bar=False,
        )
        controller.quality_manager = _manager(tmp_path)
        res = controller.completion(
            router="random",
            threshold=0.5,
            messages=[{"role": "user", "content": "hello"}],
        )

    trace = _traces(controller.quality_manager)[0]
    assert trace["routellm"]["path"] == res._hidden_params["routellm_path"]
    assert trace["routellm"]["latency_ms"] is not None
