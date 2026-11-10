"""Traces keep digests by default; prompt and response text is opt-in."""

import hashlib
import json
import logging
import os
import stat

import pytest

from routellm.quality import FineTuneConfig, QualityManager
from routellm.quality_scores import TraceContentError, score_traces

CANARY_IN = "CANARY-PROMPT-7f3a"
CANARY_OUT = "CANARY-REPLY-91bc"


def _response(text=CANARY_OUT):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _manager(tmp_path, **kwargs):
    config = FineTuneConfig(enabled=True, trace_dir=str(tmp_path / "traces"), **kwargs)
    return QualityManager(fine_tune_config=config)


def _raw(manager):
    directory = manager.fine_tune_config.trace_dir
    return [
        open(os.path.join(directory, n)).read()
        for n in sorted(os.listdir(directory))
        if n.endswith(".json")
    ]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_default_is_capture_off():
    assert FineTuneConfig().capture_content is False


def test_capture_off_writes_no_text(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace(CANARY_IN, "m", _response(), session_id="s")
    (raw,) = _raw(manager)
    assert CANARY_IN not in raw
    assert CANARY_OUT not in raw


def test_capture_off_keeps_digests_and_usage(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace(CANARY_IN, "m", _response(), session_id="s", endpoint="e")
    trace = json.loads(_raw(manager)[0])
    assert trace["input"]["prompt_sha256"] == _sha(CANARY_IN)
    assert trace["input"]["prompt_len"] == len(CANARY_IN)
    assert trace["frontier"]["output_sha256"] == _sha(CANARY_OUT)
    assert trace["frontier"]["output_len"] == len(CANARY_OUT)
    assert "prompt" not in trace["input"]
    assert "output" not in trace["frontier"]
    assert "output" not in trace
    assert trace["frontier"]["usage"]["total_tokens"] == 7
    assert trace["session_id"] == "s"
    assert trace["routellm"]["endpoint"] == "e"
    assert trace["id"] and trace["timestamp"]
    assert trace["metadata"]["content_captured"] is False


def test_capture_on_keeps_text(tmp_path):
    manager = _manager(tmp_path, capture_content=True)
    manager.record_trace(CANARY_IN, "m", _response())
    trace = json.loads(_raw(manager)[0])
    assert trace["input"]["prompt"] == CANARY_IN
    assert trace["frontier"]["output"] == CANARY_OUT
    assert trace["output"]["choices"][0]["message"]["content"] == CANARY_OUT
    assert trace["input"]["prompt_sha256"] == _sha(CANARY_IN)
    assert trace["metadata"]["content_captured"] is True


def test_may_capture_predicate_is_the_single_gate(tmp_path):
    manager = _manager(tmp_path, capture_content=True)
    assert manager._may_capture(None) is True
    manager._may_capture = lambda ctx=None: False
    manager.record_trace(CANARY_IN, "m", _response())
    assert CANARY_IN not in _raw(manager)[0]


def test_capture_on_warns_once_at_startup(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="routellm.quality"):
        manager = _manager(tmp_path, capture_content=True)
        manager.record_trace(CANARY_IN, "m", _response())
        manager.record_trace(CANARY_IN, "m", _response())
    hits = [r for r in caplog.records if "capture_content" in r.getMessage()]
    assert len(hits) == 1


def test_capture_off_does_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="routellm.quality"):
        _manager(tmp_path)
    assert not [r for r in caplog.records if "capture_content" in r.getMessage()]


def test_file_modes(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace("p", "m", _response())
    directory = manager.fine_tune_config.trace_dir
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    (name,) = [n for n in os.listdir(directory)]
    assert stat.S_IMODE(os.stat(os.path.join(directory, name)).st_mode) == 0o600


def test_existing_directory_is_tightened(tmp_path):
    (tmp_path / "traces").mkdir(mode=0o755)
    os.chmod(tmp_path / "traces", 0o755)
    _manager(tmp_path)
    assert stat.S_IMODE(os.stat(tmp_path / "traces").st_mode) == 0o700


def test_scoring_digest_only_traces_fails_clearly(tmp_path):
    manager = _manager(tmp_path)
    manager.record_trace(CANARY_IN, "m", _response())
    with pytest.raises(TraceContentError, match="capture_content"):
        score_traces(manager.fine_tune_config.trace_dir, "rubric:x.yaml", str(tmp_path / "s.jsonl"))
    assert not (tmp_path / "s.jsonl").exists()


def test_server_config_passes_capture_content(tmp_path):
    from routellm.server_config import build_quality_manager

    manager = build_quality_manager(
        {"traces": {"enabled": True, "trace_dir": "t", "capture_content": True}},
        str(tmp_path / "config.yaml"),
    )
    assert manager.fine_tune_config.capture_content is True
    off = build_quality_manager({"traces": {"enabled": True, "trace_dir": "t"}}, None)
    assert off.fine_tune_config.capture_content is False
