"""Scoring recorded traces offline with a fit scorer.

Covers the `--scorer` spec grammar `build_scorer` resolves, the traces
reader, and the JSONL scores file `score` writes. No test here makes a
network call: `judge:` is the only spec that would, and it is refused
without `--allow-llm`.
"""

import json
import logging

import pytest

from routellm.quality_scores import (
    build_scorer,
    load_traces,
    score_traces,
)


def _trace(trace_id: str, prompt: str = "hello", output: str = "hi", **routellm):
    block = {
        "endpoint": "local_fast",
        "request_model": "coding",
        "tier": "coding_fast",
        "area": None,
        "path": [],
        "cached": False,
        "is_canary": False,
        "latency_ms": 12,
        "router": "jev",
        "win_rate": 0.8,
    }
    block.update(routellm)
    return {
        "id": trace_id,
        "session_id": "sess",
        "timestamp": "2026-09-20T18:03:11Z",
        "input": {"prompt": prompt, "context": {}},
        "frontier": {"model": "m", "provider": "", "output": output, "usage": {}},
        "reward": {"score": None, "breakdown": {}},
        "routellm": block,
        "metadata": {},
    }


def _write(directory, *traces):
    directory.mkdir(parents=True, exist_ok=True)
    for trace in traces:
        (directory / f"{trace['id']}.json").write_text(json.dumps(trace))
    return directory


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_composite_spec_scores_every_trace(tmp_path):
    pytest.importorskip("fit")

    traces = _write(
        tmp_path / "traces", _trace("trace-1"), _trace("trace-2"), _trace("trace-3")
    )
    out = tmp_path / "scores.jsonl"

    written = score_traces(str(traces), "composite:accuracy,relevance,safety", str(out))

    assert written == 3
    rows = _rows(out)
    assert {row["trace_id"] for row in rows} == {"trace-1", "trace-2", "trace-3"}
    for row in rows:
        assert row["endpoint"] == "local_fast"
        assert row["scorer"] == "composite:accuracy,relevance,safety"
        assert row["score"] == pytest.approx(0.5)
        assert set(row["breakdown"]) == {"accuracy", "relevance", "safety"}


def test_missing_fit_names_the_extra(monkeypatch):
    """A machine without fit is told which extra installs it."""
    import importlib

    real = importlib.import_module

    def _refuse(name, *args, **kwargs):
        if name == "fit" or name.startswith("fit."):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(
        "routellm.quality_scores.importlib.import_module", _refuse
    )

    with pytest.raises(RuntimeError) as excinfo:
        build_scorer("composite:accuracy")

    message = str(excinfo.value)
    assert "scoring needs fit: pip install 'routellm[fit]'" in message
    # The original ImportError rides along, so a broken install is
    # distinguishable from a missing one.
    assert "fit" in message.split("routellm[fit]'")[1]
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_judge_spec_refused_without_allow_llm(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        build_scorer("judge:claude-sonnet-4", allow_llm=False, trace_count=42)

    message = str(excinfo.value)
    assert "--allow-llm" in message
    assert "42" in message


def test_rubric_scorer_is_deterministic(tmp_path, monkeypatch):
    pytest.importorskip("fit")

    def _explode(*args, **kwargs):
        raise AssertionError("a rubric scorer must not touch the network")

    import httpx

    monkeypatch.setattr(httpx, "Client", _explode)
    monkeypatch.setattr(httpx, "post", _explode)

    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(
        "patterns:\n  - [\"\\\\bbecause\\\\b\", 0.6]\n  - [\"\\\\bstep\\\\b\", 0.4]\n"
    )

    scorer = build_scorer(f"rubric:{rubric}")
    first = scorer.score("because of the first step", {})
    second = scorer.score("because of the first step", {})

    assert first.score == second.score
    assert first.breakdown == second.breakdown

    # Deterministic is not the same as constant: a rubric that scores
    # every output alike measures nothing, so the score has to move
    # with what the patterns actually match.
    both = scorer.score("because of the first step", {})
    one = scorer.score("because it is so", {})
    neither = scorer.score("no rubric word here", {})

    assert both.score == pytest.approx(1.0)
    assert one.score == pytest.approx(0.6)
    assert neither.score == pytest.approx(0.0)
    assert set(both.breakdown) == {r"\bbecause\b", r"\bstep\b"}
    assert neither.breakdown == {}


def test_unknown_spec_lists_the_four_forms():
    with pytest.raises(ValueError) as excinfo:
        build_scorer("nonsense:whatever")

    message = str(excinfo.value)
    for form in ("composite:", "rubric:", "module:", "judge:"):
        assert form in message


def test_rescore_off_skips_already_scored_ids(tmp_path):
    traces = _write(tmp_path / "traces", _trace("trace-1"), _trace("trace-2"))
    out = tmp_path / "scores.jsonl"
    spec = f"module:{__name__}:constant_scorer"

    assert score_traces(str(traces), spec, str(out)) == 2
    assert score_traces(str(traces), spec, str(out)) == 0
    assert len(_rows(out)) == 2

    assert score_traces(str(traces), spec, str(out), rescore=True) == 2
    assert len(_rows(out)) == 4


def test_unreadable_trace_is_skipped_with_a_warning(tmp_path, caplog):
    traces = _write(tmp_path / "traces", _trace("trace-1"))
    (traces / "trace-broken.json").write_text("{not json")
    (traces / "trace-partial.json.tmp").write_text("{}")

    with caplog.at_level(logging.WARNING, logger="routellm.quality_scores"):
        loaded = list(load_traces(str(traces)))

    assert [trace["id"] for trace in loaded] == ["trace-1"]
    warnings = [r for r in caplog.records if "trace-broken" in r.getMessage()]
    assert len(warnings) == 1


def test_null_score_is_written_as_null(tmp_path):
    traces = _write(tmp_path / "traces", _trace("trace-1"))
    out = tmp_path / "scores.jsonl"

    score_traces(str(traces), f"module:{__name__}:null_scorer", str(out))

    row = _rows(out)[0]
    assert row["score"] is None
    assert '"score": null' in out.read_text()


def test_an_empty_traces_directory_is_an_error(tmp_path):
    empty = tmp_path / "traces"
    empty.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        score_traces(str(empty), f"module:{__name__}:constant_scorer", str(tmp_path / "s.jsonl"))

    assert str(empty) in str(excinfo.value)


class _Reward:
    def __init__(self, score, breakdown=None):
        self.score = score
        self.breakdown = breakdown or {}
        self.metadata = {}


class _ConstantScorer:
    def score(self, output, context):
        return _Reward(0.75, {"constant": 0.75})


class _NullScorer:
    def score(self, output, context):
        return _Reward(None)


def constant_scorer():
    return _ConstantScorer()


def null_scorer():
    return _NullScorer()
