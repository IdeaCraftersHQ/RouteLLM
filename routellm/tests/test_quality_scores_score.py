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

    traces = _write(tmp_path / "traces", _trace("trace-1"), _trace("trace-2"), _trace("trace-3"))
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

    monkeypatch.setattr("routellm.quality_scores.importlib.import_module", _refuse)

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
    rubric.write_text('patterns:\n  - ["\\\\bbecause\\\\b", 0.6]\n  - ["\\\\bstep\\\\b", 0.4]\n')

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


def test_the_rubric_breakdown_agrees_with_the_score(tmp_path):
    """fit matches case-insensitively, so the breakdown must too."""
    pytest.importorskip("fit")

    rubric = tmp_path / "rubric.yaml"
    rubric.write_text('patterns:\n  - ["\\\\bfirst\\\\b", 1.0]\n')

    scorer = build_scorer(f"rubric:{rubric}")
    reward = scorer.score("First, read the spec", {})

    assert reward.score == pytest.approx(1.0)
    assert reward.breakdown == {r"\bfirst\b": 1.0}


# ---------------------------------------------------------------------------
# The command line itself
#
# Every test above calls the functions behind the CLI. These drive
# `main(argv)` the way a shell does, because a flag that parses but
# never reaches the function it names passes every test above.
# ---------------------------------------------------------------------------


def _cli(*argv) -> int:
    from routellm.quality_scores import main

    return main(list(argv))


def _traces_dir(tmp_path, count=4):
    return _write(
        tmp_path / "traces",
        *[_trace(f"trace-{i}", output="hi") for i in range(count)],
    )


def test_cli_score_writes_the_file_it_was_given(tmp_path):
    traces = _traces_dir(tmp_path)
    out = tmp_path / "elsewhere" / "scores.jsonl"

    code = _cli(
        "score",
        "--traces",
        str(traces),
        "--scorer",
        f"module:{__name__}:constant_scorer",
        "--out",
        str(out),
    )

    assert code == 0
    assert len(_rows(out)) == 4


def test_cli_score_defaults_out_beside_the_traces(tmp_path):
    traces = _traces_dir(tmp_path)

    assert (
        _cli(
            "score",
            "--traces",
            str(traces),
            "--scorer",
            f"module:{__name__}:constant_scorer",
        )
        == 0
    )
    assert len(_rows(tmp_path / "scores.jsonl")) == 4


def test_cli_limit_actually_limits(tmp_path):
    traces = _traces_dir(tmp_path)
    out = tmp_path / "scores.jsonl"

    assert (
        _cli(
            "score",
            "--traces",
            str(traces),
            "--scorer",
            f"module:{__name__}:constant_scorer",
            "--out",
            str(out),
            "--limit",
            "2",
        )
        == 0
    )
    assert len(_rows(out)) == 2


def test_cli_rescore_flag_changes_the_row_count(tmp_path):
    traces = _traces_dir(tmp_path, count=2)
    out = tmp_path / "scores.jsonl"
    argv = [
        "score",
        "--traces",
        str(traces),
        "--scorer",
        f"module:{__name__}:constant_scorer",
        "--out",
        str(out),
    ]

    assert _cli(*argv) == 0
    assert len(_rows(out)) == 2

    # Without the flag the second run is a no-op; with it, every trace
    # is scored again.
    assert _cli(*argv) == 0
    assert len(_rows(out)) == 2

    assert _cli(*argv, "--rescore") == 0
    assert len(_rows(out)) == 4


def test_cli_missing_traces_dir_exits_nonzero_naming_it(tmp_path, capsys):
    missing = tmp_path / "nope"

    code = _cli(
        "score",
        "--traces",
        str(missing),
        "--scorer",
        f"module:{__name__}:constant_scorer",
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    assert str(missing) in capsys.readouterr().err


def test_cli_unknown_scorer_exits_nonzero_listing_the_forms(tmp_path, capsys):
    code = _cli(
        "score",
        "--traces",
        str(_traces_dir(tmp_path)),
        "--scorer",
        "nonsense:x",
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    err = capsys.readouterr().err
    for form in ("composite:", "rubric:", "module:", "judge:"):
        assert form in err


def test_cli_judge_without_allow_llm_exits_nonzero(tmp_path, capsys):
    code = _cli(
        "score",
        "--traces",
        str(_traces_dir(tmp_path)),
        "--scorer",
        "judge:some-model",
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    assert "--allow-llm" in capsys.readouterr().err


@pytest.mark.parametrize(
    "spec, needle",
    [
        ("rubric:/definitely/not/here.yaml", "/definitely/not/here.yaml"),
        ("module:no.such.module:factory", "no.such.module"),
        ("module:json:not_a_name", "not_a_name"),
        ("module:json:JSONDecoder", "score"),
    ],
)
def test_cli_a_broken_scorer_spec_names_the_offending_input(tmp_path, capsys, spec, needle):
    """Every bad spec is a sentence and an exit code, never a traceback."""
    code = _cli(
        "score",
        "--traces",
        str(_traces_dir(tmp_path)),
        "--scorer",
        spec,
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    assert needle in capsys.readouterr().err


def test_cli_a_malformed_rubric_names_the_file(tmp_path, capsys):
    pytest.importorskip("fit")

    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("just a string\n")

    code = _cli(
        "score",
        "--traces",
        str(_traces_dir(tmp_path)),
        "--scorer",
        f"rubric:{rubric}",
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    assert str(rubric) in capsys.readouterr().err


def test_cli_a_rubric_with_a_bad_regex_names_the_pattern(tmp_path, capsys):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text('patterns:\n  - ["[unclosed", 1.0]\n')

    code = _cli(
        "score",
        "--traces",
        str(_traces_dir(tmp_path)),
        "--scorer",
        f"rubric:{rubric}",
        "--out",
        str(tmp_path / "s.jsonl"),
    )

    assert code == 1
    assert "[unclosed" in capsys.readouterr().err


def test_cli_a_missing_required_flag_exits_two(tmp_path):
    """argparse's own exit, kept distinct from our exit 1."""
    with pytest.raises(SystemExit) as excinfo:
        _cli("score", "--traces", str(tmp_path))

    assert excinfo.value.code == 2


def test_cli_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        _cli()

    assert excinfo.value.code == 2


def test_cli_allow_llm_lets_a_judge_spec_through(tmp_path, monkeypatch):
    """The refusal test only proves the default; this proves the flag.

    fit's judge is replaced with a local stand-in, so the flag is
    exercised without any spec that would actually call an LLM.
    """
    import routellm.quality_scores as module

    built = {}
    real = module.build_scorer

    def _spy(spec, allow_llm=False, trace_count=0):
        built["allow_llm"] = allow_llm
        if spec.startswith("judge:"):
            if not allow_llm:
                return real(spec, allow_llm=allow_llm, trace_count=trace_count)
            return _ConstantScorer()
        return real(spec, allow_llm=allow_llm, trace_count=trace_count)

    monkeypatch.setattr(module, "build_scorer", _spy)

    traces = _traces_dir(tmp_path, count=2)
    out = tmp_path / "scores.jsonl"

    assert (
        _cli(
            "score",
            "--traces",
            str(traces),
            "--scorer",
            "judge:some-model",
            "--out",
            str(out),
            "--allow-llm",
        )
        == 0
    )
    assert built["allow_llm"] is True
    assert len(_rows(out)) == 2
