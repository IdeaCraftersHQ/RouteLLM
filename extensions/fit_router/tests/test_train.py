"""Building the training dataset from scored traces.

The label is the argmax endpoint per prompt: of every endpoint that
answered a given prompt, the one whose mean score was highest. That is
the whole supervision signal, so the tests pin the argmax, the tie
break, and the rows the builder is allowed to ignore.
"""

import json

import pytest

from routellm_fit.train import (
    _means_per_prompt,
    best_endpoint_per_prompt,
    build_examples,
    load_scores,
    main,
)


def _row(endpoint, score, prompt_hash="p1", **kwargs):
    row = {
        "trace_id": kwargs.pop("trace_id", f"{endpoint}-{score}"),
        "endpoint": endpoint,
        "request_model": "coding",
        "tier": "coding_fast",
        "area": "coding",
        "cached": False,
        "is_canary": False,
        "score": score,
        "breakdown": {},
        "scorer": "module:x:y",
        "scored_at": "2026-09-20T18:03:11Z",
        "prompt_hash": prompt_hash,
    }
    row.update(kwargs)
    return row


def _scores_file(tmp_path, rows):
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_the_label_is_the_argmax_endpoint():
    rows = [
        _row("weak_one", 0.2),
        _row("strong_one", 0.9),
        _row("middling", 0.5),
    ]

    assert best_endpoint_per_prompt(rows) == {"p1": "strong_one"}


def test_the_argmax_is_over_the_mean_not_a_single_trace():
    """One lucky answer must not outrank a consistently better endpoint."""
    rows = [
        _row("lucky", 1.0, trace_id="a"),
        _row("lucky", 0.0, trace_id="b"),
        _row("steady", 0.8, trace_id="c"),
        _row("steady", 0.8, trace_id="d"),
    ]

    assert best_endpoint_per_prompt(rows) == {"p1": "steady"}


def test_each_prompt_gets_its_own_label():
    rows = [
        _row("alpha", 0.9, prompt_hash="p1"),
        _row("beta", 0.1, prompt_hash="p1"),
        _row("alpha", 0.1, prompt_hash="p2"),
        _row("beta", 0.9, prompt_hash="p2"),
    ]

    assert best_endpoint_per_prompt(rows) == {"p1": "alpha", "p2": "beta"}


def test_ties_are_broken_deterministically():
    """Same means, so the name decides; twice in a row, same answer."""
    rows = [_row("zulu", 0.7), _row("alpha", 0.7), _row("mike", 0.7)]

    first = best_endpoint_per_prompt(rows)
    second = best_endpoint_per_prompt(list(reversed(rows)))

    assert first == second == {"p1": "alpha"}


def test_cached_and_null_rows_are_ignored():
    """Both would win on their face value, and neither may."""
    rows = [
        _row("real", 0.6),
        _row("other_real", 0.5),
        _row("cache_only", 1.0, cached=True),
        _row("unscored", None),
    ]

    means = _means_per_prompt(rows)["p1"]
    assert set(means) == {"real", "other_real"}
    assert best_endpoint_per_prompt(rows) == {"p1": "real"}


def test_a_prompt_answered_by_one_endpoint_is_dropped():
    """With nothing to compare against there is no choice to learn."""
    rows = [_row("only", 0.9, prompt_hash="p1"), _row("only", 0.8, prompt_hash="p1")]

    assert best_endpoint_per_prompt(rows) == {}


def test_rows_without_a_prompt_hash_are_dropped():
    rows = [_row("a", 0.9, prompt_hash=None), _row("b", 0.1, prompt_hash=None)]

    assert best_endpoint_per_prompt(rows) == {}


def test_build_examples_carries_the_label_and_the_reward():
    rows = [
        _row("alpha", 0.9, prompt_hash="p1"),
        _row("beta", 0.1, prompt_hash="p1"),
    ]

    examples = build_examples(rows)

    assert len(examples) == 1
    example = examples[0]
    assert example["prompt_hash"] == "p1"
    assert example["best_endpoint"] == "alpha"
    assert example["reward"] == pytest.approx(0.9)
    assert example["candidates"] == {"alpha": pytest.approx(0.9), "beta": pytest.approx(0.1)}


def test_load_scores_skips_a_malformed_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(_row("a", 0.5)) + "\n{not json\n")

    rows = load_scores(str(path))

    assert len(rows) == 1


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_dry_run_prints_the_dataset_without_torch(tmp_path, capsys):
    """`--dry-run` is the path that must work with no training deps."""
    import sys

    scores = _scores_file(
        tmp_path,
        [_row("alpha", 0.9, prompt_hash="p1"), _row("beta", 0.1, prompt_hash="p1")],
    )

    code = main(["--scores", str(scores), "--out", str(tmp_path / "adv"), "--dry-run"])

    assert code == 0
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "1" in out
    assert "torch" not in sys.modules


def test_cli_dry_run_writes_no_advisor(tmp_path):
    scores = _scores_file(
        tmp_path,
        [_row("alpha", 0.9, prompt_hash="p1"), _row("beta", 0.1, prompt_hash="p1")],
    )
    out = tmp_path / "adv"

    assert main(["--scores", str(scores), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_cli_missing_scores_exits_nonzero_naming_it(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl"

    code = main(["--scores", str(missing), "--out", str(tmp_path / "adv")])

    assert code == 1
    assert str(missing) in capsys.readouterr().err


def test_cli_no_comparable_prompts_exits_nonzero(tmp_path, capsys):
    """Every prompt answered by one endpoint teaches nothing."""
    scores = _scores_file(tmp_path, [_row("only", 0.9, prompt_hash="p1")])

    code = main(["--scores", str(scores), "--out", str(tmp_path / "adv"), "--dry-run"])

    assert code == 1
    assert "prompt" in capsys.readouterr().err.lower()


def test_missing_training_modules_reports_what_is_absent(monkeypatch):
    import routellm_fit.train as train

    monkeypatch.setattr(train, "TRAINING_MODULES", ("json", "no_such_module_xyz"))

    assert train.missing_training_modules() == ["no_such_module_xyz"]


def test_training_refuses_up_front_when_a_dep_is_missing(tmp_path, monkeypatch):
    """fit's own trainer would degrade silently; this must not.

    `GRPOTrainer.train` catches its ImportError and falls back to a
    loop that trains no model, so a check that relied on fit raising
    would produce an empty advisor and report success.
    """
    import routellm_fit.train as train

    monkeypatch.setattr(train, "TRAINING_MODULES", ("no_such_module_xyz",))

    with pytest.raises(ImportError) as excinfo:
        train._train_advisor([{"prompt_hash": "p1", "best_endpoint": "a",
                               "reward": 0.9, "candidates": {}}], str(tmp_path))

    assert "no_such_module_xyz" in str(excinfo.value)


def test_cli_missing_training_deps_names_them_and_the_extra(
    tmp_path, capsys, monkeypatch
):
    import routellm_fit.train as train

    monkeypatch.setattr(train, "TRAINING_MODULES", ("no_such_module_xyz",))

    scores = _scores_file(
        tmp_path,
        [_row("alpha", 0.9, prompt_hash="p1"), _row("beta", 0.1, prompt_hash="p1")],
    )
    out = tmp_path / "adv"

    code = main(["--scores", str(scores), "--out", str(out)])

    assert code == 1
    err = capsys.readouterr().err
    assert "no_such_module_xyz" in err
    assert "training" in err
    # A refusal writes nothing: a half-made advisor is worse than none.
    assert not out.exists()


def test_cli_missing_required_flag_exits_two(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--out", str(tmp_path / "adv")])

    assert excinfo.value.code == 2
