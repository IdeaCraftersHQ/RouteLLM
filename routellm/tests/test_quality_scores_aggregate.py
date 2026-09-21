"""Aggregating scored traces into the quality sidecar.

Arithmetic over the JSONL scores file: a mean per endpoint mapped onto
the [0, 100] scale `Endpoint.quality` already validates, with the
below-threshold endpoints left out rather than written null. No fit
import anywhere in this path, which the subprocess test pins.
"""

import json
import logging
import subprocess
import sys
import textwrap

import pytest
import yaml

from routellm.quality_scores import aggregate_scores


def _row(endpoint="local_fast", score=0.8, **kwargs):
    row = {
        "trace_id": kwargs.pop("trace_id", "t"),
        "endpoint": endpoint,
        "request_model": "coding",
        "tier": kwargs.pop("tier", None),
        "area": kwargs.pop("area", None),
        "cached": kwargs.pop("cached", False),
        "is_canary": kwargs.pop("is_canary", False),
        "score": score,
        "breakdown": {},
        "scorer": "module:x:y",
        "scored_at": "2026-09-20T18:03:11Z",
    }
    row.update(kwargs)
    return row


def _rows(count, **kwargs):
    return [_row(trace_id=f"t{i}", **kwargs) for i in range(count)]


def test_mean_maps_to_the_zero_hundred_scale():
    rows = (
        _rows(10, score=0.9) + _rows(10, score=0.7) + _rows(10, score=0.8)
    )
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    assert sidecar["endpoints"]["local_fast"]["quality"] == 80
    assert sidecar["endpoints"]["local_fast"]["n"] == 30


def test_below_min_samples_is_left_out_entirely():
    rows = _rows(29, score=0.9)
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    # Absent means unrated, which the merge step never touches. A null
    # quality would mean "measured as nothing", which is different.
    assert "local_fast" not in sidecar["endpoints"]


def test_cached_traces_are_dropped():
    rows = _rows(30, score=0.9) + _rows(60, score=0.1, cached=True)
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    assert sidecar["endpoints"]["local_fast"]["quality"] == 90
    assert sidecar["endpoints"]["local_fast"]["n"] == 30


def test_null_scores_are_dropped_and_not_counted_in_n():
    rows = _rows(30, score=0.6) + _rows(20, score=None)
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    assert sidecar["endpoints"]["local_fast"]["n"] == 30
    assert sidecar["endpoints"]["local_fast"]["quality"] == 60


def test_by_area_splits_the_same_endpoint():
    rows = (
        _rows(30, score=0.9, area="coding")
        + _rows(30, score=0.5, area="copywriting")
    )
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    entry = sidecar["endpoints"]["local_fast"]
    assert entry["quality"] == 70
    assert entry["n"] == 60
    assert entry["by_area"]["coding"] == {"quality": 90, "n": 30}
    assert entry["by_area"]["copywriting"] == {"quality": 50, "n": 30}


def test_an_area_under_the_threshold_is_left_out_of_by_area_only():
    rows = (
        _rows(30, score=0.9, area="coding")
        + _rows(5, score=0.1, area="copywriting")
    )
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    entry = sidecar["endpoints"]["local_fast"]
    assert "coding" in entry["by_area"]
    assert "copywriting" not in entry["by_area"]
    # The endpoint's overall number still counts every scored trace.
    assert entry["n"] == 35


def test_percentile_transform_spreads_clustered_scores():
    rows = (
        _rows(30, score=0.50, endpoint="a")
        + _rows(30, score=0.51, endpoint="b")
        + _rows(30, score=0.52, endpoint="c")
    )
    linear = aggregate_scores(rows, min_samples=30, transform="linear")
    spread = aggregate_scores(rows, min_samples=30, transform="percentile")

    assert {e["quality"] for e in linear["endpoints"].values()} == {50, 51, 52}
    qualities = {name: e["quality"] for name, e in spread["endpoints"].items()}
    assert qualities["a"] == 0
    assert qualities["c"] == 100
    assert 0 < qualities["b"] < 100
    assert spread["transform"] == "percentile"


def test_percentile_with_one_endpoint_falls_back_and_warns(caplog):
    rows = _rows(30, score=0.8)

    with caplog.at_level(logging.WARNING, logger="routellm.quality_scores"):
        sidecar = aggregate_scores(rows, min_samples=30, transform="percentile")

    assert sidecar["transform"] == "linear"
    assert sidecar["endpoints"]["local_fast"]["quality"] == 80
    assert [r for r in caplog.records if "percentile" in r.getMessage()]


def test_sidecar_round_trips_through_yaml(tmp_path):
    rows = _rows(30, score=0.72, area="coding")
    sidecar = aggregate_scores(rows, min_samples=30, transform="linear")

    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
    loaded = yaml.safe_load(path.read_text())

    assert loaded == sidecar
    assert loaded["version"] == 1
    assert loaded["source"] == "traces"
    assert loaded["min_samples"] == 30
    # sort_keys=False means the header reads top-down as written.
    assert path.read_text().startswith("version: 1\n")


def test_aggregate_needs_no_fit(tmp_path):
    """Aggregation is arithmetic; it must run where fit is absent."""
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        "\n".join(json.dumps(row) for row in _rows(30, score=0.8)) + "\n"
    )
    out = tmp_path / "quality.yaml"

    script = textwrap.dedent(
        f"""
        import sys
        class _Block:
            def find_module(self, name, path=None):
                if name == "fit" or name.startswith("fit."):
                    raise ImportError("fit is not available here")
                return None
        sys.meta_path.insert(0, _Block())
        from routellm.quality_scores import main
        code = main(["aggregate", "--scores", {str(scores)!r},
                     "--out", {str(out)!r}, "--min-samples", "30"])
        assert "fit" not in sys.modules, sorted(
            m for m in sys.modules if m.startswith("fit")
        )
        sys.exit(code)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(out.read_text())["endpoints"]["local_fast"]["quality"] == 80


def test_an_empty_scores_file_is_an_error(tmp_path):
    from routellm.quality_scores import main

    scores = tmp_path / "scores.jsonl"
    scores.write_text("")

    assert main(["aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml")]) == 1


# ---------------------------------------------------------------------------
# The command line itself
#
# `aggregate_scores` is pure and every test above calls it directly.
# These drive `main(argv)` as a shell does, so a flag that parses but
# never reaches the function cannot pass.
# ---------------------------------------------------------------------------


def _cli(*argv) -> int:
    from routellm.quality_scores import main

    return main(list(argv))


def _scores_file(tmp_path, rows, name="scores.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _sidecar(path):
    return yaml.safe_load(path.read_text())


def test_cli_aggregate_writes_the_sidecar(tmp_path):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    out = tmp_path / "quality.yaml"

    assert _cli("aggregate", "--scores", str(scores), "--out", str(out)) == 0

    sidecar = _sidecar(out)
    assert sidecar["version"] == 1
    assert sidecar["endpoints"]["local_fast"]["quality"] == 80


def test_cli_min_samples_flag_reaches_the_aggregation(tmp_path):
    """With and without, on data that straddles the threshold."""
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    lenient, strict = tmp_path / "a.yaml", tmp_path / "b.yaml"

    assert _cli(
        "aggregate", "--scores", str(scores), "--out", str(lenient),
        "--min-samples", "30",
    ) == 0
    assert _cli(
        "aggregate", "--scores", str(scores), "--out", str(strict),
        "--min-samples", "31",
    ) == 0

    assert "local_fast" in _sidecar(lenient)["endpoints"]
    assert _sidecar(strict)["endpoints"] == {}
    assert _sidecar(strict)["min_samples"] == 31


def test_cli_transform_flag_reaches_the_aggregation(tmp_path):
    rows = (
        _rows(30, score=0.50, endpoint="a")
        + _rows(30, score=0.52, endpoint="c")
    )
    scores = _scores_file(tmp_path, rows)
    linear, spread = tmp_path / "lin.yaml", tmp_path / "pct.yaml"

    assert _cli("aggregate", "--scores", str(scores), "--out", str(linear)) == 0
    assert _cli(
        "aggregate", "--scores", str(scores), "--out", str(spread),
        "--transform", "percentile",
    ) == 0

    assert _sidecar(linear)["transform"] == "linear"
    assert _sidecar(linear)["endpoints"]["a"]["quality"] == 50
    assert _sidecar(spread)["transform"] == "percentile"
    assert _sidecar(spread)["endpoints"]["a"]["quality"] == 0
    assert _sidecar(spread)["endpoints"]["c"]["quality"] == 100


def test_cli_config_flag_rekeys_by_area_from_tier_to_area(tmp_path):
    """The flag's whole job: traces with no area of their own."""
    rows = _rows(30, score=0.9, tier="coding_fast", area=None)
    scores = _scores_file(tmp_path, rows)

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"areas": {"coding": ["coding_fast"]}}, sort_keys=False)
    )

    without, with_ = tmp_path / "w0.yaml", tmp_path / "w1.yaml"
    assert _cli("aggregate", "--scores", str(scores), "--out", str(without)) == 0
    assert _cli(
        "aggregate", "--scores", str(scores), "--out", str(with_),
        "--config", str(config),
    ) == 0

    bare = _sidecar(without)
    assert bare["area_source"] == "tier"
    assert set(bare["endpoints"]["local_fast"]["by_area"]) == {"coding_fast"}

    resolved = _sidecar(with_)
    assert resolved["area_source"] == "config"
    assert set(resolved["endpoints"]["local_fast"]["by_area"]) == {"coding"}


def test_cli_missing_scores_file_exits_nonzero_naming_it(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl"

    code = _cli("aggregate", "--scores", str(missing), "--out", str(tmp_path / "q.yaml"))

    assert code == 1
    assert str(missing) in capsys.readouterr().err


def test_cli_missing_config_file_exits_nonzero_naming_it(tmp_path, capsys):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    missing = tmp_path / "nope.yaml"

    code = _cli(
        "aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml"),
        "--config", str(missing),
    )

    assert code == 1
    assert str(missing) in capsys.readouterr().err


def test_cli_a_bad_transform_value_exits_two(tmp_path):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))

    with pytest.raises(SystemExit) as excinfo:
        _cli(
            "aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml"),
            "--transform", "bogus",
        )

    assert excinfo.value.code == 2


@pytest.mark.parametrize("missing", ["--scores", "--out"])
def test_cli_a_missing_required_flag_exits_two(tmp_path, missing):
    argv = [
        "aggregate",
        "--scores", str(_scores_file(tmp_path, _rows(30))),
        "--out", str(tmp_path / "q.yaml"),
    ]
    index = argv.index(missing)
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as excinfo:
        _cli(*argv)

    assert excinfo.value.code == 2


def test_cli_missing_config_names_the_flag(tmp_path, capsys):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    missing = tmp_path / "nope.yaml"

    code = _cli(
        "aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml"),
        "--config", str(missing),
    )

    err = capsys.readouterr().err
    assert code == 1
    assert str(missing) in err
    assert "--config" in err


def test_cli_a_malformed_config_names_the_file(tmp_path, capsys):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    config = tmp_path / "config.yaml"
    config.write_text("not a mapping\n")

    code = _cli(
        "aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml"),
        "--config", str(config),
    )

    assert code == 1
    assert str(config) in capsys.readouterr().err


def test_cli_a_config_putting_a_tier_in_two_areas_names_both(tmp_path, capsys):
    scores = _scores_file(tmp_path, _rows(30, score=0.8))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"areas": {"coding": ["shared"], "writing": ["shared"]}},
            sort_keys=False,
        )
    )

    code = _cli(
        "aggregate", "--scores", str(scores), "--out", str(tmp_path / "q.yaml"),
        "--config", str(config),
    )

    err = capsys.readouterr().err
    assert code == 1
    assert "shared" in err
    assert "coding" in err and "writing" in err
