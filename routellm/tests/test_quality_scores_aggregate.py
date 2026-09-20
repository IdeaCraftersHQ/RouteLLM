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
