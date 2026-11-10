"""The offline replay CLI: a jsonl dataset through N models, recorded.

Local fake upstreams only; the same trace recording the server uses.
"""

import json

import pytest
import yaml

from routellm import compare as replay_cli
from routellm.tests.fake_upstream import FakeUpstream


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")


@pytest.fixture
def ups():
    made = {name: FakeUpstream(name) for name in "abc"}
    yield made
    for fake in made.values():
        fake.server.shutdown()
        fake.server.server_close()


def setup(tmp_path, ups, tags=None, comparison=None, rows=None):
    tags = tags or {}
    config = {
        "endpoints": {
            name: {
                "model": f"openai/model-{name}",
                "api_base": ups[name].base,
                "api_key_env": "FAKE_KEY",
                **({"tags": tags[name]} if name in tags else {}),
            }
            for name in ups
        },
        "tiers": {"default": {"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}},
    }
    if comparison is not None:
        config["comparison"] = comparison
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    rows = rows or [
        {"messages": [{"role": "user", "content": "one"}]},
        {"prompt": "two"},
        {"messages": [{"role": "user", "content": "three"}], "temperature": 0.2},
    ]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return [
        "replay",
        "--config",
        str(tmp_path / "config.yaml"),
        "--dataset",
        str(tmp_path / "data.jsonl"),
        "--trace-dir",
        str(tmp_path / "traces"),
        "--out",
        str(tmp_path / "out.jsonl"),
    ]


def traces(tmp_path):
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "traces").glob("*.json"))]


def test_replay_runs_every_row_through_every_model(tmp_path, ups, capsys):
    argv = setup(tmp_path, ups)
    assert replay_cli.main([*argv, "--models", "a,b"]) == 0

    assert len(ups["a"].hits) == 3 and len(ups["b"].hits) == 3
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert all(sorted(x["endpoint"] for x in r["results"]) == ["a", "b"] for r in rows)
    assert ups["b"].hits[1][2]["messages"] == [{"role": "user", "content": "two"}]
    assert ups["a"].hits[2][2]["temperature"] == 0.2

    out = capsys.readouterr().out
    assert "a" in out and "b" in out and "latency" in out


def test_replay_records_into_the_trace_store(tmp_path, ups):
    argv = setup(tmp_path, ups)
    replay_cli.main([*argv, "--models", "a,b"])
    rows = traces(tmp_path)
    assert len(rows) == 6
    assert {t["routellm"]["comparison"]["role"] for t in rows} == {"candidate"}
    assert len({t["routellm"]["comparison"]["id"] for t in rows}) == 3
    assert all(t["routellm"]["comparison"]["metrics"]["latency_ms"] is not None for t in rows)


def test_replay_works_with_comparison_left_disabled_in_the_config(tmp_path, ups):
    argv = setup(tmp_path, ups, comparison={"enabled": False})
    assert replay_cli.main([*argv, "--models", "a,b"]) == 0


def test_replay_limit(tmp_path, ups):
    argv = setup(tmp_path, ups)
    replay_cli.main([*argv, "--models", "a,b", "--limit", "1"])
    assert len(ups["a"].hits) == 1


def test_replay_scores_with_a_judge_spec(tmp_path, ups):
    argv = setup(tmp_path, ups)
    spec = "module:routellm.tests.fake_scorer:make"
    replay_cli.main([*argv, "--models", "a,b", "--judge", spec])
    assert all(t["reward"]["score"] is not None for t in traces(tmp_path))


def test_replay_rejects_a_bad_dataset_line(tmp_path, ups, capsys):
    argv = setup(tmp_path, ups)
    (tmp_path / "data.jsonl").write_text('{"prompt": "ok"}\nnot json\n')
    assert replay_cli.main([*argv, "--models", "a,b"]) == 2
    assert "line 2" in capsys.readouterr().err
    assert ups["a"].hits == []


def test_replay_rejects_an_unknown_model(tmp_path, ups, capsys):
    argv = setup(tmp_path, ups)
    assert replay_cli.main([*argv, "--models", "a,ghost"]) == 2
    assert "ghost" in capsys.readouterr().err


def test_replay_needs_two_models(tmp_path, ups, capsys):
    argv = setup(tmp_path, ups)
    assert replay_cli.main([*argv, "--models", "a"]) == 2


def test_replay_obeys_tier_privacy(tmp_path, ups, capsys):
    tags = {"a": ["local"], "b": ["local"], "c": ["cloud"]}
    argv = setup(tmp_path, ups, tags=tags)
    assert replay_cli.main([*argv, "--models", "a,c", "--tier", "default"]) == 2
    assert ups["a"].hits == [] and ups["c"].hits == []
    assert "local" in capsys.readouterr().err


def test_replay_obeys_the_budget(tmp_path, ups):
    argv = setup(tmp_path, ups, comparison={"budget_usd": 0.0})
    assert replay_cli.main([*argv, "--models", "a,b"]) == 0
    assert ups["a"].hits == []
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert rows[0]["results"][0]["error"] == "budget_exhausted"
