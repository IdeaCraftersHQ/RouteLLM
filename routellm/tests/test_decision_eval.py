"""Eval CLI: recall per class, confusion table, latency, floor exit code."""

import asyncio
import json

from routellm.decision.eval import main, run_eval
from routellm.tests.decision_fakes import FakeTransport, answer, make_classifier


def labelled(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def by_text(table):
    """A transport answering from the request text; unknown text -> general."""

    def reply(messages):
        text = messages[-1]["content"]
        for needle, cls in table.items():
            if needle in text:
                return answer(cls, "chat")
        return answer("general", "chat")

    return FakeTransport(reply)


ROWS = [
    {"text": "alpha one", "data_class": "private"},
    {"text": "alpha two", "data_class": "private"},
    {"text": "beta three", "data_class": "private"},  # model misses this one
    {"text": "plain talk", "data_class": "general"},
]


def test_recall_precision_and_confusion(tmp_path):
    clf, _ = make_classifier(by_text({"alpha": "private"}))
    report = asyncio.run(run_eval(clf, ROWS))
    private = report["per_class"]["private"]
    assert private["support"] == 3
    assert abs(private["recall"] - 2 / 3) < 1e-9
    assert private["precision"] == 1.0
    assert report["confusion"]["private"] == {"private": 2, "general": 1}
    assert {"p50", "p90", "p99", "max"} <= set(report["latency_ms"])


def test_cli_prints_recall_and_fails_under_floor(tmp_path, capsys):
    clf, _ = make_classifier(by_text({"alpha": "private"}))
    code = main(["--data", labelled(tmp_path, ROWS), "--min-recall", "private=0.9"], clf)
    out = capsys.readouterr()
    assert code == 1
    assert "private" in out.out and "0.667" in out.out
    assert "recall for 'private'" in out.err


def test_cli_passes_at_or_above_floor(tmp_path, capsys):
    clf, _ = make_classifier(by_text({"alpha": "private", "beta": "private"}))
    assert main(["--data", labelled(tmp_path, ROWS), "--min-recall", "private=1.0"], clf) == 0


def test_floor_from_config(tmp_path):
    clf, _ = make_classifier(by_text({"alpha": "private"}), eval_min_recall={"private": 0.9})
    assert main(["--data", labelled(tmp_path, ROWS)], clf) == 1


def test_failures_count_as_most_restrictive(tmp_path):
    clf, _ = make_classifier(FakeTransport(ConnectionError("down")))
    report = asyncio.run(run_eval(clf, ROWS))
    assert report["errors"] == 4
    assert report["per_class"]["private"]["recall"] == 1.0
    assert report["per_class"]["general"]["recall"] == 0.0


def test_bad_data_file_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "x"}\n')
    clf, _ = make_classifier()
    assert main(["--data", str(bad)], clf) == 2
    assert "data_class" in capsys.readouterr().err


def test_floor_on_class_with_no_rows_fails(tmp_path):
    clf, _ = make_classifier(by_text({}))
    rows = [{"text": "x", "data_class": "general"}]
    assert main(["--data", labelled(tmp_path, rows), "--min-recall", "private=0.5"], clf) == 1
