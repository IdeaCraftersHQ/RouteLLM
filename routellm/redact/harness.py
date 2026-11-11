"""Recall, leak and re-identification harness for the redact treatment.

    python -m routellm.redact.harness --config server.yaml --out report.json

Measures the analyzers the config's `redact:` section builds against a
labelled corpus, runs the treatment end to end against a recording
provider, and writes a report stamped with the digest of the recognizer
config (see `routellm.redact.gate`). Exit status 0 when everything
passes, 1 when a threshold or check fails, 2 when it could not run.

Corpus: `tests/fixtures/redact/*.jsonl` (synthetic fr and en text), one
`{id, lang, text, spans: [{start, end, category, text}]}` per line;
`canaries.jsonl` is the leak and re-identification set, not scored.
A category with no entry in `THRESHOLDS` is reported, never gated: a
recognizer gained later is measured as soon as it has fixtures and gated
once it has a row.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routellm.config import ConfigError
from routellm.redact.gate import REPORT_SCHEMA, config_digest, digest_material
from routellm.redact.harness_checks import (
    linkable,
    placeholders,
    read_artifact,
    reidentifying,
    scan_artifacts,
)
from routellm.redact.harness_world import CONTROL, build_messages, run_world

__all__ = ["THRESHOLDS", "Doc", "load_corpus", "run", "score"]

CORPUS_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "redact"
CANARIES = "canaries.jsonl"

#: Minimum `(recall, precision)` per category and engine. The one table
#: that gates: a category under either number fails the report. The
#: pattern recognizers are rules, so their bar is exact; named-entity
#: recognition and presidio's phone recognizer are statistical, and their
#: bar is set under what the pinned models measure.
_RULES: dict[str, tuple[float, float]] = {
    "sin": (1.0, 0.9),
    "ramq": (1.0, 0.9),
    "postal_code": (1.0, 0.9),
    "amount": (1.0, 0.9),
    "email": (1.0, 0.9),
    "client_id": (1.0, 0.9),
    "date_of_birth": (0.9, 0.9),
    "phone": (1.0, 0.9),
    # Number-led phrases ("2024 Capital Drive") read as addresses:
    # over-detection, the safe side, so precision is not gated.
    "address": (1.0, 0.0),
}
THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "patterns": dict(_RULES),
    # Precision is not gated for names and places: the corpus labels one
    # kind of entity per line, and a model that also tags the city in a
    # postal-code line is over-detecting, which is the safe side.
    "presidio": {
        **_RULES,
        "person": (0.9, 0.0),
        "location": (0.9, 0.0),
        "phone": (0.9, 0.8),
    },
}
_MATCH_IOU = 0.5


@dataclass(frozen=True)
class Doc:
    id: str
    lang: str
    text: str
    spans: tuple[dict, ...]


def _doc(raw: dict, where: str) -> Doc:
    for span in raw["spans"]:
        if raw["text"][span["start"] : span["end"]] != span["text"]:
            raise ValueError(f"{where}: {raw['id']}: span {span} does not match the text")
    return Doc(raw["id"], raw["lang"], raw["text"], tuple(raw["spans"]))


def _load(path: Path) -> list[Doc]:
    return [
        _doc(json.loads(line), path.name)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_corpus(directory: Path | None = None) -> list[Doc]:
    """Every scored document under `directory` (the leak set excluded)."""
    directory = Path(directory or CORPUS_DIR)
    files = sorted(p for p in directory.glob("*.jsonl") if p.name != CANARIES)
    if not files:
        raise FileNotFoundError(f"no corpus (*.jsonl) under {directory}")
    return [d for f in files for d in _load(f)]


def load_canaries(directory: Path | None = None) -> list[Doc]:
    return _load(Path(directory or CORPUS_DIR) / CANARIES)


def _iou(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    return inter / (max(a["end"], b["end"]) - min(a["start"], b["start"]))


def score(docs: Iterable[Doc], analyze: Callable[[str, str], list[dict]]) -> dict[str, dict]:
    """Per category: `{gold, predicted, tp, fn, fp, recall, precision}`.

    A predicted span is right when it has the gold span's category and
    overlaps it by at least half (intersection over union); each gold span
    is claimed once. Every other prediction is a false positive of its own
    category, wherever it fell, so a negative document costs precision.
    A rate over nothing is `None`.
    """
    tally: dict[str, dict[str, int]] = {}

    def row(cat):
        return tally.setdefault(cat, {"gold": 0, "predicted": 0, "tp": 0})

    for doc in docs:
        preds = analyze(doc.text, doc.lang)
        claimed: set[int] = set()
        for g in doc.spans:
            row(g["category"])["gold"] += 1
            for i, p in enumerate(preds):
                if i not in claimed and p["category"] == g["category"] and _iou(g, p) >= _MATCH_IOU:
                    claimed.add(i)
                    row(g["category"])["tp"] += 1
                    break
        for p in preds:
            row(p["category"])["predicted"] += 1
    out = {}
    for cat, r in sorted(tally.items()):
        out[cat] = {
            "gold": r["gold"],
            "predicted": r["predicted"],
            "tp": r["tp"],
            "fn": r["gold"] - r["tp"],
            "fp": r["predicted"] - r["tp"],
            "recall": r["tp"] / r["gold"] if r["gold"] else None,
            "precision": r["tp"] / r["predicted"] if r["predicted"] else None,
        }
    return out


def judge(scores: dict[str, dict], engine: str) -> dict[str, dict]:
    """Add each category's threshold and verdict; an ungated category passes."""
    table = THRESHOLDS.get(engine, {})
    rows = {}
    for cat, s in scores.items():
        want = table.get(cat)
        row = {**s, "threshold": None, "passed": True}
        if want is not None:
            row["threshold"] = {"recall": want[0], "precision": want[1]}
            row["passed"] = (
                s["recall"] is not None
                and s["recall"] >= want[0]
                and ((s["predicted"] == 0 and want[1] == 0) or (s["precision"] or 0) >= want[1])
            )
        rows[cat] = row
    for cat, want in table.items():  # a gated category the corpus never mentions
        rows.setdefault(
            cat,
            {
                "gold": 0,
                "predicted": 0,
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "recall": None,
                "precision": None,
                "threshold": {"recall": want[0], "precision": want[1]},
                "passed": False,
            },
        )
    return rows


def _values(docs: Iterable[Doc]) -> list[str]:
    return [s["text"] for d in docs for s in d.spans]


def _check(passed: bool, **detail: Any) -> dict:
    return {"passed": passed, **detail}


def end_to_end(redact: dict, engine: str, canaries: Sequence[Doc], work: Path) -> dict[str, dict]:
    """The leak, re-identification and linkability checks, from one run."""
    gated = set(THRESHOLDS.get(engine, {}))
    docs = [d for d in canaries if {s["category"] for s in d.spans} <= gated]
    reid = [d for d in docs if {s["category"] for s in d.spans} >= {"date_of_birth", "postal_code"}]
    leak_docs = [d for d in docs if d not in reid]
    if not leak_docs or not reid:
        raise ConfigError(f"no canary fixtures for the categories gated for {engine}")
    values = _values(docs)
    conversation = build_messages([d.text for d in leak_docs])
    plan: dict[str, dict] = {
        "leak": {"class": "pii", "messages": conversation},
        "leak_stream": {"class": "pii", "messages": conversation, "stream": True},
        "control": {
            "class": "open",
            "messages": [{"role": "user", "content": f"Dossier {CONTROL}"}],
        },
    }
    for i, d in enumerate(reid):
        plan[f"reid_{i}"] = {"class": "pii", "messages": [{"role": "user", "content": d.text}]}
    for tag in ("a", "b"):
        plan[f"link_{tag}"] = {
            "class": "pii",
            "messages": [{"role": "user", "content": reid[0].text}],
        }
    world = run_world(redact, plan, work)
    sent = {name: "\n".join(reqs) for name, reqs in world["scenarios"].items()}
    files = world["files"]

    artifacts = {
        "outbound": "\n".join(v for k, v in sent.items() if k != "control"),
        "logs": "\n".join([read_artifact(files["logs"]), world["stdout"], world["stderr"]]),
        "traces": read_artifact(files["traces"]),
        "cache": read_artifact(files["cache"]),
        "audit": read_artifact(files["audit"]),
    }
    encrypted = (redact.get("store") or {}).get("encryption", "none") != "none"
    if encrypted:
        artifacts["token_store"] = read_artifact(files["token_store"])
    found = scan_artifacts(artifacts, values)
    # The scan must be able to see: the control text is in the clear in the
    # raw class's trace and cache, and goes out as is.
    blind = [
        name
        for name, text in (
            ("outbound", sent["control"]),
            ("traces", artifacts["traces"]),
            ("cache", artifacts["cache"]),
        )
        if CONTROL not in text
    ]
    last = conversation[-1]["content"]
    replies = [world["replies"][n] for n in ("leak", "leak_stream")]
    restored = all(r["status"] == 200 and r["text"] == f"echo: {last}" for r in replies)
    leak = _check(
        not found and not blind and restored,
        scanned=sorted(artifacts),
        values=len(values),
        found=found,
        blind=blind,
        client_restored=restored,
    )

    reid_sent = [sent[f"reid_{i}"] for i in range(len(reid))]
    flagged = reidentifying(reid_sent, _values(reid))
    reident = _check(not flagged, requests=len(reid_sent), flagged=len(flagged))

    pa, pb = placeholders(sent["link_a"]), placeholders(sent["link_b"])
    shared = linkable([pa, pb])
    link = _check(not shared and bool(pa) and bool(pb), shared=shared, placeholders=len(pa))
    return {"leak": leak, "reidentification": reident, "linkability": link}


def run(
    redact: dict | None,
    *,
    corpus: Path | None = None,
    config_path: str | None = None,
    work: Path | None = None,
) -> dict:
    """Measure the config's analyzers and the treatment; return the report."""
    from routellm.redact.config import _analyzer

    redact = dict(redact or {})
    engine = redact.get("engine", "presidio")
    languages = tuple(redact.get("languages") or ("fr", "en"))
    analyze = _analyzer(redact, languages)
    categories = judge(score(load_corpus(corpus), analyze), engine)
    with tempfile.TemporaryDirectory(prefix="redact-harness-") as tmp:
        checks = end_to_end(redact, engine, load_canaries(corpus), work or Path(tmp))
    failed = [c for c, r in categories.items() if not r["passed"]]
    failed += [n for n, c in checks.items() if not c["passed"]]
    return {
        "schema": REPORT_SCHEMA,
        "digest": config_digest(redact),
        "config": digest_material(redact),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": not failed,
        "failed": failed,
        "categories": categories,
        "checks": checks,
    }


def _table(report: dict) -> str:
    def pct(x):
        return "  -  " if x is None else f"{x:5.2f}"

    lines = ["category        gold   recall  precision  need(r/p)   "]
    for cat, r in report["categories"].items():
        t = r["threshold"]
        need = f"{t['recall']:.2f}/{t['precision']:.2f}" if t else "ungated"
        mark = "ok  " if r["passed"] else "FAIL"
        lines.append(
            f"{cat:<15} {r['gold']:>4}   {pct(r['recall'])}   {pct(r['precision'])}    "
            f"{need:<10} {mark}"
        )
    for name, c in report["checks"].items():
        lines.append(f"{name:<15} {'ok' if c['passed'] else 'FAIL'}")
    lines.append(f"digest {report['digest']}  {'PASS' if report['passed'] else 'FAIL'}")
    return "\n".join(lines)


def _load_config(path: Path) -> dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    redact = data.get("redact")
    if not isinstance(redact, dict):
        raise ConfigError(f"{path}: no `redact:` section")
    return redact


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m routellm.redact.harness", description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="server config with `redact:`")
    ap.add_argument("--out", required=True, type=Path, help="where to write the JSON report")
    ap.add_argument("--corpus", type=Path, help="fixture directory (default: the shipped one)")
    args = ap.parse_args(argv)
    try:
        report = run(_load_config(args.config), corpus=args.corpus, config_path=str(args.config))
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"harness: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(_table(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
