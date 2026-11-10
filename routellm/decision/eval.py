"""Eval CLI: labelled JSONL in, per-class precision / recall and latency out.

    python -m routellm.decision.eval --config cfg.yaml --data labelled.jsonl \\
        [--min-recall private=0.95] [--model-only]

Each row is `{"text": "...", "data_class": "<label>"}` or
`{"messages": [...], "data_class": "<label>"}`; an optional `"intent"`
adds an intent confusion table. Floors come from `--min-recall` and
`decision.eval_min_recall`; exit status is 1 when any named class's
recall is under its floor, 2 on a usage or config error.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from routellm.decision.classifier import DecisionClassifier
from routellm.decision.models import DecisionRefused

REFUSED = "<refused>"


def load_rows(path: str) -> list[dict[str, Any]]:
    """Read a labelled JSONL file; blank lines are skipped."""
    rows = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not JSON") from exc
            if "data_class" not in row or not ("text" in row or "messages" in row):
                raise ValueError(f"{path}:{number}: needs data_class and text or messages")
            rows.append(row)
    return rows


async def run_eval(
    classifier: DecisionClassifier, rows: list[dict[str, Any]], use_recognizers: bool = True
) -> dict[str, Any]:
    """Classify every row and return the report as a dict.

    Rows are independent samples, so the sticky floor is off for the run.
    """
    classifier.sticky = None
    results = []
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["text"]}]
        started = time.perf_counter()
        try:
            decision = await classifier.classify(messages, use_recognizers=use_recognizers)
            predicted, intent, error = decision.data_class, decision.intent, decision.error
        except DecisionRefused as refusal:
            predicted, intent, error = REFUSED, None, refusal.error.kind
        results.append(
            {
                "expected": row["data_class"],
                "predicted": predicted,
                "expected_intent": row.get("intent"),
                "intent": intent,
                "error": error,
                "ms": (time.perf_counter() - started) * 1000,
            }
        )
    return summarize(results, list(classifier.config.data_classes))


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def summarize(results: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """Fold per-row results into per-class metrics, a confusion table, latency."""
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        confusion[r["expected"]][r["predicted"]] += 1
    per_class = {}
    for name in classes:
        tp = confusion[name][name]
        support = sum(confusion[name].values())
        predicted = sum(row[name] for row in confusion.values())
        per_class[name] = {
            "support": support,
            "recall": tp / support if support else None,
            "precision": tp / predicted if predicted else None,
        }
    times = [r["ms"] for r in results]
    intent_rows = [r for r in results if r["expected_intent"] is not None]
    return {
        "total": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "intent_accuracy": (
            sum(r["intent"] == r["expected_intent"] for r in intent_rows) / len(intent_rows)
            if intent_rows
            else None
        ),
        "latency_ms": (
            {f"p{p}": _percentile(times, p) for p in (50, 90, 99)} | {"max": max(times)}
            if times
            else {}
        ),
    }


def failed_floors(report: dict[str, Any], floors: dict[str, float]) -> list[str]:
    """Return a message per class whose recall is under its floor."""
    out = []
    for name, floor in floors.items():
        stats = report["per_class"].get(name)
        recall = stats["recall"] if stats else None
        if recall is None or recall < floor:
            shown = "no labelled rows" if recall is None else f"{recall:.3f}"
            out.append(f"recall for {name!r} is {shown}, floor {floor}")
    return out


def render(report: dict[str, Any]) -> str:
    """Return the human-readable report."""
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"rows {report['total']}  errors {report['errors']}", ""]
    lines.append(f"{'class':<16}{'support':>8}{'recall':>9}{'precision':>11}")
    for name, s in report["per_class"].items():
        lines.append(f"{name:<16}{s['support']:>8}{fmt(s['recall']):>9}{fmt(s['precision']):>11}")
    lines += ["", "confusion (expected -> predicted):"]
    for expected, row in report["confusion"].items():
        lines.append(f"  {expected}: " + ", ".join(f"{k}={v}" for k, v in sorted(row.items())))
    if report["intent_accuracy"] is not None:
        lines.append(f"\nintent accuracy {report['intent_accuracy']:.3f}")
    if report["latency_ms"]:
        lines.append(
            "latency ms: " + "  ".join(f"{k} {v:.0f}" for k, v in report["latency_ms"].items())
        )
    return "\n".join(lines)


def _parse_floors(items: list[str]) -> dict[str, float]:
    floors = {}
    for item in items:
        name, _, value = item.partition("=")
        floors[name] = float(value)
    return floors


def main(argv: list[str] | None = None, classifier: DecisionClassifier | None = None) -> int:
    """CLI entry; `classifier` lets tests supply a bound one."""
    parser = argparse.ArgumentParser(prog="routellm.decision.eval", description=__doc__)
    parser.add_argument("--config", help="config file; default is the discovered chain")
    parser.add_argument("--data", required=True, help="labelled JSONL")
    parser.add_argument("--min-recall", action="append", default=[], metavar="CLASS=FLOOR")
    parser.add_argument("--model-only", action="store_true", help="skip the recognizers")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    try:
        if classifier is None:
            from routellm.config import load_config
            from routellm.decision.build import build_decision
            from routellm.decision.transport import litellm_transport
            from routellm.endpoints import EndpointRegistry

            data = load_config(explicit=args.config).data
            registry = EndpointRegistry.from_config(data)
            classifier = build_decision(data, registry)
            if classifier is None:
                raise ValueError("config has no `decision:` section")
            classifier.bind(litellm_transport(registry))
        floors = {**classifier.config.eval_min_recall, **_parse_floors(args.min_recall)}
        rows = load_rows(args.data)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = asyncio.run(run_eval(classifier, rows, use_recognizers=not args.model_only))
    print(json.dumps(report, indent=2) if args.json else render(report))
    failures = failed_floors(report, floors)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
