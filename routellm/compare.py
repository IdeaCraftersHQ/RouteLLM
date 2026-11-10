"""Offline replay: run a dataset through N models and record every call.

    python -m routellm.compare replay --dataset chats.jsonl \\
        --models cloud_strong,local_glm --config config.yaml

Each dataset line is a JSON object: either a full chat request
(`{"messages": [...], "temperature": 0.2, ...}`) or `{"prompt": "text"}`,
which becomes one user message. Every row goes to every model, and each
call is recorded in the trace store exactly as a live comparison is
(role `candidate`, one shared comparison id per row), so the traces feed
the same scoring and aggregation as live traffic. One result row per
dataset row lands in `--out`, and a per-model summary is printed.

The config supplies the endpoints, and its `comparison:` section the
timeout, budget and judge; `comparison.enabled` is not consulted, since
replay is an explicit run, not served traffic. Exit codes: 0 done,
2 a usage or dataset problem (nothing is called when the dataset or the
model selection is wrong).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

from routellm.capabilities import catalog_records
from routellm.comparison import ComparisonConfig, ComparisonManager, JudgeConfig
from routellm.config import ConfigError, load_config
from routellm.endpoints import EndpointRegistry
from routellm.quality import FineTuneConfig, QualityManager
from routellm.server_config import build_comparison_config

logger = logging.getLogger(__name__)


class DatasetError(ValueError):
    """A dataset line that is not a usable chat request."""


def load_dataset(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a jsonl dataset into chat request dicts.

    Parameters
    ----------
    path : str
        The jsonl file.
    limit : int, optional
        Keep only this many rows.

    Raises
    ------
    DatasetError
        On a line that is not a JSON object, or that carries neither
        `messages` nor `prompt`. The message names the line.
    """
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise DatasetError(f"line {number}: not valid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"line {number}: expected a JSON object")
            if "messages" not in row:
                if not isinstance(row.get("prompt"), str):
                    raise DatasetError(f"line {number}: needs `messages` or a string `prompt`")
                prompt = row.pop("prompt")
                row["messages"] = [{"role": "user", "content": prompt}]
            for key in ("model", "models", "stream"):
                row.pop(key, None)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


async def replay(
    manager: ComparisonManager,
    rows: list[dict[str, Any]],
    models: list[str],
    *,
    tier: str | None = None,
    concurrency: int = 1,
    measure_ttft: bool | None = None,
) -> list[dict[str, Any]]:
    """Run every row through every model; one result dict per row, in order."""
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            body = await manager.compare(models, row, tier=tier, measure_ttft=measure_ttft)
        return {"index": index, **body}

    return list(await asyncio.gather(*(one(i, row) for i, row in enumerate(rows))))


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-model means over a replay: latency, tokens per second, cost, errors."""
    stats: dict[str, dict[str, list]] = {}
    for result in results:
        for item in result["results"]:
            slot = stats.setdefault(
                item["endpoint"], {"latency": [], "tps": [], "ttft": [], "cost": [], "errors": []}
            )
            metrics = item["metrics"]
            if item["error"]:
                slot["errors"].append(item["error"])
                continue
            for key, field in (
                ("latency", "latency_ms"),
                ("tps", "tokens_per_second"),
                ("ttft", "ttft_ms"),
                ("cost", "cost_usd"),
            ):
                if metrics[field] is not None:
                    slot[key].append(metrics[field])

    def mean(values: list) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    return {
        name: {
            "calls": len(slot["latency"]) + len(slot["errors"]),
            "errors": len(slot["errors"]),
            "latency_ms": mean(slot["latency"]),
            "ttft_ms": mean(slot["ttft"]),
            "tokens_per_second": mean(slot["tps"]),
            "cost_usd": round(sum(slot["cost"]), 6) if slot["cost"] else None,
        }
        for name, slot in sorted(stats.items())
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m routellm.compare",
        description="Run a jsonl dataset of chat requests through N models and record each call.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("replay", help="replay a dataset through several models")
    rep.add_argument("--dataset", required=True, help="jsonl file of chat requests")
    rep.add_argument("--models", required=True, help="comma-separated endpoint names or a profile")
    rep.add_argument("--config", default=None, help="explicit config file, merged last")
    rep.add_argument("--trace-dir", default=".routellm_traces", help="where traces are recorded")
    rep.add_argument("--out", default=None, help="jsonl file for the per-row results")
    rep.add_argument("--tier", default=None, help="apply this tier's privacy rules")
    rep.add_argument("--judge", default=None, help="scorer spec, overriding comparison.judge")
    rep.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    rep.add_argument("--concurrency", type=int, default=1, help="rows in flight at once")
    rep.add_argument("--measure-ttft", action="store_true", help="stream upstream to time TTFT")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)

    def fail(message: str) -> int:
        print(f"error: {message}", file=sys.stderr)
        return 2

    try:
        loaded = load_config(explicit=args.config)
        origin = str(loaded.layers[-1].path) if loaded.layers else None
        registry = EndpointRegistry.from_config(loaded.data or {}, config_path=origin)
        config = build_comparison_config(loaded.data)
        raw = config.model_dump()
        raw["enabled"] = True
        if args.judge:
            raw["judge"] = JudgeConfig(scorer=args.judge).model_dump()
        config = ComparisonConfig(**raw)
        rows = load_dataset(args.dataset, args.limit)
    except (ConfigError, FileNotFoundError, DatasetError, ValueError) as exc:
        return fail(str(exc))

    host = SimpleNamespace(
        endpoints=registry,
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(enabled=True, trace_dir=args.trace_dir)
        ),
        api_base=None,
        api_key=None,
        _catalog_records=catalog_records(registry),
    )
    try:
        manager = ComparisonManager(config, host)
        manager.check_models(args.models, args.tier)
    except ValueError as exc:
        return fail(str(exc))

    results = asyncio.run(
        replay(
            manager,
            rows,
            args.models,
            tier=args.tier,
            concurrency=args.concurrency,
            measure_ttft=True if args.measure_ttft else None,
        )
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")

    print(f"{len(rows)} rows x {len(next(iter(results), {'results': []})['results'])} models")
    header = f"{'endpoint':<20} {'calls':>5} {'errors':>6} {'latency_ms':>11} {'ttft_ms':>8} {'tok/s':>8} {'cost_usd':>10}"
    print(header)
    for name, row in summarize(results).items():
        print(
            f"{name:<20} {row['calls']:>5} {row['errors']:>6} {_cell(row['latency_ms']):>11} "
            f"{_cell(row['ttft_ms']):>8} {_cell(row['tokens_per_second']):>8} "
            f"{_cell(row['cost_usd']):>10}"
        )
    return 0


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


if __name__ == "__main__":
    sys.exit(main())
