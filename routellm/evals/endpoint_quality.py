"""Measure each endpoint's quality and write the sidecar a config reads.

A script, not a library. It runs the existing coding benchmark against
each named endpoint through `Controller.completion`, with
`strong_model` and `weak_model` both set to that one endpoint, so no
router runs and every call lands where it is meant to. The benchmark's
own scorer produces an accuracy, which is scaled linearly onto the
0-100 `quality` scale a `quality_desc` selector orders on::

    python -m routellm.evals.endpoint_quality --config c.yaml \\
        --endpoints cloud_strong,local_server --limit 50 --out quality.yaml

COST: this spends REAL TOKENS on REAL PROVIDERS. `--limit` defaults to
50 prompts per endpoint, so a run over four endpoints is 200 paid
completions plus whatever the benchmark's scorer costs. Start small.

The sidecar it writes is the shared format `routellm.quality_scores`
reads, tagged `source: evals`. Scored live traffic produces the same
shape with a different `source`, and a config referencing either with
`quality_from:` cannot tell them apart, which is the point.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["main"]

#: Prompts per endpoint when `--limit` names none.
DEFAULT_LIMIT = 50

#: How a raw accuracy in [0, 1] becomes a quality in [0, 100]. Named in
#: the sidecar so a later reader knows what the number means.
TRANSFORM = "linear"


def _load_config(config_path: str) -> dict[str, Any]:
    """Return the parsed YAML config at `config_path`."""
    import yaml

    with open(config_path) as handle:
        return yaml.safe_load(handle) or {}


def _build_controller(config: dict[str, Any], name: str, config_path: str):
    """Return a controller pinned to one endpoint on both sides.

    Both sides naming the same endpoint means no router decision can
    send a prompt anywhere else, which is what makes the resulting
    score a measurement OF that endpoint rather than of a policy.
    """
    from routellm.caching import CacheConfig
    from routellm.controller import Controller
    from routellm.endpoints import EndpointRegistry

    registry = EndpointRegistry.from_config(
        {"endpoints": config.get("endpoints") or {}}, config_path=config_path
    )

    return Controller(
        routers=["random"],
        strong_model=name,
        weak_model=name,
        endpoints=registry,
        cache_config=CacheConfig(enabled=False),
    )


def _score_endpoint(controller, name: str, limit: int) -> tuple[float, int]:
    """Return one endpoint's raw accuracy and the sample count.

    Runs the coding benchmark from `routellm.evals.benchmarks` and
    scores with the benchmark's own scorer, so the number is
    comparable with everything else that benchmark produces.

    Parameters
    ----------
    controller : Controller
        A controller pinned to this one endpoint on both sides.
    name : str
        The endpoint's name, for logs.
    limit : int
        How many prompts to run.

    Returns
    -------
    tuple[float, int]
        Accuracy in [0, 1], and the number of prompts actually scored.
    """
    from routellm.evals.benchmarks import GSM8K

    benchmark = GSM8K(routed_pair=None, overwrite_cache=[])

    scored = 0
    correct = 0
    for prompt, expected in _prompts(benchmark, limit):
        try:
            response = controller.completion(
                model=name, messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            logger.warning("endpoint %s failed one prompt: %s", name, exc)
            continue

        scored += 1
        if (
            benchmark.check(answer, expected)
            if hasattr(benchmark, "check")
            else (str(expected).strip() in str(answer))
        ):
            correct += 1

    if scored == 0:
        return 0.0, 0
    return correct / scored, scored


def _prompts(benchmark, limit: int):
    """Yield `(prompt, expected)` pairs from a benchmark, up to `limit`."""
    rows = getattr(benchmark, "all_data", None)
    if rows is None:
        return

    for index, row in enumerate(rows.itertuples()):
        if index >= limit:
            return
        yield getattr(row, "prompt", ""), getattr(row, "answer", "")


def _sidecar(scores: dict[str, tuple[float, int]], limit: int) -> dict[str, Any]:
    """Return the sidecar payload for a run's raw scores."""
    return {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_samples": limit,
        "transform": TRANSFORM,
        "source": "evals",
        "endpoints": {
            name: {
                "quality": max(0, min(100, round(accuracy * 100))),
                "n": samples,
                "by_area": {},
            }
            for name, (accuracy, samples) in scores.items()
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Measure each named endpoint and write the quality sidecar.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        0 on success, 1 when an endpoint is not configured or the run
        could not be written.
    """
    parser = argparse.ArgumentParser(
        prog="python -m routellm.evals.endpoint_quality",
        description=(
            "Measure each endpoint on a small coding set and write the "
            "quality sidecar a config references with quality_from:."
        ),
        epilog=(
            "This spends real tokens on real providers: --limit prompts "
            "per endpoint, each a paid completion. Start small."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--endpoints",
        required=True,
        help="Comma-separated endpoint names to measure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Prompts per endpoint. Default {DEFAULT_LIMIT}.",
    )
    parser.add_argument("--out", required=True, help="Sidecar path to write.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    try:
        config = _load_config(args.config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    configured = set((config.get("endpoints") or {}))
    names = [name.strip() for name in args.endpoints.split(",") if name.strip()]

    unknown = [name for name in names if name not in configured]
    if unknown:
        print(
            f"Not configured in {args.config}: {', '.join(unknown)}. "
            f"Configured endpoints: {', '.join(sorted(configured)) or '<none>'}",
            file=sys.stderr,
        )
        return 1

    if not names:
        print("--endpoints named nothing to measure", file=sys.stderr)
        return 1

    scores: dict[str, tuple[float, int]] = {}
    for name in names:
        logger.info("measuring %s on %d prompts", name, args.limit)
        controller = _build_controller(config, name, args.config)
        scores[name] = _score_endpoint(controller, name, args.limit)

    import yaml

    payload = _sidecar(scores, args.limit)
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(payload, sort_keys=False))
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
