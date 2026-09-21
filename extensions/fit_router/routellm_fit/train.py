"""Build a training dataset from scored traces, and train an advisor.

    python -m routellm_fit.train --scores scores.jsonl --out advisor/
    python -m routellm_fit.train --scores scores.jsonl --out advisor/ --dry-run

The supervision signal is the argmax endpoint per prompt: of every
endpoint that answered a given prompt, the one whose mean score was
highest. A prompt only one endpoint ever answered teaches nothing and
is dropped, because there was no choice to make.

The mean matters rather than the best single trace: one lucky answer
should not outrank an endpoint that is consistently better.

`--dry-run` builds and prints the dataset and stops. It is the path
that works without training deps, mirroring fit's own
`python -m examples.train_advisor --dry-run`.

Training proper runs fit's GRPO, which needs `trl`, `transformers` and
torch. Those are fit's dependencies, not routellm's. fit's
`GRPOTrainer.train` does NOT raise when they are missing: it logs and
falls back to a simplified in-process loop that trains no model and
downloads nothing. That is a reasonable default for fit and a bad one
here, because a command that says it trained an advisor and produced
none is worse than one that refuses, so the deps are checked up front
and their absence is refused with the extra that installs them.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

#: What fit's GRPO actually needs. `trl` is the one usually missing:
#: torch arrives with routellm itself.
TRAINING_MODULES = ("torch", "transformers", "trl")

TRAINING_HINT = (
    "training needs {missing}, which fit's training extra installs: "
    "pip install -e '.[training]'. Use --dry-run to build and inspect "
    "the dataset without them."
)


def load_scores(path: str) -> list:
    """Read a scores JSONL file, skipping malformed lines.

    Parameters
    ----------
    path : str
        The scores file `routellm.quality_scores score` wrote.

    Returns
    -------
    list[dict]
        One row per scored trace.

    Raises
    ------
    FileNotFoundError
        If there is no file at `path`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scores file at {path}")

    rows = []
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                logger.warning("skipping malformed score at %s:%s: %s", path, number, exc)
    return rows


def _means_per_prompt(rows: list) -> dict:
    """Return `{prompt_hash: {endpoint: mean_score}}` over usable rows.

    A row is usable when it was scored, was not served from cache, and
    names both a prompt and an endpoint. Cached rows measure the cache,
    not the endpoint, so training on them would teach the router about
    its own cache.
    """
    totals: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("cached"):
            continue
        score = row.get("score")
        prompt = row.get("prompt_hash")
        endpoint = row.get("endpoint")
        if score is None or not prompt or not endpoint:
            continue
        totals[prompt][endpoint].append(float(score))

    return {
        prompt: {
            endpoint: sum(scores) / len(scores)
            for endpoint, scores in endpoints.items()
        }
        for prompt, endpoints in totals.items()
    }


def best_endpoint_per_prompt(rows: list) -> dict:
    """Return `{prompt_hash: best_endpoint}` by mean score.

    Ties break on the endpoint name, so the same scores file always
    produces the same labels however the rows were ordered on disk.

    Parameters
    ----------
    rows : list[dict]
        Rows of a scores file.

    Returns
    -------
    dict[str, str]
        The argmax endpoint per prompt, over prompts at least two
        endpoints answered.
    """
    labels = {}
    for prompt, means in _means_per_prompt(rows).items():
        if len(means) < 2:
            continue
        labels[prompt] = min(means, key=lambda name: (-means[name], name))
    return labels


def build_examples(rows: list) -> list:
    """Return one training example per comparable prompt.

    Each example carries the label, the reward that won it, and every
    candidate's mean, so a reader can see how close the decision was.

    Parameters
    ----------
    rows : list[dict]
        Rows of a scores file.

    Returns
    -------
    list[dict]
        `{prompt_hash, best_endpoint, reward, candidates}` per prompt,
        ordered by prompt hash.
    """
    means = _means_per_prompt(rows)
    labels = best_endpoint_per_prompt(rows)

    return [
        {
            "prompt_hash": prompt,
            "best_endpoint": labels[prompt],
            "reward": means[prompt][labels[prompt]],
            "candidates": dict(sorted(means[prompt].items())),
        }
        for prompt in sorted(labels)
    ]


def missing_training_modules() -> list:
    """Return the training modules this interpreter cannot import.

    Checked with `find_spec` rather than a real import, so asking the
    question costs nothing when the answer is yes.

    Returns
    -------
    list[str]
        Module names, in `TRAINING_MODULES` order.
    """
    import importlib.util

    missing = []
    for name in TRAINING_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _train_advisor(examples: list, out_dir: str) -> str:
    """Train a fit advisor on the examples and export it.

    Separated from `main` so the refusal path is testable without
    installing gigabytes of training dependencies.

    Parameters
    ----------
    examples : list[dict]
        What `build_examples` returned.
    out_dir : str
        Where to write the advisor.

    Returns
    -------
    str
        The output directory.

    Raises
    ------
    ImportError
        If any of `TRAINING_MODULES` is absent. Raised here rather
        than left to fit, whose trainer catches its own ImportError
        and silently degrades to a loop that trains nothing.
    """
    missing = missing_training_modules()
    if missing:
        raise ImportError(", ".join(missing))

    from fit.training import DatasetBuilder, GRPOConfig, GRPOTrainer
    from fit.training.tracer import TraceRecord

    records = [
        TraceRecord(
            id=example["prompt_hash"],
            session_id=example["prompt_hash"],
            timestamp="",
            prompt=example["prompt_hash"],
            context={"candidates": example["candidates"]},
            advice_text=example["best_endpoint"],
            advice_domain="routellm",
            advice_confidence=float(example["reward"]),
            frontier_output="",
            frontier_model=example["best_endpoint"],
            reward_score=float(example["reward"]),
            reward_breakdown={},
            metadata={},
        )
        for example in examples
    ]

    dataset = DatasetBuilder(records).build()
    trainer = GRPOTrainer(GRPOConfig())
    trainer.train(dataset)
    trainer.save(out_dir)
    return out_dir


def main(argv: Optional[list] = None) -> int:
    """Run the CLI, returning its exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m routellm_fit.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores", required=True, help="scores JSONL")
    parser.add_argument("--out", required=True, help="advisor output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print the dataset, then stop; needs no torch",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_scores(args.scores)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    examples = build_examples(rows)
    if not examples:
        print(
            "error: no prompt in "
            f"{args.scores} was answered by two or more endpoints, so "
            "there is no choice to learn. Score more traffic, or route "
            "the same prompts through more than one endpoint.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(f"{len(examples)} training examples")
        for example in examples:
            candidates = ", ".join(
                f"{name}={mean:.3f}"
                for name, mean in example["candidates"].items()
            )
            print(
                f"  {example['prompt_hash']} -> {example['best_endpoint']} "
                f"(reward {example['reward']:.3f}; {candidates})"
            )
        return 0

    try:
        out = _train_advisor(examples, args.out)
    except ImportError as exc:
        print(
            "error: " + TRAINING_HINT.format(missing=exc),
            file=sys.stderr,
        )
        return 1

    print(f"wrote advisor to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
