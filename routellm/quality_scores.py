"""Score recorded traces offline and aggregate them into a sidecar.

Two steps, both offline, neither of them making a model call on the
serving path:

    python -m routellm.quality_scores score --traces <dir> --scorer <spec>
    python -m routellm.quality_scores aggregate --scores <file> --out quality.yaml

`score` reads the JSON traces `QualityManager.record_trace` wrote and
runs a fit `RewardScorer` over the `(prompt, output)` pairs already on
disk. fit's `Session.run` is deliberately not used: it calls the
adapter, which would re-run the LLM on traffic that already happened.
`scorer.score(output, context)` is called directly instead.

This is not the evals harness. `routellm/evals/evaluate.py` imports
matplotlib, pandas, pandarallel and psutil at module load, wants a
labelled benchmark with ground truth, and drives a Controller to make
live calls. It still owns benchmark evaluation. This module scores
traffic that already happened, against no ground truth at all.

fit is imported inside `build_scorer` and nowhere else, so `aggregate`
runs on a machine that has never heard of fit.
"""

import argparse
import fnmatch
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

FIT_HINT = "scoring needs fit: pip install 'routellm[fit]'"

SCORER_FORMS = (
    "composite:<dim>,<dim>,...  fit's built-in dimension scorers",
    "rubric:<path.yaml>         regex patterns and weights, deterministic",
    "module:<pkg.mod>:<factory> your own scorer",
    "judge:<model>              an LLM judge, needs --allow-llm",
)


def resolve_sidecar_path(
    quality_from: str, config_path: Optional[Union[str, Path]]
) -> Path:
    """Return the sidecar path `quality_from` names.

    A relative path is resolved against the CONFIG FILE's directory,
    not the current working directory: a config is a document that
    refers to its neighbours, and a server started from elsewhere must
    read the same file the operator wrote next to it.

    Parameters
    ----------
    quality_from : str
        The path as the config writes it.
    config_path : str or Path, optional
        The config file this came from. None falls back to the CWD,
        which is only ever the case for a registry built in memory.

    Returns
    -------
    Path
        The resolved sidecar path.
    """
    candidate = Path(quality_from)
    if candidate.is_absolute() or config_path is None:
        return candidate

    return Path(config_path).resolve().parent / candidate


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _import_fit(name: str):
    """Import a fit module, naming the extra when it is absent.

    Parameters
    ----------
    name : str
        Dotted module name under `fit`.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    RuntimeError
        If fit is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f"{FIT_HINT} ({exc})") from exc


class RubricScorer:
    """A `RewardScorer` over fit's regex `RubricJudgeReward`.

    fit's reward functions are callables shaped `(context, advice,
    output) -> float`; a scorer is an object shaped `score(output,
    context) -> Reward`. This adapts the first to the second so a
    rubric can be passed anywhere a scorer is expected. routellm gives
    no advice, so that argument is always empty.

    Deterministic and offline: the underlying reward is regex matching
    with weights, and nothing here opens a socket.
    """

    def __init__(self, reward_fn, patterns: list):
        self._reward_fn = reward_fn
        self._patterns = patterns

    def score(self, output: str, context: dict):
        """Return the rubric's `Reward` for one output."""
        from fit.types import Reward

        # fit's RewardFn is `(context, advice, output)`. routellm gives
        # no advice, so the middle argument is empty.
        value = float(self._reward_fn(context.get("prompt", ""), "", output))
        # fit's RubricJudgeReward matches case-insensitively by
        # default, so the breakdown has to as well: a breakdown that
        # disagrees with the score it explains is worse than none.
        breakdown = {
            pattern: weight
            for pattern, weight in self._patterns
            if re.search(pattern, output or "", re.IGNORECASE)
        }
        return Reward(score=value, breakdown=breakdown, metadata={"scorer": "rubric"})


def _build_rubric(path: str):
    """Build a `RubricScorer` from a `{patterns: [[regex, weight]]}` file.

    Every way the file can be wrong is reported as a ValueError naming
    the path, so a typo in a rubric reaches the operator as a sentence
    rather than a traceback.

    Parameters
    ----------
    path : str
        The rubric YAML.

    Returns
    -------
    RubricScorer
        The scorer.

    Raises
    ------
    ValueError
        If the file is missing, unreadable, not a mapping, carries no
        usable `patterns:`, or a pattern is not a `[regex, weight]`
        pair.
    RuntimeError
        If fit is not installed.
    """
    import yaml

    if not os.path.exists(path):
        raise ValueError(
            f"No rubric file at {path}. A rubric is a YAML mapping with "
            "a `patterns:` list of [regex, weight] pairs."
        )

    try:
        with open(path) as handle:
            spec = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unreadable rubric at {path}: {exc}") from exc

    if not isinstance(spec, dict):
        raise ValueError(
            f"Malformed rubric at {path}: expected a mapping with a "
            f"`patterns:` key, found {type(spec).__name__}."
        )

    raw = spec.get("patterns")
    if not raw:
        raise ValueError(
            f"Rubric at {path} has no `patterns:`. Without patterns "
            "every output scores 0.0, which measures nothing."
        )

    patterns = []
    for entry in raw:
        try:
            pattern, weight = entry
            patterns.append((str(pattern), float(weight)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed pattern {entry!r} in rubric {path}: each "
                f"entry is a [regex, weight] pair ({exc})."
            ) from exc

    for pattern, _ in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex {pattern!r} in rubric {path}: {exc}"
            ) from exc

    reward_fn = _import_fit("fit.training.reward_fn")
    return RubricScorer(reward_fn.RubricJudgeReward(patterns), patterns)


def build_scorer(spec: str, allow_llm: bool = False, trace_count: int = 0):
    """Return the scorer a `--scorer` spec names.

    Parameters
    ----------
    spec : str
        One of the four forms listed in `SCORER_FORMS`.
    allow_llm : bool, optional
        Whether a spec that calls an LLM is permitted (default False).
    trace_count : int, optional
        How many traces are about to be scored, quoted back in the
        refusal of a `judge:` spec so the cost is visible before it is
        paid (default 0).

    Returns
    -------
    object
        Something with `score(output, context) -> Reward`.

    Raises
    ------
    ValueError
        If the spec is not one of the four forms, or a `judge:` spec
        was given without `allow_llm`.
    RuntimeError
        If the spec needs fit and fit is not installed.
    """
    kind, _, rest = spec.partition(":")

    if kind == "composite":
        reward = _import_fit("fit.reward")
        dimensions = [part.strip() for part in rest.split(",") if part.strip()]
        logger.warning(
            "composite: measures nothing. fit's DimensionScorer returns a "
            "constant 0.5 for every input, so %r is a wiring smoke test, "
            "not a measurement. Use rubric: or module: for real numbers.",
            spec,
        )
        return reward.CompositeScorer.composite(dimensions)

    if kind == "rubric":
        return _build_rubric(rest)

    if kind == "module":
        module_name, _, factory_name = rest.rpartition(":")
        if not module_name:
            raise ValueError(
                f"Scorer spec {spec!r} is malformed; module: wants "
                "module:pkg.mod:factory"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ValueError(
                f"Scorer spec {spec!r} names module {module_name!r}, "
                f"which cannot be imported ({exc}). It must be on the "
                "Python path of this process."
            ) from exc

        try:
            factory = getattr(module, factory_name)
        except AttributeError as exc:
            raise ValueError(
                f"Scorer spec {spec!r} names {factory_name!r}, which "
                f"module {module_name!r} does not define."
            ) from exc

        try:
            scorer = factory()
        except Exception as exc:
            raise ValueError(
                f"Scorer spec {spec!r}: calling "
                f"{module_name}:{factory_name}() raised "
                f"{type(exc).__name__}: {exc}. A factory takes no "
                "arguments and returns the scorer."
            ) from exc

        if not hasattr(scorer, "score"):
            raise ValueError(
                f"Scorer spec {spec!r}: {module_name}:{factory_name} "
                f"returned {type(scorer).__name__}, which has no "
                "`score(output, context)`."
            )
        return scorer

    if kind == "judge":
        if not allow_llm:
            raise ValueError(
                f"Scorer spec {spec!r} calls an LLM once per trace "
                f"(about {trace_count} calls). Pass --allow-llm to "
                "authorise the spend, or use rubric: or module:, which "
                "make no calls."
            )
        reward_fn = _import_fit("fit.training.reward_fn")
        return _JudgeScorer(reward_fn.LLMJudgeReward(model=rest))

    raise ValueError(
        f"Unknown scorer spec {spec!r}. The four forms are:\n  "
        + "\n  ".join(SCORER_FORMS)
    )


class _JudgeScorer:
    """A `RewardScorer` over fit's `LLMJudgeReward`. Calls an LLM."""

    def __init__(self, reward_fn):
        self._reward_fn = reward_fn

    def score(self, output: str, context: dict):
        from fit.types import Reward

        value = float(self._reward_fn(context.get("prompt", ""), "", output))
        return Reward(score=value, breakdown={}, metadata={"scorer": "judge"})


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def load_traces(directory: str) -> Iterator[dict]:
    """Yield every readable trace in `directory`, oldest name first.

    Reads `*.json` directly rather than through fit's `TraceIngester`,
    which normalizes into a `TraceRecord` and drops the `routellm`
    block this module exists to read. Half-written `.json.tmp` files
    are skipped; an unreadable file logs one WARNING and does not stop
    the run.

    Parameters
    ----------
    directory : str
        Trace directory.

    Yields
    ------
    dict
        One parsed trace.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"No trace directory at {directory}")

    for name in sorted(os.listdir(directory)):
        if not fnmatch.fnmatch(name, "*.json"):
            continue
        full = os.path.join(directory, name)
        try:
            with open(full) as handle:
                yield json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("skipping unreadable trace %s: %s", name, exc)


def _scored_ids(out_path: str) -> set:
    """Return the trace ids an existing scores file already holds."""
    if not os.path.exists(out_path):
        return set()
    ids = set()
    with open(out_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["trace_id"])
            except (ValueError, KeyError):
                continue
    return ids


def score_traces(
    traces_dir: str,
    spec: str,
    out_path: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    rescore: bool = False,
    allow_llm: bool = False,
) -> int:
    """Score every trace in a directory into a JSONL scores file.

    One row per trace: `{trace_id, endpoint, request_model, tier, area,
    cached, is_canary, score, breakdown, scorer, scored_at}`. `score`
    is null when the scorer failed, which is fit's own failure
    semantics (reward-schema-v1), and a null is kept rather than
    dropped so the count of attempts stays honest.

    Cached responses are scored and marked. The aggregator drops them:
    they measure the cache, not the endpoint.

    Appending is the default; a trace id the file already holds is
    skipped unless `rescore`.

    Parameters
    ----------
    traces_dir : str
        Directory of recorded traces.
    spec : str
        A `--scorer` spec.
    out_path : str, optional
        Scores file. Defaults to `scores.jsonl` beside `traces_dir`.
    limit : int, optional
        Score at most this many traces.
    rescore : bool, optional
        Score ids the file already holds (default False).
    allow_llm : bool, optional
        Permit a spec that calls an LLM (default False).

    Returns
    -------
    int
        How many rows were written.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or holds no trace.
    """
    out_path = out_path or os.path.join(
        os.path.dirname(os.path.abspath(traces_dir)), "scores.jsonl"
    )

    traces = list(load_traces(traces_dir))
    if not traces:
        raise FileNotFoundError(f"No traces to score in {traces_dir}")

    already = set() if rescore else _scored_ids(out_path)
    pending = [trace for trace in traces if trace.get("id") not in already]
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        logger.info("every trace in %s is already scored", traces_dir)
        return 0

    scorer = build_scorer(spec, allow_llm=allow_llm, trace_count=len(pending))

    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "a") as handle:
        for trace in pending:
            handle.write(json.dumps(_score_row(trace, scorer, spec)) + "\n")
            written += 1
    logger.info("wrote %s scored traces to %s", written, out_path)
    return written


def _score_row(trace: dict, scorer, spec: str) -> dict:
    """Return the scores-file row for one trace."""
    block = trace.get("routellm") or {}
    output = (trace.get("frontier") or {}).get("output", "")
    prompt = (trace.get("input") or {}).get("prompt", "")

    try:
        reward = scorer.score(output, {"prompt": prompt})
        score = None if reward.score is None else float(reward.score)
        breakdown = {k: float(v) for k, v in (reward.breakdown or {}).items()}
    except Exception as exc:
        logger.warning("scorer failed on %s: %s", trace.get("id"), exc)
        score, breakdown = None, {}

    return {
        "trace_id": trace.get("id"),
        "endpoint": block.get("endpoint"),
        "request_model": block.get("request_model"),
        "tier": block.get("tier"),
        "area": block.get("area"),
        "cached": bool(block.get("cached", False)),
        "is_canary": bool(block.get("is_canary", False)),
        "score": score,
        "breakdown": breakdown,
        "scorer": spec,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_scores(
    rows: list,
    min_samples: int = 30,
    transform: str = "linear",
    areas: Optional[dict] = None,
) -> dict:
    """Turn scored traces into the quality sidecar, as a dict.

    Pure: no I/O, no fit import, no network. The CLI is a thin wrapper
    around this.

    A trace counts when it was scored (`score` is not null) and was not
    served from cache: a cached answer measures the cache, not the
    endpoint that once produced it. Canary traces DO count. They are
    real answers from a real endpoint, and excluding them would make an
    endpoint's measured quality depend on how much canary traffic it
    happened to get.

    An endpoint under `min_samples` is left OUT of `endpoints:` rather
    than written with a null quality. Absent means unrated, and the
    merge step then leaves the YAML `quality` alone; a null would mean
    "measured, and the measurement is nothing". The same threshold
    applies per area: an area under it is dropped from `by_area` while
    the endpoint's overall number may still be present.

    Transforms
    ----------
    linear
        `quality = round(100 * mean_score)`, clamped to [0, 100]. fit
        scores live in [0, 1] (reward-schema-v1) and `Endpoint.quality`
        is `ge=0, le=100`, so this is the identity times 100 and the
        number means "the mean reward, as a percentage".
    percentile
        Ranks the rated endpoints against each other and spreads them
        over [0, 100]. Useful when every scorer output clusters, which
        fit's constant `DimensionScorer` guarantees. It compares
        endpoints to each other, so it needs at least two; with fewer
        it falls back to linear and warns.

    Parameters
    ----------
    rows : list[dict]
        Rows of the scores file.
    min_samples : int, optional
        Below this an endpoint or area is omitted (default 30).
    transform : str, optional
        `linear` or `percentile` (default "linear").
    areas : dict, optional
        Tier name to area name. When given, a trace with no `area` of
        its own takes the area of its tier. When None, `by_area` is
        keyed by the raw tier name and the sidecar says so.

    Returns
    -------
    dict
        The sidecar, ready for `yaml.safe_dump(..., sort_keys=False)`.
    """
    overall: dict = {}
    per_area: dict = {}

    for row in rows:
        if row.get("cached"):
            continue
        score = row.get("score")
        if score is None:
            continue
        endpoint = row.get("endpoint")
        if not endpoint:
            continue

        overall.setdefault(endpoint, []).append(float(score))

        area = row.get("area")
        if area is None and areas is not None:
            area = areas.get(row.get("tier"))
        if area is None:
            area = row.get("tier")
        if area is not None:
            per_area.setdefault(endpoint, {}).setdefault(area, []).append(
                float(score)
            )

    rated = {
        name: values
        for name, values in overall.items()
        if len(values) >= min_samples
    }

    effective = transform
    if transform == "percentile" and len(rated) < 2:
        logger.warning(
            "percentile ranks endpoints against each other and needs at "
            "least 2 rated ones; %s rated, falling back to linear",
            len(rated),
        )
        effective = "linear"

    if effective == "percentile":
        means = {name: sum(v) / len(v) for name, v in rated.items()}
        qualities = _percentile_scale(means)
    else:
        qualities = {
            name: _linear(sum(v) / len(v)) for name, v in rated.items()
        }

    endpoints: dict = {}
    for name in sorted(rated):
        by_area = {
            area: {"quality": _linear(sum(v) / len(v)), "n": len(v)}
            for area, v in sorted((per_area.get(name) or {}).items())
            if len(v) >= min_samples
        }
        endpoints[name] = {
            "quality": qualities[name],
            "n": len(rated[name]),
            "by_area": by_area,
        }

    return {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_samples": min_samples,
        "transform": effective,
        "source": "traces",
        "area_source": "config" if areas is not None else "tier",
        "endpoints": endpoints,
    }


def _linear(mean: float) -> int:
    """Return a mean reward on the [0, 100] scale `quality` validates."""
    return max(0, min(100, round(100 * mean)))


def _percentile_scale(means: dict) -> dict:
    """Spread rated endpoints over [0, 100] by their rank.

    Ties share a rank, so two endpoints that scored the same get the
    same quality rather than an arbitrary order between them.
    """
    ordered = sorted(set(means.values()))
    last = len(ordered) - 1
    return {
        name: _clamp_int(100 * ordered.index(mean) / last) if last else 100
        for name, mean in means.items()
    }


def _clamp_int(value: float) -> int:
    """Round to an int inside [0, 100]."""
    return max(0, min(100, round(value)))


def load_scores(path: str) -> list:
    """Read a scores JSONL file into a list of rows.

    Parameters
    ----------
    path : str
        Scores file.

    Returns
    -------
    list[dict]
        One row per scored trace.

    Raises
    ------
    FileNotFoundError
        If the file is missing or holds no row.
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

    if not rows:
        raise FileNotFoundError(f"No scores to aggregate in {path}")
    return rows


def _areas_from_config(path: str) -> dict:
    """Invert a config's `areas:` map into tier name to area name.

    Parameters
    ----------
    path : str
        A routellm config YAML. Only its `areas:` key is read.

    Returns
    -------
    dict[str, str]
        Tier name to area name. Empty when the config has no `areas:`,
        which is not an error: the sidecar then keys `by_area` on the
        raw tier names.

    Raises
    ------
    FileNotFoundError
        If there is no file at `path`, naming the flag that asked.
    ValueError
        If the file is unreadable or is not a mapping.
    """
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No config file at {path} (given as --config, read only "
            "for its `areas:` map)"
        )

    try:
        with open(path) as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unreadable config at {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Malformed config at {path}: expected a mapping, found "
            f"{type(config).__name__}"
        )

    areas = config.get("areas") or {}
    if not areas:
        logger.warning(
            "config %s has no `areas:`; by_area will be keyed on the "
            "raw tier names",
            path,
        )

    inverted: dict = {}
    for area, tiers in areas.items():
        for tier in tiers or []:
            if tier in inverted and inverted[tier] != area:
                raise ValueError(
                    f"Tier {tier!r} is in two areas in {path}, "
                    f"{inverted[tier]!r} and {area!r}."
                )
            inverted[tier] = area
    return inverted


# ---------------------------------------------------------------------------
# Sidecar consumption
# ---------------------------------------------------------------------------

#: A sidecar older than this logs a staleness warning. Never an error:
#: an old measurement still beats no measurement.
STALE_AFTER_DAYS = 30

SIDECAR_VERSION = 1


class AreaQuality(BaseModel):
    """One area's measured quality for one endpoint."""

    quality: int = Field(ge=0, le=100)
    n: int = 0


class EndpointQuality(BaseModel):
    """One endpoint's measured quality, overall and per area."""

    quality: int = Field(ge=0, le=100)
    n: int = 0
    by_area: dict = Field(default_factory=dict)


class Sidecar(BaseModel):
    """The quality sidecar `quality_from` points at."""

    version: int
    generated_at: str = ""
    min_samples: int = 0
    transform: str = "linear"
    source: str = "traces"
    area_source: str = "tier"
    endpoints: dict = Field(default_factory=dict)


def load_sidecar(path: str) -> Sidecar:
    """Read and validate a quality sidecar.

    Parameters
    ----------
    path : str
        Sidecar YAML, already resolved to an absolute or cwd-relative
        path by the caller.

    Returns
    -------
    Sidecar
        The parsed sidecar.

    Raises
    ------
    FileNotFoundError
        If there is no file at `path`. The message names the config key
        that asked for it, so the fix is obvious from the error alone.
    ValueError
        If the version is not the one this code reads, or the YAML is
        malformed.
    """
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No quality sidecar at {path} (asked for by the config key "
            "`quality_from`). Generate one with: python -m "
            "routellm.quality_scores aggregate"
        )

    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed quality sidecar at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Malformed quality sidecar at {path}: expected a mapping, "
            f"found {type(raw).__name__}"
        )

    version = raw.get("version")
    if version != SIDECAR_VERSION:
        raise ValueError(
            f"Quality sidecar at {path} is version {version!r}; this "
            f"routellm reads version {SIDECAR_VERSION}. Regenerate it "
            "with: python -m routellm.quality_scores aggregate"
        )

    try:
        sidecar = Sidecar(**raw)
        sidecar.endpoints = {
            name: EndpointQuality(**(spec or {}))
            for name, spec in (raw.get("endpoints") or {}).items()
        }
    except ValidationError as exc:
        raise ValueError(f"Malformed quality sidecar at {path}: {exc}") from exc

    _warn_if_stale(path, sidecar.generated_at)
    return sidecar


def _warn_if_stale(path: str, generated_at: str) -> None:
    """Log one WARNING when a sidecar is older than `STALE_AFTER_DAYS`."""
    if not generated_at:
        return
    try:
        stamp = datetime.strptime(generated_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        logger.warning(
            "quality sidecar %s has an unreadable generated_at %r",
            path,
            generated_at,
        )
        return

    age = (datetime.now(timezone.utc) - stamp).days
    if age > STALE_AFTER_DAYS:
        logger.warning(
            "quality sidecar %s is stale: measured %s days ago (%s). "
            "Regenerate it; an old measurement still beats none.",
            path,
            age,
            generated_at,
        )


def apply_sidecar(registry, sidecar: Sidecar, override: bool = True) -> int:
    """Merge measured quality into a registry's endpoints, in place.

    Precedence: THE SIDECAR WINS. It is measured; the YAML number is a
    guess someone typed once and forgot. Every override logs one INFO
    line naming both numbers and the sample count, so a surprising
    ordering can be read out of the startup log.

    `override=False` flips it: an explicit YAML `quality` wins and the
    sidecar only fills endpoints that set none. That mode logs one INFO
    per endpoint it declined to override.

    Under either setting an endpoint the sidecar does not name is never
    touched, so it keeps exactly the ordering it has today.

    Per-area numbers land on `registry.area_quality` as
    `{endpoint: {area: quality}}`, which pairing reads when a selector
    sits inside an area's tier.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry to merge into.
    sidecar : Sidecar
        The parsed sidecar.
    override : bool, optional
        Whether a measured number beats an explicit YAML one
        (default True).

    Returns
    -------
    int
        How many endpoints changed.
    """
    changed = 0
    area_quality: dict = {}

    for name, measured in sidecar.endpoints.items():
        if name not in registry.names():
            logger.warning(
                "quality sidecar names %r, which is not a configured "
                "endpoint; ignored",
                name,
            )
            continue
        endpoint = registry.get(name)
        current = endpoint.quality

        area_quality[name] = {
            area: int(spec["quality"])
            for area, spec in (measured.by_area or {}).items()
            if isinstance(spec, dict) and "quality" in spec
        }

        if current is not None and not override:
            logger.info(
                "quality: %s declined to override explicit %s with %s "
                "(n=%s); quality_from_override is false",
                name,
                current,
                measured.quality,
                measured.n,
            )
            continue

        if current == measured.quality:
            continue

        endpoint.quality = measured.quality
        endpoint.quality_measured = True
        changed += 1
        logger.info(
            "quality: %s %s -> %s (n=%s, from the sidecar)",
            name,
            current,
            measured.quality,
            measured.n,
        )

    registry.area_quality = area_quality
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m routellm.quality_scores",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scorer = sub.add_parser("score", help="score recorded traces")
    scorer.add_argument("--traces", required=True, help="trace directory")
    scorer.add_argument(
        "--scorer",
        required=True,
        help="scorer spec; one of:\n  " + "\n  ".join(SCORER_FORMS),
    )
    scorer.add_argument("--out", default=None, help="scores JSONL (default beside --traces)")
    scorer.add_argument("--limit", type=int, default=None)
    scorer.add_argument("--rescore", action="store_true")
    scorer.add_argument(
        "--allow-llm",
        action="store_true",
        help="authorise a scorer spec that calls an LLM per trace",
    )

    agg = sub.add_parser("aggregate", help="aggregate scores into the sidecar")
    agg.add_argument("--scores", required=True, help="scores JSONL")
    agg.add_argument("--config", default=None, help="routellm YAML, read for areas:")
    agg.add_argument("--out", required=True, help="sidecar YAML to write")
    agg.add_argument("--min-samples", type=int, default=30)
    agg.add_argument(
        "--transform", choices=("linear", "percentile"), default="linear"
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    """Run the CLI, returning its exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)

    if args.command == "score":
        try:
            score_traces(
                args.traces,
                args.scorer,
                args.out,
                limit=args.limit,
                rescore=args.rescore,
                allow_llm=args.allow_llm,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    elif args.command == "aggregate":
        import yaml

        try:
            rows = load_scores(args.scores)
            areas = _areas_from_config(args.config) if args.config else None
            sidecar = aggregate_scores(
                rows,
                min_samples=args.min_samples,
                transform=args.transform,
                areas=areas,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        with open(args.out, "w") as handle:
            yaml.safe_dump(sidecar, handle, sort_keys=False)
        logger.info(
            "wrote %s rated endpoints to %s",
            len(sidecar["endpoints"]),
            args.out,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
