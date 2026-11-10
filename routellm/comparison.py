"""Comparison routing: shadow and compare fan-out for benchmarking.

Send one request to two or more models at once and record how each did:
time to first token, total latency, tokens per second, token counts,
cost, error and output. Three ways in, all off until `comparison:` is
enabled in the config:

* Shadow mode. The normal chat-completions request is answered by its
  primary, unchanged, streamed or not. The shadow models get the same
  request concurrently as background tasks. Their results are recorded
  and nothing else: they never delay or alter the primary's response,
  never touch the cache, and a failure in one is logged and recorded,
  never shown to the client.
* The compare endpoint, `POST /v1/compare`. All named models answer and
  the results come back side by side.
* The offline replay CLI, `python -m routellm.compare`.

What triggers a shadow, in priority order, at most one plan per request:

1. the `X-RouteLLM-Compare` request header, naming an experiment, a
   profile, or a comma-separated list of endpoints;
2. a named experiment, sampled at its `rate`;
3. the sampling rule of the tier the request entered.

Safety, enforced here and nowhere else:

* A tier whose reachable endpoints all carry the `local` tag may only be
  shadowed to `local`-tagged endpoints. Config that breaks this is
  refused at load; a header or an untiered experiment is filtered at
  request time.
* A remote LLM judge never sees data from such a tier.
* Endpoints with `pay: true` are never comparison targets: a shadow must
  not spend the wallet.
* A global and a per-experiment USD budget, and a global and a
  per-experiment concurrency cap. A plan that does not fit is dropped,
  never queued.
* Every outbound call carries the hop header.

Budget accounting is in USD, in memory, per process: a restart clears
it. A call's cost is its token usage priced from the catalog record of
the endpoint; an unpriced endpoint costs nothing against the budget. A
call is charged when it finishes, so in-flight calls can overshoot a cap
by their own cost.
"""

import asyncio
import copy
import logging
import random
import secrets
import time
from typing import Any

from litellm import acompletion, stream_chunk_builder
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from routellm.leaf_params import drop_kwargs, log_dropped
from routellm.transcription import LOCAL_TAG

logger = logging.getLogger(__name__)

#: Request header naming what to compare: an experiment, a profile, or
#: a comma-separated list of endpoint names.
COMPARE_HEADER = "X-RouteLLM-Compare"

#: Roles a recorded trace can have in a comparison.
ROLE_PRIMARY = "primary"
ROLE_SHADOW = "shadow"
ROLE_CANDIDATE = "candidate"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingRule(_Strict):
    """Shadow a fraction of one tier's requests.

    Attributes
    ----------
    rate : float
        Probability in [0, 1] that a request to the tier is shadowed.
    models : list[str] or str
        Endpoint names, or the name of a profile.
    """

    rate: float = Field(ge=0, le=1)
    models: list[str] | str


class Experiment(_Strict):
    """A named experiment: models, a sample rate and its own limits.

    Attributes
    ----------
    models : list[str] or str
        Endpoint names, or the name of a profile.
    rate : float
        Probability in [0, 1] that a request is shadowed. Naming the
        experiment in the header ignores it.
    tiers : list[str], optional
        Only requests that entered one of these tiers are sampled.
        Unset samples every request.
    budget_usd : float, optional
        Stop this experiment once it has spent this much.
    max_concurrency : int, optional
        At most this many of its shadow calls in flight at once.
    """

    models: list[str] | str
    rate: float = Field(default=0.0, ge=0, le=1)
    tiers: list[str] | None = None
    budget_usd: float | None = Field(default=None, ge=0)
    max_concurrency: int | None = Field(default=None, gt=0)


class JudgeConfig(_Strict):
    """Score each recorded output with an existing scorer spec.

    Attributes
    ----------
    scorer : str
        A `routellm.quality_scores` spec: `rubric:`, `module:`,
        `composite:` or `judge:<model>`. Only `judge:` calls an LLM; the
        others run offline code. `judge:<model>` names an endpoint (or
        its model string), and counts as local only if that endpoint
        carries the `local` tag.
    """

    scorer: str


class ComparisonConfig(_Strict):
    """The `comparison:` section. Everything is off until `enabled`.

    Attributes
    ----------
    enabled : bool
        Master switch. Off: no shadow, no endpoint, header ignored.
    endpoint : bool
        Serve `POST /v1/compare` when enabled (default True).
    allow_header : bool
        Honour the `X-RouteLLM-Compare` header (default True).
    max_concurrency : int
        Shadow calls in flight at once, across everything.
    max_models : int
        Most models one compare request or plan may name.
    timeout_ms : int
        Per-call timeout for a comparison call.
    measure_ttft : bool
        Stream shadow calls upstream to measure time to first token.
        Off by default: shadows are non-streaming and report no TTFT.
    budget_usd : float, optional
        Stop all comparison calls once this much is spent.
    compare_allowed_models : list[str]
        Endpoints `POST /v1/compare` may name without a `tier`. Models
        in `profiles` and `experiments` are allowed too, since the
        operator named them.
    allow_untiered_compare : bool
        DANGEROUS. Lets an untiered `/v1/compare` name any configured
        endpoint, so any caller can send any prompt to any model,
        including paid and cloud ones. Off by default.
    profiles : dict
        Named endpoint lists, usable in the header and in `models:`.
    sampling : dict
        Per-tier sampling rules.
    experiments : dict
        Named experiments.
    judge : JudgeConfig, optional
        Score recorded outputs.
    """

    enabled: StrictBool = False
    endpoint: StrictBool = True
    allow_header: StrictBool = True
    max_concurrency: int = Field(default=4, gt=0)
    max_models: int = Field(default=4, gt=0)
    timeout_ms: int = Field(default=60000, gt=0)
    measure_ttft: StrictBool = False
    budget_usd: float | None = Field(default=None, ge=0)
    compare_allowed_models: list[str] = Field(default_factory=list)
    allow_untiered_compare: StrictBool = False
    profiles: dict[str, list[str]] = Field(default_factory=dict)
    sampling: dict[str, SamplingRule] = Field(default_factory=dict)
    experiments: dict[str, Experiment] = Field(default_factory=dict)
    judge: JudgeConfig | None = None


class Plan:
    """One fan-out the manager decided on, holding its reserved slots."""

    def __init__(
        self,
        models: list[str],
        trigger: str,
        experiment: str | None,
        tier: str | None,
        local_only: bool,
    ):
        self.id = f"cmp-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
        self.models = models
        self.trigger = trigger
        self.experiment = experiment
        self.tier = tier
        self.local_only = local_only


def is_local(registry, name: str) -> bool:
    """Whether endpoint `name` is registered and tagged `local`."""
    return name in registry.names() and LOCAL_TAG in registry.get(name).tags


def leaves_of(registry, tier: str) -> set[str] | None:
    """Endpoints reachable from a tier; None while a side is unresolved."""
    return registry._leaves(tier)


def tier_is_local_only(registry, tier: str) -> bool:
    """Whether every endpoint a tier can reach is tagged `local`."""
    leaves = leaves_of(registry, tier)
    return bool(leaves) and all(is_local(registry, leaf) for leaf in leaves)


class ComparisonManager:
    """Plans, runs and records comparisons for one controller.

    Parameters
    ----------
    config : ComparisonConfig
        The validated section.
    host : object
        Carries `endpoints`, `quality_manager`, `api_base`, `api_key`
        and `_catalog_records`, and optionally `_can_serve(name, reqs)`.
        Read at use, so a host that swaps its quality manager is
        followed.
    rng : callable, optional
        Returns a float in [0, 1); a test seam, `random.random` by
        default.

    Raises
    ------
    ValueError
        If the config names an unknown endpoint, tier, profile or
        experiment, an endpoint with `pay: true`, or breaks the
        local-only rules.
    """

    def __init__(self, config: ComparisonConfig, host, rng=random.random):
        self.config = config
        self.host = host
        self._rng = rng
        self._tasks: set[asyncio.Future] = set()
        self._inflight = 0
        self._exp_inflight: dict[str, int] = {}
        self.spent_usd = 0.0
        self._exp_spent: dict[str, float] = {}
        self.dropped = 0
        self._scorer = None
        self._scorer_failed = False
        self._validate()

    # ----------------------------------------------------------- load time

    @property
    def registry(self):
        return self.host.endpoints

    def _validate(self) -> None:
        registry, config = self.registry, self.config

        for name, models in config.profiles.items():
            self._check_endpoints(models, f"profile {name!r}")
        self._check_endpoints(config.compare_allowed_models, "compare_allowed_models")

        local_only_tiers: set[str] = set()

        for tier, rule in config.sampling.items():
            if not registry.has_tier(tier):
                raise ValueError(f"comparison.sampling names unknown tier {tier!r}")
            models = self._expand(rule.models, f"sampling of tier {tier!r}")
            self._check_endpoints(models, f"sampling of tier {tier!r}")
            self._check_local(tier, models, f"sampling of tier {tier!r}", local_only_tiers)

        for name, experiment in config.experiments.items():
            label = f"experiment {name!r}"
            models = self._expand(experiment.models, label)
            self._check_endpoints(models, label)
            for tier in experiment.tiers or []:
                if not registry.has_tier(tier):
                    raise ValueError(f"comparison {label} names unknown tier {tier!r}")
                self._check_local(tier, models, label, local_only_tiers)

        self._judge_local = True
        if config.judge is not None:
            self._judge_local = self._spec_is_local(config.judge.scorer)
            if not self._judge_local and local_only_tiers:
                raise ValueError(
                    f"comparison.judge {config.judge.scorer!r} is not local, but "
                    f"data from local-only tier(s) {', '.join(sorted(local_only_tiers))} "
                    "would be judged by it. Use a judge model tagged `local`, an "
                    "offline scorer, or drop those tiers from sampling/experiments."
                )

    def _expand(self, models: list[str] | str, where: str) -> list[str]:
        if isinstance(models, str):
            if models not in self.config.profiles:
                raise ValueError(f"comparison {where} names unknown profile {models!r}")
            return list(self.config.profiles[models])
        return list(models)

    def _check_endpoints(self, models: list[str], where: str) -> None:
        for name in models:
            if name not in self.registry.names():
                raise ValueError(
                    f"comparison {where} names {name!r}, which is not a configured endpoint"
                )
            if self.registry.get(name).pay:
                raise ValueError(
                    f"comparison {where} names {name!r}, which has `pay: true`: "
                    "a comparison call must never spend the wallet"
                )

    def _check_local(self, tier: str, models: list[str], where: str, found: set[str]) -> None:
        if not tier_is_local_only(self.registry, tier):
            return
        found.add(tier)
        bad = [m for m in models if not is_local(self.registry, m)]
        if bad:
            raise ValueError(
                f"comparison {where}: tier {tier!r} is local-only (every endpoint it "
                f"reaches is tagged {LOCAL_TAG!r}), so it may only be shadowed to "
                f"{LOCAL_TAG!r} endpoints; not {', '.join(bad)}"
            )

    def _spec_is_local(self, spec: str) -> bool:
        """Whether a scorer spec keeps its data on this machine.

        Only `judge:<model>` calls a model; every other form runs
        offline code. A judge is local when its model is an endpoint
        tagged `local`, named directly or by its model string.
        """
        kind, _, rest = spec.partition(":")
        if kind != "judge":
            return True
        if is_local(self.registry, rest):
            return True
        return any(
            LOCAL_TAG in self.registry.get(n).tags
            for n in self.registry.names()
            if self.registry.get(n).model == rest
        )

    # --------------------------------------------------------- planning

    def local_only_scope(self, path: list[dict] | None, pair) -> bool:
        """Whether the request's primary can only reach `local` endpoints."""
        tier = (path or [{}])[0].get("tier")
        if tier is not None and self.registry.has_tier(tier):
            return tier_is_local_only(self.registry, tier)
        if pair is None:
            return False
        sides = [str(pair.strong), str(pair.weak)]
        return all(is_local(self.registry, side) for side in sides)

    def plan(
        self,
        *,
        header: str | None,
        path: list[dict] | None,
        pair,
        picked: str,
    ) -> Plan | None:
        """Decide whether this request is shadowed, and to which models.

        Reserves the concurrency slots of the plan it returns; they are
        released when `launch` finishes it.

        Parameters
        ----------
        header : str, optional
            The `X-RouteLLM-Compare` value.
        path : list[dict], optional
            The primary's decision path; its first entry names the tier
            the request entered.
        pair : ModelPair, optional
            The flat pair, when the request was routed flat.
        picked : str
            The endpoint the primary was routed to; never shadowed.

        Returns
        -------
        Plan or None
            None when nothing triggers, nothing survives the filters,
            or a budget or concurrency cap stops it.
        """
        tier = (path or [{}])[0].get("tier")
        local_only = self.local_only_scope(path, pair)

        trigger = experiment = None
        models: list[str] = []
        if header and self.config.allow_header:
            trigger, experiment, models = self._from_header(header)
        if not models:
            trigger, experiment, models = self._from_config(tier)
        if not models:
            return None

        models = self._filter(models, picked, local_only)
        if not models:
            return None

        if not self._admit(models, experiment):
            return None

        self._reserve(len(models), experiment)
        return Plan(models, trigger, experiment, tier, local_only)

    def _from_header(self, header: str):
        tokens = [t.strip() for t in header.split(",") if t.strip()]
        if len(tokens) == 1 and tokens[0] in self.config.experiments:
            name = tokens[0]
            models = self._expand(self.config.experiments[name].models, name)
            return "header", name, models
        if len(tokens) == 1 and tokens[0] in self.config.profiles:
            return "header", None, list(self.config.profiles[tokens[0]])
        known = [t for t in tokens if t in self.registry.names()]
        for token in set(tokens) - set(known):
            logger.warning("%s names %r, which is not an endpoint; ignored", COMPARE_HEADER, token)
        return "header", None, known

    def _from_config(self, tier: str | None):
        for name, experiment in self.config.experiments.items():
            if experiment.tiers and tier not in experiment.tiers:
                continue
            if self._rng() < experiment.rate:
                return "experiment", name, self._expand(experiment.models, name)
        rule = self.config.sampling.get(tier) if tier else None
        if rule is not None and self._rng() < rule.rate:
            return "sampling", None, self._expand(rule.models, f"tier {tier}")
        return None, None, []

    def _filter(self, models: list[str], picked: str, local_only: bool) -> list[str]:
        kept: list[str] = []
        for name in dict.fromkeys(models):
            if name == picked:
                continue
            if name not in self.registry.names() or self.registry.get(name).pay:
                continue
            if local_only and not is_local(self.registry, name):
                logger.warning("comparison skips %s: the request is local-only", name)
                continue
            kept.append(name)
        return kept[: self.config.max_models]

    def _admit(self, models: list[str], experiment: str | None) -> bool:
        config = self.config
        if config.budget_usd is not None and self.spent_usd >= config.budget_usd:
            logger.info("comparison skipped: global budget spent")
            return False
        if experiment is not None:
            spec = self.config.experiments.get(experiment)
            if spec and spec.budget_usd is not None:
                if self._exp_spent.get(experiment, 0.0) >= spec.budget_usd:
                    logger.info("comparison skipped: experiment %s budget spent", experiment)
                    return False
            if spec and spec.max_concurrency is not None:
                if self._exp_inflight.get(experiment, 0) + len(models) > spec.max_concurrency:
                    self.dropped += 1
                    return False
        if self._inflight + len(models) > config.max_concurrency:
            self.dropped += 1
            logger.info("comparison dropped: concurrency cap reached")
            return False
        return True

    def _reserve(self, count: int, experiment: str | None) -> None:
        self._inflight += count
        if experiment is not None:
            self._exp_inflight[experiment] = self._exp_inflight.get(experiment, 0) + count

    def _release(self, count: int, experiment: str | None) -> None:
        self._inflight -= count
        if experiment is not None:
            self._exp_inflight[experiment] = self._exp_inflight.get(experiment, 0) - count

    def _charge(self, cost: float | None, experiment: str | None) -> None:
        if not cost:
            return
        self.spent_usd += cost
        if experiment is not None:
            self._exp_spent[experiment] = self._exp_spent.get(experiment, 0.0) + cost

    # ----------------------------------------------------------- running

    def launch(
        self,
        plan: Plan,
        *,
        kwargs: dict[str, Any],
        prompt: str,
        request_model: str | None,
        path: list[dict] | None,
        session_id: str,
        reqs=None,
    ) -> None:
        """Run a plan in the background. Returns at once, never raises."""
        job = self._run_plan(
            plan,
            copy.deepcopy(kwargs),
            prompt,
            request_model,
            path,
            session_id,
            reqs,
        )
        task = asyncio.ensure_future(job)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for every background comparison to finish (tests, shutdown)."""
        while self._tasks:
            await asyncio.wait(list(self._tasks))

    async def _run_plan(self, plan, kwargs, prompt, request_model, path, session_id, reqs):
        try:
            await asyncio.gather(
                *(
                    self._one(
                        name,
                        kwargs,
                        prompt,
                        plan=plan,
                        role=ROLE_SHADOW,
                        request_model=request_model,
                        path=path,
                        session_id=session_id,
                        reqs=reqs,
                        measure_ttft=self.config.measure_ttft,
                    )
                    for name in plan.models
                ),
                return_exceptions=True,
            )
        except Exception as exc:  # never reaches the client
            logger.error("comparison %s failed: %s", plan.id, exc)
        finally:
            self._release(len(plan.models), plan.experiment)

    async def compare(
        self,
        models: list[str] | str,
        kwargs: dict[str, Any],
        *,
        tier: str | None = None,
        measure_ttft: bool | None = None,
        operator: bool = False,
    ) -> dict[str, Any]:
        """Run every model on one request and return the side-by-side results.

        Backs `POST /v1/compare`. Unlike a shadow it is awaited, and it
        runs the models even when one of them is the model a router
        would have picked.

        Parameters
        ----------
        models : list[str] or str
            Endpoint names, or a profile or experiment name.
        kwargs : dict
            The request: `messages` and any sampling params.
        tier : str, optional
            A tier whose privacy applies: when it is local-only every
            model must be `local`-tagged.
        measure_ttft : bool, optional
            Stream upstream to measure TTFT; the config default if None.
        operator : bool, optional
            The caller is the operator (the replay CLI), not a client,
            so the untiered allow-list does not apply.

        Raises
        ------
        ValueError
            On an unknown model, too many models, a privacy breach, or
            an untiered request naming a model the operator did not
            allow-list.
        """
        names, local_only = self.check_models(models, tier)
        if tier is None and not operator:
            self._require_allowed(names)

        from routellm import requirements as requirements_module

        messages = kwargs["messages"]
        prompt = requirements_module._prompt_text(messages)
        reqs = requirements_module.derive(messages, kwargs, None)
        plan = Plan(names, "compare", None, tier, local_only)
        results = await asyncio.gather(
            *(
                self._one(
                    name,
                    kwargs,
                    prompt,
                    plan=plan,
                    role=ROLE_CANDIDATE,
                    request_model=None,
                    path=[{"tier": tier}] if tier else None,
                    session_id=str(kwargs.get("user") or plan.id),
                    reqs=reqs,
                    measure_ttft=(
                        self.config.measure_ttft if measure_ttft is None else measure_ttft
                    ),
                )
                for name in names
            )
        )
        return {"object": "comparison", "id": plan.id, "results": list(results)}

    def _require_allowed(self, names: list[str]) -> None:
        """Refuse an untiered request for models the operator did not list."""
        config = self.config
        if config.allow_untiered_compare:
            return
        allowed = set(config.compare_allowed_models)
        for models in config.profiles.values():
            allowed.update(models)
        for experiment in config.experiments.values():
            allowed.update(self._expand(experiment.models, "experiment"))
        bad = [n for n in names if n not in allowed]
        if bad:
            raise ValueError(
                f"{', '.join(bad)} may not be compared without a `tier`: name a tier "
                "(its privacy rule then applies), or list the model in "
                "comparison.compare_allowed_models or a profile"
            )

    def check_models(
        self, models: list[str] | str, tier: str | None = None
    ) -> tuple[list[str], bool]:
        """Validate a model selection for an explicit comparison.

        Returns
        -------
        tuple[list[str], bool]
            The endpoint names, and whether `tier` is local-only.

        Raises
        ------
        ValueError
            On an unknown model, too few or too many models, an
            endpoint with `pay: true`, or a non-`local` model for a
            local-only tier.
        """
        names = self._compare_models(models)
        local_only = False
        if tier is not None:
            if not self.registry.has_tier(tier):
                raise ValueError(f"unknown tier {tier!r}")
            local_only = tier_is_local_only(self.registry, tier)
            bad = [n for n in names if local_only and not is_local(self.registry, n)]
            if bad:
                raise ValueError(
                    f"tier {tier!r} is local-only; {', '.join(bad)} is not tagged "
                    f"{LOCAL_TAG!r} and may not receive its data"
                )
        return names, local_only

    def _compare_models(self, models: list[str] | str) -> list[str]:
        if isinstance(models, str):
            if models in self.config.experiments:
                names = self._expand(self.config.experiments[models].models, models)
            elif models in self.config.profiles:
                names = list(self.config.profiles[models])
            else:
                names = [m.strip() for m in models.split(",") if m.strip()]
        else:
            names = list(models)
        names = list(dict.fromkeys(names))
        if len(names) < 2:
            raise ValueError("compare needs at least two models")
        if len(names) > self.config.max_models:
            raise ValueError(f"compare takes at most {self.config.max_models} models")
        for name in names:
            if name not in self.registry.names():
                raise ValueError(f"{name!r} is not a configured endpoint")
            if self.registry.get(name).pay:
                raise ValueError(f"{name!r} has `pay: true` and cannot be compared")
        return names

    def over_budget(self) -> bool:
        budget = self.config.budget_usd
        return budget is not None and self.spent_usd >= budget

    # ---------------------------------------------------------- one call

    async def _one(
        self,
        name,
        kwargs,
        prompt,
        *,
        plan,
        role,
        request_model,
        path,
        session_id,
        reqs,
        measure_ttft,
    ) -> dict[str, Any]:
        """Call one endpoint, judge, record, and return its result."""
        result = await self._call(name, kwargs, reqs, measure_ttft, plan)
        self._charge(result["metrics"]["cost_usd"], plan.experiment)

        if result["error"] is None:
            result["score"] = await self._judge(prompt, result["output"], plan.local_only)

        comparison = {
            "id": plan.id,
            "role": role,
            "trigger": plan.trigger,
            "experiment": plan.experiment,
            "error": result["error"],
            "metrics": result["metrics"],
            "score": result["score"],
        }
        try:
            self.host.quality_manager.record_trace(
                prompt,
                name,
                result.pop("_response"),
                None,
                path=path,
                request_model=request_model,
                endpoint=name,
                session_id=session_id,
                latency_ms=result["metrics"]["latency_ms"],
                comparison=comparison,
                score=result["score"],
            )
        except Exception as exc:
            logger.warning("recording comparison trace failed: %s", exc)
        result.pop("_response", None)
        return result

    async def _call(self, name, kwargs, reqs, measure_ttft, plan) -> dict[str, Any]:
        endpoint = self.registry.get(name)
        metrics: dict[str, Any] = {
            "ttft_ms": None,
            "latency_ms": None,
            "tokens_per_second": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
        }
        result: dict[str, Any] = {
            "endpoint": name,
            "model": endpoint.model,
            "output": "",
            "finish_reason": None,
            "error": None,
            "score": None,
            "metrics": metrics,
            "_response": {},
        }

        can_serve = getattr(self.host, "_can_serve", None)
        if reqs is not None and can_serve is not None:
            try:
                reason = can_serve(name, reqs)
            except Exception:
                reason = None
            if reason:
                result["error"] = f"unsupported: {reason}"
                return result

        if self.over_budget():
            result["error"] = "budget_exhausted"
            return result

        started = time.monotonic()
        try:
            res, ttft = await asyncio.wait_for(
                self._request(name, endpoint, kwargs, measure_ttft),
                timeout=self.config.timeout_ms / 1000,
            )
        except TimeoutError:
            result["error"] = "timeout"
            metrics["latency_ms"] = int((time.monotonic() - started) * 1000)
            return result
        except Exception as exc:
            logger.warning("comparison call to %s failed: %s: %s", name, type(exc).__name__, exc)
            result["error"] = "upstream_error"
            metrics["latency_ms"] = int((time.monotonic() - started) * 1000)
            return result

        elapsed = time.monotonic() - started
        body = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        result["output"] = message.get("content") or ""
        result["finish_reason"] = choice.get("finish_reason")
        result["_response"] = body
        metrics.update(
            ttft_ms=None if ttft is None else int(ttft * 1000),
            latency_ms=int(elapsed * 1000),
            tokens_per_second=(
                round(completion_tokens / elapsed, 3) if completion_tokens and elapsed > 0 else None
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(name, prompt_tokens, completion_tokens),
        )
        return result

    async def _request(self, name, endpoint, kwargs, measure_ttft):
        """One non-cached, non-retried upstream call; returns (response, ttft)."""
        host = self.host
        api_base, api_key = endpoint.credentials(host.api_base, host.api_key)
        extra = {
            **endpoint.extra,
            **drop_kwargs(endpoint.model, api_base, endpoint.drop_params, endpoint.extra),
        }
        from routellm.controller import outbound_headers

        call = {**extra, **kwargs}
        call.pop("stream", None)
        call.pop("stream_options", None)
        call.pop("extra_headers", None)
        call["model"] = endpoint.model
        log_dropped(name, endpoint.model, api_base, call)
        headers = outbound_headers(extra.get("extra_headers"), kwargs.get("extra_headers"))

        if not measure_ttft:
            res = await acompletion(
                api_base=api_base, api_key=api_key, extra_headers=headers, **call
            )
            return res, None

        started = time.monotonic()
        stream = await acompletion(
            api_base=api_base,
            api_key=api_key,
            extra_headers=headers,
            stream=True,
            stream_options={"include_usage": True},
            **call,
        )
        chunks = []
        ttft = None
        async for chunk in stream:
            if ttft is None and _has_content(chunk):
                ttft = time.monotonic() - started
            chunks.append(chunk)
        res = stream_chunk_builder(chunks, messages=kwargs.get("messages"))
        if res is None:
            raise RuntimeError("empty stream")
        return res, ttft

    def _cost(self, name: str, prompt_tokens, completion_tokens) -> float | None:
        record = (getattr(self.host, "_catalog_records", None) or {}).get(name)
        if record is None:
            return None
        cost_in = getattr(record, "cost_input", None)
        cost_out = getattr(record, "cost_output", None)
        if cost_in is None and cost_out is None:
            return None
        return (
            (prompt_tokens or 0) * (cost_in or 0.0) + (completion_tokens or 0) * (cost_out or 0.0)
        ) / 1e6

    # ------------------------------------------------------------- judge

    async def _judge(self, prompt: str, output: str, local_only: bool) -> float | None:
        judge = self.config.judge
        if judge is None or self._scorer_failed:
            return None
        if local_only and not self._judge_local:
            logger.warning("not judging local-only data: the judge is not local")
            return None

        loop = asyncio.get_running_loop()
        try:
            if self._scorer is None:
                from routellm.quality_scores import build_scorer

                self._scorer = await loop.run_in_executor(
                    None, lambda: build_scorer(judge.scorer, allow_llm=True)
                )
            reward = await loop.run_in_executor(
                None, lambda: self._scorer.score(output, {"prompt": prompt})
            )
            return None if reward.score is None else float(reward.score)
        except Exception as exc:
            if self._scorer is None:
                self._scorer_failed = True
            logger.warning("comparison judge failed: %s", exc)
            return None


def _has_content(chunk) -> bool:
    try:
        return bool(chunk.choices[0].delta.content)
    except (AttributeError, IndexError, TypeError):
        return False
