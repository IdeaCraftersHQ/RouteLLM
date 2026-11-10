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

from routellm.comparison_call import ROLE_CANDIDATE, ROLE_PRIMARY, ROLE_SHADOW, CallMixin
from routellm.comparison_config import (
    ComparisonConfig,
    Experiment,
    JudgeConfig,
    SamplingRule,
    expand_models,
    is_local,
    leaves_of,
    tier_is_local_only,
    validate_config,
)
from routellm.transcription import LOCAL_TAG

__all__ = [
    "COMPARE_HEADER",
    "ROLE_CANDIDATE",
    "ROLE_PRIMARY",
    "ROLE_SHADOW",
    "ComparisonConfig",
    "ComparisonManager",
    "Experiment",
    "JudgeConfig",
    "Plan",
    "SamplingRule",
    "is_local",
    "leaves_of",
    "tier_is_local_only",
]

logger = logging.getLogger(__name__)

#: Request header naming what to compare: an experiment, a profile, or
#: a comma-separated list of endpoint names.
COMPARE_HEADER = "X-RouteLLM-Compare"


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


class ComparisonManager(CallMixin):
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
        self._judge_local = validate_config(config, host.endpoints)

    @property
    def registry(self):
        return self.host.endpoints

    def _expand(self, models: list[str] | str, where: str) -> list[str]:
        return expand_models(self.config, models, where)

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
