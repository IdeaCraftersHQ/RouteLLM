"""The `comparison:` config section and its load-time validation.

Split from `routellm.comparison` so the settings and the rules that
refuse a bad config sit apart from the code that runs comparisons.
`routellm.comparison` re-exports everything public here.
"""

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from routellm.transcription import LOCAL_TAG


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


def expand_models(config: ComparisonConfig, models: list[str] | str, where: str) -> list[str]:
    """Return endpoint names for a `models:` value: a list, or a profile name."""
    if isinstance(models, str):
        if models not in config.profiles:
            raise ValueError(f"comparison {where} names unknown profile {models!r}")
        return list(config.profiles[models])
    return list(models)


def check_endpoints(registry, models: list[str], where: str) -> None:
    """Refuse a name that is not an endpoint, or that has `pay: true`."""
    for name in models:
        if name not in registry.names():
            raise ValueError(
                f"comparison {where} names {name!r}, which is not a configured endpoint"
            )
        if registry.get(name).pay:
            raise ValueError(
                f"comparison {where} names {name!r}, which has `pay: true`: "
                "a comparison call must never spend the wallet"
            )


def _check_local(registry, tier: str, models: list[str], where: str, found: set[str]) -> None:
    if not tier_is_local_only(registry, tier):
        return
    found.add(tier)
    bad = [m for m in models if not is_local(registry, m)]
    if bad:
        raise ValueError(
            f"comparison {where}: tier {tier!r} is local-only (every endpoint it "
            f"reaches is tagged {LOCAL_TAG!r}), so it may only be shadowed to "
            f"{LOCAL_TAG!r} endpoints; not {', '.join(bad)}"
        )


def spec_is_local(registry, spec: str) -> bool:
    """Whether a scorer spec keeps its data on this machine.

    Only `judge:<model>` calls a model; every other form runs
    offline code. A judge is local when its model is an endpoint
    tagged `local`, named directly or by its model string.
    """
    kind, _, rest = spec.partition(":")
    if kind != "judge":
        return True
    if is_local(registry, rest):
        return True
    return any(
        LOCAL_TAG in registry.get(n).tags for n in registry.names() if registry.get(n).model == rest
    )


def validate_config(config: ComparisonConfig, registry) -> bool:
    """Refuse a config that names unknown things or breaks the local-only rules.

    Returns
    -------
    bool
        Whether the configured judge is local (True when there is none).

    Raises
    ------
    ValueError
        On an unknown endpoint, tier, profile or experiment, a `pay`
        endpoint, a non-local model for a local-only tier, or a
        non-local judge while a local-only tier is sampled.
    """
    for name, models in config.profiles.items():
        check_endpoints(registry, models, f"profile {name!r}")
    check_endpoints(registry, config.compare_allowed_models, "compare_allowed_models")

    local_only_tiers: set[str] = set()

    for tier, rule in config.sampling.items():
        if not registry.has_tier(tier):
            raise ValueError(f"comparison.sampling names unknown tier {tier!r}")
        models = expand_models(config, rule.models, f"sampling of tier {tier!r}")
        check_endpoints(registry, models, f"sampling of tier {tier!r}")
        _check_local(registry, tier, models, f"sampling of tier {tier!r}", local_only_tiers)

    for name, experiment in config.experiments.items():
        label = f"experiment {name!r}"
        models = expand_models(config, experiment.models, label)
        check_endpoints(registry, models, label)
        for tier in experiment.tiers or []:
            if not registry.has_tier(tier):
                raise ValueError(f"comparison {label} names unknown tier {tier!r}")
            _check_local(registry, tier, models, label, local_only_tiers)

    judge_local = True
    if config.judge is not None:
        judge_local = spec_is_local(registry, config.judge.scorer)
        if not judge_local and local_only_tiers:
            raise ValueError(
                f"comparison.judge {config.judge.scorer!r} is not local, but "
                f"data from local-only tier(s) {', '.join(sorted(local_only_tiers))} "
                "would be judged by it. Use a judge model tagged `local`, an "
                "offline scorer, or drop those tiers from sampling/experiments."
            )
    return judge_local
