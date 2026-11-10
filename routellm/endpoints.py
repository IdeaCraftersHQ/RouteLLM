"""Named model endpoints with their own base URL and credential.

An endpoint gives a reachable model a stable name, so routing,
caching, and traces refer to "cloud_strong" rather than to whatever
litellm model string happens to sit behind it. The registry is built
from the `endpoints:` key of the YAML passed to `--config`::

    endpoints:
      cloud_strong:
        model: gpt-4o
        api_key_env: OPENAI_API_KEY
        tags: [tools, vision]
        quality: 90
        extra: {timeout: 60}
      local_fast:
        model: ollama_chat/qwen3:8b
        api_base: http://127.0.0.1:11500
        tags: [local]

Credentials are never stored: `api_key_env` names an environment
variable that is read at call time, so a config loads on a machine
that holds none of the keys.

An endpoint may also be authorised to charge the configured wallet::

    endpoints:
      metered:
        model: openai/some-metered-model
        api_base: https://metered.example.com/v1
        pay: true

`pay:` is False everywhere unless written, and enabling a payment
provider does not change that: the provider supplies a wallet, this
names who may spend it. A 402 from any other endpoint is returned
unpaid, which matters because litellm's HTTP session is process-global
and would otherwise put every upstream in front of the wallet.

That says who may charge, never how much. `--max-payment` sets a
process-wide ceiling on a single payment, and a payable endpoint may
lower it further::

    endpoints:
      metered:
        model: openai/some-metered-model
        api_base: https://metered.example.com/v1
        pay: true
        max_payment: "$0.01"

The effective cap is the smaller of the two, so `max_payment:` can
only tighten the ceiling, never raise it. Setting neither leaves the
payment library's own per-payment default standing, which is not the
same as leaving a payment unbounded.

A tier names a strong/weak pair whose sides may themselves be tiers, so
a request addressed to a tier walks a tree, one router call per level::

    tiers:
      premium: {router: jev, threshold: 0.33, strong: cloud_strong, weak: frontier_local}
      default: {router: mf,  threshold: 0.12, strong: premium,      weak: local_fast}

Tier and endpoint names share one namespace and are validated together
at load: every side must name a known endpoint or tier, the graph must
be acyclic, and it may not nest deeper than `MAX_TIER_DEPTH` levels.

A side may instead be a `Selector`, saying what it wants rather than
naming it. `routellm.pairing` resolves every selector to an endpoint
name at controller construction, before this validation runs a second
time, so nothing downstream sees one::

    tiers:
      default:
        strong: {select: "tool_call:true reasoning:true", order: quality_desc}
        weak:   {select: "tag:local", order: cost_asc}

Names are restricted to `[A-Za-z0-9_]+`; hyphens are reserved for the
model-name grammar that splits on '-'.
"""

import difflib
import logging
import os
import re
from functools import cached_property
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from routellm.capabilities import Capabilities
from routellm.governance import (
    check_namespaces,
    check_tag_namespaces_config,
    parse_jur,
    parse_reach,
)
from routellm.requirements import DEFAULT_AUDIO_TOKENS_PER_SECOND
from routellm.transcription import LOCAL_TAG, TranscriberConfig

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

MAX_TIER_DEPTH = 4

#: Orderings a `Selector` may ask for over its candidates.
SelectorOrder = Literal[
    "cost_asc",
    "cost_desc",
    "quality_asc",
    "quality_desc",
    "context_desc",
    "max_output_desc",
]


class Selector(BaseModel):
    """A policy standing in for an endpoint name on a tier side.

    Instead of naming one endpoint, a side may say what it wants: a
    `select` expression over endpoint tags and models.dev facts, and
    an `order` deciding which matching endpoint wins. Selectors are
    resolved to endpoint names once, at controller construction, by
    `routellm.pairing`; nothing downstream ever sees one.

    Attributes
    ----------
    select : str
        Space-separated terms, all ANDed. `tag:<label>` matches an
        endpoint's own tags; every other `key:value` term is a
        models.dev catalog term.
    order : str
        Which matching endpoint wins, one of `SelectorOrder`.
        Default `quality_desc`. Endpoints missing the value an order
        reads always sort last, whichever direction it runs. Both
        quality orders break ties on release date newest-first, so
        `quality_asc` reverses only the quality scores, not the dates.
    pool : int
        How many matching endpoints the side keeps, best first. The
        first is the side itself; the rest are tried in order, after it
        and before the other side, when it fails or is cooling after a
        rate limit. Default 1: no pool.
    """

    select: str
    order: SelectorOrder = "quality_desc"
    pool: int = Field(default=1, ge=1)

    @field_validator("select")
    @classmethod
    def _validate_select(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Selector select must be a non-empty string")
        return value


class Endpoint(BaseModel):
    """A named, individually addressable model endpoint.

    Attributes
    ----------
    name : str
        Registry name, matching `[A-Za-z0-9_]+`.
    model : str
        Non-empty litellm model name to send to the provider.
    api_base : str, optional
        Base URL for this endpoint; falls back to the controller default.
    api_key_env : str, optional
        Name of the environment variable holding this endpoint's key.
        Never the key itself.
    tags : list[str]
        Free-form labels for selection by later routing layers.
    quality : int, optional
        Manual quality score in [0, 100].
    extra : dict
        Extra keyword arguments passed to litellm. Request kwargs win.
    capabilities : Capabilities, optional
        What this endpoint can do, stated explicitly. Highest
        precedence source; what it leaves None is filled from the
        deprecated tags, then from the models.dev catalog.
    strict : bool
        Whether an unknown capability refuses a request that needs it.
        Default False: an unknown capability serves by default, so a
        config that declares nothing keeps routing as it always did.
    quality_measured : bool
        Whether `quality` came from a `quality_from:` sidecar rather
        than from this endpoint's own `quality:`. Read only by the
        `--capabilities` matrix, which marks each score accordingly.
    pay : bool
        Whether this endpoint is authorised to charge the configured
        wallet. Default False: an endpoint that says nothing about
        payment never has a payment signed for it, however
        well-formed the 402 it answers with. Enabling a payment
        provider supplies a wallet; this is what names who may spend
        it.
    timeout_ms : int, optional
        Per-attempt timeout for this endpoint, overriding the server's
        `resilience.timeout_ms`. None keeps the global value.
    drop_params : bool or list[str], optional
        What to do with a request param this endpoint's provider does
        not support. Unset drops the advisory flags the provider lacks
        (`parallel_tool_calls`, `reasoning_effort`, `stream_options`)
        and fails on anything else; `true` drops anything the provider
        rejects; `false` drops nothing; a list of names drops those
        for this endpoint as well. See `routellm.leaf_params`.
    max_payment : str, optional
        The most a single payment to this endpoint may be, as human
        money ("$0.01"). It may only lower the process-wide
        `--max-payment` ceiling, never raise it, so the effective cap
        is the smaller of the two. None means this endpoint names no
        cap of its own and the ceiling alone applies. Requires
        `pay: true`: a cap on an endpoint that can never spend is at
        best dead config, and at worst a belief that writing it is
        what authorises paying.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    tags: list[str] = Field(default_factory=list)
    quality: int | None = Field(default=None, ge=0, le=100)
    extra: dict[str, Any] = Field(default_factory=dict)
    capabilities: Capabilities | None = None
    strict: bool = False
    quality_measured: bool = False
    pay: StrictBool = False
    max_payment: str | None = None
    timeout_ms: int | None = Field(default=None, gt=0)
    drop_params: StrictBool | list[Annotated[str, StringConstraints(min_length=1)]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _refuse_unknown_keys(cls, data: Any) -> Any:
        """Name the endpoint and the key, with a near-miss suggestion.

        A mistyped key that was silently ignored (`api_base_env:` for
        `api_base:`) would let the endpoint fall back to its provider's
        public default base. `extra="forbid"` is the backstop; this runs
        first so the message says which endpoint and what was meant.
        """
        if not isinstance(data, dict):
            return data
        known = sorted(cls.model_fields)
        for key in data:
            if key in cls.model_fields:
                continue
            hint = difflib.get_close_matches(str(key), known, n=1)
            raise ValueError(
                f"endpoint {data.get('name')!r}: unknown key {key!r}"
                + (f" (did you mean {hint[0]!r}?)" if hint else "")
                + f"; allowed keys: {', '.join(known)}"
            )
        return data

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(
                f"Invalid endpoint name: {value!r}. Names must match [A-Za-z0-9_]+ (no hyphens)."
            )
        return value

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Endpoint model must be a non-empty string")
        return value

    @field_validator("max_payment")
    @classmethod
    def _validate_max_payment(cls, value: str | None) -> str | None:
        """Refuse a cap the payment library could not act on.

        A cap nobody can parse has to fail at load. Dropping it back
        to None would leave the operator believing a limit is in
        force, and the first they would hear of it is a payment that
        went through.
        """
        if value is None:
            return None

        from routellm.payment.limits import parse_cap

        parse_cap(value)
        return value

    @model_validator(mode="after")
    def _validate_governance_tags(self) -> "Endpoint":
        """Refuse a malformed or repeated `jur:` tag at construction."""
        parse_jur(self.name, self.tags)
        return self

    @cached_property
    def reach(self) -> frozenset[str]:
        """The subset of `local`, `vpn`, `cloud` this endpoint carries."""
        return parse_reach(self.tags)

    @cached_property
    def jur(self) -> str | None:
        """The lowercase country code of its one `jur:` tag, or None."""
        return parse_jur(self.name, self.tags)

    @model_validator(mode="after")
    def _validate_local_has_base(self) -> "Endpoint":
        """A `local` endpoint names where it lives, or it does not load.

        Without `api_base` the call goes to the provider's default base
        (public OpenAI for an `openai/` model), so a leaf meant to stay
        on the machine would send its prompts to the cloud. Tag-only
        enforcement of the privacy intent: no implicit fallback.
        """
        if LOCAL_TAG in self.tags and not self.api_base:
            raise ValueError(
                f"endpoint {self.name!r} is tagged `local` but sets no `api_base`; "
                "without one the call would go to the provider's default (public) "
                "base. Set `api_base`, or drop the `local` tag."
            )
        return self

    @model_validator(mode="after")
    def _validate_cap_needs_pay(self) -> "Endpoint":
        """A cap on an unpayable endpoint cannot mean what it says."""
        if self.max_payment is not None and not self.pay:
            raise ValueError(
                f"endpoint {self.name!r} sets max_payment but not `pay: true`, "
                "so it can never spend and the cap bounds nothing. Add "
                "`pay: true` to authorise it, or drop max_payment."
            )
        return self

    def credentials(
        self,
        default_base: str | None,
        default_key: str | None,
    ) -> tuple[str | None, str | None]:
        """Resolve the base URL and API key for a call to this endpoint.

        The environment is read here rather than at config load, so a
        config that names variables loads on a machine that holds none
        of them.

        Parameters
        ----------
        default_base : str, optional
            Base URL to use when this endpoint sets none.
        default_key : str, optional
            API key to use when this endpoint names no environment
            variable.

        Returns
        -------
        tuple[str or None, str or None]
            The `(api_base, api_key)` pair for this call.

        Raises
        ------
        ValueError
            If `api_key_env` is set but that variable is not in the
            environment.
        """
        api_base = self.api_base or default_base

        if self.api_key_env is None:
            return api_base, default_key

        api_key = os.environ.get(self.api_key_env)
        if api_key is None:
            raise ValueError(
                f"Endpoint {self.name!r} needs environment variable "
                f"{self.api_key_env!r}, which is not set."
            )
        return api_base, api_key


class Tier(BaseModel):
    """A named strong/weak pair whose sides may be tiers themselves.

    Router and threshold are optional: a tier that names neither takes
    them from the level above it, then from the request, then from the
    controller defaults.

    Attributes
    ----------
    name : str
        Tier name, matching `[A-Za-z0-9_]+`.
    router : str, optional
        Router to run at this level. Inherited when None.
    threshold : float, optional
        Decision threshold in [0, 1] for this level. Inherited when None.
    intent_routing : bool, optional
        Whether a request addressed to this tier may be re-routed to
        the tier an intent classifier picks. Unset means True for the
        tier named `default` and False for every other, so an app that
        chose a tier by name is never overruled by a classifier.
    strong : str or Selector
        Endpoint or tier taken when the win rate clears the threshold,
        or a `Selector` resolved to one at controller construction.
    weak : str or Selector
        Endpoint or tier taken otherwise, or a `Selector`.
    transcriber : dict, optional
        Keys of the top-level `transcriber:` section this tier
        overrides, key by key. Validated, merged with the global
        section, when the registry is built.
    """

    name: str
    router: str | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)
    intent_routing: bool | None = None
    strong: str | Selector
    weak: str | Selector
    transcriber: dict[str, Any] | None = None

    def accepts_intent_routing(self) -> bool:
        """Return whether an intent may choose a tier in place of this one.

        Returns
        -------
        bool
            The declared `intent_routing` when this tier sets one,
            otherwise True for the tier named `default` and False for
            any other.
        """
        if self.intent_routing is not None:
            return self.intent_routing
        return self.name == "default"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(
                f"Invalid tier name: {value!r}. Names must match [A-Za-z0-9_]+ (no hyphens)."
            )
        return value

    @field_validator("strong", "weak")
    @classmethod
    def _validate_reference(cls, value: str | Selector) -> str | Selector:
        if isinstance(value, Selector):
            return value
        if not value.strip():
            raise ValueError("Tier strong and weak must be non-empty strings")
        return value


def _invert_areas(raw: dict[str, Any], tiers: dict[str, "Tier"]) -> dict[str, str]:
    """Invert `{area: [tier, ...]}` into `{tier: area}`.

    Parameters
    ----------
    raw : dict
        The config's `areas:` section.
    tiers : dict[str, Tier]
        The configured tiers, checked against so a typo in an area is
        caught at load rather than silently grouping nothing.

    Returns
    -------
    dict[str, str]
        Tier name to area name.

    Raises
    ------
    ValueError
        If an area names a tier that does not exist, or a tier appears
        in two areas.
    """
    inverted: dict[str, str] = {}
    for area, names in raw.items():
        for tier in names or []:
            if tier not in tiers:
                raise ValueError(
                    f"Area {area!r} names tier {tier!r}, which is not "
                    f"configured. Known tiers: "
                    f"{', '.join(sorted(tiers)) or '<none>'}."
                )
            if tier in inverted:
                raise ValueError(
                    f"Tier {tier!r} is in two areas, {inverted[tier]!r} "
                    f"and {area!r}; a tier belongs to at most one."
                )
            inverted[tier] = area
    return inverted


class EndpointRegistry:
    """Lookup of endpoints by name, with raw-model passthrough.

    A name the registry does not know is treated as a raw litellm model
    name and wrapped in an anonymous endpoint carrying no base URL and
    no credential, so calls written against model names keep working.
    """

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        tiers: dict[str, Tier] | None = None,
        transcriber: dict[str, Any] | None = None,
        intent_targets: list[str] | None = None,
        audio_tokens_per_second: float = DEFAULT_AUDIO_TOKENS_PER_SECOND,
        governance: bool = False,
        tag_namespaces: list[str] | None = None,
    ):
        """Initialize the registry.

        Parameters
        ----------
        endpoints : dict[str, Endpoint], optional
            Endpoints keyed by name. Empty when None.
        tiers : dict[str, Tier], optional
            Tiers keyed by name. Empty when None. Validated here, so a
            registry built directly is held to the same rules as one
            built from a config.

        Raises
        ------
        ValueError
            If a name is both an endpoint and a tier, a tier side names
            neither, the tier graph has a cycle, or it nests deeper than
            `MAX_TIER_DEPTH`.
        """
        self._endpoints: dict[str, Endpoint] = dict(endpoints or {})
        self._tiers: dict[str, Tier] = dict(tiers or {})
        self._warned: set[str] = set()

        # Measured per-area quality, `{endpoint: {area: quality}}`,
        # filled by `routellm.quality_scores.apply_sidecar` when the
        # config points at a sidecar. Empty means nothing was measured,
        # and pairing then orders on the overall number as it always
        # has.
        self.area_quality: dict[str, dict[str, int]] = {}

        # Tier name to area name, inverted from the config's
        # `areas: {area: [tier, ...]}`. Empty means no tier has an
        # area, and pairing then orders on overall quality as always.
        self.areas: dict[str, str] = {}

        # `(tier, side)` to the ordered endpoint names a selector with
        # `pool: N` resolved to, head first. A side with no pool has no
        # entry; `pool_for` answers it with the side itself.
        self._pools: dict[tuple[str, str], list[str]] = {}

        self._transcriber_raw: dict[str, Any] | None = transcriber
        self._transcribers: dict[str | None, TranscriberConfig] = {}
        self.intent_targets: list[str] = list(intent_targets or [])

        # Advisory tokens charged per second of audio in the context
        # estimate; see `routellm.requirements`.
        self.audio_tokens_per_second: float = audio_tokens_per_second

        if not isinstance(governance, bool):
            raise ValueError("`governance:` must be true or false")
        self.governance: bool = governance
        self.known_namespaces: frozenset[str] = check_tag_namespaces_config(tag_namespaces)

        self._validate_governance_tags()
        self._validate_tiers()
        self._validate_transcribers()

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Any | None = None
    ) -> "EndpointRegistry":
        """Build a registry from a loaded config dict.

        Reads the `endpoints:`, `tiers:` and `quality_from:` keys;
        every other top-level key is left to its own owner.

        `quality_from:` names a measured-quality sidecar, resolved
        relative to the CONFIG FILE rather than the CWD. The sidecar
        WINS over a hand-written `quality:`, because it is measured and
        the YAML number is a guess someone typed once; set
        `quality_from_override: false` to flip that, so an explicit
        number survives and the sidecar only fills endpoints that set
        none. Either way an endpoint the sidecar does not name keeps
        exactly the ordering it has today.

        Parameters
        ----------
        config : dict
            Loaded YAML config. Absent keys yield empty collections.
        config_path : str or Path, optional
            The file `config` was read from, used to resolve a relative
            `quality_from:`. Without it a relative path falls back to
            the CWD, which is only ever right for an in-memory config.

        Returns
        -------
        EndpointRegistry
            Registry holding one `Endpoint` and one `Tier` per
            configured name.

        Raises
        ------
        ValueError
            If any endpoint or tier is malformed, the tier graph fails
            validation, or the sidecar is malformed or written by a
            version this code does not read.
        FileNotFoundError
            If `quality_from:` names a file that is not there. The
            message names the config key, so the fix is readable from
            the error alone.
        """
        config = config or {}

        raw_endpoints = config.get("endpoints") or {}
        endpoints = {
            name: Endpoint(name=name, **(spec or {})) for name, spec in raw_endpoints.items()
        }

        raw_tiers = config.get("tiers") or {}
        tiers = {name: Tier(name=name, **(spec or {})) for name, spec in raw_tiers.items()}

        raw_transcriber = config.get("transcriber")
        if raw_transcriber is not None and not isinstance(raw_transcriber, dict):
            raise ValueError("`transcriber:` must be a mapping")

        intents = config.get("intents")
        targets = (intents.get("tiers") or {}).values() if isinstance(intents, dict) else []
        rate = config.get("audio_tokens_per_second")
        if rate is None:
            rate = DEFAULT_AUDIO_TOKENS_PER_SECOND
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
            raise ValueError("`audio_tokens_per_second` must be a positive number")

        registry = cls(
            endpoints,
            tiers,
            raw_transcriber,
            [str(t) for t in targets],
            audio_tokens_per_second=rate,
            governance=config.get("governance", False),
            tag_namespaces=config.get("tag_namespaces"),
        )

        # The sidecar merges HERE, on the built registry, so every
        # consumer that goes through from_config -- the server and both
        # pairing CLIs -- orders on the same numbers. Per-area quality
        # needs the registry itself, which is why this cannot run on the
        # raw endpoint mapping above.
        #
        # Precedence: the sidecar wins, because it is measured and a
        # hand-written `quality:` is a guess someone typed once. Set
        # `quality_from_override: false` to flip it.
        quality_from = config.get("quality_from")
        if quality_from:
            from routellm.quality_scores import (
                apply_sidecar,
                load_sidecar,
                resolve_sidecar_path,
            )

            override = config.get("quality_from_override", True)
            apply_sidecar(
                registry,
                load_sidecar(str(resolve_sidecar_path(quality_from, config_path))),
                override=bool(override),
            )

        registry.areas = _invert_areas(config.get("areas") or {}, tiers)
        return registry

    def area_of(self, tier: str | None) -> str | None:
        """Return the area a tier belongs to, or None.

        A null tier is tolerated: a flat pair has no tier at all
        (`routing.py` writes `tier: None` for one), and asking for its
        area must answer None rather than raise.

        Parameters
        ----------
        tier : str or None
            Tier name.

        Returns
        -------
        str or None
            The area name, or None when the tier has none.
        """
        if tier is None:
            return None
        return self.areas.get(tier)

    def revalidate(self) -> None:
        """Re-run tier validation after the tiers were rewritten in place.

        `routellm.pairing` replaces selector sides with endpoint names
        at controller construction; this re-checks the graph those names
        now form.

        Raises
        ------
        ValueError
            On the same conditions as construction.
        """
        self._validate_tiers()
        self._validate_transcribers()

    def _validate_governance_tags(self) -> None:
        """Hold every endpoint's tags to the governance grammar.

        Namespaced tags need a known namespace always; under
        `governance: true` every endpoint must also carry a reach tag
        and exactly one `jur:` tag.
        """
        for ep in self._endpoints.values():
            check_namespaces(ep.name, ep.tags, self.known_namespaces)
            if not self.governance:
                continue
            if not ep.reach:
                raise ValueError(
                    f"endpoint {ep.name!r} carries no reach tag under `governance: true`; "
                    "add at least one of local, vpn, cloud."
                )
            if ep.jur is None:
                raise ValueError(
                    f"endpoint {ep.name!r} carries no jur: tag under `governance: true`; "
                    "add exactly one, e.g. `jur:ca`."
                )

    def _validate_tiers(self) -> None:
        """Check the tier namespace, its references, cycles, and depth."""
        if not self._tiers:
            return

        collisions = sorted(set(self._tiers) & set(self._endpoints))
        if collisions:
            raise ValueError(
                "Names are both a tier and an endpoint: "
                f"{', '.join(collisions)}. Tier and endpoint names must "
                "be distinct."
            )

        for tier in self._tiers.values():
            for side, reference in (("strong", tier.strong), ("weak", tier.weak)):
                # A Selector names nothing yet; `routellm.pairing`
                # rewrites it into an endpoint name before the
                # controller validates the resolved registry.
                if isinstance(reference, Selector):
                    continue
                if reference in self._tiers or reference in self._endpoints:
                    continue
                # A tier may not name itself: that is a one-node cycle
                # the walk below rejects, so never suggest it.
                others = sorted(set(self._tiers) - {tier.name})
                known = ", ".join(others + self.names()) or "<none>"
                raise ValueError(
                    f"Tier {tier.name!r} names unknown {side} {reference!r}. "
                    f"Known tiers and endpoints: {known}"
                )

        # Every tier is walked as its own root, so a tier reachable only
        # from another still has its own nesting checked. Repeated work
        # on shared subtrees is bounded by the depth cap.
        for name in sorted(self._tiers):
            self._walk(name, [])

    def _walk(self, name: str, ancestors: list[str]) -> None:
        """Depth-first check of one tier for cycles and excess depth.

        Parameters
        ----------
        name : str
            Tier being entered.
        ancestors : list[str]
            Tiers already on this path, outermost first.

        Raises
        ------
        ValueError
            If `name` is already on the path, or entering it would
            exceed `MAX_TIER_DEPTH` tier levels.
        """
        if name in ancestors:
            cycle = ancestors[ancestors.index(name) :] + [name]
            raise ValueError(f"Tier cycle: {' -> '.join(cycle)}")

        path = ancestors + [name]
        if len(path) > MAX_TIER_DEPTH:
            raise ValueError(
                f"Tier nesting exceeds the maximum depth of {MAX_TIER_DEPTH}: {' -> '.join(path)}"
            )

        tier = self._tiers[name]
        for reference in (tier.strong, tier.weak):
            if not isinstance(reference, Selector) and reference in self._tiers:
                self._walk(reference, path)

    @property
    def transcriber_raw(self) -> dict[str, Any] | None:
        """Return the raw `transcriber:` section the registry was built with."""
        return self._transcriber_raw

    def transcriber_for(self, tier: str | None) -> TranscriberConfig | None:
        """Return the transcriber settings a request to `tier` runs under.

        The tier's own `transcriber:` keys are laid over the global
        section. A request addressing no tier gets the global section.

        Parameters
        ----------
        tier : str, optional
            The tier the request addresses.

        Returns
        -------
        TranscriberConfig or None
            None when neither the tier nor the config names a
            transcriber. A config in mode `passthrough` is returned as
            is; its `active` is False.
        """
        return self._transcribers.get(tier)

    def _validate_transcribers(self) -> None:
        """Build every tier's transcriber config and apply the local-only rule.

        A tier is local-only when every endpoint it can reach carries
        the `local` tag, the label endpoints already use for `tag:local`
        selection. Such a tier may not transcribe through a service not
        tagged `local` either: its audio would leave the place its text
        is confined to. A tier whose sides are still unresolved
        selectors cannot be judged yet; `revalidate` judges it once
        pairing has named them.

        Raises
        ------
        ValueError
            On an invalid transcriber section, or a local-only tier
            using a non-local transcriber that would transcribe.
        """
        built: dict[str | None, TranscriberConfig] = {}
        base = self._transcriber_raw or {}

        def build(label: str, spec: dict[str, Any]) -> TranscriberConfig:
            try:
                return TranscriberConfig(**spec)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc']) or '<section>'}: {err['msg']}"
                    for err in exc.errors()
                )
                raise ValueError(f"invalid {label}: {problems}") from exc

        if self._transcriber_raw is not None:
            built[None] = build("`transcriber:` section", base)

        for name, tier in self._tiers.items():
            if self._transcriber_raw is None and tier.transcriber is None:
                continue
            built[name] = build(
                f"`transcriber:` of tier {name!r}", {**base, **(tier.transcriber or {})}
            )

        for name, config in built.items():
            if name is None or not config.active or config.is_local:
                continue
            leaves = self._leaves(name)
            if leaves and all(LOCAL_TAG in self._endpoints[leaf].tags for leaf in leaves):
                raise ValueError(
                    f"Tier {name!r} is local-only (every endpoint it can reach is "
                    f"tagged {LOCAL_TAG!r}) but its transcriber is not tagged "
                    f"{LOCAL_TAG!r}, so audio would leave the machine to be "
                    f"transcribed. Point the tier at a local transcriber and tag it "
                    f"`tags: [{LOCAL_TAG}]`, or set `mode: passthrough`."
                )

        # Intent routing re-routes AFTER audio would already have been
        # transcribed (it classifies the prompt, which includes the
        # transcript). A tier that may be re-routed to a local-only tier
        # must therefore itself transcribe locally.
        for name, config in built.items():
            if name is None or not config.active or config.is_local:
                continue
            if not self._tiers[name].accepts_intent_routing():
                continue
            for target in self.intent_targets:
                if target == name or target not in self._tiers:
                    continue
                leaves = self._leaves(target)
                if leaves and all(LOCAL_TAG in self._endpoints[leaf].tags for leaf in leaves):
                    raise ValueError(
                        f"Tier {name!r} accepts intent routing and an intent can send it "
                        f"to local-only tier {target!r}, but its transcriber is not tagged "
                        f"{LOCAL_TAG!r}: audio would be transcribed off-machine before the "
                        f"move. Use a local transcriber on {name!r}, set "
                        f"`intent_routing: false` on it, or `mode: passthrough`."
                    )

        self._transcribers = built

    def _leaves(self, name: str) -> set[str] | None:
        """Return the endpoint names reachable from tier `name`.

        Returns None when the tier still has an unresolved selector
        below it, because its leaves are not yet known.
        """
        leaves: set[str] = set()
        stack = [name]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            tier = self._tiers[current]
            for side in (tier.strong, tier.weak):
                if isinstance(side, Selector):
                    return None
                if side in self._tiers:
                    stack.append(side)
                else:
                    leaves.add(side)
        return leaves

    @property
    def tiers(self) -> dict[str, Tier]:
        """Return a shallow copy of the configured tiers, keyed by name.

        The mapping is fresh, so adding or removing a key leaves the
        registry alone. The `Tier` objects in it are the registry's own:
        pairing rewrites a resolved side through this mapping, and every
        holder sees that change.
        """
        return dict(self._tiers)

    def names(self) -> list[str]:
        """Return the configured endpoint names, sorted."""
        return sorted(self._endpoints)

    def tier_names(self) -> list[str]:
        """Return the configured tier names, sorted."""
        return sorted(self._tiers)

    def has_tier(self, name: str) -> bool:
        """Return whether `name` is a configured tier."""
        return name in self._tiers

    def payable_bases(self, default_base: str | None = None) -> list[str]:
        """Return the base URLs the configured endpoints may be charged on.

        Only endpoints carrying `pay: true` contribute, so an endpoint
        that says nothing about payment never appears here and never
        has a payment signed for it. A payable endpoint setting no
        `api_base` is reached at `default_base`, which is therefore
        what gets authorised for it; when there is no default either,
        it contributes nothing rather than an empty authorisation
        nobody could read.

        A raw model name resolved through the registry's passthrough is
        never payable: it carries no `pay` flag, because no config line
        ever named it.

        Parameters
        ----------
        default_base : str, optional
            The controller's own base URL, used by any payable endpoint
            that sets none of its own.

        Returns
        -------
        list[str]
            The distinct base URLs, in configuration order. Empty when
            no endpoint asked to pay, which authorises nothing.
        """
        bases: list[str] = []
        for endpoint in self._endpoints.values():
            if not endpoint.pay:
                continue
            base = endpoint.api_base or default_base
            if not base:
                logger.warning(
                    "endpoint %r is marked payable but has no api_base and "
                    "there is no default one, so nothing is authorised for "
                    "it and a 402 from it will not be paid",
                    endpoint.name,
                )
                continue
            if base not in bases:
                bases.append(base)
        return bases

    def payment_caps(self, default_base: str | None = None) -> dict[str, str]:
        """Return each payable endpoint's own cap, keyed on its base URL.

        The scope a payment is enforced against is keyed on base URLs,
        so the caps have to be too. Only endpoints carrying both
        `pay: true` and a `max_payment:` contribute; everything else
        is left to the process-wide ceiling.

        Two payable endpoints may share one base URL, which the scope
        already collapses into a single authorisation. There is only
        one cap to enforce for it, so the lowest wins: taking the
        higher would let one endpoint's config raise the other's
        limit.

        Parameters
        ----------
        default_base : str, optional
            The controller's own base URL, used by a payable endpoint
            that sets none of its own, exactly as `payable_bases`
            does.

        Returns
        -------
        dict
            Base URL to the cap in force for it. Empty when no payable
            endpoint named one.
        """
        from routellm.payment.limits import parse_cap

        caps: dict[str, str] = {}
        for endpoint in self._endpoints.values():
            if not endpoint.pay or endpoint.max_payment is None:
                continue
            base = endpoint.api_base or default_base
            if not base:
                logger.warning(
                    "endpoint %r sets max_payment but has no api_base and "
                    "there is no default one, so there is no base URL to "
                    "enforce the cap on",
                    endpoint.name,
                )
                continue
            existing = caps.get(base)
            if existing is None or parse_cap(endpoint.max_payment) < parse_cap(existing):
                caps[base] = endpoint.max_payment
        return caps

    def set_pool(self, tier: str, side: str, names: list[str]) -> None:
        """Record the ordered endpoints a tier side's selector resolved to."""
        self._pools[(tier, side)] = list(names)

    def pool_for(self, tier: str, side: str) -> list[str]:
        """Return the endpoints a tier side tries, in order, head first.

        A side that names one endpoint or tier, or a selector with no
        pool, answers with just that side.
        """
        pooled = self._pools.get((tier, side))
        if pooled is not None:
            return list(pooled)
        return [getattr(self.get_tier(tier), side)]

    def get_tier(self, name: str) -> Tier:
        """Return the tier registered under `name`.

        Parameters
        ----------
        name : str
            Tier name.

        Returns
        -------
        Tier
            The registered tier.

        Raises
        ------
        KeyError
            If no tier carries that name. The message lists the known
            names.
        """
        try:
            return self._tiers[name]
        except KeyError:
            known = ", ".join(self.tier_names()) or "<none>"
            raise KeyError(f"Unknown tier: {name}. Configured tiers: {known}") from None

    def get(self, name: str) -> Endpoint:
        """Return the endpoint registered under `name`.

        Parameters
        ----------
        name : str
            Endpoint name.

        Returns
        -------
        Endpoint
            The registered endpoint.

        Raises
        ------
        KeyError
            If no endpoint carries that name. The message lists the
            known names.
        """
        try:
            return self._endpoints[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"Unknown endpoint: {name}. Configured endpoints: {known}") from None

    def resolve(self, name_or_model: str) -> Endpoint:
        """Return an endpoint for a name or a raw litellm model name.

        A configured name returns its endpoint. Anything else becomes an
        anonymous endpoint whose model is the given string and which
        carries no base URL and no credential, so the controller
        defaults apply. The first anonymous resolution of each distinct
        string logs one warning; repeats stay quiet. The set of
        already-warned names is per registry instance, so a new
        registry warns afresh.

        Parameters
        ----------
        name_or_model : str
            Endpoint name, or a litellm model name.

        Returns
        -------
        Endpoint
            The configured or anonymous endpoint.

        Raises
        ------
        ValueError
            If `name_or_model` is empty or only whitespace.
        """
        if not name_or_model or not name_or_model.strip():
            raise ValueError("Endpoint name or model must be a non-empty string")

        endpoint = self._endpoints.get(name_or_model)
        if endpoint is not None:
            return endpoint

        if name_or_model not in self._warned:
            self._warned.add(name_or_model)
            logger.warning(
                "no endpoint named %s; using it as a raw model with the controller defaults",
                name_or_model,
            )
        return Endpoint.model_construct(
            name=name_or_model,
            model=name_or_model,
            api_base=None,
            api_key_env=None,
            tags=[],
            quality=None,
            extra={},
            capabilities=None,
            strict=False,
            quality_measured=False,
        )
