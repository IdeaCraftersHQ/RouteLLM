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

import logging
import os
import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictBool, field_validator

from routellm.capabilities import Capabilities

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
    """

    select: str
    order: SelectorOrder = "quality_desc"

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
    """

    name: str
    model: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    quality: Optional[int] = Field(default=None, ge=0, le=100)
    extra: dict[str, Any] = Field(default_factory=dict)
    capabilities: Optional[Capabilities] = None
    strict: bool = False
    quality_measured: bool = False
    pay: StrictBool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(
                f"Invalid endpoint name: {value!r}. "
                "Names must match [A-Za-z0-9_]+ (no hyphens)."
            )
        return value

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Endpoint model must be a non-empty string")
        return value

    def credentials(
        self,
        default_base: Optional[str],
        default_key: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
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
    """

    name: str
    router: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0, le=1)
    intent_routing: Optional[bool] = None
    strong: Union[str, Selector]
    weak: Union[str, Selector]

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
                f"Invalid tier name: {value!r}. "
                "Names must match [A-Za-z0-9_]+ (no hyphens)."
            )
        return value

    @field_validator("strong", "weak")
    @classmethod
    def _validate_reference(
        cls, value: Union[str, Selector]
    ) -> Union[str, Selector]:
        if isinstance(value, Selector):
            return value
        if not value.strip():
            raise ValueError("Tier strong and weak must be non-empty strings")
        return value


def _invert_areas(
    raw: dict[str, Any], tiers: dict[str, "Tier"]
) -> dict[str, str]:
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
        endpoints: Optional[dict[str, Endpoint]] = None,
        tiers: Optional[dict[str, Tier]] = None,
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

        self._validate_tiers()

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Optional[Any] = None
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
            name: Endpoint(name=name, **(spec or {}))
            for name, spec in raw_endpoints.items()
        }

        raw_tiers = config.get("tiers") or {}
        tiers = {
            name: Tier(name=name, **(spec or {})) for name, spec in raw_tiers.items()
        }

        registry = cls(endpoints, tiers)

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

    def area_of(self, tier: Optional[str]) -> Optional[str]:
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
            cycle = ancestors[ancestors.index(name):] + [name]
            raise ValueError(f"Tier cycle: {' -> '.join(cycle)}")

        path = ancestors + [name]
        if len(path) > MAX_TIER_DEPTH:
            raise ValueError(
                f"Tier nesting exceeds the maximum depth of {MAX_TIER_DEPTH}: "
                f"{' -> '.join(path)}"
            )

        tier = self._tiers[name]
        for reference in (tier.strong, tier.weak):
            if not isinstance(reference, Selector) and reference in self._tiers:
                self._walk(reference, path)

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

    def payable_bases(self, default_base: Optional[str] = None) -> list[str]:
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
            raise KeyError(
                f"Unknown tier: {name}. Configured tiers: {known}"
            ) from None

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
            raise KeyError(
                f"Unknown endpoint: {name}. Configured endpoints: {known}"
            ) from None

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
            raise ValueError(
                "Endpoint name or model must be a non-empty string"
            )

        endpoint = self._endpoints.get(name_or_model)
        if endpoint is not None:
            return endpoint

        if name_or_model not in self._warned:
            self._warned.add(name_or_model)
            logger.warning(
                "no endpoint named %s; using it as a raw model with the "
                "controller defaults",
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
