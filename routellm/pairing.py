"""Policy-based pairing: a tier side that selects instead of naming.

A tier side normally names one endpoint. With a `Selector` it says what
it wants and the winner is chosen once, at controller construction,
from the configured endpoints' own tags plus the models.dev catalog::

    tiers:
      default:
        router: mf
        threshold: 0.12
        strong: {select: "tool_call:true reasoning:true", order: quality_desc}
        weak:   {select: "tag:local", order: cost_asc}

`select` is a space-separated list of terms, all ANDed. A `tag:<label>`
term is answered locally from `Endpoint.tags`; every other term is a
models.dev fact handed to `hop.aim.parse_query`, so the key set is the
one aim documents and an unknown key surfaces as a config error naming
the term. Bare free-text tokens are rejected: a policy says what a
model *is*, never what its name looks like.

An endpoint reaches the catalog through its litellm model name:
`litellm.get_llm_provider` splits it, a small alias table maps the
litellm provider onto the models.dev provider id, and the record is
matched by `(provider, id)`. Names litellm cannot split, and providers
with no models.dev counterpart such as ollama, become tag-only
candidates: they satisfy `tag:` terms and fail every catalog term.

Fetching is synchronous and never touches the caller's event loop: the
async `hop.aim.Registry` runs on a one-shot helper thread with its own
loop, because the server builds its Controller inside an async lifespan
where `asyncio.run` is not allowed. The result is snapshotted under
`$XDG_CACHE_HOME/routellm/models_dev.json` with a 24h TTL; a failed
fetch falls back to a stale snapshot with a warning, and only a policy
that actually needs a catalog term fails when neither is available.

The pick is explainable without a server, and the same command prints
the capability matrix, which is why there is deliberately no
`python -m routellm.capabilities`::

    python -m routellm.pairing            # the discovered config
    python -m routellm.pairing --config config.yaml
    python -m routellm.pairing --config config.yaml --capabilities
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from routellm.config import ConfigError, load_config
from routellm.capabilities import (
    CAPABILITY_KEYS,
    build_tier_index,
    RANGE_KEYS,
    Capabilities,
    CapabilityQuery,
    capabilities_for,
    matches,
    parse_capability_terms,
)
from routellm.endpoints import Endpoint, EndpointRegistry, Selector

logger = logging.getLogger(__name__)

__all__ = [
    "CATALOG_CACHE_ENV",
    "CATALOG_TTL_SECONDS",
    "CatalogUnavailable",
    "ModelRecord",
    "Selector",
    "catalog_provider_for",
    "rank_candidates",
    "records_for_registry",
    "resolve_pairing",
    "resolve_registry_pairings",
]

#: Environment variable overriding the snapshot path. Tests point it
#: at a temporary file; operators rarely need it.
CATALOG_CACHE_ENV = "ROUTELLM_CATALOG_CACHE"

#: How long a snapshot counts as fresh.
CATALOG_TTL_SECONDS = 24 * 60 * 60

#: Timeout for the helper thread that runs the async catalog fetch.
CATALOG_FETCH_TIMEOUT = 30

#: Name of the extra that installs the optional catalog client.
PAIRING_EXTRA = "pairing"

#: litellm provider name -> models.dev provider id. None marks a
#: provider models.dev does not list (local runtimes), whose endpoints
#: stay tag-only candidates.
PROVIDER_ALIASES: dict[str, Optional[str]] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "vertex_ai": "google",
    "groq": "groq",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "xai": "xai",
    "together_ai": "togetherai",
    "fireworks_ai": "fireworks-ai",
    # models.dev namespaces every openrouter id by its vendor, and
    # litellm strips only its own `openrouter/` prefix, so what is left
    # of `openrouter/deepseek/deepseek-chat` is already the catalog id
    # `deepseek/deepseek-chat`. No extra splitting is needed.
    "openrouter": "openrouter",
    # litellm routes chat completions as `cohere_chat` and embeddings as
    # bare `cohere`; models.dev lists both under `cohere`, with bare ids.
    "cohere": "cohere",
    "cohere_chat": "cohere",
    "ollama": None,
    "ollama_chat": None,
}


class CatalogUnavailable(RuntimeError):
    """Raised when a policy needs the catalog and none can be had."""


@dataclass
class ModelRecord:
    """One models.dev model, flattened to the facts pairing reads.

    Attributes
    ----------
    provider : str
        models.dev provider id.
    id : str
        Model id within that provider.
    cost_input : float, optional
        USD per 1M input tokens. None when models.dev lists none.
    cost_output : float, optional
        USD per 1M output tokens. None when models.dev lists none.
    tool_call : bool
        Whether the model supports tool calls.
    reasoning : bool
        Whether the model is a reasoning model.
    context : int, optional
        Context window in tokens.
    release_date : str, optional
        ISO date string, compared lexically as a quality tiebreak.
    structured_output : bool
        Whether the model honours a response schema.
    open_weights : bool
        Whether the model's weights are published.
    modalities_in : list[str]
        Input modalities the model accepts, e.g. `["text", "image"]`.
    max_output : int, optional
        Maximum output tokens in one response.

    Every field added after the first release carries a default, so a
    snapshot written before it existed still reads: `ModelRecord(**entry)`
    over an older dict fills the missing keys from these defaults.
    """

    provider: str
    id: str
    cost_input: Optional[float] = None
    cost_output: Optional[float] = None
    tool_call: bool = False
    reasoning: bool = False
    context: Optional[int] = None
    release_date: Optional[str] = None
    structured_output: bool = False
    open_weights: bool = False
    modalities_in: list[str] = field(default_factory=list)
    max_output: Optional[int] = None


@dataclass
class Candidate:
    """One endpoint considered for a side, with its catalog record.

    Attributes
    ----------
    name : str
        Endpoint name.
    endpoint : Endpoint
        The configured endpoint.
    record : ModelRecord, optional
        Its models.dev record, None when the model maps to no catalog
        entry. A candidate without one fails every catalog term.
    capabilities : Capabilities, optional
        The endpoint's merged capabilities, built once per candidate by
        `rank_candidates` and read by the capability terms and the
        `max_output_desc` order.
    effective_quality : int, optional
        The quality this candidate was actually ordered on: the
        per-area measurement when there is one, the endpoint's overall
        number otherwise, and None when it has neither.
    quality_area : str, optional
        The area `effective_quality` came from, None when it is the
        overall number. Shown in the candidate table so a surprising
        pick names its own source.
    """

    name: str
    endpoint: Endpoint
    record: Optional[ModelRecord] = None
    capabilities: Optional[Capabilities] = None
    effective_quality: Optional[int] = None
    quality_area: Optional[str] = None

    @property
    def total_cost(self) -> Optional[float]:
        """Return input plus output USD per 1M, or None when unpriced."""
        if self.record is None:
            return None
        if self.record.cost_input is None and self.record.cost_output is None:
            return None
        return (self.record.cost_input or 0.0) + (self.record.cost_output or 0.0)


# ---------------------------------------------------------------------------
# Catalog access
# ---------------------------------------------------------------------------


def _import_aim():
    """Import `hop.aim`, naming the extra when it is absent.

    Returns
    -------
    module
        The imported `hop.aim` package.

    Raises
    ------
    CatalogUnavailable
        If the optional dependency is not installed.
    """
    try:
        import hop.aim as aim
    except ImportError as exc:
        raise CatalogUnavailable(
            "Policy-based pairing needs the models.dev client. Install it "
            f"with: pip install 'routellm[{PAIRING_EXTRA}]'"
        ) from exc
    return aim


def _fetch_catalog() -> list[ModelRecord]:
    """Fetch the whole models.dev catalog, synchronously.

    The async registry is driven on a one-shot helper thread with its
    own event loop, so this is safe to call from inside a running loop
    where `asyncio.run` would raise.

    Returns
    -------
    list[ModelRecord]
        Every model models.dev lists, flattened.

    Raises
    ------
    CatalogUnavailable
        If the client is missing, the fetch fails, or it does not
        finish within `CATALOG_FETCH_TIMEOUT` seconds.
    """
    aim = _import_aim()

    def _work() -> list[Any]:
        return asyncio.run(aim.Registry().models())

    # Deliberately not a `with` block: leaving the context manager calls
    # shutdown(wait=True), which blocks until the worker finishes, so a
    # hung fetch would hold startup far past the timeout. Shut the pool
    # down without waiting instead and let the stuck thread die on its
    # own; the daemon-free worker holds nothing the caller needs.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        models = pool.submit(_work).result(timeout=CATALOG_FETCH_TIMEOUT)
    except Exception as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise CatalogUnavailable(f"models.dev fetch failed: {exc}") from exc
    pool.shutdown(wait=False)

    return [
        ModelRecord(
            provider=model.provider,
            id=model.id,
            cost_input=getattr(model.cost, "input", None) if model.cost else None,
            cost_output=getattr(model.cost, "output", None) if model.cost else None,
            tool_call=bool(model.tool_call),
            reasoning=bool(model.reasoning),
            context=model.limit.context if model.limit else None,
            release_date=model.release_date,
            structured_output=bool(getattr(model, "structured_output", False)),
            open_weights=bool(getattr(model, "open_weights", False)),
            modalities_in=list(getattr(model.modalities, "input", []) or [])
            if getattr(model, "modalities", None)
            else [],
            max_output=model.limit.output if model.limit else None,
        )
        for model in models
    ]


def snapshot_path() -> Path:
    """Return the snapshot path, honouring the override and XDG."""
    override = os.environ.get(CATALOG_CACHE_ENV)
    if override:
        return Path(override)

    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "routellm" / "models_dev.json"


def write_snapshot(
    path: Path, records: list[ModelRecord], fetched_at: Optional[float] = None
) -> None:
    """Write `records` to `path` with a fetch timestamp.

    Parameters
    ----------
    path : Path
        Destination file; parent directories are created.
    records : list[ModelRecord]
        Catalog to store.
    fetched_at : float, optional
        Epoch seconds to stamp. Defaults to now.
    """
    payload = {
        "fetched_at": time.time() if fetched_at is None else fetched_at,
        "models": [asdict(record) for record in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def read_snapshot(path: Path) -> Optional[tuple[float, list[ModelRecord]]]:
    """Read a snapshot, or None when there is no readable one.

    A corrupt or partially written file is treated as no snapshot
    rather than as a failure: the caller can still fetch.

    Parameters
    ----------
    path : Path
        Snapshot file.

    Returns
    -------
    tuple[float, list[ModelRecord]] or None
        The fetch timestamp and the stored records.
    """
    try:
        payload = json.loads(path.read_text())
        records = [ModelRecord(**entry) for entry in payload["models"]]
        return float(payload["fetched_at"]), records
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("no usable catalog snapshot at %s: %s", path, exc)
        return None


def load_catalog() -> list[ModelRecord]:
    """Return the catalog, from a fresh snapshot, the network, or a stale one.

    Returns
    -------
    list[ModelRecord]
        The models.dev catalog.

    Raises
    ------
    CatalogUnavailable
        If the fetch fails and no snapshot exists at all.
    """
    path = snapshot_path()
    stored = read_snapshot(path)

    if stored is not None and time.time() - stored[0] <= CATALOG_TTL_SECONDS:
        logger.debug("using fresh models.dev snapshot from %s", path)
        return stored[1]

    try:
        records = _fetch_catalog()
    except CatalogUnavailable:
        if stored is None:
            raise
        age_hours = (time.time() - stored[0]) / 3600
        logger.warning(
            "models.dev fetch failed; using the stale snapshot at %s "
            "(%.1f hours old)",
            path,
            age_hours,
        )
        return stored[1]

    write_snapshot(path, records)
    return records


# ---------------------------------------------------------------------------
# Provider mapping
# ---------------------------------------------------------------------------


def catalog_provider_for(model: str) -> Optional[tuple[str, str]]:
    """Map a litellm model name onto a models.dev `(provider, id)` pair.

    Parameters
    ----------
    model : str
        litellm model name, as an endpoint configures it.

    Returns
    -------
    tuple[str, str] or None
        The models.dev provider id and model id, or None when litellm
        cannot split the name or the provider has no models.dev
        counterpart. None means "tag-only candidate", never an error.

    Examples
    --------
    >>> catalog_provider_for("gpt-4o")
    ('openai', 'gpt-4o')
    >>> catalog_provider_for("ollama_chat/qwen3:8b") is None
    True
    """
    import litellm

    try:
        model_id, provider = litellm.get_llm_provider(model)[:2]
    except Exception as exc:
        logger.debug("no litellm provider for %r (%s); tag-only", model, exc)
        return None

    if provider not in PROVIDER_ALIASES:
        logger.debug(
            "litellm provider %r for %r has no models.dev alias; tag-only",
            provider,
            model,
        )
        return None

    alias = PROVIDER_ALIASES[provider]
    if alias is None:
        logger.debug("provider %r for %r is local; tag-only", provider, model)
        return None

    logger.debug("mapped %r to models.dev %s/%s", model, alias, model_id)
    return alias, model_id


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------


def _split_terms(select: str) -> tuple[list[str], list[str], str]:
    """Split a `select` expression into tags, capability terms, and a query.

    A term is routed by its key: `tag:` goes to the tag bucket; a key
    in `CAPABILITY_KEYS`, `RANGE_KEYS`, `tool_call`, or `input` goes to
    the capability bucket routellm judges itself; everything else is
    passed through unchanged to `hop.aim.parse_query`.

    The capability terms MUST be stripped before aim sees the query:
    aim's key set is exactly {in, out, provider, family, tool_call,
    reasoning, open_weights, structured_output, temperature} and it
    raises on `vision:`, `context:` and `input:`.

    `tool_call` stays in the capability bucket as an alias of `tools`,
    answered from `Capabilities.tools`, which already merged the
    catalog's own `tool_call` value.

    Parameters
    ----------
    select : str
        The raw expression.

    Returns
    -------
    tuple[list[str], list[str], str]
        Labels from the `tag:` terms, the capability terms verbatim,
        and the remaining terms joined back into an aim query string.

    Raises
    ------
    ValueError
        If a term carries no colon, which would be free text.
    """
    tags: list[str] = []
    capability_terms: list[str] = []
    catalog_terms: list[str] = []

    for term in select.split():
        if ":" not in term:
            raise ValueError(
                f"Selector term {term!r} in {select!r} is bare free text. "
                "Every term must be key:value, either tag:<label> or a "
                "models.dev key."
            )
        key, _, value = term.partition(":")
        if key == "tag":
            tags.append(value)
        elif key in CAPABILITY_KEYS or key in RANGE_KEYS or key in (
            "tool_call",
            "input",
        ):
            capability_terms.append(term)
        else:
            catalog_terms.append(term)

    return tags, capability_terms, " ".join(catalog_terms)


def _catalog_filter(query: str, select: str):
    """Parse the catalog half of a `select` expression.

    Parameters
    ----------
    query : str
        The non-`tag:` terms, space-joined.
    select : str
        The whole expression, quoted in any error.

    Returns
    -------
    hop.aim.Filter or None
        The parsed filter, None when there are no catalog terms.

    Raises
    ------
    ValueError
        If aim rejects a key or a value, with the offending term named.
    """
    if not query:
        return None

    aim = _import_aim()
    try:
        return aim.parse_query(query)
    except ValueError as exc:
        raise ValueError(
            f"Selector {select!r} has an invalid term: {exc}"
        ) from exc


def _unsupported_term(name: str) -> ValueError:
    """Return the error for a catalog term pairing cannot judge."""
    return ValueError(
        f"Selector term {name!r} is not supported by routellm pairing; "
        "supported catalog terms are tool_call, reasoning, provider, "
        "open_weights, structured_output, and input, plus routellm's own "
        "vision, tools, context, and max_output."
    )


def _record_matches(record: Optional[ModelRecord], filter_) -> bool:
    """Return whether a catalog record satisfies an aim filter.

    A candidate with no record fails any catalog term, so the absence
    of a models.dev entry is a rejection rather than a pass.
    """
    if record is None:
        return False

    for field, value in (
        ("tool_call", filter_.tool_call),
        ("reasoning", filter_.reasoning),
    ):
        if value is not None and getattr(record, field) != value:
            return False

    if filter_.provider and record.provider != filter_.provider:
        return False

    # Facts this flattened record does not carry cannot be judged
    # locally; matching them would silently pass. Reject instead.
    #
    # The tri-states are checked with `is not None`, not for
    # truthiness: `open_weights:false` parses to False, which is a real
    # constraint pairing still cannot honour. The string and list
    # fields carry no such distinction, so empty means unset there.
    # `open_weights`, `structured_output` and `input` are answered from
    # the merged `Capabilities` instead, so they never reach here: the
    # splitter routes them into the capability bucket. `temperature`,
    # `family` and `query` remain unjudgeable locally.
    if getattr(filter_, "temperature", None) is not None:
        raise _unsupported_term("temperature")

    for name in ("output", "family", "query"):
        if getattr(filter_, name, None):
            raise _unsupported_term(name)

    return True


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def _sort_key(candidate: Candidate, order: str, area: Optional[str] = None):
    """Return the sort key placing `candidate` under `order`.

    Missing values always sort last, whichever direction the order
    runs, so an unpriced or unlisted endpoint is never promoted to the
    front by the absence of data.

    Under a `quality_*` order the number read is the per-area one when
    `area` is set and the registry measured one for this endpoint,
    then the endpoint's overall `quality`, and only then the unrated
    bucket. Falling back to the overall number rather than straight to
    unrated matters: an endpoint measured everywhere except this one
    area is still better known than one measured nowhere.
    """
    record = candidate.record

    if order in ("cost_asc", "cost_desc"):
        cost = candidate.total_cost
        if cost is None:
            return (1, 0.0, candidate.name)
        return (0, cost if order == "cost_asc" else -cost, candidate.name)

    if order in ("quality_asc", "quality_desc"):
        quality = candidate.effective_quality
        if quality is None:
            quality = candidate.endpoint.quality
        release = _release_key(record)
        if quality is None:
            # Unrated endpoints sort after every rated one, and among
            # themselves by release date, newest first.
            return (1, 0.0, release, candidate.name)
        signed = quality if order == "quality_asc" else -quality
        return (0, float(signed), release, candidate.name)

    if order == "max_output_desc":
        # Read the merged capabilities, not the record, so an endpoint
        # with an explicit block orders alongside a catalogued one.
        caps = candidate.capabilities
        max_output = caps.max_output if caps else None
        if max_output is None:
            return (1, 0.0, candidate.name)
        return (0, -float(max_output), candidate.name)

    # context_desc
    context = record.context if record else None
    if context is None:
        return (1, 0.0, candidate.name)
    return (0, -float(context), candidate.name)


def _release_key(record: Optional[ModelRecord]) -> tuple:
    """Return the release-date tiebreak key, newest first, undated last.

    An absent date gets its own trailing group rather than an empty
    descending key, which would otherwise sort ahead of every real
    date because `()` precedes any non-empty tuple.
    """
    release = (record.release_date if record else None) or ""
    if not release:
        return (1, ())
    return (0, _reverse_text(release))


def _reverse_text(text: str) -> tuple:
    """Return a key sorting `text` in descending lexical order."""
    return tuple(-ord(char) for char in text)


def _index_catalog(records: list[ModelRecord]) -> dict[tuple[str, str], ModelRecord]:
    """Key catalog records by their `(provider, id)` pair."""
    return {(record.provider, record.id): record for record in records}


def rank_candidates(
    registry: EndpointRegistry,
    selector: Selector,
    area: Optional[str] = None,
) -> list[tuple[str, Candidate]]:
    """Return the endpoints matching a selector, best first.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints are the candidate pool.
    selector : Selector
        The policy to apply.
    area : str, optional
        The area the selector sits in. When given, a `quality_*` order
        reads each endpoint's measured number for that area in
        preference to its overall one.

    Returns
    -------
    list[tuple[str, Candidate]]
        Matching `(name, candidate)` pairs in `selector.order`.

    Raises
    ------
    ValueError
        If `select` is malformed or names a term pairing cannot judge.
    CatalogUnavailable
        If `select` carries a catalog term and no catalog can be had,
        or a capability term that some endpoint can only answer from
        its record.

    Notes
    -----
    The catalog is mandatory when a term could only be answered by it.
    If EVERY endpoint in the registry answers the capability query from
    its explicit `capabilities:` block alone, a failed fetch is a
    warning rather than an error, so a config whose endpoints all
    declare themselves resolves with no network at all.
    """
    tags, capability_terms, query = _split_terms(selector.select)
    capability_query = parse_capability_terms(capability_terms)
    filter_ = _catalog_filter(query, selector.select)

    # A catalog term makes the catalog mandatory; an order that reads
    # catalog facts only makes it useful, so a failed fetch there
    # degrades to name-ordered rather than aborting startup.
    by_key: dict[tuple[str, str], ModelRecord] = {}
    try:
        by_key = _index_catalog(load_catalog())
    except CatalogUnavailable as exc:
        if filter_ is not None:
            raise
        if not _blocks_answer(registry, capability_query):
            raise
        logger.warning(
            "no models.dev catalog for %r; every endpoint answers its "
            "capability terms from its own block, so ordering and "
            "selection continue on local facts only (%s)",
            selector.select,
            exc,
        )

    candidates: list[Candidate] = []
    for name in registry.names():
        endpoint = registry.get(name)
        if not set(tags).issubset(set(endpoint.tags)):
            continue

        record = None
        key = catalog_provider_for(endpoint.model)
        if key is not None:
            record = by_key.get(key)

        capabilities = capabilities_for(endpoint, record)

        # Cheapest check first: tags are already done, then the local
        # capability query, then the aim filter, which may raise.
        if not matches(capabilities, capability_query):
            continue

        if filter_ is not None and not _record_matches(record, filter_):
            continue

        candidates.append(
            Candidate(
                name=name,
                endpoint=endpoint,
                record=record,
                capabilities=capabilities,
            )
        )
        measured = None
        if area is not None:
            measured = (getattr(registry, "area_quality", {}) or {}).get(
                name, {}
            ).get(area)

        candidates.append(
            Candidate(
                name=name,
                endpoint=endpoint,
                record=record,
                effective_quality=(
                    measured if measured is not None else endpoint.quality
                ),
                quality_area=area if measured is not None else None,
            )
        )

    candidates.sort(key=lambda c: _sort_key(c, selector.order, area))
    return [(candidate.name, candidate) for candidate in candidates]


def _blocks_answer(
    registry: EndpointRegistry, capability_query: CapabilityQuery
) -> bool:
    """Return whether every endpoint answers a query without the catalog.

    A capability term is answerable locally when each endpoint's own
    explicit block, or its deprecated tags, already state every key the
    query reads. When one does not, only the catalog could, so a failed
    fetch stays an error.
    """
    if capability_query.is_empty():
        return True

    wanted = capability_query.keys()
    for name in registry.names():
        caps = capabilities_for(registry.get(name), None)
        if any(getattr(caps, key) is None for key in wanted):
            return False

    return True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def records_for_registry(
    registry: EndpointRegistry,
) -> dict[str, ModelRecord]:
    """Return the catalog record for each endpoint that has one.

    Never raises: the capability index is an optimisation, and an
    endpoint with no record simply answers from its own block. A failed
    fetch yields an empty mapping and one DEBUG line.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints are looked up.

    Returns
    -------
    dict[str, ModelRecord]
        Endpoint name to its models.dev record.
    """
    try:
        by_key = _index_catalog(load_catalog())
    except CatalogUnavailable as exc:
        logger.debug("no models.dev catalog for the capability index: %s", exc)
        return {}

    found: dict[str, ModelRecord] = {}
    for name in registry.names():
        key = catalog_provider_for(registry.get(name).model)
        if key is None:
            continue
        record = by_key.get(key)
        if record is not None:
            found[name] = record

    return found


def resolve_pairing(
    registry: EndpointRegistry,
    selector: Selector,
    area: Optional[str] = None,
) -> str:
    """Return the endpoint name a selector picks.

    The ordered candidate table is logged at INFO, so a surprising pick
    can be read out of the startup log without rerunning anything.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints are the candidate pool.
    selector : Selector
        The policy to apply.
    area : str, optional
        The area of the tier this selector sits in.

    Returns
    -------
    str
        The winning endpoint name.

    Raises
    ------
    ValueError
        If `select` is malformed, or no endpoint matches it.
    CatalogUnavailable
        If `select` carries a catalog term and no catalog can be had.
    """
    ranked = rank_candidates(registry, selector, area)
    if not ranked:
        raise ValueError(
            f"Selector {selector.select!r} matches no configured endpoint. "
            f"Configured endpoints: {', '.join(registry.names()) or '<none>'}"
        )

    logger.info(
        "selector %r order %s%s picked %s from: %s",
        selector.select,
        selector.order,
        f" in area {area}" if area else "",
        ranked[0][0],
        ", ".join(describe_candidate(candidate) for _, candidate in ranked),
    )
    return ranked[0][0]


def describe_candidate(candidate: Candidate) -> str:
    """Return a one-line description of a candidate for logs and tables."""
    cost = candidate.total_cost
    record = candidate.record

    quality = candidate.effective_quality
    source = f" [{candidate.quality_area}]" if candidate.quality_area else ""
    if quality is None:
        quality = candidate.endpoint.quality
        source = ""

    return (
        f"{candidate.name}(model={candidate.endpoint.model}, "
        f"quality={quality}{source}, "
        f"cost={'-' if cost is None else f'{cost:.4g}'}, "
        f"context={record.context if record else '-'})"
    )


def resolve_registry_pairings(registry: EndpointRegistry) -> None:
    """Replace every `Selector` in the registry's tiers with a name.

    Runs once, at controller construction, before the registry's tier
    references are validated, so the validator only ever sees names.
    Tiers are mutated in place.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose tiers may carry selectors.

    Raises
    ------
    ValueError
        If a selector is malformed, matches nothing, or both sides of
        one tier resolve to the same endpoint.
    CatalogUnavailable
        If a selector carries a catalog term and no catalog can be had.
    """
    for tier in registry.tiers.values():
        area = registry.area_of(tier.name)
        resolved: dict[str, str] = {}
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            if isinstance(value, Selector):
                resolved[side] = resolve_pairing(registry, value, area)

        if not resolved:
            continue

        strong = resolved.get("strong", tier.strong)
        weak = resolved.get("weak", tier.weak)
        if strong == weak:
            raise ValueError(
                f"Tier {tier.name!r} resolves both sides to {strong!r}; a "
                "tier must choose between two distinct endpoints. Narrow "
                "one side's select or its order."
            )

        for side, name in resolved.items():
            setattr(tier, side, name)
            logger.info("tier %s %s resolved to %s", tier.name, side, name)


# ---------------------------------------------------------------------------
# Explain surface
# ---------------------------------------------------------------------------


#: Columns the capability matrix prints, in order.
_MATRIX_COLUMNS = (
    "vision",
    "tools",
    "structured_output",
    "reasoning",
    "context",
    "max_output",
)


def _cell(value) -> str:
    """Render one capability value as `yes`, `no`, `?`, or a number."""
    if value is None:
        return "?"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def _quality_cell(endpoint: Endpoint) -> str:
    """Render one endpoint's quality, saying where the number came from.

    `(measured)` means a `quality_from:` sidecar supplied it,
    `(manual)` means the endpoint's own `quality:` did. The distinction
    is the point: a measurement that quietly replaced a decision would
    be worse than no measurement.
    """
    quality = endpoint.quality
    if quality is None:
        return "?"

    source = "measured" if getattr(endpoint, "quality_measured", False) else "manual"
    return f"{quality} ({source})"


def _capability_matrix(
    registry: EndpointRegistry, selects: Optional[list[str]] = None
) -> str:
    """Return the capability matrix for a registry, as fixed-width text.

    One row per endpoint, then one row per tier carrying the union over
    its reachable leaves and a `via` column naming the leaf that
    supplied each yes. Then an "Unknown:" block, one line per endpoint
    listing the keys it cannot answer, and a final line naming the
    capability terms the config's own selectors use that some endpoint
    is unknown on: those are the endpoints a selector silently drops,
    which is the actionable part.

    Unknown renders `?`, never `no`: "not known" and "known false" are
    different answers, and collapsing them would hide exactly the gap
    this table exists to show.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints and tiers are tabulated. Its selectors
        must already be resolved.
    selects : list[str], optional
        The `select` expressions the config declares, read before
        resolution replaced them with names.

    Returns
    -------
    str
        The rendered matrix.
    """
    records = records_for_registry(registry)
    caps_by_name = {
        name: capabilities_for(registry.get(name), records.get(name))
        for name in registry.names()
    }
    tier_index = build_tier_index(registry, records)

    width = max(
        [len(name) for name in registry.names()]
        + [len(name) + len(" (tier)") for name in registry.tier_names()]
        + [8]
    )
    header = "  ".join(
        [f"{'endpoint':<{width}}"]
        + [f"{column:>18}" for column in _MATRIX_COLUMNS]
        + [f"{'quality':>18}", "via"]
    )
    lines = ["Capabilities", header, "-" * len(header)]

    for name in registry.names():
        caps = caps_by_name[name]
        cells = [f"{_cell(getattr(caps, column)):>18}" for column in _MATRIX_COLUMNS]
        cells.append(f"{_quality_cell(registry.get(name)):>18}")
        lines.append("  ".join([f"{name:<{width}}"] + cells + [""]).rstrip())

    for tier_name in registry.tier_names():
        caps = tier_index.get(tier_name, Capabilities())
        cells = [f"{_cell(getattr(caps, column)):>18}" for column in _MATRIX_COLUMNS]
        cells.append(f"{'-':>18}")
        via = _via(registry, tier_name, caps_by_name)
        lines.append(
            "  ".join([f"{tier_name + ' (tier)':<{width}}"] + cells + [via]).rstrip()
        )

    lines.append("")
    lines.append("Unknown:")
    any_unknown = False
    for name in registry.names():
        missing = caps_by_name[name].unknown_fields()
        if missing:
            any_unknown = True
            lines.append(f"  {name}: {', '.join(missing)}")
    if not any_unknown:
        lines.append("  (none)")

    risky = _terms_at_risk(selects or [], caps_by_name)
    lines.append("")
    if risky:
        lines.append(
            "Selector terms some endpoint cannot answer (those endpoints "
            "are dropped silently): " + ", ".join(risky)
        )
    else:
        lines.append(
            "Every capability term the selectors use is answered by every "
            "endpoint."
        )

    return "\n".join(lines)


def _leaves(registry: EndpointRegistry, name: str) -> list[str]:
    """Return every endpoint name reachable from a tier or endpoint."""
    if not registry.has_tier(name):
        return [name]

    tier = registry.get_tier(name)
    found: list[str] = []
    for side in (tier.strong, tier.weak):
        for leaf in _leaves(registry, str(side)):
            if leaf not in found:
                found.append(leaf)
    return found


def _via(
    registry: EndpointRegistry,
    tier_name: str,
    caps_by_name: dict[str, Capabilities],
) -> str:
    """Return `key=leaf` for each capability a tier gets from one leaf."""
    parts: list[str] = []
    leaves = _leaves(registry, tier_name)

    for column in _MATRIX_COLUMNS:
        for leaf in leaves:
            caps = caps_by_name.get(leaf)
            if caps is not None and getattr(caps, column) is True:
                parts.append(f"{column}={leaf}")
                break

    return " ".join(parts)


def _terms_at_risk(
    selects: list[str], caps_by_name: dict[str, Capabilities]
) -> list[str]:
    """Return the selector terms some endpoint cannot answer.

    An unknown capability fails its term at selection time, so these
    are exactly the terms that drop candidates without saying so.
    """
    wanted: list[str] = []
    for select in selects:
        _, capability_terms, _ = _split_terms(select)
        for key in parse_capability_terms(capability_terms).keys():
            if key not in wanted:
                wanted.append(key)

    return [
        key
        for key in wanted
        if any(getattr(caps, key) is None for caps in caps_by_name.values())
    ]


def _explain(config_path: Optional[str] = None) -> str:
    """Return the candidate table and pick for every selector in a config.

    Parameters
    ----------
    config_path : str, optional
        Explicit config file, merged last over the discovered chain.
        None explains whatever discovery finds.

    Returns
    -------
    str
        A plain-text report, one block per tier side that selects,
        each preceded by its area when it has one and each candidate
        naming the quality it was ordered on, then one line per
        configured intent naming the tier it enters.
    """
    config = load_config(explicit=config_path).data

    registry = EndpointRegistry.from_config(config, config_path=config_path)
    lines: list[str] = []

    for name in registry.tier_names():
        tier = registry.get_tier(name)
        area = registry.area_of(name)
        if area:
            lines.append(f"{name}: area {area}")
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            if not isinstance(value, Selector):
                lines.append(f"{name}.{side}: {value} (named)")
                continue

            ranked = rank_candidates(registry, value, area)
            if not ranked:
                raise ValueError(
                    f"Tier {name!r} {side} selector {value.select!r} matches "
                    "no configured endpoint."
                )

            lines.append(
                f"{name}.{side}: select={value.select!r} order={value.order}"
            )
            for position, (_, candidate) in enumerate(ranked):
                marker = "->" if position == 0 else "  "
                lines.append(f"  {marker} {describe_candidate(candidate)}")
        lines.append("")

    intent_tiers = ((config.get("intents") or {}).get("tiers")) or {}
    for intent in sorted(intent_tiers):
        lines.append(f"intent {intent} -> tier {intent_tiers[intent]}")

    return "\n".join(lines).rstrip()


def main(argv: Optional[list[str]] = None) -> int:
    """Print the candidate table and pick for a config's selectors.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        0 on success, 1 when a selector cannot be resolved.
    """
    parser = argparse.ArgumentParser(
        prog="python -m routellm.pairing",
        description="Explain how each tier's selectors resolve.",
        epilog=(
            "The capability matrix is a flag here rather than a "
            "`python -m routellm.capabilities` module, which is "
            "deliberately not a thing: this command already loads the "
            "YAML, builds the registry, and resolves selectors, which "
            "is everything the matrix needs. A second entry point "
            "would duplicate all of it and drift from it."
        ),
    )
    parser.add_argument(
        "--capabilities",
        action="store_true",
        help=(
            "Print the capability matrix instead of the selector tables. "
            "With --explain, print both, selectors first."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the selector tables. Implied when --capabilities is absent.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Explicit config file, merged last over the discovered chain "
            "(system, user, project, ROUTELLM_CONFIG). Omit it to explain "
            "whatever discovery finds."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    try:
        blocks: list[str] = []
        if args.explain or not args.capabilities:
            blocks.append(_explain(args.config))
        if args.capabilities:
            registry, selects = _registry_from(args.config)
            blocks.append(_capability_matrix(registry, selects))
        print("\n\n".join(block for block in blocks if block))
    except (ValueError, CatalogUnavailable, OSError) as exc:
        print(_explain(args.config))
    except (ValueError, CatalogUnavailable, OSError, ConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _registry_from(
    config_path: Optional[str] = None,
) -> tuple[EndpointRegistry, list[str]]:
    """Build a registry from a YAML config with its selectors resolved.

    A tier's capabilities are the union over the leaves it actually
    reaches, so the selectors have to be resolved first: an unresolved
    side names no endpoint and the tier's whole row would read unknown.
    """
    config = load_config(explicit=config_path).data

    registry = EndpointRegistry.from_config(config, config_path=config_path)
    selects = [
        side.select
        for tier in registry.tiers.values()
        for side in (tier.strong, tier.weak)
        if isinstance(side, Selector)
    ]
    resolve_registry_pairings(registry)
    registry.revalidate()
    return registry, selects


if __name__ == "__main__":
    sys.exit(main())
