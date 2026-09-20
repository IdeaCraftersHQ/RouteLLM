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

The pick is explainable without a server::

    python -m routellm.pairing --config config.yaml
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

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
    """

    provider: str
    id: str
    cost_input: Optional[float] = None
    cost_output: Optional[float] = None
    tool_call: bool = False
    reasoning: bool = False
    context: Optional[int] = None
    release_date: Optional[str] = None


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
    """

    name: str
    endpoint: Endpoint
    record: Optional[ModelRecord] = None

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


def _split_terms(select: str) -> tuple[list[str], str]:
    """Split a `select` expression into tag labels and a catalog query.

    Parameters
    ----------
    select : str
        The raw expression.

    Returns
    -------
    tuple[list[str], str]
        Labels from the `tag:` terms, and the remaining terms joined
        back into a query string for `hop.aim.parse_query`.

    Raises
    ------
    ValueError
        If a term carries no colon, which would be free text.
    """
    tags: list[str] = []
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
        else:
            catalog_terms.append(term)

    return tags, " ".join(catalog_terms)


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
        "supported catalog terms are tool_call, reasoning, and provider."
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
    for name in ("open_weights", "structured_output", "temperature"):
        if getattr(filter_, name, None) is not None:
            raise _unsupported_term(name)

    for name in ("input", "output", "family", "query"):
        if getattr(filter_, name, None):
            raise _unsupported_term(name)

    return True


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def _sort_key(candidate: Candidate, order: str):
    """Return the sort key placing `candidate` under `order`.

    Missing values always sort last, whichever direction the order
    runs, so an unpriced or unlisted endpoint is never promoted to the
    front by the absence of data.
    """
    record = candidate.record

    if order in ("cost_asc", "cost_desc"):
        cost = candidate.total_cost
        if cost is None:
            return (1, 0.0, candidate.name)
        return (0, cost if order == "cost_asc" else -cost, candidate.name)

    if order in ("quality_asc", "quality_desc"):
        quality = candidate.endpoint.quality
        release = _release_key(record)
        if quality is None:
            # Unrated endpoints sort after every rated one, and among
            # themselves by release date, newest first.
            return (1, 0.0, release, candidate.name)
        signed = quality if order == "quality_asc" else -quality
        return (0, float(signed), release, candidate.name)

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
    registry: EndpointRegistry, selector: Selector
) -> list[tuple[str, Candidate]]:
    """Return the endpoints matching a selector, best first.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints are the candidate pool.
    selector : Selector
        The policy to apply.

    Returns
    -------
    list[tuple[str, Candidate]]
        Matching `(name, candidate)` pairs in `selector.order`.

    Raises
    ------
    ValueError
        If `select` is malformed or names a term pairing cannot judge.
    CatalogUnavailable
        If `select` carries a catalog term and no catalog can be had.
    """
    tags, query = _split_terms(selector.select)
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
        logger.warning(
            "no models.dev catalog for order %s; ordering %r on local "
            "facts only (%s)",
            selector.order,
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

        if filter_ is not None and not _record_matches(record, filter_):
            continue

        candidates.append(Candidate(name=name, endpoint=endpoint, record=record))

    candidates.sort(key=lambda c: _sort_key(c, selector.order))
    return [(candidate.name, candidate) for candidate in candidates]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_pairing(registry: EndpointRegistry, selector: Selector) -> str:
    """Return the endpoint name a selector picks.

    The ordered candidate table is logged at INFO, so a surprising pick
    can be read out of the startup log without rerunning anything.

    Parameters
    ----------
    registry : EndpointRegistry
        Registry whose endpoints are the candidate pool.
    selector : Selector
        The policy to apply.

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
    ranked = rank_candidates(registry, selector)
    if not ranked:
        raise ValueError(
            f"Selector {selector.select!r} matches no configured endpoint. "
            f"Configured endpoints: {', '.join(registry.names()) or '<none>'}"
        )

    logger.info(
        "selector %r order %s picked %s from: %s",
        selector.select,
        selector.order,
        ranked[0][0],
        ", ".join(describe_candidate(candidate) for _, candidate in ranked),
    )
    return ranked[0][0]


def describe_candidate(candidate: Candidate) -> str:
    """Return a one-line description of a candidate for logs and tables."""
    cost = candidate.total_cost
    record = candidate.record
    return (
        f"{candidate.name}(model={candidate.endpoint.model}, "
        f"quality={candidate.endpoint.quality}, "
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
        resolved: dict[str, str] = {}
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            if isinstance(value, Selector):
                resolved[side] = resolve_pairing(registry, value)

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


def _explain(config_path: str) -> str:
    """Return the candidate table and pick for every selector in a config.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.

    Returns
    -------
    str
        A plain-text report, one block per tier side that selects,
        then one line per configured intent naming the tier it enters.
    """
    import yaml

    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}

    registry = EndpointRegistry.from_config(config)
    lines: list[str] = []

    for name in registry.tier_names():
        tier = registry.get_tier(name)
        for side in ("strong", "weak"):
            value = getattr(tier, side)
            if not isinstance(value, Selector):
                lines.append(f"{name}.{side}: {value} (named)")
                continue

            ranked = rank_candidates(registry, value)
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
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    try:
        print(_explain(args.config))
    except (ValueError, CatalogUnavailable, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
