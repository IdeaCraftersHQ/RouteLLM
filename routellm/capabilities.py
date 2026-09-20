"""Typed capabilities per endpoint, merged from config and catalog.

Routing picks on difficulty and policy; this module adds *fitness*. An
endpoint's capabilities answer "can this one take an image?" as a typed
value rather than as a hand-written tag a router cannot read.

Three sources, highest precedence first::

    endpoints:
      local_server:
        model: openai/qwen3-coder
        api_base: http://127.0.0.1:11800/v1
        capabilities:
          vision: false
          tools: true
          structured_output: true
          reasoning: false
          context: 262144
          max_output: 32768
          modalities_in: [text]
        strict: false

1. the explicit `capabilities:` block on the endpoint,
2. the deprecated capability tags (`tools`, `vision`, `long_context`),
3. the models.dev record the pairing path already fetches.

`None` is the unknown marker on every field. "Known false" is `False`,
"not known" is `None`; there is no separate sentinel and no separate
`unknown` field.

This module deliberately does NOT import `routellm.endpoints` at module
level: `endpoints.py` imports `Capabilities` from here, so the reverse
import would close a cycle. `Endpoint` arrives as a typed argument
under `TYPE_CHECKING` only.
"""

import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from routellm.endpoints import Endpoint
    from routellm.pairing import ModelRecord

logger = logging.getLogger(__name__)

__all__ = [
    "CAPABILITY_KEYS",
    "LONG_CONTEXT_TOKENS",
    "RANGE_KEYS",
    "Capabilities",
    "CapabilityQuery",
    "capabilities_for",
    "from_tags",
    "matches",
    "merge",
    "parse_capability_terms",
]

#: Context window the deprecated `long_context` tag stands for. It is a
#: FLOOR an operator asserts, not a measurement: a tag says "at least
#: this much", and the real number belongs in `capabilities.context`.
LONG_CONTEXT_TOKENS = 200_000

#: Deprecated tag -> the capability field it populates, and its value.
_TAG_ALIASES: dict[str, tuple[str, object]] = {
    "tools": ("tools", True),
    "vision": ("vision", True),
    "long_context": ("context", LONG_CONTEXT_TOKENS),
}

#: Aliases already warned about, one warning per alias per process.
_warned: set[str] = set()

#: Every capability field, in the order the matrix and merge read them.
_FIELDS = (
    "vision",
    "tools",
    "structured_output",
    "reasoning",
    "open_weights",
    "context",
    "max_output",
    "modalities_in",
)


class Capabilities(BaseModel):
    """What one endpoint can do, with `None` meaning "not known".

    Every field is optional and tri-state: `True` and `False` are both
    knowledge, `None` is its absence. Nothing here is inferred from
    anything else; a field is filled only by a source that states it.

    Attributes
    ----------
    vision : bool, optional
        Whether the endpoint accepts image parts in message content.
    tools : bool, optional
        Whether it accepts tool/function definitions. The catalog
        spells this `tool_call`; both names mean the same thing.
    structured_output : bool, optional
        Whether it honours a `response_format` schema.
    reasoning : bool, optional
        Whether it is a reasoning model.
    open_weights : bool, optional
        Whether its weights are published.
    context : int, optional
        Context window in tokens.
    max_output : int, optional
        Maximum output tokens in one response.
    modalities_in : list[str], optional
        Input modalities the endpoint accepts, e.g. `["text", "image"]`.
    """

    vision: Optional[bool] = None
    tools: Optional[bool] = None
    structured_output: Optional[bool] = None
    reasoning: Optional[bool] = None
    open_weights: Optional[bool] = None
    context: Optional[int] = None
    max_output: Optional[int] = None
    modalities_in: Optional[list[str]] = None

    def unknown_fields(self) -> list[str]:
        """Return the capability names this record cannot answer.

        Returns
        -------
        list[str]
            Field names whose value is `None`, in declaration order.
            The `--capabilities` matrix prints exactly these under its
            "Unknown:" block.
        """
        return [name for name in _FIELDS if getattr(self, name) is None]


def merge(
    explicit: Optional[Capabilities],
    record: Optional["ModelRecord"],
    tag_derived: Optional[Capabilities],
) -> Capabilities:
    """Merge the three capability sources, field by field.

    Per field the first non-None of `explicit`, `tag_derived`, `record`
    wins. Tags sit BETWEEN the explicit block and the catalog because a
    hand-written capability tag on a catalogued model is almost always
    a correction to a catalog the operator found wrong; an operator who
    wants to override a tag has the explicit block above it.

    Parameters
    ----------
    explicit : Capabilities, optional
        The endpoint's own `capabilities:` block.
    record : ModelRecord, optional
        Its models.dev record, None when the model maps to no entry.
    tag_derived : Capabilities, optional
        What `from_tags` read out of the deprecated aliases.

    Returns
    -------
    Capabilities
        The merged record, `None` wherever no source knew.
    """
    from_record = _from_record(record)

    values = {}
    for name in _FIELDS:
        for source in (explicit, tag_derived, from_record):
            if source is None:
                continue
            value = getattr(source, name)
            if value is not None:
                values[name] = value
                break

    return Capabilities(**values)


def _from_record(record: Optional["ModelRecord"]) -> Optional[Capabilities]:
    """Flatten a catalog record into capabilities, or None when absent.

    The catalog states `tool_call`, `reasoning`, `structured_output`,
    `open_weights`, `context`, `max_output` and `modalities.input`; it
    says nothing about vision directly, which is read off
    `modalities_in` instead.
    """
    if record is None:
        return None

    modalities = list(record.modalities_in or [])
    return Capabilities(
        vision=("image" in modalities) if modalities else None,
        tools=bool(record.tool_call),
        structured_output=bool(record.structured_output),
        reasoning=bool(record.reasoning),
        open_weights=bool(record.open_weights),
        context=record.context,
        max_output=record.max_output,
        modalities_in=modalities or None,
    )


def from_tags(tags: list[str]) -> Capabilities:
    """Read the deprecated capability tags into a typed record.

    `tools` means `tools=True`, `vision` means `vision=True`, and
    `long_context` means `context=LONG_CONTEXT_TOKENS`, a floor rather
    than a measurement. Every other tag is ignored here: tags still
    carry governance, performance and expertise meaning and stay
    selectable with `tag:`.

    Each distinct alias logs ONE DeprecationWarning per process, via a
    module-level warned set, so a registry with twenty tagged endpoints
    does not print twenty lines.

    Parameters
    ----------
    tags : list[str]
        The endpoint's own tags, untouched by this call.

    Returns
    -------
    Capabilities
        Only the fields the aliases state; everything else stays None.
    """
    values: dict[str, object] = {}

    for tag in tags:
        alias = _TAG_ALIASES.get(tag)
        if alias is None:
            continue

        field, value = alias
        values.setdefault(field, value)

        if tag not in _warned:
            _warned.add(tag)
            logger.warning(
                "endpoint tag %r is deprecated; write capabilities: "
                "{%s: %s} instead. It will stop populating capabilities "
                "in the next release.",
                tag,
                field,
                "true" if value is True else value,
            )

    return Capabilities(**values)


def capabilities_for(
    endpoint: "Endpoint", record: Optional["ModelRecord"]
) -> Capabilities:
    """Return one endpoint's merged capabilities.

    The composition of `from_tags` over the endpoint's tags and `merge`
    over its explicit block, those tags, and its catalog record.

    Parameters
    ----------
    endpoint : Endpoint
        The configured or anonymous endpoint.
    record : ModelRecord, optional
        Its models.dev record, None when the model maps to no entry.

    Returns
    -------
    Capabilities
        The merged record, `None` wherever no source knew.
    """
    explicit = getattr(endpoint, "capabilities", None)
    tag_derived = from_tags(getattr(endpoint, "tags", None) or [])
    return merge(explicit, record, tag_derived)


# ---------------------------------------------------------------------------
# Selector grammar
# ---------------------------------------------------------------------------

#: Boolean capability terms a selector may carry, e.g. `vision:true`.
#: `tools` is an alias of the catalog's `tool_call`; both spellings are
#: answered from `Capabilities.tools`, which already merged the catalog.
CAPABILITY_KEYS = {
    "vision",
    "tools",
    "structured_output",
    "reasoning",
    "open_weights",
}

#: Numeric capability terms, spelled `>=N` or `<=N` only.
RANGE_KEYS = {"context", "max_output"}

#: The catalog spelling routellm accepts as an alias of `tools`.
_TOOL_CALL_ALIAS = "tool_call"

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


class CapabilityQuery(BaseModel):
    """The capability half of a `select` expression, already parsed.

    Attributes
    ----------
    booleans : dict[str, bool]
        Capability field name to the value the term demands.
    ranges : dict[str, tuple[str, int]]
        Capability field name to an `(op, value)` pair, `op` being
        `">="` or `"<="`.
    modalities_in : list[str]
        Input modalities every candidate must accept, from `input:`.
    """

    booleans: dict[str, bool] = {}
    ranges: dict[str, tuple[str, int]] = {}
    modalities_in: list[str] = []

    def is_empty(self) -> bool:
        """Return whether this query constrains nothing at all."""
        return not (self.booleans or self.ranges or self.modalities_in)

    def keys(self) -> list[str]:
        """Return every capability field this query reads."""
        names = list(self.booleans) + list(self.ranges)
        if self.modalities_in:
            names.append("modalities_in")
        return names


def parse_capability_terms(terms: list[str]) -> CapabilityQuery:
    """Parse routellm's own `key:value` capability terms.

    `hop.aim.parse_query` accepts none of these keys, so they are split
    out of a `select` expression before aim ever sees it. Accepted
    spellings::

        vision:true  tools:false  structured_output:true
        reasoning:true  open_weights:true
        context:>=200000  max_output:<=8192
        input:image

    Parameters
    ----------
    terms : list[str]
        The capability terms, each already `key:value`.

    Returns
    -------
    CapabilityQuery
        The parsed constraints.

    Raises
    ------
    ValueError
        If a boolean term carries a non-boolean value, a range term
        omits its comparison operator, or a key is not a capability.
    """
    booleans: dict[str, bool] = {}
    ranges: dict[str, tuple[str, int]] = {}
    modalities: list[str] = []

    for term in terms:
        key, _, value = term.partition(":")

        if key == _TOOL_CALL_ALIAS:
            key = "tools"

        if key in CAPABILITY_KEYS:
            booleans[key] = _as_bool(key, value)
        elif key in RANGE_KEYS:
            ranges[key] = _as_range(key, value)
        elif key == "input":
            modalities.append(value)
        else:  # pragma: no cover - the splitter never routes anything else
            raise ValueError(f"Selector term {term!r} is not a capability term.")

    return CapabilityQuery(
        booleans=booleans, ranges=ranges, modalities_in=modalities
    )


def _as_bool(key: str, value: str) -> bool:
    """Return a boolean term's value, or raise naming the term."""
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(
        f"Selector term {key}:{value!r} needs a boolean value; write "
        f"{key}:true or {key}:false."
    )


def _as_range(key: str, value: str) -> tuple[str, int]:
    """Return a range term's `(op, number)`, or raise naming the term.

    A bare `context:200000` is rejected rather than read as equality:
    "exactly this context window" is never what an operator means, and
    silently treating it as `>=` would hide the typo.
    """
    text = value.strip()
    for op in (">=", "<="):
        if text.startswith(op):
            number = text[len(op):].strip()
            try:
                return op, int(number)
            except ValueError:
                raise ValueError(
                    f"Selector term {key}:{value!r} needs an integer after "
                    f"{op}; write {key}:{op}200000."
                ) from None

    raise ValueError(
        f"Selector term {key}:{value!r} needs a comparison: write "
        f"{key}:>={text or 'N'} or {key}:<={text or 'N'}. A bare "
        f"{key}:N would mean an exact window, which is never the intent."
    )


def matches(caps: Capabilities, query: CapabilityQuery) -> bool:
    """Return whether an endpoint's capabilities satisfy a selector query.

    UNKNOWN FAILS THE TERM here. A candidate whose `vision` is `None`
    does not satisfy `vision:true`. This is the OPPOSITE of the
    request-time default in `satisfies`, deliberately: startup is where
    an operator can see the gap and fix it, `--capabilities` prints
    exactly which endpoint is unknown on which key, and silently
    selecting a model that might not do the job is worse than a clear
    "no candidate matched".

    Parameters
    ----------
    caps : Capabilities
        The candidate's merged capabilities.
    query : CapabilityQuery
        The parsed capability terms, all ANDed.

    Returns
    -------
    bool
        Whether every term holds.
    """
    for name, wanted in query.booleans.items():
        if getattr(caps, name) is not wanted:
            return False

    for name, (op, bound) in query.ranges.items():
        value = getattr(caps, name)
        if value is None:
            return False
        if op == ">=" and value < bound:
            return False
        if op == "<=" and value > bound:
            return False

    if query.modalities_in:
        available = caps.modalities_in
        if available is None:
            return False
        if not set(query.modalities_in).issubset(set(available)):
            return False

    return True
