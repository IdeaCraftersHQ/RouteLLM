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
    "LONG_CONTEXT_TOKENS",
    "Capabilities",
    "capabilities_for",
    "from_tags",
    "merge",
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
