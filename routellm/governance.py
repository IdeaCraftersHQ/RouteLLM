"""Governance tag grammar: reach set, single jurisdiction, namespaces.

Endpoint tags stay free-form, with one exception: a tag of the form
`<ns>:<value>` is namespaced, and a namespace has to be known. Two
conventions are parsed rather than merely matched:

- reach, the subset of `local`, `vpn`, `cloud` an endpoint carries
  (how far a request sent to it travels);
- `jur:<cc>`, at most one per endpoint, a lowercase ISO 3166-1
  alpha-2 country code (where it is processed).

Nothing here reads a request. The registry applies these rules at load,
before any tier is resolved, so a typo in a tag fails the config rather
than silently widening a selector.
"""

import re
from dataclasses import dataclass

from routellm.transcription import LOCAL_TAG

__all__ = [
    "BUILTIN_NAMESPACES",
    "JUR_NAMESPACE",
    "REACH_TAGS",
    "TagTerm",
    "check_namespaces",
    "check_tag_namespaces_config",
    "check_tag_term",
    "parse_jur",
    "parse_reach",
    "parse_tag_term",
    "split_tag",
]

VPN_TAG = "vpn"
CLOUD_TAG = "cloud"

#: Reach values, nearest to furthest.
REACH_TAGS: frozenset[str] = frozenset({LOCAL_TAG, VPN_TAG, CLOUD_TAG})

JUR_NAMESPACE = "jur"

#: Namespaces always known; `tag_namespaces:` adds to these.
BUILTIN_NAMESPACES: frozenset[str] = frozenset({JUR_NAMESPACE})

_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_JUR_PATTERN = re.compile(r"^[a-z]{2}$")


def split_tag(tag: str) -> tuple[str, str] | None:
    """Split `<ns>:<value>` on its first colon; None for a plain tag."""
    ns, sep, value = tag.partition(":")
    return (ns, value) if sep else None


def parse_reach(tags: list[str]) -> frozenset[str]:
    """Return the reach values among `tags`."""
    return REACH_TAGS.intersection(tags)


def parse_jur(name: str, tags: list[str]) -> str | None:
    """Return the single `jur:` value among `tags`, or None.

    Raises
    ------
    ValueError
        If there is more than one `jur:` tag, or a value is not a
        lowercase two-letter country code.
    """
    values = []
    for tag in tags:
        parts = split_tag(tag)
        if parts and parts[0] == JUR_NAMESPACE:
            if not _JUR_PATTERN.match(parts[1]):
                raise ValueError(
                    f"endpoint {name!r} tag {tag!r}: jur must be a lowercase "
                    "ISO 3166-1 alpha-2 country code, e.g. `jur:ca`."
                )
            values.append(parts[1])
    if len(values) > 1:
        raise ValueError(
            f"endpoint {name!r} carries {len(values)} jur: tags ({', '.join(values)}); "
            "an endpoint is processed in one jurisdiction."
        )
    return values[0] if values else None


def check_namespaces(name: str, tags: list[str], known: frozenset[str]) -> None:
    """Refuse a namespaced tag whose namespace is not in `known`."""
    for tag in tags:
        parts = split_tag(tag)
        if parts is None:
            continue
        if parts[0] not in known or not parts[1]:
            raise ValueError(
                f"endpoint {name!r} tag {tag!r}: unknown tag namespace {parts[0]!r} "
                f"(known: {', '.join(sorted(known))}). Declare it under `tag_namespaces:` "
                "if it is intended."
            )


def check_tag_namespaces_config(raw: object) -> frozenset[str]:
    """Validate the `tag_namespaces:` config value; return the known set."""
    if raw is None:
        return BUILTIN_NAMESPACES
    if not isinstance(raw, list):
        raise ValueError("`tag_namespaces:` must be a list of names")
    for item in raw:
        if not isinstance(item, str) or not _NAMESPACE_PATTERN.match(item):
            raise ValueError(
                f"`tag_namespaces:` entry {item!r} must match [a-z][a-z0-9_]* "
                "(lowercase, no colon)."
            )
        if item in REACH_TAGS:
            raise ValueError(f"`tag_namespaces:` entry {item!r} collides with a reach tag.")
    return BUILTIN_NAMESPACES | frozenset(raw)


@dataclass(frozen=True)
class TagTerm:
    """One `tag:` selector term.

    `any_of` holds the alternatives (one for a plain `tag:x`); the term
    holds for an endpoint carrying at least one. `negated` flips it, so
    `!tag:a|b` means carrying neither.
    """

    any_of: tuple[str, ...]
    negated: bool = False

    def matches(self, tags: "list[str] | set[str] | frozenset[str]") -> bool:
        hit = any(alt in tags for alt in self.any_of)
        return hit != self.negated


def parse_tag_term(term: str, select: str = "") -> TagTerm:
    """Parse `tag:<a>[|<b>...]` or `!tag:<a>[|<b>...]`.

    Raises
    ------
    ValueError
        On an empty label or alternative, or on alternatives that share
        no namespace and are not all reach labels. A malformed term
        never degrades to a match-all.
    """
    where = f"Selector term {term!r}" + (f" in {select!r}" if select else "")
    negated = term.startswith("!")
    body = term[1:] if negated else term
    if not body.startswith("tag:"):
        raise ValueError(f"{where}: `!` negates `tag:` terms only.")
    alts = tuple(body[len("tag:") :].split("|"))
    if any(not alt for alt in alts):
        raise ValueError(f"{where}: empty tag label or alternative.")
    if len(alts) > 1:
        spaces = {(split_tag(a) or (None,))[0] for a in alts}
        all_reach = all(a in REACH_TAGS for a in alts)
        if not all_reach and (len(spaces) != 1 or None in spaces):
            raise ValueError(
                f"{where}: `|` alternatives must share one namespace "
                "(e.g. `jur:ca|jur:fr`) or all be reach labels (`local|vpn|cloud`)."
            )
    for alt in alts:
        parts = split_tag(alt)
        if parts is not None and not parts[1]:
            raise ValueError(f"{where}: empty value in {alt!r}.")
    return TagTerm(alts, negated)


def check_tag_term(term: TagTerm, known: frozenset[str], select: str = "") -> None:
    """Refuse a negated or any-of term naming an unknown namespace.

    A plain positive term with a typo matches nothing, which fails
    closed; a negated or alternative one would widen the match.
    """
    if not term.negated and len(term.any_of) == 1:
        return
    for alt in term.any_of:
        parts = split_tag(alt)
        if parts is not None and parts[0] not in known:
            raise ValueError(
                f"Selector tag {alt!r}"
                + (f" in {select!r}" if select else "")
                + f": unknown tag namespace {parts[0]!r} "
                f"(known: {', '.join(sorted(known))})."
            )
