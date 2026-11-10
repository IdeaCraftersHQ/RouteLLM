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

from routellm.transcription import LOCAL_TAG

__all__ = [
    "BUILTIN_NAMESPACES",
    "JUR_NAMESPACE",
    "REACH_TAGS",
    "check_namespaces",
    "check_tag_namespaces_config",
    "parse_jur",
    "parse_reach",
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
