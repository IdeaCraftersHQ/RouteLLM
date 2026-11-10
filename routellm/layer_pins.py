"""Governance pinned to the system config layer.

Layers merge by key, so without a guard a project file, the user file,
`ROUTELLM_CONFIG` or `--config` could re-tag an endpoint the operator
defined globally (add `local` to a cloud endpoint) or point `policy:`
at a file of its own. Governance is therefore honoured only from the
system layer.

The rules, applied to every non-system layer as it is merged:

* `policy:` may be set by the system layer only, whether or not the
  system layer sets one.
* When the system layer sets a `policy:`, these are pinned too: the keys
  in `PINNED_KEYS`, and the governance tags of every endpoint, meaning
  the reach labels (`local`, `vpn`, `cloud`) and every namespaced tag
  (`jur:ca`, `team:x`). A higher layer may not add, remove or change
  them on any endpoint, including one it defines itself. Plain tags,
  every other endpoint key and deleting an endpoint outright stay
  mergeable. An endpoint a higher layer adds with no governance tag is
  allowed; carrying none, no policy class can reach it.
* With no system `policy:` nothing but `policy:` is pinned and layers
  merge as they always did.

A violation is a `ConfigError` naming the layer, its file, the key or
endpoint, and the system file the key is pinned to.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from routellm.governance import REACH_TAGS, split_tag

POLICY_KEY = "policy"

#: Top-level keys a non-system layer may not set while a system `policy:`
#: is loaded. `policy` itself is always system-only.
PINNED_KEYS: tuple[str, ...] = (
    "policy",
    "governance",
    "strict_registry",
    "tag_namespaces",
    "egress_gates",
)

#: Display name of the endpoint-tag pin, listed beside `PINNED_KEYS`.
TAGS_PIN = "endpoints.*.tags"


def pinned_labels() -> tuple[str, ...]:
    """Every pin, as `config show` lists them."""
    return (*PINNED_KEYS, TAGS_PIN)


def is_pinning(system_layer: Mapping[str, Any]) -> bool:
    """True when the system layer loads a policy, which switches the pins on."""
    return system_layer.get(POLICY_KEY) is not None


def governance_tags(endpoints: Any) -> dict[str, frozenset[str]]:
    """Governance tags per endpoint: reach labels and namespaced tags."""
    found: dict[str, frozenset[str]] = {}
    if not isinstance(endpoints, Mapping):
        return found
    for name, spec in endpoints.items():
        tags = spec.get("tags") if isinstance(spec, Mapping) else None
        found[str(name)] = frozenset(
            t
            for t in (tags if isinstance(tags, list) else [])
            if isinstance(t, str) and (t in REACH_TAGS or split_tag(t) is not None)
        )
    return found


def check_layer(
    entry,
    parsed: Mapping[str, Any],
    merged_before: Mapping[str, Any],
    merged_after: Mapping[str, Any],
    pinned: bool,
    system_path: Path,
) -> None:
    """Refuse a non-system layer that touches pinned governance.

    Parameters
    ----------
    entry : ResolvedPath
        The layer being merged; names the file and the layer in errors.
    parsed : Mapping
        What the layer set.
    merged_before, merged_after : Mapping
        The merged config before and after this layer.
    pinned : bool
        Whether the system layer loaded a policy.
    system_path : Path
        The system layer file, named as the pin's source.

    Raises
    ------
    ConfigError
        On the first violation.
    """
    from routellm.config import ConfigError

    where = f"{entry.path}: the {entry.source} config layer may not"
    if POLICY_KEY in parsed:
        raise ConfigError(
            f"{where} set `{POLICY_KEY}`: it is honoured only from the system layer ({system_path})"
        )
    if not pinned:
        return
    for key in PINNED_KEYS:
        if key in parsed:
            raise ConfigError(
                f"{where} set `{key}`: pinned to the system layer ({system_path}) "
                "while a `policy:` is loaded"
            )
    before = governance_tags(merged_before.get("endpoints"))
    after = governance_tags(merged_after.get("endpoints"))
    for name, tags in after.items():
        old = before.get(name, frozenset())
        if tags == old:
            continue
        parts = []
        if tags - old:
            parts.append(f"add {', '.join(sorted(tags - old))}")
        if old - tags:
            parts.append(f"remove {', '.join(sorted(old - tags))}")
        raise ConfigError(
            f"{where} {' and '.join(parts)} (governance tags) on endpoint {name!r}: "
            f"pinned to the system layer ({system_path}) while a `policy:` is loaded"
        )
