"""The measured-quality sidecar a config references with `quality_from:`.

An endpoint's `quality:` is a number an operator writes by hand, which
orders `quality_desc` selectors and nothing else. A sidecar lets that
number be *measured* instead, by the evals harness in
`routellm.evals.endpoint_quality` or from scored live traffic, without
either producer having to edit the config:

    version: 1
    generated_at: "2026-09-21T10:00:00Z"
    min_samples: 50
    transform: linear
    source: evals
    endpoints:
      cloud_strong: {quality: 91, n: 50, by_area: {}}
      local_server: {quality: 78, n: 50, by_area: {}}

Reading it is deliberately one small module shared by every producer:
the shape is the contract, and `source:` says which one wrote it.

PRECEDENCE, fixed: an endpoint's own `quality:` always wins over the
sidecar, so a hand-set number is never silently overwritten by a
measurement. Only an endpoint that sets none is filled.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["apply_sidecar", "load_sidecar", "resolve_sidecar_path"]


def resolve_sidecar_path(
    quality_from: str, config_path: Optional[Union[str, Path]]
) -> Path:
    """Return the sidecar path `quality_from` names.

    A relative path is resolved against the CONFIG FILE's directory,
    not the current working directory: a config is a document that
    refers to its neighbours, and a server started from elsewhere must
    read the same file the operator wrote next to it.

    Parameters
    ----------
    quality_from : str
        The path as the config writes it.
    config_path : str or Path, optional
        The config file this came from. None falls back to the CWD,
        which is only ever the case for a registry built in memory.

    Returns
    -------
    Path
        The resolved sidecar path.
    """
    candidate = Path(quality_from)
    if candidate.is_absolute() or config_path is None:
        return candidate

    return Path(config_path).resolve().parent / candidate


def load_sidecar(path: Union[str, Path]) -> dict[str, int]:
    """Read a sidecar and return its endpoint-to-quality mapping.

    Parameters
    ----------
    path : str or Path
        The resolved sidecar path.

    Returns
    -------
    dict[str, int]
        Quality score per endpoint name, clamped to [0, 100].

    Raises
    ------
    ValueError
        If the file does not exist, naming the resolved path, or its
        contents are not the documented shape.
    """
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(
            f"quality_from names {resolved}, which does not exist. Write the "
            "sidecar with `python -m routellm.evals.endpoint_quality`, or "
            "remove the key."
        )

    try:
        payload = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"quality sidecar {resolved} is not valid YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"quality sidecar {resolved} is not a mapping")

    entries = payload.get("endpoints") or {}
    if not isinstance(entries, dict):
        raise ValueError(
            f"quality sidecar {resolved} has no `endpoints:` mapping"
        )

    scores: dict[str, int] = {}
    for name, entry in entries.items():
        value = entry.get("quality") if isinstance(entry, dict) else entry
        if value is None:
            continue
        try:
            scores[name] = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            logger.warning(
                "quality sidecar %s: %s has a non-numeric quality %r; ignored",
                resolved,
                name,
                value,
            )

    return scores


def apply_sidecar(endpoints: dict[str, Any], path: Union[str, Path]) -> None:
    """Fill each endpoint's missing `quality` from a sidecar, in place.

    An endpoint that sets its own `quality:` keeps it: a hand-set
    number is a decision, and a measurement never overrides a decision
    silently. A sidecar naming an endpoint the registry does not carry
    is a WARNING naming it, not an error: endpoints come and go faster
    than measurements do.

    Parameters
    ----------
    endpoints : dict[str, Endpoint]
        The registry's endpoints, mutated in place.
    path : str or Path
        The resolved sidecar path.

    Raises
    ------
    ValueError
        If the sidecar is missing or malformed.
    """
    scores = load_sidecar(path)

    for name, score in scores.items():
        endpoint = endpoints.get(name)
        if endpoint is None:
            logger.warning(
                "quality sidecar %s names %r, which is not a configured "
                "endpoint; ignored",
                path,
                name,
            )
            continue

        if endpoint.quality is not None:
            continue

        endpoint.quality = score
        endpoint.quality_measured = True
