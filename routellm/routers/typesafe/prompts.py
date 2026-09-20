"""YAML prompt file for the Jev router and the Jev intent detector.

Keeps the Noul question and the Choice instructions editable without
code changes. Both sections and every key inside them are optional; an
absent key stays ``None``, which callers read as "use the default".

File format::

    router:
      instructions: <string>
      criteria:
        "true": <string>
        "false": <string>
    intent_detector:
      instructions: <string>
      general_description: <string>

This module deliberately imports no typesafe_sdk: loading a prompt file
must not require the optional dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import yaml

_ROUTER_SECTION = "router"
_DETECTOR_SECTION = "intent_detector"
_ROUTER_KEYS = frozenset({"instructions", "criteria"})
_DETECTOR_KEYS = frozenset({"instructions", "general_description"})
_CRITERIA_KEYS = frozenset({"true", "false"})


@dataclass
class RouterPrompt:
    """Router prompt values read from a prompt file, ``None`` when absent."""

    instructions: Optional[str] = None
    criteria: Optional[Dict[str, str]] = None


@dataclass
class DetectorPrompt:
    """Detector prompt values read from a prompt file, ``None`` when absent."""

    instructions: Optional[str] = None
    general_description: Optional[str] = None


def _resolve(kwarg, file_value, default):
    """Pick the most specific of a kwarg, a file value and a default.

    Parameters
    ----------
    kwarg : Any
        Value passed explicitly by the caller, or None.
    file_value : Any
        Value read from the prompt file, or None.
    default : Any
        Built-in fallback.

    Returns
    -------
    tuple[Any, str]
        The winning value and its source, one of "kwarg", "file" or
        "default", for logging.
    """
    if kwarg is not None:
        return kwarg, "kwarg"
    if file_value is not None:
        return file_value, "file"
    return default, "default"


def _require_mapping(value, label, path):
    """Return `value` when it is a mapping, else raise ValueError."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{path}: {label} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_str(value, label, path):
    """Return `value` when it is a str, else raise ValueError."""
    if not isinstance(value, str):
        raise ValueError(
            f"{path}: {label} must be a string, got {type(value).__name__}"
        )
    return value


def _section(document, name, allowed, path):
    """Validate one top-level section and return it as a mapping."""
    section = document.get(name)
    if section is None:
        return {}
    _require_mapping(section, name, path)
    for key in section:
        if key not in allowed:
            raise ValueError(f"{path}: unknown key {key!r} in {name!r}")
    return section


def _criteria(section, path):
    """Validate and return the router `criteria` mapping, or None."""
    criteria = section.get("criteria")
    if criteria is None:
        return None
    _require_mapping(criteria, "router.criteria", path)
    for key, value in criteria.items():
        if key not in _CRITERIA_KEYS:
            raise ValueError(f"{path}: unknown key {key!r} in 'router.criteria'")
        _require_str(value, f"router.criteria.{key}", path)
    return dict(criteria)


def _optional_str(section, key, section_name, path):
    """Return `section[key]` as a str, or None when the key is absent."""
    value = section.get(key)
    if value is None:
        return None
    return _require_str(value, f"{section_name}.{key}", path)


def load_prompt_file(path) -> Tuple[RouterPrompt, DetectorPrompt]:
    """Load and validate a YAML prompt file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the YAML prompt file.

    Returns
    -------
    tuple[RouterPrompt, DetectorPrompt]
        Router and detector values; every field is None when the file
        does not set it.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the document is not a mapping, carries an unknown key, or
        gives a value of the wrong type. The message names the file and
        the offending key.
    """
    with open(path) as handle:
        document = yaml.safe_load(handle)
    if document is None:
        return RouterPrompt(), DetectorPrompt()
    _require_mapping(document, "prompt file", path)
    for key in document:
        if key not in (_ROUTER_SECTION, _DETECTOR_SECTION):
            raise ValueError(f"{path}: unknown top-level key {key!r}")

    router = _section(document, _ROUTER_SECTION, _ROUTER_KEYS, path)
    detector = _section(document, _DETECTOR_SECTION, _DETECTOR_KEYS, path)
    return (
        RouterPrompt(
            instructions=_optional_str(router, "instructions", _ROUTER_SECTION, path),
            criteria=_criteria(router, path),
        ),
        DetectorPrompt(
            instructions=_optional_str(
                detector, "instructions", _DETECTOR_SECTION, path
            ),
            general_description=_optional_str(
                detector, "general_description", _DETECTOR_SECTION, path
            ),
        ),
    )
