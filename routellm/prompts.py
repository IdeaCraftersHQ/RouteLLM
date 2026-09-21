"""Generic YAML prompt file shared by every adapter.

Keeps model-facing wording editable without code changes. The file is a
mapping of section names to mappings; each adapter declares the section
it owns and the types its keys take, and asks for it by name::

    router:
      instructions: <string>
      criteria:
        "true": <string>
        "false": <string>
    intent_detector:
      instructions: <string>
      general_description: <string>

Sections nobody requests are ignored, so one file can carry sections for
every adapter. This module imports nothing from routers or from any
optional SDK: loading a prompt file must never pull in a heavy
dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Tuple

import yaml

logger = logging.getLogger(__name__)


def resolve(kwarg, file_value, default) -> Tuple[Any, str]:
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
        The winning value and the name of its source, one of "kwarg",
        "file" or "default", so callers can log it without re-deriving.
    """
    if kwarg is not None:
        return kwarg, "kwarg"
    if file_value is not None:
        return file_value, "file"
    return default, "default"


class PromptFile:
    """A parsed YAML prompt file, validated one section at a time.

    Parsing checks only that the document is a mapping of sections.
    Per-key validation happens in `section`, against the schema the
    calling adapter declares, so the loader stays adapter-agnostic.
    """

    def __init__(self, sections: Mapping[str, Any], path):
        """Wrap an already-parsed mapping of sections read from `path`."""
        self.sections = sections
        self.path = path

    @classmethod
    def load(cls, path) -> "PromptFile":
        """Read and parse a YAML prompt file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to the YAML prompt file.

        Returns
        -------
        PromptFile
            The parsed file. An empty file yields no sections.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.
        ValueError
            If the document is not a mapping.
        """
        with open(path) as handle:
            document = yaml.safe_load(handle)
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ValueError(
                f"{path}: prompt file must be a mapping of sections, got {type(document).__name__}"
            )
        return cls(document, path)

    def section(self, name: str, schema: Mapping[str, type]) -> Dict[str, Any]:
        """Return one section, validated against `schema`.

        Parameters
        ----------
        name : str
            Section name, e.g. "router".
        schema : Mapping[str, type]
            Allowed keys mapped to the type each value must be.

        Returns
        -------
        dict
            The section's keys and values. Empty when the file has no
            such section.

        Raises
        ------
        ValueError
            If the section is not a mapping, carries a key outside
            `schema`, or gives a value of the wrong type. The message
            names the file, the section and the offending key.
        """
        raw = self.sections.get(name)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"{self.path}: section {name!r} must be a mapping, got {type(raw).__name__}"
            )
        for key, value in raw.items():
            expected = schema.get(key)
            if expected is None:
                raise ValueError(f"{self.path}: unknown key {key!r} in section {name!r}")
            wrong_type = not isinstance(value, expected)
            # bool subclasses int, so isinstance(True, int) is True; an
            # int field must still reject a bool value.
            is_bool_for_int = expected is int and isinstance(value, bool)
            if wrong_type or is_bool_for_int:
                raise ValueError(
                    f"{self.path}: {name}.{key} must be {expected.__name__}, "
                    f"got {type(value).__name__}"
                )
        return dict(raw)
