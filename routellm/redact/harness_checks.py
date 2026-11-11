"""Pure checks of the harness: what leaked, what re-identifies, what links.

Each takes observations (text, files) and returns the violations; none
runs a server. `harness_world` produces the observations.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "ARTIFACTS",
    "leaked",
    "linkable",
    "placeholders",
    "read_artifact",
    "reidentifying",
    "scan_artifacts",
]

PLACEHOLDER = re.compile(r"\[\[[A-Za-z_]+_\d+\]\]")
_SEPARATORS = re.compile(r"[\s\-.\u00a0\u202f]+")
# Independent of the recognizers on purpose: a recognizer that misses a
# date or a postal code must not also hide it from the check.
_POSTAL = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d(?![A-Za-z0-9])")
_DATES = (
    re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-]\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}(?:er)?\s+[A-Za-zéèûôîÀ-ÿ]{3,9}\.?\s+\d{4}(?!\d)"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}(?!\d)"),
)
#: What a run leaves behind that must hold no value in clear.
ARTIFACTS = ("outbound", "logs", "traces", "cache", "audit", "token_store")


def _squash(text: str) -> str:
    return _SEPARATORS.sub("", text).lower()


def leaked(text: str, values: Iterable[str]) -> list[str]:
    """The values found in `text`, as written or with separators removed.

    The second form catches `046454286` for `046 454 286`; it is used
    only for values of six or more characters, where a chance match is
    not a concern.
    """
    low, squashed = text.lower(), _squash(text)
    found = []
    for value in dict.fromkeys(values):
        if value.lower() in low or (len(_squash(value)) >= 6 and _squash(value) in squashed):
            found.append(value)
    return found


def read_artifact(path: str | Path) -> str:
    """The text of a file, or of every file under a directory (undecodable bytes replaced)."""
    path = Path(path)
    if path.is_dir():
        return "\n".join(read_artifact(p) for p in sorted(path.rglob("*")) if p.is_file())
    if not path.is_file():
        return ""
    return path.read_bytes().decode("utf-8", "replace")


def scan_artifacts(artifacts: Mapping[str, str], values: Iterable[str]) -> dict[str, list[str]]:
    """`{artifact: values found}` for each artifact that holds any value."""
    values = list(values)
    return {name: hits for name, text in artifacts.items() if (hits := leaked(text, values))}


def reidentifying(texts: Sequence[str], values: Iterable[str] = ()) -> list[str]:
    """Outbound texts that carry a date and a postal code together, name or no name.

    A birth date with a postal code narrows a person to a handful of
    people, so the pair must not travel even when each alone might. The
    shapes are matched here, not by the recognizers; `values` (labelled
    dates and postal codes) are matched too.
    """
    values = list(values)
    out = []
    for text in texts:
        pair = _POSTAL.search(text) and any(rx.search(text) for rx in _DATES)
        if pair or len(leaked(text, values)) >= 2:
            out.append(text)
    return out


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def linkable(per_request: Sequence[Iterable[str]]) -> list[str]:
    """Placeholders that appear in more than one request (sorted).

    Two requests that carry the same value must not carry the same
    placeholder, or an observer links them.
    """
    seen: dict[str, int] = {}
    for texts in per_request:
        for p in set(texts):
            seen[p] = seen.get(p, 0) + 1
    return sorted(p for p, n in seen.items() if n > 1)
