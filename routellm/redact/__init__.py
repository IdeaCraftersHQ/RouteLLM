"""Local PII analysis: `analyze(text, language)` -> spans.

    from routellm.redact import analyze
    analyze("Né le 12 mars 1985, NAS 046 454 286", "fr")
    # [{"start": 6, "end": 18, "category": "date_of_birth", "score": 0.85}, ...]

A span is `{start, end, category, score}`: character offsets into the
text (end exclusive), a category name, a score in [0, 1].

Two tiers, one call:

- Base install: the pattern recognizers only (SIN, RAMQ, postal code,
  amounts, date of birth, email, configured client ids).
  `analyze_patterns` is that tier on its own.
- `pip install 'routellm[redact]'` plus the spaCy models: adds person,
  location, phone, IBAN and card through Presidio. `analyze` uses this
  tier and raises `RedactUnavailable` if it is not installed.

Offline: no call here touches the network. See docs/redact.md.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from routellm.redact.recognizers import (
    LANGUAGES,
    Span,
    analyze_patterns,
    compile_client_id_patterns,
)

__all__ = [
    "LANGUAGES",
    "RedactUnavailable",
    "Span",
    "analyze",
    "analyze_patterns",
    "build_analyzer",
    "compile_client_id_patterns",
]

_cache: dict[tuple, Any] = {}


def __getattr__(name: str):
    # RedactUnavailable lives with the adapter but is importable from
    # here without Presidio: the module that defines it imports Presidio
    # only inside the engine builder.
    if name == "RedactUnavailable":
        from routellm.redact.presidio_adapter import RedactUnavailable

        return RedactUnavailable
    raise AttributeError(name)


def build_analyzer(config: Mapping[str, Any] | None = None):
    """A `PresidioAnalyzer` from a `redact:` config section (or none).

    Keys: `client_id_patterns` (list of regex), `models`
    (`{fr: <spaCy model>, en: <spaCy model>}`).
    """
    from routellm.redact.presidio_adapter import DEFAULT_MODELS, PresidioAnalyzer

    config = dict(config or {})
    unknown = set(config) - {"client_id_patterns", "models"}
    if unknown:
        raise ValueError(f"redact: unknown key(s) {sorted(unknown)}")
    return PresidioAnalyzer(
        client_id_patterns=tuple(config.get("client_id_patterns") or ()),
        models={**DEFAULT_MODELS, **(config.get("models") or {})},
    )


def analyze(
    text: str,
    language: str,
    *,
    client_id_patterns: Iterable[str] = (),
    models: Mapping[str, str] | None = None,
    score_threshold: float = 0.4,
) -> list[dict]:
    """Spans for `text`, as dicts, in text order. `language` is `fr` or `en`.

    The engine (and its spaCy models) is built on first use and reused
    for the same `client_id_patterns` and `models`.
    """
    key = (tuple(client_id_patterns), tuple(sorted((models or {}).items())))
    if key not in _cache:
        _cache[key] = build_analyzer({"client_id_patterns": key[0], "models": dict(key[1])})
    return [s.to_dict() for s in _cache[key].analyze(text, language, score_threshold)]
