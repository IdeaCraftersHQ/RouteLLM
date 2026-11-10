"""Deterministic recognizers: a class from the text, no model call.

A recognizer is any object with `name`, `data_class`, a `blocking`
flag and `recognize(text, language=None) -> Match | None`. Regex and
marker recognizers come from config. A heavier analyzer (an NER or PII
engine exposing `analyze(text, language) -> spans`) plugs in through
`SpanRecognizer` without this module knowing it. A recognizer that does
real work sets `blocking = True`, and the classifier runs it off the
event loop.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from routellm.decision.config import RecognizerSpec
from routellm.decision.plugins import RECOGNIZER_GROUP, load_entry_point

__all__ = [
    "MarkerRecognizer",
    "Match",
    "Recognizer",
    "RegexRecognizer",
    "SpanRecognizer",
    "build_recognizer",
]


@dataclass(frozen=True)
class Match:
    """A recognizer hit: which recognizer, which class, why."""

    name: str
    data_class: str
    detail: dict[str, Any] = field(default_factory=dict)


class Recognizer(Protocol):
    """Interface every recognizer meets."""

    name: str
    data_class: str
    blocking: bool

    def recognize(self, text: str, language: str | None = None) -> Match | None: ...


class RegexRecognizer:
    """Hits when any configured pattern matches."""

    blocking = False

    def __init__(self, name: str, data_class: str, patterns: list[str], case_sensitive=False):
        if not patterns:
            raise ValueError(f"recognizer {name!r}: type regex needs `patterns`")
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, flags))
            except re.error as exc:
                raise ValueError(f"recognizer {name!r}: bad pattern {pattern!r}: {exc}") from exc
        self.name, self.data_class, self._patterns = name, data_class, compiled

    def recognize(self, text: str, language: str | None = None) -> Match | None:
        for pattern in self._patterns:
            if pattern.search(text):
                return Match(self.name, self.data_class, {"pattern": pattern.pattern})
        return None


class MarkerRecognizer:
    """Hits when any literal marker occurs."""

    blocking = False

    def __init__(self, name: str, data_class: str, markers: list[str], case_sensitive=False):
        if not markers or any(not m for m in markers):
            raise ValueError(f"recognizer {name!r}: type markers needs non-empty `markers`")
        self.name, self.data_class = name, data_class
        self._fold = not case_sensitive
        self._markers = [m.lower() if self._fold else m for m in markers]

    def recognize(self, text: str, language: str | None = None) -> Match | None:
        haystack = text.lower() if self._fold else text
        for marker in self._markers:
            if marker in haystack:
                return Match(self.name, self.data_class, {"marker": marker})
        return None


class SpanRecognizer:
    """Adapt an analyzer exposing `analyze(text, language) -> spans`.

    A span carries `entity_type` and `score` (attribute or key). It
    hits when any span of a type in `entity_types` (all types when
    empty) scores at least `min_score`. The analyzer is called off the
    event loop.
    """

    blocking = True

    def __init__(
        self,
        name: str,
        data_class: str,
        analyzer: Any,
        entity_types: list[str] | None = None,
        min_score: float = 0.5,
        language: str = "en",
    ):
        self.name, self.data_class = name, data_class
        self._analyzer = analyzer
        self._types = set(entity_types or [])
        self._min_score = min_score
        self._language = language

    @staticmethod
    def _get(span: Any, key: str):
        return span.get(key) if isinstance(span, dict) else getattr(span, key, None)

    def recognize(self, text: str, language: str | None = None) -> Match | None:
        for span in self._analyzer.analyze(text, language or self._language):
            kind, score = self._get(span, "entity_type"), self._get(span, "score")
            if self._types and kind not in self._types:
                continue
            if score is not None and score < self._min_score:
                continue
            return Match(self.name, self.data_class, {"entity_type": kind, "score": score})
        return None


def build_recognizer(spec: RecognizerSpec) -> Recognizer:
    """Build one recognizer from its config entry.

    Raises
    ------
    ValueError
        If the spec is malformed or its `type` is neither built in nor
        registered under the `routellm.decision_recognizers` group.
    """
    if spec.type == "regex":
        return RegexRecognizer(spec.name, spec.data_class, spec.patterns, spec.case_sensitive)
    if spec.type == "markers":
        return MarkerRecognizer(spec.name, spec.data_class, spec.markers, spec.case_sensitive)
    factory = load_entry_point(RECOGNIZER_GROUP, spec.type)
    if factory is None:
        raise ValueError(
            f"recognizer {spec.name!r}: unknown type {spec.type!r} "
            f"(built in: regex, markers; others register under {RECOGNIZER_GROUP})"
        )
    return factory(spec)
