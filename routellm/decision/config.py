"""The `decision:` config section, validated at load."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ["DecisionConfig", "InputWindow", "RecognizerSpec", "StickyConfig"]

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputWindow(_Strict):
    """Which part of the request the model reads.

    `messages`: `last_user` (the latest user message), `user` (every
    user message) or `all`. `keep` says which end survives truncation
    to `max_chars`.
    """

    messages: Literal["last_user", "user", "all"] = "last_user"
    max_chars: int = Field(default=4000, gt=0)
    keep: Literal["head", "tail"] = "tail"


class StickyConfig(_Strict):
    """Per-conversation floor: a class never drops below the strictest seen."""

    enabled: bool = False
    ttl_s: float = Field(default=3600.0, gt=0)
    max_entries: int = Field(default=10000, gt=0)


class RecognizerSpec(_Strict):
    """One deterministic recognizer.

    `type` is `regex` (`patterns`), `markers` (`markers`, literal
    substrings), or the name of a recognizer registered under the
    `routellm.decision_recognizers` entry-point group, which reads
    `options`.
    """

    name: str
    type: str
    data_class: str = Field(alias="class")
    patterns: list[str] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)
    case_sensitive: bool = False
    options: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DecisionConfig(_Strict):
    """The `decision:` section.

    `data_classes` is ordered least to most restrictive: the last one
    is the class `on_error: most_restrictive` falls to and the class
    that skips the model. Labels and descriptions are all config.

    `declarable` names a subset of `data_classes` that only a caller can
    declare: the model is never offered them and never accepted naming
    one, no recognizer may emit one, and none is carried sticky. They
    keep their place in the order, so strictest-wins still applies; a
    declared one can be raised by the model but never lowered. The last
    class cannot be one: it is the fail-closed class.
    """

    provider: str = "llm"
    endpoint: str
    require_tags: list[str] = Field(default_factory=list)
    data_classes: dict[str, str]
    declarable: list[str] = Field(default_factory=list)
    intents: dict[str, str] = Field(default_factory=dict)
    areas: dict[str, str] = Field(default_factory=dict)
    input: InputWindow = Field(default_factory=InputWindow)
    timeout_ms: int = Field(default=2000, gt=0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    on_error: Literal["most_restrictive", "refuse"] = "most_restrictive"
    recognizers: list[RecognizerSpec] = Field(default_factory=list)
    sticky: StickyConfig = Field(default_factory=StickyConfig)
    options: dict[str, Any] = Field(default_factory=dict)
    eval_min_recall: dict[str, float] = Field(default_factory=dict)

    @field_validator("data_classes", "intents", "areas")
    @classmethod
    def _labels(cls, value: dict[str, str]) -> dict[str, str]:
        for label in value:
            if not _LABEL.match(str(label)):
                raise ValueError(f"label {label!r} must match {_LABEL.pattern}")
        return {str(k): "" if v is None else str(v) for k, v in value.items()}

    @model_validator(mode="after")
    def _consistent(self) -> "DecisionConfig":
        if not self.data_classes:
            raise ValueError("data_classes must name at least one class")
        known = set(self.data_classes)
        for name in self.declarable:
            if name not in known:
                raise ValueError(f"declarable: {name!r} is not in data_classes")
        if len(set(self.declarable)) != len(self.declarable):
            raise ValueError("declarable: lists a class twice")
        if self.most_restrictive in self.declarable:
            raise ValueError(
                f"declarable: {self.most_restrictive!r} is the last data class, the fail-closed "
                "class; the model must be able to choose it"
            )
        for spec in self.recognizers:
            if spec.data_class in self.declarable:
                raise ValueError(
                    f"recognizer {spec.name!r}: class {spec.data_class!r} is declarable; "
                    "only a caller may name it"
                )
            if spec.data_class not in known:
                raise ValueError(
                    f"recognizer {spec.name!r}: class {spec.data_class!r} is not in data_classes"
                )
        for name, floor in self.eval_min_recall.items():
            if name not in known:
                raise ValueError(f"eval_min_recall: {name!r} is not in data_classes")
            if not 0.0 <= floor <= 1.0:
                raise ValueError(f"eval_min_recall: {name!r} floor must be within 0..1")
        return self

    @property
    def most_restrictive(self) -> str:
        """The strictest class: the last one declared."""
        return list(self.data_classes)[-1]

    @property
    def selectable(self) -> dict[str, str]:
        """The classes the model may choose: `data_classes` less `declarable`."""
        return {k: v for k, v in self.data_classes.items() if k not in self.declarable}

    def rank(self, data_class: str) -> int:
        """Return the restrictiveness rank of `data_class`; higher is stricter."""
        return list(self.data_classes).index(data_class)
