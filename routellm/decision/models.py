"""Value types of the decision step: the Decision, the request view, the errors."""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Decision", "DecisionError", "DecisionRefused", "RequestView"]


@dataclass(frozen=True)
class Decision:
    """What the decision step concluded about one request.

    Attributes
    ----------
    data_class : str
        Effective data class: the most restrictive of every source.
    intent : str, optional
        Intent label. None when no model ran (a declaration, a
        recognizer hit, a sticky floor, or a failure).
    area : str, optional
        Topic area label, when the config declares any.
    confidence : float, optional
        The model's stated confidence; None when no model answered.
    source : str
        Where `data_class` came from: `declared`, `recognizer`,
        `sticky`, `model`, or `error` (a failure resolved by `on_error`).
    error : str, optional
        Failure kind when `source` is `error`: `timeout`, `transport`,
        `invalid_answer`, `low_confidence`, `unknown_declared_class`.
    detail : dict
        Free-form evidence: the recognizer that fired, the raw
        failure message, the sticky floor that applied.
    """

    data_class: str
    intent: str | None = None
    area: str | None = None
    confidence: float | None = None
    source: str = "model"
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the record written into the routing path."""
        record: dict[str, Any] = {"data_class": self.data_class, "source": self.source}
        for key in ("intent", "area", "confidence", "error"):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        if self.detail:
            record["detail"] = dict(self.detail)
        return record


@dataclass(frozen=True)
class RequestView:
    """What a provider or recognizer may read of a request.

    Attributes
    ----------
    messages : list[dict]
        The windowed messages the model should judge.
    text : str
        The windowed text, already truncated to the configured size.
    full_text : str
        Every message's text, for deterministic recognizers.
    declared_class : str, optional
        A class the caller declared, already validated.
    conversation_id : str, optional
        Key of the conversation, when one is known.
    """

    messages: list[dict[str, Any]]
    text: str
    full_text: str
    declared_class: str | None = None
    conversation_id: str | None = None


class DecisionError(Exception):
    """A decision attempt failed; `kind` names how."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.message = message


class DecisionRefused(Exception):
    """The decision step failed and `on_error: refuse` is configured."""

    def __init__(self, error: DecisionError):
        super().__init__(f"decision step failed ({error}); on_error is refuse")
        self.error = error
