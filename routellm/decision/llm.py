"""Built-in `llm` provider: a typed answer from a model, validated.

The model gets the label sets and a JSON schema whose fields are enums
of those labels. Whatever it returns is validated again here: an answer
outside the label sets is an error, never coerced.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from routellm.decision.config import DecisionConfig
from routellm.decision.models import Decision, DecisionError, RequestView

#: `transport(endpoint, messages, response_format, timeout_ms) -> answer text`
Transport = Callable[[str, list[dict[str, Any]], dict[str, Any], int], Awaitable[str]]


def answer_schema(config: DecisionConfig) -> dict[str, Any]:
    """Return the JSON schema the answer must satisfy."""
    props: dict[str, Any] = {
        "data_class": {"type": "string", "enum": list(config.data_classes)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    if config.intents:
        props["intent"] = {"type": "string", "enum": list(config.intents)}
    if config.areas:
        props["area"] = {"type": "string", "enum": list(config.areas)}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _labels(title: str, labels: dict[str, str]) -> str:
    return title + ":\n" + "\n".join(f"- {k}: {v}" for k, v in labels.items())


def build_prompt(config: DecisionConfig, text: str) -> list[dict[str, Any]]:
    """Return the messages asking the model to label `text`."""
    sections = [_labels("data_class (pick the most restrictive that applies)", config.data_classes)]
    if config.intents:
        sections.append(_labels("intent", config.intents))
    if config.areas:
        sections.append(_labels("area", config.areas))
    system = (
        "You label a request for a router. Answer with one JSON object using only "
        "the labels below, plus confidence between 0 and 1. Do not answer the "
        "request itself.\n\n" + "\n\n".join(sections)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Request to label:\n<<<\n{text}\n>>>"},
    ]


class LLMProvider:
    """Ask a registry endpoint for a typed label set."""

    def __init__(self, config: DecisionConfig, transport: Transport):
        self._config, self._transport = config, transport
        self._format = {
            "type": "json_schema",
            "json_schema": {"name": "decision", "strict": True, "schema": answer_schema(config)},
        }

    async def decide(self, view: RequestView) -> Decision:
        cfg = self._config
        raw = await self._transport(
            cfg.endpoint, build_prompt(cfg, view.text), self._format, cfg.timeout_ms
        )
        return self.parse(raw)

    def parse(self, raw: Any) -> Decision:
        """Validate a raw answer into a Decision.

        Raises
        ------
        DecisionError
            `invalid_answer` for non-JSON, a wrong shape, a label outside
            its set, or a missing or out-of-range confidence.
        """
        cfg = self._config
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError as exc:
            raise DecisionError("invalid_answer", "not JSON") from exc
        if not isinstance(data, dict):
            raise DecisionError("invalid_answer", "not an object")

        def label(key: str, allowed: dict[str, str], required: bool) -> str | None:
            value = data.get(key)
            if value is None and not required:
                return None
            if not isinstance(value, str) or value not in allowed:
                raise DecisionError("invalid_answer", f"{key} {value!r} is not a configured label")
            return value

        data_class = label("data_class", cfg.data_classes, True)
        intent = label("intent", cfg.intents, True) if cfg.intents else None
        area = label("area", cfg.areas, True) if cfg.areas else None
        confidence = data.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise DecisionError("invalid_answer", "confidence missing or not a number")
        if not 0 <= confidence <= 1:
            raise DecisionError("invalid_answer", "confidence outside 0..1")
        return Decision(
            data_class=data_class,
            intent=intent,
            area=area,
            confidence=float(confidence),
            source="model",
        )


def create(config: DecisionConfig, transport: Transport) -> LLMProvider:
    """Provider factory for the `llm` kind."""
    return LLMProvider(config, transport)
