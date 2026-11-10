"""Fakes for the decision step: a scripted provider and a scripted transport."""

import asyncio
import json

from routellm.decision import DecisionClassifier, DecisionConfig
from routellm.decision.llm import LLMProvider

CLASSES = {
    "general": "nothing sensitive",
    "internal": "company-internal material",
    "private": "personal or secret data; must stay local",
}
INTENTS = {"code": "software work", "chat": "small talk", "hard": "difficult reasoning"}


def make_config(**over) -> DecisionConfig:
    base = {
        "endpoint": "decider",
        "data_classes": CLASSES,
        "intents": INTENTS,
        "timeout_ms": 500,
        "min_confidence": 0.6,
    }
    return DecisionConfig.model_validate({**base, **over})


def answer(data_class="general", intent="chat", confidence=0.9, **extra) -> str:
    return json.dumps(
        {"data_class": data_class, "intent": intent, "confidence": confidence, **extra}
    )


class FakeTransport:
    """Records every call; `reply` is a string, an exception, or a callable."""

    def __init__(self, reply=None, delay=0.0):
        self.reply = answer() if reply is None else reply
        self.delay = delay
        self.calls = []

    async def __call__(self, endpoint, messages, response_format, timeout_ms):
        self.calls.append({"endpoint": endpoint, "messages": messages, "format": response_format})
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.reply(messages) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_classifier(transport=None, recognizers=(), **over):
    config = make_config(**over)
    transport = transport or FakeTransport()
    return (
        DecisionClassifier(config, recognizers, provider=LLMProvider(config, transport)),
        transport,
    )
