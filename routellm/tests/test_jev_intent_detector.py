"""Tests for JevIntentDetector, backed by a MockTransport TypeSafe client."""
import json

import httpx2
import pytest

from routellm.controller import ModelPair
from routellm.middleware.intent_model_selector import (
    IntentModelMapping,
    IntentModelSelector,
)
from routellm.middleware.jev_intent_detector import JevIntentDetector

CODING_MAPPING = IntentModelMapping(
    intent="coding",
    model_pair=ModelPair(strong="claude-3-opus", weak="mistral-medium"),
    description="Coding questions",
)


def _choice_response(choice: str, confidence: float, probabilities: dict) -> dict:
    """Build a wire-format /v1/systemone response body for one Choice answer."""
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 1, "output_tokens": 0},
        "answers": {
            "intent": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
    }


def _make_transport(body: dict, captured: dict):
    """Build a MockTransport that records the request body and returns `body`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=body)

    return httpx2.MockTransport(handler)


@pytest.fixture(autouse=True)
def typesafe_api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")


def test_detect_intent_returns_choice():
    captured = {}
    body = _choice_response(
        "coding", 0.91, {"coding": 0.91, "general": 0.09}
    )
    transport = _make_transport(body, captured)
    detector = JevIntentDetector([CODING_MAPPING], transport=transport)

    intent = detector.detect_intent("write a for loop")

    assert intent == "coding"


def test_confidence_floor_returns_general():
    captured = {}
    body = _choice_response(
        "coding", 0.3, {"coding": 0.3, "general": 0.7}
    )
    transport = _make_transport(body, captured)
    detector = JevIntentDetector([CODING_MAPPING], transport=transport)

    intent = detector.detect_intent("write a for loop")

    assert intent == "general"


def test_probabilities_map_matches_intents():
    captured = {}
    body = _choice_response(
        "coding", 0.91, {"coding": 0.91, "general": 0.09}
    )
    transport = _make_transport(body, captured)
    detector = JevIntentDetector([CODING_MAPPING], transport=transport)

    probabilities = detector.get_intent_confidence("write a for loop")

    assert set(probabilities.keys()) == {"coding", "general"}


def test_criteria_built_from_mappings():
    captured = {}
    body = _choice_response(
        "coding", 0.91, {"coding": 0.91, "general": 0.09}
    )
    transport = _make_transport(body, captured)
    detector = JevIntentDetector([CODING_MAPPING], transport=transport)

    detector.detect_intent("write a for loop")

    questions = captured["body"]["questions"]
    criteria = questions["intent"]["criteria"]
    assert criteria == {
        "coding": "Coding questions",
        "general": "none of the listed intents fit",
    }


def test_selector_integration(monkeypatch):
    from unittest.mock import patch

    captured = {}
    body = _choice_response(
        "coding", 0.91, {"coding": 0.91, "general": 0.09}
    )
    transport = _make_transport(body, captured)
    detector = JevIntentDetector([CODING_MAPPING], transport=transport)
    default_pair = ModelPair(strong="default-strong", weak="default-weak")
    selector = IntentModelSelector(
        intent_mappings=[CODING_MAPPING],
        default_model_pair=default_pair,
        intent_detector=detector,
    )

    with patch("litellm.completion") as mock_completion:
        model_pair = selector.get_model_pair("write a for loop")

    assert model_pair == CODING_MAPPING.model_pair
    mock_completion.assert_not_called()


