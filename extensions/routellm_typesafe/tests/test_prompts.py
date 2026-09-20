"""Tests for Jev prompt-file wiring: JevRouter and JevIntentDetector.

The generic loader itself is covered by test_prompt_file.py. Every
request here goes through httpx2.MockTransport, never the network, and
every fixture sets TYPESAFE_API_KEY because TypeSafeClient checks for a
key before it sends anything.
"""
import json
import pathlib
import textwrap

import httpx2
import pytest

from routellm.controller import ModelPair
from routellm.middleware.intent_model_selector import IntentModelMapping
from routellm.prompts import PromptFile
from routellm_typesafe.intent_detector import (
    DEFAULT_INSTRUCTIONS as DETECTOR_DEFAULT_INSTRUCTIONS,
)
from routellm_typesafe.intent_detector import (
    GENERAL_DESCRIPTION,
    JevIntentDetector,
)
from routellm_typesafe.intent_detector import PROMPT_SCHEMA as DETECTOR_SCHEMA
from routellm_typesafe.router import (
    DEFAULT_CRITERIA,
    DEFAULT_INSTRUCTIONS,
    JevRouter,
)
from routellm_typesafe.router import PROMPT_SCHEMA as ROUTER_SCHEMA

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PROMPT_FILE = PACKAGE_ROOT / "prompts" / "jev.example.yaml"

FULL_FILE = textwrap.dedent(
    """\
    router:
      instructions: file router question
      criteria:
        "true": file yes case
        "false": file no case
    intent_detector:
      instructions: file detector instructions
      general_description: file general description
    """
)

ROUTER_RESPONSE_BODY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 1, "output_tokens": 0},
    "answers": {"strong": {"type": "noul", "noul": 0.83}},
}

DETECTOR_RESPONSE_BODY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 1, "output_tokens": 0},
    "answers": {
        "intent": {
            "type": "choice",
            "choice": "coding",
            "confidence": 0.91,
            "probabilities": {"coding": 0.91, "general": 0.09},
        }
    },
}

CODING_MAPPING = IntentModelMapping(
    intent="coding",
    model_pair=ModelPair(strong="claude-3-opus", weak="mistral-medium"),
    description="Coding questions",
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")


@pytest.fixture
def captured():
    """Mutable box the mock handler stashes the decoded request body into."""
    return {}


def _transport(body, captured):
    """Build a MockTransport that records the request body and returns `body`."""

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=body)

    return httpx2.MockTransport(handler)


def _write(tmp_path, text, name="prompts.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- router ---------------------------------------------------------------


def test_router_file_values_reach_request(tmp_path, captured):
    router = JevRouter(
        prompt_file=_write(tmp_path, FULL_FILE),
        transport=_transport(ROUTER_RESPONSE_BODY, captured),
    )
    try:
        router.calculate_strong_win_rate("hello world")
    finally:
        router.close()

    question = captured["body"]["questions"]["strong"]
    assert question["instructions"] == "file router question"
    assert question["criteria"] == {
        "true": "file yes case",
        "false": "file no case",
    }


def test_router_kwarg_beats_file(tmp_path, captured):
    router = JevRouter(
        instructions="kwarg question",
        criteria={"true": "kwarg yes", "false": "kwarg no"},
        prompt_file=_write(tmp_path, FULL_FILE),
        transport=_transport(ROUTER_RESPONSE_BODY, captured),
    )
    try:
        router.calculate_strong_win_rate("hello world")
    finally:
        router.close()

    question = captured["body"]["questions"]["strong"]
    assert question["instructions"] == "kwarg question"
    assert question["criteria"] == {"true": "kwarg yes", "false": "kwarg no"}


def test_router_without_file_uses_defaults(captured):
    router = JevRouter(transport=_transport(ROUTER_RESPONSE_BODY, captured))
    try:
        router.calculate_strong_win_rate("hello world")
    finally:
        router.close()

    question = captured["body"]["questions"]["strong"]
    assert question["instructions"] == DEFAULT_INSTRUCTIONS
    assert question["criteria"] == DEFAULT_CRITERIA


def test_router_file_criteria_only_keeps_default_instructions(tmp_path, captured):
    text = textwrap.dedent(
        """\
        router:
          criteria:
            "true": file yes case
            "false": file no case
        """
    )
    router = JevRouter(
        prompt_file=_write(tmp_path, text),
        transport=_transport(ROUTER_RESPONSE_BODY, captured),
    )
    try:
        router.calculate_strong_win_rate("hello world")
    finally:
        router.close()

    question = captured["body"]["questions"]["strong"]
    assert question["instructions"] == DEFAULT_INSTRUCTIONS
    assert question["criteria"]["true"] == "file yes case"


def test_router_rejects_unknown_criteria_key(tmp_path):
    text = 'router:\n  criteria:\n    "maybe": hmm\n'
    path = _write(tmp_path, text)

    with pytest.raises(ValueError) as excinfo:
        JevRouter(prompt_file=path, transport=httpx2.MockTransport(lambda r: None))

    assert "maybe" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_router_rejects_non_string_criteria_value(tmp_path):
    text = 'router:\n  criteria:\n    "true": 3\n'
    path = _write(tmp_path, text)

    with pytest.raises(ValueError) as excinfo:
        JevRouter(prompt_file=path, transport=httpx2.MockTransport(lambda r: None))

    assert "true" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_router_empty_prompt_file_path_raises(captured):
    """An empty path is a bad path, not "no prompt file"."""
    with pytest.raises(FileNotFoundError):
        JevRouter(
            prompt_file="",
            transport=_transport(ROUTER_RESPONSE_BODY, captured),
        )


# --- detector -------------------------------------------------------------


def test_detector_file_values_reach_request(tmp_path, captured):
    detector = JevIntentDetector(
        [CODING_MAPPING],
        prompt_file=_write(tmp_path, FULL_FILE),
        transport=_transport(DETECTOR_RESPONSE_BODY, captured),
    )
    try:
        detector.detect_intent("write a for loop")
    finally:
        detector.close()

    question = captured["body"]["questions"]["intent"]
    assert question["instructions"] == "file detector instructions"
    assert question["criteria"]["general"] == "file general description"


def test_detector_instructions_kwarg_beats_file(tmp_path, captured):
    detector = JevIntentDetector(
        [CODING_MAPPING],
        instructions="kwarg detector instructions",
        prompt_file=_write(tmp_path, FULL_FILE),
        transport=_transport(DETECTOR_RESPONSE_BODY, captured),
    )
    try:
        detector.detect_intent("write a for loop")
    finally:
        detector.close()

    question = captured["body"]["questions"]["intent"]
    assert question["instructions"] == "kwarg detector instructions"


def test_detector_descriptions_general_beats_file(tmp_path, captured):
    detector = JevIntentDetector(
        [CODING_MAPPING],
        descriptions={"general": "kwarg general description"},
        prompt_file=_write(tmp_path, FULL_FILE),
        transport=_transport(DETECTOR_RESPONSE_BODY, captured),
    )
    try:
        detector.detect_intent("write a for loop")
    finally:
        detector.close()

    question = captured["body"]["questions"]["intent"]
    assert question["criteria"]["general"] == "kwarg general description"


def test_detector_without_file_uses_defaults(captured):
    detector = JevIntentDetector(
        [CODING_MAPPING],
        transport=_transport(DETECTOR_RESPONSE_BODY, captured),
    )
    try:
        detector.detect_intent("write a for loop")
    finally:
        detector.close()

    question = captured["body"]["questions"]["intent"]
    assert "Classify the user prompt" in question["instructions"]
    assert question["criteria"]["general"] == GENERAL_DESCRIPTION


def test_detector_empty_prompt_file_path_raises(captured):
    """An empty path is a bad path, not "no prompt file"."""
    with pytest.raises(FileNotFoundError):
        JevIntentDetector(
            [CODING_MAPPING],
            prompt_file="",
            transport=_transport(DETECTOR_RESPONSE_BODY, captured),
        )


# --- example file drift ----------------------------------------------------


def test_example_prompt_file_matches_built_in_defaults():
    """prompts/jev.example.yaml must never drift from the code defaults.

    Resolved relative to this test file, not the working directory, so
    it passes regardless of where pytest is invoked from.
    """
    prompt_file = PromptFile.load(EXAMPLE_PROMPT_FILE)

    router_section = prompt_file.section("router", ROUTER_SCHEMA)
    assert router_section["instructions"] == DEFAULT_INSTRUCTIONS
    assert router_section["criteria"] == DEFAULT_CRITERIA

    detector_section = prompt_file.section("intent_detector", DETECTOR_SCHEMA)
    assert detector_section["instructions"] == DETECTOR_DEFAULT_INSTRUCTIONS
    assert detector_section["general_description"] == GENERAL_DESCRIPTION
