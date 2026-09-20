"""Tests for the YAML prompt file: loader, router wiring, detector wiring.

Every request goes through httpx2.MockTransport, never the network, and
every fixture sets TYPESAFE_API_KEY because TypeSafeClient checks for a
key before it sends anything.
"""
import json
import textwrap

import httpx2
import pytest

from routellm.controller import ModelPair
from routellm.middleware.intent_model_selector import IntentModelMapping
from routellm.middleware.jev_intent_detector import (
    GENERAL_DESCRIPTION,
    JevIntentDetector,
)
from routellm.routers.typesafe.prompts import load_prompt_file
from routellm.routers.typesafe.router import (
    DEFAULT_CRITERIA,
    DEFAULT_INSTRUCTIONS,
    JevRouter,
)

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


# --- loader ---------------------------------------------------------------


def test_full_file_round_trips(tmp_path):
    router, detector = load_prompt_file(_write(tmp_path, FULL_FILE))

    assert router.instructions == "file router question"
    assert router.criteria == {"true": "file yes case", "false": "file no case"}
    assert detector.instructions == "file detector instructions"
    assert detector.general_description == "file general description"


def test_partial_file_leaves_absent_keys_none(tmp_path):
    text = textwrap.dedent(
        """\
        router:
          instructions: only the question
        """
    )

    router, detector = load_prompt_file(_write(tmp_path, text))

    assert router.instructions == "only the question"
    assert router.criteria is None
    assert detector.instructions is None
    assert detector.general_description is None


def test_empty_file_yields_all_none(tmp_path):
    router, detector = load_prompt_file(_write(tmp_path, ""))

    assert router.instructions is None
    assert router.criteria is None
    assert detector.instructions is None
    assert detector.general_description is None


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompt_file(tmp_path / "absent.yaml")


def test_non_mapping_document_raises(tmp_path):
    path = _write(tmp_path, "- just\n- a list\n")

    with pytest.raises(ValueError, match=str(path)):
        load_prompt_file(path)


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(tmp_path, "rooter:\n  instructions: typo\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "rooter" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_unknown_section_key_raises(tmp_path):
    path = _write(tmp_path, "router:\n  criterion: typo\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "criterion" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_unknown_criteria_key_raises(tmp_path):
    path = _write(tmp_path, 'router:\n  criteria:\n    "maybe": hmm\n')

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "maybe" in str(excinfo.value)


def test_non_mapping_criteria_raises(tmp_path):
    path = _write(tmp_path, "router:\n  criteria: not a mapping\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "criteria" in str(excinfo.value)


def test_non_string_criteria_value_raises(tmp_path):
    path = _write(tmp_path, 'router:\n  criteria:\n    "true": 3\n')

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "true" in str(excinfo.value)


def test_non_string_instructions_raises(tmp_path):
    path = _write(tmp_path, "router:\n  instructions: 7\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "instructions" in str(excinfo.value)


def test_non_mapping_section_raises(tmp_path):
    path = _write(tmp_path, "intent_detector: nope\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "intent_detector" in str(excinfo.value)


def test_non_string_general_description_raises(tmp_path):
    path = _write(tmp_path, "intent_detector:\n  general_description: 5\n")

    with pytest.raises(ValueError) as excinfo:
        load_prompt_file(path)

    assert "general_description" in str(excinfo.value)


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
