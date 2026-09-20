"""Tests for the generic prompt file loader.

These exercise routellm.prompts on its own: no adapter, no SDK, no
router. Adapter wiring lives in test_jev_prompts.py.
"""
import pathlib
import subprocess
import sys
import textwrap

import pytest

from routellm.prompts import PromptFile, resolve

ROUTER_SCHEMA = {"instructions": str, "criteria": dict}
DETECTOR_SCHEMA = {"instructions": str, "general_description": str}

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


def _write(tmp_path, text, name="prompts.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- resolve --------------------------------------------------------------


def test_resolve_prefers_kwarg():
    assert resolve("kwarg", "file", "default") == ("kwarg", "kwarg")


def test_resolve_falls_back_to_file():
    assert resolve(None, "file", "default") == ("file", "file")


def test_resolve_falls_back_to_default():
    assert resolve(None, None, "default") == ("default", "default")


# --- load -----------------------------------------------------------------


def test_load_full_file(tmp_path):
    prompt_file = PromptFile.load(_write(tmp_path, FULL_FILE))

    assert prompt_file.section("router", ROUTER_SCHEMA) == {
        "instructions": "file router question",
        "criteria": {"true": "file yes case", "false": "file no case"},
    }
    assert prompt_file.section("intent_detector", DETECTOR_SCHEMA) == {
        "instructions": "file detector instructions",
        "general_description": "file general description",
    }


def test_empty_file_has_no_sections(tmp_path):
    prompt_file = PromptFile.load(_write(tmp_path, ""))

    assert prompt_file.sections == {}
    assert prompt_file.section("router", ROUTER_SCHEMA) == {}


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        PromptFile.load(tmp_path / "absent.yaml")


def test_non_mapping_document_raises(tmp_path):
    path = _write(tmp_path, "- just\n- a list\n")

    with pytest.raises(ValueError) as excinfo:
        PromptFile.load(path)

    assert str(path) in str(excinfo.value)


# --- section --------------------------------------------------------------


def test_absent_section_returns_empty(tmp_path):
    prompt_file = PromptFile.load(_write(tmp_path, "router:\n  instructions: hi\n"))

    assert prompt_file.section("intent_detector", DETECTOR_SCHEMA) == {}


def test_unrequested_section_ignored(tmp_path):
    text = textwrap.dedent(
        """\
        router:
          instructions: file router question
        intent_selector:
          anything: at all
        """
    )
    prompt_file = PromptFile.load(_write(tmp_path, text))

    assert prompt_file.section("router", ROUTER_SCHEMA) == {
        "instructions": "file router question"
    }


def test_partial_section_returns_only_present_keys(tmp_path):
    prompt_file = PromptFile.load(_write(tmp_path, "router:\n  instructions: hi\n"))

    assert prompt_file.section("router", ROUTER_SCHEMA) == {"instructions": "hi"}


def test_section_wrong_type_raises(tmp_path):
    path = _write(tmp_path, "router: not a mapping\n")
    prompt_file = PromptFile.load(path)

    with pytest.raises(ValueError) as excinfo:
        prompt_file.section("router", ROUTER_SCHEMA)

    assert "router" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_unknown_section_key_raises(tmp_path):
    path = _write(tmp_path, "router:\n  criterion: typo\n")
    prompt_file = PromptFile.load(path)

    with pytest.raises(ValueError) as excinfo:
        prompt_file.section("router", ROUTER_SCHEMA)

    assert "criterion" in str(excinfo.value)
    assert "router" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_wrong_value_type_raises(tmp_path):
    path = _write(tmp_path, "router:\n  instructions: 7\n")
    prompt_file = PromptFile.load(path)

    with pytest.raises(ValueError) as excinfo:
        prompt_file.section("router", ROUTER_SCHEMA)

    assert "instructions" in str(excinfo.value)
    assert "str" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_wrong_dict_value_type_raises(tmp_path):
    path = _write(tmp_path, "router:\n  criteria: not a mapping\n")
    prompt_file = PromptFile.load(path)

    with pytest.raises(ValueError) as excinfo:
        prompt_file.section("router", ROUTER_SCHEMA)

    assert "criteria" in str(excinfo.value)
    assert "dict" in str(excinfo.value)


def test_schema_is_per_call(tmp_path):
    """The same file serves different adapters with different schemas."""
    prompt_file = PromptFile.load(_write(tmp_path, FULL_FILE))

    with pytest.raises(ValueError, match="criteria"):
        prompt_file.section("router", DETECTOR_SCHEMA)


def test_import_pulls_in_no_sdk_or_torch():
    """Importing the loader must not drag in typesafe_sdk or torch.

    Checked in a subprocess: this process has already imported both via
    other test modules, so sys.modules here proves nothing.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import routellm.prompts, sys; "
            "print('typesafe_sdk' in sys.modules, 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=pathlib.Path(__file__).resolve().parents[2],
    )

    assert result.stdout.strip() == "False False"
