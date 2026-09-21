"""Tests for the capability picture readable from outside the process.

Covers the additive `routellm` object on every `/v1/models` entry, and
the `--capabilities` flag on the existing explain command: the matrix,
its `?` for unknown, the `via` column naming the leaf behind a tier's
yes, the line flagging selector terms some endpoint cannot answer, and
that `--explain` alone is byte-identical to what it printed before.

Server cases follow test_openai_server_models.py: `sys.argv` is set
before importing `routellm.openai_server`, which parses argv at import,
and the app is driven under `TestClient` so the lifespan runs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import routellm.pairing as pairing
from routellm.endpoints import EndpointRegistry
from routellm.pairing import ModelRecord, _capability_matrix, main

REPO_ROOT = Path(__file__).resolve().parents[2]

_PREAMBLE = """
import json, sys
from unittest.mock import AsyncMock, MagicMock
sys.argv = {argv!r}
import routellm.openai_server as server
from fastapi.testclient import TestClient

res = MagicMock()
res._hidden_params = {{}}
res.model_dump.return_value = {{"id": "c1", "choices": []}}
server.acompletion = AsyncMock(return_value=res)
import routellm.controller as controller
controller.acompletion = AsyncMock(return_value=res)
"""

_LIST_MODELS = """
with TestClient(server.app) as client:
    print(json.dumps(client.get("/v1/models").json()))
"""

MATRIX_CONFIG = {
    "endpoints": {
        "seeing": {
            "model": "ollama_chat/llava:13b",
            "capabilities": {
                "vision": True,
                "tools": True,
                "structured_output": True,
                "reasoning": False,
                "context": 128000,
                "max_output": 8192,
                "modalities_in": ["text", "image"],
            },
        },
        "mystery": {"model": "ollama_chat/qwen3:8b"},
    },
    "tiers": {
        "default": {
            "router": "random",
            "threshold": 0.5,
            "strong": "seeing",
            "weak": "mystery",
        }
    },
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """Keep every test off the network and out of the real cache."""
    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])


@pytest.fixture
def matrix_config(tmp_path):
    """The matrix config, written to disk for the CLI."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(MATRIX_CONFIG))
    return path


def _run_server(tmp_path, argv, body):
    """Run `body` against a server built from `argv` and parse its JSON."""
    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE.format(argv=argv) + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ROUTELLM_CATALOG_CACHE": str(tmp_path / "catalog.json"),
        },
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def server_config(tmp_path):
    """A config with one tier over a seeing and an unknown endpoint."""
    path = tmp_path / "server.yaml"
    path.write_text(
        "endpoints:\n"
        "  seeing:\n"
        "    model: ollama_chat/llava:13b\n"
        "    capabilities: {vision: true, tools: true, context: 128000}\n"
        "  mystery:\n"
        "    model: ollama_chat/qwen3:8b\n"
        "tiers:\n"
        "  default:\n"
        "    router: random\n"
        "    threshold: 0.5\n"
        "    strong: seeing\n"
        "    weak: mystery\n"
    )
    return path


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_models_lists_tier_capabilities(tmp_path, server_config):
    argv = ["openai_server", "--config", str(server_config), "--routers", "random"]

    payload = _run_server(tmp_path, argv, _LIST_MODELS)
    entry = next(e for e in payload["data"] if e["id"] == "default")

    assert entry["routellm"]["kind"] == "tier"
    assert entry["routellm"]["capabilities"]["vision"] is True
    assert entry["routellm"]["capabilities"]["context"] == 128000


@pytest.mark.slow
def test_models_entry_keeps_the_openai_keys_untouched(tmp_path, server_config):
    argv = ["openai_server", "--config", str(server_config), "--routers", "random"]

    payload = _run_server(tmp_path, argv, _LIST_MODELS)
    entry = next(e for e in payload["data"] if e["id"] == "default")

    assert entry["object"] == "model"
    assert entry["owned_by"] == "routellm"
    assert isinstance(entry["created"], int)
    assert list(entry)[:4] == ["id", "object", "created", "owned_by"]


@pytest.mark.slow
def test_models_marks_unknown_capabilities(tmp_path, server_config):
    argv = ["openai_server", "--config", str(server_config), "--routers", "random"]

    payload = _run_server(tmp_path, argv, _LIST_MODELS)
    entry = next(e for e in payload["data"] if e["id"] == "default")

    # `mystery` knows nothing, and the union is optimistic, so the
    # keys neither leaf answers are the unknown ones.
    assert "structured_output" in entry["routellm"]["unknown"]
    assert "vision" not in entry["routellm"]["unknown"]


@pytest.mark.slow
def test_router_entry_has_no_tier_capabilities(tmp_path, server_config):
    argv = ["openai_server", "--config", str(server_config), "--routers", "random"]

    payload = _run_server(tmp_path, argv, _LIST_MODELS)
    entry = next(e for e in payload["data"] if e["id"].startswith("router-"))

    assert entry["routellm"]["kind"] == "router"
    # This controller carries a tier, not a flat pair, so there is no
    # pair to report capabilities for.
    assert "capabilities" not in entry["routellm"]


# ---------------------------------------------------------------------------
# --capabilities
# ---------------------------------------------------------------------------


def test_capabilities_flag_prints_the_matrix(matrix_config, capsys):
    assert main(["--config", str(matrix_config), "--capabilities"]) == 0

    out = capsys.readouterr().out

    assert "Capabilities" in out
    assert "seeing" in out
    assert "default (tier)" in out
    assert "Unknown:" in out


def test_capabilities_flag_marks_unknown_with_a_question_mark(matrix_config, capsys):
    main(["--config", str(matrix_config), "--capabilities"])

    out = capsys.readouterr().out
    mystery = next(line for line in out.splitlines() if line.startswith("mystery"))

    assert "?" in mystery
    assert "no" not in mystery.split()


def test_capabilities_and_explain_print_both(matrix_config, capsys):
    assert main(["--config", str(matrix_config), "--capabilities", "--explain"]) == 0

    out = capsys.readouterr().out

    assert "default.strong: seeing (named)" in out
    assert "Capabilities" in out
    assert out.index("default.strong") < out.index("Capabilities")


def test_matrix_names_the_leaf_behind_a_tier_capability(matrix_config, capsys):
    main(["--config", str(matrix_config), "--capabilities"])

    out = capsys.readouterr().out
    tier_row = next(line for line in out.splitlines() if line.startswith("default (tier)"))

    assert "vision=seeing" in tier_row


def test_matrix_flags_terms_used_by_selectors_with_unknowns(tmp_path, capsys):
    import yaml

    config = dict(MATRIX_CONFIG)
    config["tiers"] = {
        "default": {
            "router": "random",
            "threshold": 0.5,
            "strong": {"select": "vision:true", "order": "quality_desc"},
            "weak": "mystery",
        }
    }
    path = tmp_path / "selector.yaml"
    path.write_text(yaml.safe_dump(config))

    main(["--config", str(path), "--capabilities"])

    out = capsys.readouterr().out

    assert "dropped silently" in out
    assert "vision" in out.rsplit("dropped silently", 1)[-1]


def test_explain_without_the_flag_is_byte_identical_to_today(matrix_config, capsys):
    assert main(["--config", str(matrix_config)]) == 0
    default_out = capsys.readouterr().out

    assert main(["--config", str(matrix_config), "--explain"]) == 0
    explicit_out = capsys.readouterr().out

    assert default_out == explicit_out
    assert "Capabilities" not in default_out


def test_matrix_renders_a_known_false_as_no(tmp_path):
    registry = EndpointRegistry.from_config(MATRIX_CONFIG)

    rendered = _capability_matrix(registry, [])
    seeing = next(line for line in rendered.splitlines() if line.startswith("seeing"))

    # `reasoning: false` is knowledge, not absence: it must not read `?`.
    assert " no " in f" {seeing} "
    assert "yes" in seeing


def test_matrix_reports_no_unknowns_when_every_block_is_complete():
    config = {
        "endpoints": {
            "full": {
                "model": "ollama_chat/a:1b",
                "capabilities": {
                    "vision": True,
                    "tools": True,
                    "structured_output": True,
                    "reasoning": True,
                    "open_weights": True,
                    "context": 1000,
                    "max_output": 100,
                    "modalities_in": ["text"],
                },
            }
        }
    }
    registry = EndpointRegistry.from_config(config)

    rendered = _capability_matrix(registry, [])

    assert "(none)" in rendered


def test_record_without_modalities_leaves_vision_unknown():
    record = ModelRecord(provider="openai", id="gpt-4o", tool_call=True)
    config = {"endpoints": {"cloud": {"model": "gpt-4o"}}}
    registry = EndpointRegistry.from_config(config)

    from routellm.capabilities import capabilities_for

    caps = capabilities_for(registry.get("cloud"), record)

    assert caps.vision is None
    assert caps.tools is True
