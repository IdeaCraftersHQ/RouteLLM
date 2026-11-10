"""The ML stack is optional: only the routers that run a model need it.

`routellm.routers.routers` used to import torch, transformers and
datasets at module top, so a server configured with only the `random`
router (or a threshold over config tiers) still needed several GB of
wheels. Each case runs in a subprocess with an import blocker standing
in for an environment where the `ml` extra is absent, so it holds
whether or not those packages happen to be installed here.
"""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCKER = """
import importlib.abc, sys

BLOCKED = ("torch", "transformers", "datasets", "huggingface_hub")


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)


sys.meta_path.insert(0, _Blocker())
"""


def _spawn(tmp_path, body):
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def test_the_blocker_really_blocks(tmp_path):
    result = _spawn(tmp_path, "import torch")
    assert result.returncode != 0
    assert "blocked" in result.stderr


@pytest.mark.slow
def test_server_routes_a_random_router_request_without_the_ml_stack(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  default: {strong: big, weak: small}\n"
    )
    body = f"""
import json, sys
from unittest.mock import MagicMock
sys.argv = ["x", "--config", {str(config)!r}, "--routers", "random"]
import routellm.openai_server as server
from fastapi.testclient import TestClient

seen = {{}}
res = MagicMock()
res._hidden_params = {{}}
res.model_dump.return_value = {{"id": "c1", "choices": []}}


async def _fake_acompletion(**kwargs):
    seen.update(kwargs)
    return res


import routellm.controller as controller
controller.acompletion = _fake_acompletion

with TestClient(server.app) as client:
    server.CONTROLLER.cache.config.enabled = False
    reply = client.post(
        "/v1/chat/completions",
        json={{"model": "default", "messages": [{{"role": "user", "content": "hi"}}]}},
    )
heavy = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
print(json.dumps({{"status": reply.status_code, "model": seen.get("model"), "heavy": heavy}}))
"""
    result = _spawn(tmp_path, body)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == 200
    assert payload["model"] in ("m_big", "m_small")
    assert payload["heavy"] == []


def test_router_classes_stay_importable_without_the_ml_stack(tmp_path):
    body = """
from routellm.routers.routers import (
    ROUTER_CLS, BERTRouter, CausalLLMRouter, MatrixFactorizationRouter,
    RandomRouter, SWRankingRouter,
)
print(sorted(ROUTER_CLS))
"""
    result = _spawn(tmp_path, body)
    assert result.returncode == 0, result.stderr
    for name in ("bert", "causal_llm", "mf", "random", "sw_ranking"):
        assert name in result.stdout


@pytest.mark.parametrize(
    "construct",
    [
        "BERTRouter(checkpoint_path='x')",
        "MatrixFactorizationRouter(checkpoint_path='x')",
        "CausalLLMRouter(checkpoint_path='x')",
        "SWRankingRouter(arena_battle_datasets=['x'], arena_embedding_datasets=['y'])",
    ],
)
def test_a_heavy_router_without_the_extra_names_it(tmp_path, construct):
    body = f"""
import routellm.routers.routers as r
try:
    r.{construct}
except ImportError as exc:
    print("IMPORT_ERROR", exc)
else:
    print("NO_ERROR")
"""
    result = _spawn(tmp_path, body)
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("IMPORT_ERROR"), line
    assert "routellm[ml]" in line


def test_the_ml_stack_is_an_extra_not_a_core_dependency():
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    core = {
        d.split("[")[0].split("<")[0].split(">")[0].split("=")[0].strip(" '\"").lower()
        for d in meta["project"]["dependencies"]
    }
    ml = {
        d.split("[")[0].split("<")[0].split(">")[0].split("=")[0].strip(" '\"").lower()
        for d in meta["project"]["optional-dependencies"]["ml"]
    }
    for heavy in ("torch", "transformers", "datasets"):
        assert heavy not in core
        assert heavy in ml
