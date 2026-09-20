"""Tests for the server's model listing and its implicit default tier.

`routellm.openai_server` parses `sys.argv` at import, so one process
cannot import it twice under different flags. Each case therefore runs
its own subprocess, setting `sys.argv` before the import and driving the
app through `TestClient` exactly as `test_tiers.py` does, with litellm's
`acompletion` patched so nothing leaves the machine.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from routellm.endpoints import EndpointRegistry

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


def _run(tmp_path, argv, body):
    """Run `body` in a subprocess against a server built from `argv`."""
    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE.format(argv=argv) + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def tiered_config(tmp_path):
    """A config carrying two endpoints and its own `default` tier."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  premium: {strong: big, weak: small}\n"
        "  default: {strong: premium, weak: small}\n"
    )
    return config


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_models_lists_tiers_and_routers(tmp_path, tiered_config):
    """Every tier name and one id per router, at the default threshold."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(tiered_config), "--routers", "random",
         "--default-threshold", "0.5"],
        """
with TestClient(server.app) as client:
    reply = client.get("/v1/models")
print(json.dumps({"status": reply.status_code, "body": reply.json()}))
""",
    )

    assert payload["status"] == 200
    body = payload["body"]
    assert body["object"] == "list"
    assert [item["id"] for item in body["data"]] == [
        "default",
        "premium",
        "router-random-0.5",
    ]
    for item in body["data"]:
        assert item["object"] == "model"
        assert item["owned_by"] == "routellm"
        assert isinstance(item["created"], int)


@pytest.mark.slow
def test_models_ids_are_routable(tmp_path, tiered_config):
    """Each listed id is accepted as the model of a completion."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(tiered_config), "--routers", "random",
         "--default-threshold", "0.5"],
        """
with TestClient(server.app) as client:
    ids = [item["id"] for item in client.get("/v1/models").json()["data"]]
    codes = {
        model: client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        for model in ids
    }
print(json.dumps({"codes": codes}))
""",
    )

    assert payload["codes"] == {
        "default": 200,
        "premium": 200,
        "router-random-0.5": 200,
    }


# ---------------------------------------------------------------------------
# The implicit default tier
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_flags_build_an_implicit_default_tier(tmp_path):
    """No config `default` tier: the two flags become one."""
    config = tmp_path / "flat.yaml"
    config.write_text("endpoints:\n  big: {model: m_big}\n")

    payload = _run(
        tmp_path,
        ["x", "--config", str(config), "--routers", "random",
         "--strong-model", "gpt-4o", "--weak-model", "ollama_chat/qwen3:8b",
         "--default-threshold", "0.25"],
        """
with TestClient(server.app) as client:
    ids = [item["id"] for item in client.get("/v1/models").json()["data"]]
    reply = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
print(json.dumps({"ids": ids, "path": reply.json()["routellm"]["path"]}))
""",
    )

    assert payload["ids"] == ["default", "router-random-0.25"]
    entry = payload["path"][0]
    assert entry["tier"] == "default"
    # The tier the flags built carries them as its own values, so both
    # are sourced from the tier rather than inherited.
    assert (entry["router"], entry["router_from"]) == ("random", "tier")
    assert (entry["threshold"], entry["threshold_from"]) == (0.25, "tier")
    # `random` scores each prompt afresh, so only the two sides of the
    # implicit tier are reachable.
    assert entry["picked"] in ("implicit_strong", "implicit_weak")


@pytest.mark.slow
def test_config_default_tier_wins_over_the_flags(tmp_path, tiered_config):
    """A config `default` tier is used even when both flags are given."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(tiered_config), "--routers", "random",
         "--strong-model", "gpt-4o", "--weak-model", "ollama_chat/qwen3:8b",
         "--default-threshold", "0.25"],
        """
with TestClient(server.app) as client:
    ids = [item["id"] for item in client.get("/v1/models").json()["data"]]
    reply = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
print(json.dumps({"ids": ids, "path": reply.json()["routellm"]["path"]}))
""",
    )

    assert "implicit_strong" not in payload["ids"]
    assert payload["ids"] == ["default", "premium", "router-random-0.25"]
    # The config tier names no router or threshold, so both are inherited
    # from the controller defaults rather than written onto the tier.
    root = payload["path"][0]
    assert (root["router_from"], root["threshold_from"]) == ("default", "default")
    assert root["picked"] in ("premium", "small")


@pytest.mark.slow
def test_implicit_tier_reuses_a_configured_endpoint(tmp_path):
    """A flag naming a configured endpoint keeps that endpoint."""
    config = tmp_path / "named.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small, api_base: 'http://127.0.0.1:11500'}\n"
    )

    payload = _run(
        tmp_path,
        ["x", "--config", str(config), "--routers", "random",
         "--strong-model", "big", "--weak-model", "small"],
        """
with TestClient(server.app) as client:
    reply = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
    tier = server.CONTROLLER.endpoints.get_tier("default")
    sides = [tier.strong, tier.weak]
    names = server.CONTROLLER.endpoints.names()
print(json.dumps({"sides": sides, "names": names,
                  "path": reply.json()["routellm"]["path"]}))
""",
    )

    # Neither side was wrapped, so the endpoints keep their own api_base.
    assert payload["sides"] == ["big", "small"]
    assert payload["names"] == ["big", "small"]
    assert payload["path"][0]["picked"] in ("big", "small")


# ---------------------------------------------------------------------------
# config.example.yaml
# ---------------------------------------------------------------------------


def test_config_example_loads_into_the_registry():
    """Every documented `endpoints:`/`tiers:` key survives a real load."""
    with open(REPO_ROOT / "config.example.yaml") as handle:
        config = yaml.safe_load(handle)

    registry = EndpointRegistry.from_config(config)

    assert registry.names() == [
        "cloud_strong",
        "colibri_glm",
        "embedding",
        "ollama_qwen",
    ]
    assert registry.tier_names() == ["default", "premium"]

    # Construction validates the tier graph; re-running it states that
    # the example's references, namespace, and depth all hold.
    registry.revalidate()

    default = registry.get_tier("default")
    assert (default.strong, default.weak) == ("premium", "ollama_qwen")
    assert (default.router, default.threshold) == ("mf", 0.12)

    colibri = registry.get("colibri_glm")
    assert colibri.api_base == "http://127.0.0.1:11800/v1"
    assert colibri.api_key_env == "COLI_API_KEY"
    assert colibri.tags == ["local", "tools"]
    assert colibri.quality == 85
