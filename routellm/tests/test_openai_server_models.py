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


def _spawn(tmp_path, argv, body, env=None, cwd=None):
    """Run `body` in a subprocess against a server built from `argv`.

    Parameters
    ----------
    env : dict, optional
        Extra environment entries, merged over the isolated base. Used by
        the discovery cases to point `XDG_CONFIG_HOME` at `tmp_path`.
    cwd : path-like, optional
        Working directory. Defaults to the repo root; a discovery case
        moves it under `tmp_path` so the walk-up finds no marker.
    """
    base = {
        "PYTHONPATH": str(REPO_ROOT),
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    base.update(env or {})
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE.format(argv=argv) + body],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=base,
    )


def _run(tmp_path, argv, body, env=None, cwd=None):
    """Run `body` and return the JSON its last line printed."""
    result = _spawn(tmp_path, argv, body, env=env, cwd=cwd)
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


@pytest.fixture
def no_default_config(tmp_path):
    """A config carrying a tier, but not one named `default`."""
    config = tmp_path / "no-default.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  premium: {strong: big, weak: small}\n"
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
        ["x", "--config", str(tiered_config), "--routers", "random", "--default-threshold", "0.5"],
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
        ["x", "--config", str(tiered_config), "--routers", "random", "--default-threshold", "0.5"],
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
        [
            "x",
            "--config",
            str(config),
            "--routers",
            "random",
            "--strong-model",
            "gpt-4o",
            "--weak-model",
            "ollama_chat/qwen3:8b",
            "--default-threshold",
            "0.25",
        ],
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
        [
            "x",
            "--config",
            str(tiered_config),
            "--routers",
            "random",
            "--strong-model",
            "gpt-4o",
            "--weak-model",
            "ollama_chat/qwen3:8b",
            "--default-threshold",
            "0.25",
        ],
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
        [
            "x",
            "--config",
            str(config),
            "--routers",
            "random",
            "--strong-model",
            "big",
            "--weak-model",
            "small",
        ],
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


@pytest.mark.slow
def test_no_flags_invents_no_default_tier(tmp_path, no_default_config):
    """Neither flag and no config `default`: nothing is derived.

    The tier is not invented, so it is not advertised either, and no
    `implicit_*` endpoint is registered. The legacy flat form still
    routes, against the historic pair, and that assumption is announced
    at WARNING so it is visible without `--verbose`.
    """
    result = _spawn(
        tmp_path,
        ["x", "--config", str(no_default_config), "--routers", "random"],
        """
with TestClient(server.app) as client:
    ids = [item["id"] for item in client.get("/v1/models").json()["data"]]
    names = server.CONTROLLER.endpoints.names()
    pair = server.CONTROLLER.model_pair
    reply = client.post(
        "/v1/chat/completions",
        json={"model": "router-random-0.5",
              "messages": [{"role": "user", "content": "hi"}]},
    )
print(json.dumps({"ids": ids, "names": names, "status": reply.status_code,
                  "pair": [pair.strong, pair.weak]}))
""",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert "default" not in payload["ids"]
    assert payload["ids"] == ["premium", "router-random-0.5"]
    assert payload["names"] == ["big", "small"]
    assert not [name for name in payload["names"] if name.startswith("implicit")]

    # The legacy flat form keeps answering, against the historic pair.
    assert payload["status"] == 200
    assert payload["pair"] == [
        "gpt-4-1106-preview",
        "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]

    assert "gpt-4-1106-preview" in result.stderr
    assert "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1" in result.stderr
    assert "define a 'default' tier" in result.stderr


@pytest.mark.slow
def test_one_model_flag_alone_is_an_argparse_error(tmp_path, no_default_config):
    """One side of a pair is never enough; argparse rejects it."""
    for flag, value in (
        ("--strong-model", "gpt-4o"),
        ("--weak-model", "ollama_chat/qwen3:8b"),
    ):
        result = _spawn(
            tmp_path,
            ["x", "--config", str(no_default_config), "--routers", "random", flag, value],
            "print('{}')\n",
        )

        assert result.returncode != 0
        assert "must be given together" in result.stderr


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


def test_config_example_builds_a_controller_on_a_stock_install(monkeypatch):
    """Nothing in the example needs an optional extension to construct.

    Every router the example's tiers name must be registered by the base
    package, so a stock install can serve the shipped config rather than
    failing in the lifespan.
    """
    import routellm.controller
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    class _Stub:
        def calculate_strong_win_rate(self, prompt):
            return 0.5

    monkeypatch.setitem(routellm.controller.ROUTER_CLS, "mf", lambda **kw: _Stub())

    with open(REPO_ROOT / "config.example.yaml") as handle:
        config = yaml.safe_load(handle)

    registry = EndpointRegistry.from_config(config)

    controller = Controller(
        routers=["mf"],
        strong_model=None,
        weak_model=None,
        config={},
        endpoints=registry,
        cache_config=CacheConfig(enabled=False),
    )

    assert set(controller.routers) == {"mf"}


def test_unknown_jev_router_names_the_typesafe_extra():
    """A tier on `jev` without the extension says how to install it."""
    from routellm.caching import CacheConfig
    from routellm.controller import Controller
    from routellm.hints import TYPESAFE_INSTALL_HINT

    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                "a": {"model": "gpt-4o"},
                "b": {"model": "ollama_chat/qwen3:8b"},
            },
            "tiers": {
                "premium": {
                    "router": "jev",
                    "threshold": 0.33,
                    "strong": "a",
                    "weak": "b",
                },
                "default": {
                    "router": "random",
                    "threshold": 0.12,
                    "strong": "premium",
                    "weak": "b",
                },
            },
        }
    )

    with pytest.raises(ValueError) as excinfo:
        Controller(
            routers=["random"],
            strong_model=None,
            weak_model=None,
            config={},
            endpoints=registry,
            cache_config=CacheConfig(enabled=False),
        )

    message = str(excinfo.value)
    assert "unknown router 'jev'" in message
    assert TYPESAFE_INSTALL_HINT in message


def test_unknown_other_router_carries_no_install_hint():
    """The hint is specific to `jev`; other names get the plain error."""
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"a": {"model": "gpt-4o"}, "b": {"model": "m"}},
            "tiers": {
                "premium": {"router": "nope", "strong": "a", "weak": "b"},
                "default": {
                    "router": "random",
                    "threshold": 0.12,
                    "strong": "premium",
                    "weak": "b",
                },
            },
        }
    )

    with pytest.raises(ValueError) as excinfo:
        Controller(
            routers=["random"],
            strong_model=None,
            weak_model=None,
            config={},
            endpoints=registry,
            cache_config=CacheConfig(enabled=False),
        )

    assert "pip install" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_server_discovers_the_user_config_without_a_flag(tmp_path):
    """No `--config`: the user layer under XDG still supplies the tiers."""
    xdg = tmp_path / "xdgconf"
    user = xdg / "routellm" / "config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text(
        "endpoints:\n"
        "  discovered_big: {model: m_big}\n"
        "  discovered_small: {model: m_small}\n"
        "tiers:\n"
        "  discovered: {strong: discovered_big, weak: discovered_small}\n"
        "  default: {strong: discovered, weak: discovered_small}\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    payload = _run(
        tmp_path,
        ["x", "--routers", "random", "--default-threshold", "0.5"],
        """
with TestClient(server.app) as client:
    reply = client.get("/v1/models")
print(json.dumps({"ids": [item["id"] for item in reply.json()["data"]]}))
""",
        env={"XDG_CONFIG_HOME": str(xdg)},
        cwd=elsewhere,
    )

    assert payload["ids"] == ["default", "discovered", "router-random-0.5"]


@pytest.mark.slow
def test_flag_still_overrides_discovery(tmp_path, tiered_config):
    """`--config` merges last, so its tiers join the discovered ones."""
    xdg = tmp_path / "xdgconf"
    user = xdg / "routellm" / "config.yaml"
    user.parent.mkdir(parents=True)
    user.write_text(
        "endpoints:\n"
        "  discovered_small: {model: m_small}\n"
        "tiers:\n"
        "  discovered: {strong: discovered_small, weak: discovered_small}\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    payload = _run(
        tmp_path,
        ["x", "--config", str(tiered_config), "--routers", "random", "--default-threshold", "0.5"],
        """
with TestClient(server.app) as client:
    reply = client.get("/v1/models")
print(json.dumps({"ids": [item["id"] for item in reply.json()["data"]]}))
""",
        env={"XDG_CONFIG_HOME": str(xdg)},
        cwd=elsewhere,
    )

    assert payload["ids"] == [
        "default",
        "discovered",
        "premium",
        "router-random-0.5",
    ]
