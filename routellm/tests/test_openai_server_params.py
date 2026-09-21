"""Tests for what the server forwards to litellm, and for `--host`.

`routellm.openai_server` parses `sys.argv` at import, so one process
cannot import it twice under different flags. Each case therefore runs
its own subprocess, setting `sys.argv` before the import and driving the
app through `TestClient`, with litellm's `acompletion` patched so
nothing leaves the machine and the call's kwargs come back as JSON.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_PREAMBLE = """
import json, sys
from unittest.mock import MagicMock
sys.argv = {argv!r}
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
"""


def _spawn(tmp_path, argv, body):
    """Run `body` in a subprocess against a server built from `argv`."""
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE.format(argv=argv) + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def _run(tmp_path, argv, body):
    """Run `body` and return the JSON its last line printed."""
    result = _spawn(tmp_path, argv, body)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def flat_config(tmp_path):
    """A config carrying one endpoint and a `default` tier over it."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "endpoints:\n"
        "  big: {model: m_big}\n"
        "  small: {model: m_small}\n"
        "tiers:\n"
        "  default: {strong: big, weak: small}\n"
    )
    return config


_POST = """
with TestClient(server.app) as client:
    # The completion cache is a repo-relative SQLite file shared by every
    # run, and a hit never reaches litellm at all.
    server.CONTROLLER.cache.config.enabled = False
    client.post("/v1/chat/completions", json={body})
print(json.dumps({{"seen": sorted(seen), "values": {{
    k: seen[k] for k in seen if k in (
        "presence_penalty", "frequency_penalty", "temperature", "top_p",
        "n", "stream", "model", "max_tokens",
    )
}}}}))
"""


# ---------------------------------------------------------------------------
# Only client-set sampling parameters reach litellm
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_unset_sampling_params_are_not_forwarded(tmp_path, flat_config):
    """A body naming no sampling field forwards none of them.

    Pydantic defaults would otherwise manufacture `presence_penalty=0.0`
    and friends, which several providers reject outright.
    """
    payload = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(body='{"model": "default", "messages": [{"role": "user", "content": "hi"}]}'),
    )

    for field in (
        "presence_penalty",
        "frequency_penalty",
        "temperature",
        "top_p",
        "n",
        "max_tokens",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "seed",
        "stop",
        "tools",
        "tool_choice",
        "user",
    ):
        assert field not in payload["seen"], field


@pytest.mark.slow
def test_client_set_sampling_params_are_forwarded_verbatim(tmp_path, flat_config):
    """A body naming a field forwards exactly that value."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(
            body=(
                '{"model": "default", '
                '"messages": [{"role": "user", "content": "hi"}], '
                '"presence_penalty": 0.5, "temperature": 0.2}'
            )
        ),
    )

    assert payload["values"]["presence_penalty"] == 0.5
    assert payload["values"]["temperature"] == 0.2
    assert "frequency_penalty" not in payload["seen"]


@pytest.mark.slow
def test_explicit_zero_penalty_is_forwarded(tmp_path, flat_config):
    """A client asking for `0.0` gets `0.0`, not a dropped field."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(
            body=(
                '{"model": "default", '
                '"messages": [{"role": "user", "content": "hi"}], '
                '"presence_penalty": 0.0}'
            )
        ),
    )

    assert payload["values"]["presence_penalty"] == 0.0


@pytest.mark.slow
def test_explicit_null_is_not_forwarded(tmp_path, flat_config):
    """An explicit `null` is dropped: litellm rejects the key itself."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(
            body=(
                '{"model": "default", '
                '"messages": [{"role": "user", "content": "hi"}], '
                '"presence_penalty": None}'
            )
        ),
    )

    assert "presence_penalty" not in payload["seen"]


@pytest.mark.slow
def test_the_routed_endpoint_model_replaces_the_tier_name(tmp_path, flat_config):
    """Routing rewrites `model`; `messages` is passed through."""
    payload = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(body='{"model": "default", "messages": [{"role": "user", "content": "hi"}]}'),
    )

    assert payload["values"]["model"] in ("m_big", "m_small")
    assert "messages" in payload["seen"]


@pytest.mark.slow
def test_stream_is_forwarded_only_when_the_client_set_it(tmp_path, flat_config):
    """The server reads `stream`, but litellm is what makes the chunks."""
    unset = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(body='{"model": "default", "messages": [{"role": "user", "content": "hi"}]}'),
    )
    assert "stream" not in unset["seen"]

    explicit = _run(
        tmp_path,
        ["x", "--config", str(flat_config), "--routers", "random"],
        _POST.format(
            body=(
                '{"model": "default", '
                '"messages": [{"role": "user", "content": "hi"}], '
                '"stream": False}'
            )
        ),
    )
    assert explicit["values"]["stream"] is False


# ---------------------------------------------------------------------------
# --host
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_help_lists_host_with_a_loopback_default():
    """`--help` documents `--host` and names its loopback default."""
    result = subprocess.run(
        [sys.executable, "-m", "routellm.openai_server", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert "--host" in result.stdout
    assert "127.0.0.1" in result.stdout


@pytest.mark.slow
def test_host_flag_is_accepted_and_defaults_to_loopback():
    """`--host` parses, and omitting it yields `127.0.0.1`."""
    probe = (
        "import json, sys;"
        "sys.argv = {argv!r};"
        "import routellm.openai_server as server;"
        "print(json.dumps({{'host': server.args.host}}))"
    )

    def _host(argv):
        result = subprocess.run(
            [sys.executable, "-c", probe.format(argv=argv)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])["host"]

    assert _host(["x", "--routers", "random"]) == "127.0.0.1"
    assert _host(["x", "--routers", "random", "--host", "0.0.0.0"]) == "0.0.0.0"


# ---------------------------------------------------------------------------
# Import cycle
# ---------------------------------------------------------------------------


_CYCLE_PROBE = """
import json, sys
sys.argv = ["x", "--routers", "random"]
from importlib.metadata import EntryPoint
import routellm.routers.registry as registry

# An entry point that pulls `routellm.middleware`, as the `jev` router
# of the typesafe extension does. Discovery runs at `routers.py` import
# time, so this loads while `routellm.controller` is still half-built --
# the chain the live smoke hit once the extension was installed.
real = registry.entry_points


def fake(group=None):
    if group == registry.ENTRY_POINT_GROUP:
        return [
            EntryPoint(
                name="probe",
                value="routellm.middleware:IntentModelSelector",
                group=group,
            )
        ]
    return real(group=group)


registry.entry_points = fake

import routellm.openai_server  # noqa: F401

print(json.dumps({"failures": registry.discovery_failures}))
"""


@pytest.mark.slow
def test_middleware_entry_point_loads_during_controller_import():
    """A middleware-backed entry point must not re-enter a half-built module.

    `routers.py` runs discovery at import time, so a middleware module
    importing from `routellm.controller` closes a cycle the moment any
    such extension is installed. Discovery swallows the ImportError, so
    the symptom is a recorded failure rather than a raised one.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CYCLE_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    failures = json.loads(result.stdout.strip().splitlines()[-1])["failures"]
    assert failures == {}


def test_middleware_does_not_import_model_pair_from_the_controller():
    """`ModelPair` lives in `routellm.types`; the controller only re-exports."""
    offenders = [
        path
        for path in (REPO_ROOT / "routellm" / "middleware").rglob("*.py")
        if "from routellm.controller import" in path.read_text()
    ]

    assert offenders == []
