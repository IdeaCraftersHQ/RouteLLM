"""Params a leaf's provider rejects are dropped for that leaf, not the request.

Agent clients send `parallel_tool_calls` whether or not the routed
provider takes it, and litellm's `ollama_chat/` transport refuses the
whole call over it. Every case drives the real app and the real litellm
against local fakes: an Ollama-shaped upstream for the `ollama_chat/`
leaf and an OpenAI-shaped one for the `openai/` leaf.

The server parses `sys.argv` at import, so each case runs in its own
subprocess and prints one JSON line.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from routellm.endpoints import Endpoint

REPO_ROOT = Path(__file__).resolve().parents[2]

_PREAMBLE = """
import json, logging, sys
from routellm.tests.fake_ollama import FakeOllama
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

O = FakeOllama("O")      # an ollama_chat/ leaf
A = FakeUpstream("A")    # an openai/ leaf
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{
        "o": {{"model": "ollama_chat/model-o", "api_base": O.base, {o_extra}}},
        "a": {{"model": "openai/model-a", "api_base": A.base, "api_key_env": "FAKE_KEY",
              {a_extra}}},
    }},
    "tiers": {{
        "to_o": {{"router": "random", "threshold": 0.0, "strong": "o", "weak": "a"}},
        "to_a": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "o"}},
    }},
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient

class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []
    def emit(self, record):
        self.lines.append(record.getMessage())

logs = Capture()
logging.getLogger("routellm").addHandler(logs)
logging.getLogger("routellm").setLevel(logging.DEBUG)

TOOLS = [{{"type": "function", "function": {{
    "name": "read_file", "description": "d",
    "parameters": {{"type": "object", "properties": {{"p": {{"type": "string"}}}}}},
}}}}]

with TestClient(server.app) as client:
    server.CONTROLLER.cache.config.enabled = False
    server.CONTROLLER.resilience.config.max_retries = 0
"""


def _run(tmp_path, body, o_extra="", a_extra=""):
    """Run `body` inside the `with` block and return its last JSON line."""
    indented = "\n".join("    " + line for line in body.strip("\n").splitlines())
    script = (
        _PREAMBLE.format(
            config=str(tmp_path / "config.json"),
            o_extra=o_extra or '"tags": []',
            a_extra=a_extra or '"tags": []',
        )
        + indented
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "FAKE_KEY": "sk-fake",
        },
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


_CALL = """
r = client.post("/v1/chat/completions", json={{
    "model": {model!r}, "messages": [{{"role": "user", "content": "SECRET-PROMPT"}}],
    "tools": TOOLS, {extra}}})
fake = {fake}
print(json.dumps({{"status": r.status_code, "seen": fake.hits[-1][2] if fake.hits else None,
                  "logs": logs.lines}}))
"""


def _call(model, fake, extra):
    return _CALL.format(model=model, fake=fake, extra=extra)


@pytest.mark.slow
@pytest.mark.parametrize("value", ["True", "False"])
def test_ollama_leaf_is_served_with_parallel_tool_calls_dropped(tmp_path, value):
    out = _run(tmp_path, _call("to_o", "O", f'"parallel_tool_calls": {value}'))

    assert out["status"] == 200, out
    # The leaf itself served it, not its sibling.
    assert out["seen"] is not None
    assert "parallel_tool_calls" not in json.dumps(out["seen"])


@pytest.mark.slow
def test_ollama_leaf_streams_with_parallel_tool_calls_dropped(tmp_path):
    out = _run(
        tmp_path,
        _call("to_o", "O", '"parallel_tool_calls": True, "stream": True'),
    )

    assert out["status"] == 200, out
    assert out["seen"] is not None
    assert "parallel_tool_calls" not in json.dumps(out["seen"])


@pytest.mark.slow
@pytest.mark.parametrize("value", ["True", "False"])
def test_openai_leaf_keeps_parallel_tool_calls(tmp_path, value):
    out = _run(tmp_path, _call("to_a", "A", f'"parallel_tool_calls": {value}'))

    assert out["status"] == 200, out
    assert out["seen"]["parallel_tool_calls"] is (value == "True")


@pytest.mark.slow
def test_dropped_params_are_logged_by_name_and_leaf_without_content(tmp_path):
    out = _run(tmp_path, _call("to_o", "O", '"parallel_tool_calls": True'))

    dropped = [line for line in out["logs"] if "parallel_tool_calls" in line]
    assert dropped, out["logs"]
    assert "o" in dropped[0].split()  # the leaf, by endpoint name
    assert not any("SECRET-PROMPT" in line for line in out["logs"])


@pytest.mark.slow
def test_nothing_is_logged_when_nothing_is_dropped(tmp_path):
    out = _run(tmp_path, _call("to_a", "A", '"parallel_tool_calls": True'))

    assert not [line for line in out["logs"] if "dropp" in line.lower()], out["logs"]


@pytest.mark.slow
def test_a_param_the_default_does_not_cover_still_fails_loudly(tmp_path):
    """The default covers advisory flags only; `tools`-class params stay loud."""
    out = _run(tmp_path, _call("to_o", "O", '"presence_penalty": 0.5'))

    assert out["seen"] is None, out  # the leaf refused; its sibling answered


@pytest.mark.slow
def test_drop_params_true_drops_whatever_the_provider_rejects(tmp_path):
    out = _run(
        tmp_path,
        _call("to_o", "O", '"presence_penalty": 0.5, "parallel_tool_calls": True'),
        o_extra='"drop_params": True',
    )

    assert out["status"] == 200, out
    assert out["seen"] is not None
    assert "presence_penalty" not in out["seen"]
    assert "parallel_tool_calls" not in json.dumps(out["seen"])


@pytest.mark.slow
def test_drop_params_false_restores_the_strict_behaviour(tmp_path):
    out = _run(
        tmp_path,
        _call("to_o", "O", '"parallel_tool_calls": True'),
        o_extra='"drop_params": False',
    )

    assert out["seen"] is None, out  # the leaf refused; its sibling answered


@pytest.mark.slow
def test_drop_params_list_drops_those_names_even_where_supported(tmp_path):
    out = _run(
        tmp_path,
        _call("to_a", "A", '"parallel_tool_calls": True, "reasoning_effort": "low"'),
        a_extra='"drop_params": ["parallel_tool_calls"]',
    )

    assert out["status"] == 200, out
    assert "parallel_tool_calls" not in out["seen"]
    assert out["seen"]["reasoning_effort"] == "low"
    assert any("parallel_tool_calls" in line for line in out["logs"])


def test_drop_params_accepts_a_bool_or_a_list_of_names():
    assert Endpoint(name="e", model="m").drop_params is None
    assert Endpoint(name="e", model="m", drop_params=True).drop_params is True
    assert Endpoint(name="e", model="m", drop_params=["a", "b"]).drop_params == ["a", "b"]


@pytest.mark.parametrize("bad", ["yes", 3, [1], [""], {"a": 1}])
def test_drop_params_rejects_anything_else(bad):
    with pytest.raises(ValidationError):
        Endpoint(name="e", model="m", drop_params=bad)
