"""What the server logs and what it tells a client about failures.

Prompts are the user's data: they must not reach INFO logs, and an
upstream's URL or error text must not reach the client. Every case runs
in a subprocess against local fake upstreams (see the agent-fidelity
tests for why), and prints one JSON line.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from routellm.tests.subproc import SUBPROCESS_TIMEOUT

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow

_PREAMBLE = """
import io, json, logging, sys
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random", "--verbose"]

A = FakeUpstream("A")
B = FakeUpstream("B")
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{
        "a": {{"model": "openai/model-a", "api_base": A.base, "api_key_env": "FAKE_KEY"}},
        "b": {{"model": "openai/model-b", "api_base": B.base, "api_key_env": "FAKE_KEY"}},
    }},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}}}},
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient

# Only this project's own records: the root logger (where the server
# logs) and routellm.*; third-party libraries log what they like.
buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
handler.addFilter(lambda r: r.name == "root" or r.name.startswith("routellm"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel({level})

SECRET = "SECRET-PROMPT-9f3a"
MESSAGES = [{{"role": "system", "content": "sys " + SECRET}},
            {{"role": "user", "content": "hello " + SECRET}}]


def emit(obj):
    print("@@RESULT@@" + json.dumps(obj))


def sse(response):
    return [line[6:] for line in response.iter_lines() if line.startswith("data: ")]


with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.cache.config.enabled = {cache}
    server.CONTROLLER.cache.config.db_path = {db!r}
    server.CONTROLLER.cache._init_db()
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000  # starved CPU is not an upstream timeout
"""


def _run(tmp_path, body, level="logging.INFO", cache=False):
    preamble = _PREAMBLE.format(
        config=str(tmp_path / "config.json"),
        level=level,
        cache=cache,
        db=str(tmp_path / "cache.db"),
    )
    indented = "\n".join("    " + line for line in body.strip("\n").splitlines())
    result = subprocess.run(
        [sys.executable, "-c", preamble + indented + "\n"],
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
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    return json.loads(line.split("@@RESULT@@", 1)[1])


def test_info_logs_carry_no_message_content(tmp_path):
    out = _run(
        tmp_path,
        """
client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES})
sse(client.post("/v1/chat/completions",
                json={"model": "default", "messages": MESSAGES, "stream": True}))
emit({"log": buffer.getvalue()})
""",
        cache=True,
    )

    assert "SECRET-PROMPT" not in out["log"]


def test_the_request_summary_is_logged_without_content(tmp_path):
    out = _run(
        tmp_path,
        """
client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES,
            "stream": True, "tools": [{"type": "function", "function": {"name": "f"}}]})
emit({"log": buffer.getvalue()})
""",
    )

    line = next(ln for ln in out["log"].splitlines() if "Received request" in ln)
    for expected in ("model=default", "messages=2", "tools=True", "images=False", "stream=True"):
        assert expected in line, line


def test_debug_logs_carry_no_prompt_either(tmp_path):
    out = _run(
        tmp_path,
        """
client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES})
client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES})
emit({"log": buffer.getvalue(), "hits": len(A.hits)})
""",
        level="logging.DEBUG",
        cache=True,
    )

    assert "SECRET-PROMPT" not in out["log"]


def test_stream_error_event_is_generic_with_a_request_id(tmp_path):
    out = _run(
        tmp_path,
        """
A.mode = "die_after"
with client.stream("POST", "/v1/chat/completions",
                   json={"model": "default", "messages": MESSAGES, "stream": True}) as r:
    events = sse(r)
emit({"events": events, "a": A.base, "log": buffer.getvalue()})
""",
    )

    last = json.loads(out["events"][-1])
    message = last["error"]["message"]
    assert last["error"]["type"] == "server_error"
    assert "request id" in message
    rid = last["error"]["request_id"]
    assert rid in message
    # Nothing of the upstream: not its address, not the exception.
    raw = out["events"][-1]
    assert "127.0.0.1" not in raw
    assert "Error" not in raw and "Exception" not in raw
    # The detail is kept server-side, keyed by the same id.
    assert rid in out["log"]


def test_non_stream_failure_is_generic_with_a_request_id(tmp_path):
    out = _run(
        tmp_path,
        """
A.mode = B.mode = "500"
r = client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES})
emit({"status": r.status_code, "text": r.text, "log": buffer.getvalue()})
""",
    )

    assert out["status"] == 502
    body = json.loads(out["text"])
    assert "request id" in body["message"]
    assert "127.0.0.1" not in out["text"] and "boom" not in out["text"]
    assert "SECRET-PROMPT" not in out["log"]
