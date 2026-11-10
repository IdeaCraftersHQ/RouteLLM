"""Every candidate cooling is a 503 with `Retry-After`, never a 502.

`routellm.openai_server` parses `sys.argv` at import, so each case runs
in its own subprocess and prints one JSON line. Fake controller only.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PREAMBLE = """
import json, sys
sys.argv = ["x", "--routers", "random"]
from fastapi.testclient import TestClient
import routellm.openai_server as server
from routellm.resilience import EndpointsCooling

MESSAGES = [{"role": "user", "content": "hi"}]


class Fake:
    model_counts = {}
    decision = None

    def __init__(self, error):
        self.error = error

    async def acompletion(self, **kwargs):
        raise self.error


def post(error, **extra):
    server.CONTROLLER = Fake(error)
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": MESSAGES, **extra},
    )
    return {"status": r.status_code, "headers": dict(r.headers), "text": r.text}

"""


def _run(tmp_path, body):
    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cooling_is_a_503_with_the_soonest_expiry(tmp_path):
    out = _run(
        tmp_path,
        """
err = EndpointsCooling({"a": 1030.2, "b": 1100.0}, now=1000.0)
print(json.dumps(post(err)))
""",
    )

    assert out["status"] == 503
    assert out["headers"]["retry-after"] == "31"  # ceil(30.2)
    body = json.loads(out["text"])
    assert body["type"] == "server_error"
    assert body["request_id"] in body["message"]
    # Generic: no endpoint names, no timings.
    assert "a (" not in out["text"] and "cooling" not in out["text"]


def test_retry_after_is_at_least_one_second(tmp_path):
    out = _run(
        tmp_path,
        """
err = EndpointsCooling({"a": 999.0}, now=1000.0)
print(json.dumps(post(err)))
""",
    )

    assert out["status"] == 503
    assert out["headers"]["retry-after"] == "1"


def test_streaming_request_failing_before_first_chunk_is_a_503(tmp_path):
    out = _run(
        tmp_path,
        """
err = EndpointsCooling({"a": 1007.0}, now=1000.0)
print(json.dumps(post(err, stream=True)))
""",
    )

    assert out["status"] == 503
    assert out["headers"]["retry-after"] == "7"
    assert out["headers"]["content-type"].startswith("application/json")


def test_other_failures_stay_502_without_retry_after(tmp_path):
    out = _run(
        tmp_path,
        """
print(json.dumps(post(RuntimeError("boom"))))
""",
    )

    assert out["status"] == 502
    assert "retry-after" not in out["headers"]
