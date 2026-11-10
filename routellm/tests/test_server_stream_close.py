"""`stream_response` closes the routed stream however the client leaves.

`routellm.openai_server` parses `sys.argv` at import, so each case runs
in its own subprocess and prints one JSON line. Fake sources only.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PREAMBLE = """
import asyncio, json, sys
sys.argv = ["x", "--routers", "random"]
import routellm.openai_server as server


class Chunk:
    def model_dump_json(self):
        return '{"id": "c"}'


class Source:
    def __init__(self, items=1, error=None):
        self.items, self.error, self.closed = items, error, 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.items:
            self.items -= 1
            return Chunk()
        if self.error:
            raise self.error
        raise StopAsyncIteration

    async def aclose(self):
        self.closed += 1

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


def test_a_client_that_disconnects_early_closes_the_stream(tmp_path):
    out = _run(
        tmp_path,
        """
async def go():
    src = Source(items=5)
    gen = server.stream_response(src)
    await gen.__anext__()
    await gen.aclose()          # what a client disconnect does
    return src.closed
print(json.dumps({"closed": asyncio.run(go())}))
""",
    )

    assert out["closed"] == 1


def test_a_cancelled_consumer_closes_the_stream(tmp_path):
    out = _run(
        tmp_path,
        """
class Slow(Source):
    async def __anext__(self):
        await asyncio.sleep(30)

async def go():
    src = Slow()
    async def consume():
        async for _ in server.stream_response(src):
            pass
    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return src.closed
print(json.dumps({"closed": asyncio.run(go())}))
""",
    )

    assert out["closed"] == 1


def test_a_stream_that_fails_midway_is_closed(tmp_path):
    out = _run(
        tmp_path,
        """
async def go():
    src = Source(items=1, error=ConnectionError("dropped"))
    events = [e async for e in server.stream_response(src)]
    return src.closed, len(events)
closed, n = asyncio.run(go())
print(json.dumps({"closed": closed, "events": n}))
""",
    )

    assert out["closed"] == 1
    assert out["events"] == 2


def test_a_finished_stream_is_closed(tmp_path):
    out = _run(
        tmp_path,
        """
async def go():
    src = Source(items=2)
    events = [e async for e in server.stream_response(src)]
    return src.closed, events[-1]
closed, last = asyncio.run(go())
print(json.dumps({"closed": closed, "last": last}))
""",
    )

    assert out["closed"] == 1
    assert out["last"].strip() == "data: [DONE]"
