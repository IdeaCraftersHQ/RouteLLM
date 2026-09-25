"""End-to-end scenarios driven through xrr record/replay cassettes.

These tests replay committed cassettes by default. `XRR_MODE=record`
re-records them against the module's stubbed transport; nothing here ever
reaches a network in either mode.

Why replay is the default: CI must never record. A recording run writes
whatever it observes and passes, so a suite that records by default
cannot fail on a changed response -- it just adopts it. Replay makes the
committed cassette the expectation, which is the point of having one.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# xrr is a declared dev dependency (see the `dev` extra). It is imported
# unconditionally and on purpose: the previous version of this file fell
# back to an inline fake whose replay returned a hardcoded string, so the
# test passed without the real library ever running. A skip states the
# truth; a fake green does not.
xrr = pytest.importorskip(
    "xrr",
    reason="xrr is not installed -- install the dev extra: pip install -e '.[dev]'",
)

RECORD = xrr.RECORD
REPLAY = xrr.REPLAY
FileCassette = xrr.FileCassette
Session = xrr.Session

# Cassettes live beside the test and are committed. A cassette that is
# not in the repo makes replay mode impossible and the test meaningless.
CASSETTE_DIR = Path(__file__).parent / "cassettes"


class XrrLiteLLMAdapter:
    """Addresses a cassette by the routed request and round-trips the
    response payload through YAML."""

    id = "litellm"

    def fingerprint(self, req: dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                "model": req.get("model"),
                "messages": req.get("messages"),
                "params": {k: v for k, v in req.items() if k not in ("model", "messages")},
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]

    def serialize_req(self, req: dict[str, Any]) -> dict[str, Any]:
        return req

    def serialize_resp(self, resp: Any) -> dict[str, Any]:
        return resp if isinstance(resp, dict) else resp.model_dump()

    def deserialize_resp(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


def _mode() -> str:
    """Session mode. Replay unless an operator opts into recording."""
    return os.getenv("XRR_MODE", REPLAY)


@pytest.fixture
def xrr_session():
    if _mode() == RECORD:
        CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    return Session(_mode(), FileCassette(str(CASSETTE_DIR)))


def _controller():
    """A controller with caching off.

    `.routellm_cache.db` is shared across the suite, so an enabled cache
    lets a previous test's entry satisfy this one -- a pass for the wrong
    reason. The cassette is this test's only source of a response.
    """
    from routellm.caching import CacheConfig
    from routellm.controller import Controller

    return Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        cache_config=CacheConfig(enabled=False),
    )


def _stub_payload(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-xrr-fixture",
        "object": "chat.completion",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def _stub_response(content: str) -> MagicMock:
    """The transport's stand-in during a recording run.

    The controller calls `.model_dump()` on whatever the transport
    returns, so the stub has to be an object rather than a bare dict.
    """
    payload = _stub_payload(content)
    res = MagicMock()
    res.choices = [MagicMock()]
    res.choices[0].message.content = content
    res.model_dump.return_value = payload
    return res


@pytest.mark.asyncio
async def test_completion_replays_from_cassette(xrr_session, monkeypatch):
    """The routed completion round-trips through a cassette.

    In replay mode `do_call` must never run: xrr serves the committed
    payload and the stub below would not be consulted at all. The
    `called` flag asserts exactly that, so a silent fall-through to the
    live path cannot pass.
    """
    controller = _controller()
    adapter = XrrLiteLLMAdapter()
    req = {
        "model": "router-random-0.5",
        "messages": [{"role": "user", "content": "hello"}],
    }

    called = {"n": 0}

    async def do_call():
        called["n"] += 1
        monkeypatch.setattr(
            "routellm.controller.acompletion",
            AsyncMock(return_value=_stub_response("hello there")),
        )
        return await controller.acompletion(**req)

    if _mode() == REPLAY:
        # xrr serves the cassette payload; `do` is never invoked.
        payload = xrr_session.record(adapter, req, lambda: None)
        assert called["n"] == 0, "replay must not execute the live path"
    else:
        live = await do_call()
        assert called["n"] == 1
        # Record writes the serialized response, so compare against the
        # same shape replay will hand back rather than the live object.
        xrr_session.record(adapter, req, lambda: live)
        payload = adapter.serialize_resp(live)

    assert payload["choices"][0]["message"]["content"] == "hello there"
    assert payload["id"] == "chatcmpl-xrr-fixture"


@pytest.mark.asyncio
async def test_cassette_is_committed():
    """Replay has something to replay.

    Guards the defect this file used to have: no cassette in the repo,
    RECORD as the default, so the first run recorded into a temp dir and
    a re-run passed off its own output as an expectation.
    """
    adapter = XrrLiteLLMAdapter()
    req = {
        "model": "router-random-0.5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    fp = adapter.fingerprint(req)
    for kind in ("req", "resp"):
        path = CASSETTE_DIR / f"{adapter.id}-{fp}.{kind}.yaml"
        assert path.exists(), f"missing committed cassette: {path.name}"


def test_cassettes_carry_no_secrets():
    """No cassette in the repo contains a credential.

    xrr redacts at record time, before anything reaches disk. This
    asserts the outcome rather than trusting it, so a cassette recorded
    with redaction disabled cannot be committed unnoticed.
    """
    from xrr.redact import Redactor

    redactor = Redactor()
    files = sorted(CASSETTE_DIR.glob("*.yaml"))
    assert files, "no cassettes found"

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not redactor.is_secret_value(text), f"{path.name} matches a credential pattern"
        for marker in ("api_key", "authorization", "bearer ", "sk-"):
            assert marker not in text.lower(), f"{path.name} contains {marker!r}"
