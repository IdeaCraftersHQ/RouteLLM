"""Conformance harness: one set of round-trip cases, run per dialect.

A dialect's test module supplies a `Driver` (the client's side of the
dialect: how to say each neutral `Scenario` on the wire, and how to read
the answer back) and subclasses `ConformanceSuite`:

    class TestMyDialect(ConformanceSuite):
        driver = MyDriver()
        skip = {"image_input": "dialect has no image blocks"}

The cases run against a real server in a child process (it parses
`sys.argv` at import) whose only upstream is a fake on loopback; see
`conftest.py` for the `harness` fixture. What each case asserts is
dialect-neutral: what the upstream received (system prompt, history,
images, tool calls and results arrive as chat completions) and what the
`Outcome` read back from the dialect's answer holds.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.subproc import SUBPROCESS_TIMEOUT, WAIT_DEADLINE

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The key the harness server knows. Every authorized case sends it.
KEY = "sk-harness-0000"
#: What the fake upstream answers in `echo` mode, for the user text "hi".
ECHO = "echo: hi"
#: A 1x1 PNG, base64: the image every image case sends.
PNG_B64 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
    )
).decode()


# -- neutral scenario ---------------------------------------------------------


@dataclass(frozen=True)
class ToolUse:
    """A tool call: `arguments` is the parsed JSON object."""

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    id: str
    content: str


@dataclass(frozen=True)
class Message:
    """One turn. `role` is `user` or `assistant`. A user turn may carry tool
    results (the dialect decides whether that is a role or a block); an
    assistant turn may carry tool calls."""

    role: str
    text: str = ""
    image_png_b64: str | None = None
    tool_uses: tuple[ToolUse, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class Scenario:
    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[ToolDef, ...] = ()
    stream: bool = False
    model: str = "default"


READ_FILE = ToolDef(
    "read_file",
    "Read a file",
    {"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]},
)
HI = Message("user", "hi")


@dataclass
class Http:
    """One request on the wire. `json` and `content` are exclusive."""

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json: Any = None
    content: bytes | None = None


@dataclass
class Outcome:
    """What a dialect's answer says, in neutral terms.

    `finish` is `stop`, `tool_calls` or `length`. Token counts are None
    when the answer carries none.
    """

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    finish: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Driver:
    """The client side of one dialect. Implement all four methods."""

    def build(self, scenario: Scenario) -> Http:
        """The request that says `scenario` in this dialect, without credentials."""
        raise NotImplementedError

    def authorize(self, http: Http, key: str | None) -> Http:
        """`http` carrying `key` the way the dialect does (header, query, signature).

        `key` None sends no credentials at all.
        """
        raise NotImplementedError

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        """Read a 200 answer (JSON body, or the whole stream when `scenario.stream`)."""
        raise NotImplementedError

    def parse_error(self, response: httpx.Response) -> str:
        """The message of a dialect error response. Fail if the shape is wrong."""
        raise NotImplementedError

    def unsupported(self) -> Http | None:
        """A request using a field the pipeline cannot honour, or None to skip.

        Expected: a 400 in the dialect's error shape, nothing sent upstream.
        """
        return None


# -- the server under test ----------------------------------------------------


class Harness:
    """A routellm server in a child process, one fake upstream behind it."""

    def __init__(self, workdir: Path, modules: str = ""):
        self.upstream = FakeUpstream("A")
        self.other = FakeUpstream("B")
        self.workdir = workdir
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        keys = workdir / "keys.yaml"
        import hashlib

        digest = hashlib.sha256(KEY.encode()).hexdigest()
        keys.write_text(f'keys:\n  - key_sha256: "{digest}"\n    principal: harness\n')
        config = workdir / "config.json"

        def ep(name, up):
            return {"model": f"openai/model-{name}", "api_base": up.base, "api_key_env": "FAKE_KEY"}

        config.write_text(
            json.dumps(
                {
                    "endpoints": {"a": ep("a", self.upstream), "b": ep("b", self.other)},
                    "tiers": {
                        "default": {
                            "router": "random",
                            "threshold": 0.0,
                            "strong": "a",
                            "weak": "b",
                        }
                    },
                    "auth": {"keys_file": str(keys)},
                    "resilience": {"max_retries": 0, "timeout_ms": 600_000},
                }
            )
        )
        self.log = (workdir / "server.log").open("w")
        env = {
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "XDG_CONFIG_HOME": str(workdir / "xdg"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "FAKE_KEY": "sk-fake",
            "ROUTELLM_ADAPTER_MODULES": modules,
        }
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "routellm.openai_server",
                "--config",
                str(config),
                "--routers",
                "random",
                "--port",
                str(port),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self._wait()

    def _wait(self) -> None:
        deadline = time.monotonic() + min(WAIT_DEADLINE, SUBPROCESS_TIMEOUT)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.3)
        self.close()
        raise RuntimeError(f"server did not come up:\n{self.tail()}")

    def tail(self, lines: int = 40) -> str:
        self.log.flush()
        return "\n".join((self.workdir / "server.log").read_text().splitlines()[-lines:])

    def reset(self) -> None:
        for up in (self.upstream, self.other):
            up.hits.clear()
            up.mode = "ok"
            up.reply = None

    def send(self, http: Http) -> httpx.Response:
        return httpx.request(
            http.method,
            self.base + http.path,
            headers=http.headers,
            params=http.params,
            json=http.json,
            content=http.content,
            timeout=60,
        )

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def start_harness(workdir: Path, modules: str = "") -> Harness:
    extra = os.environ.get("ROUTELLM_ADAPTER_MODULES", "")
    return Harness(workdir, ",".join(filter(None, (modules, extra))))


# -- upstream-side helpers ----------------------------------------------------


def text_of(message: dict) -> str:
    """The text of a chat message: a string, or its text parts joined."""
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def images_of(message: dict) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [p["image_url"]["url"] for p in content if p.get("type") == "image_url"]


# -- the cases ----------------------------------------------------------------


class ConformanceSuite:
    """Subclass, set `driver`, optionally `skip = {case: reason}`.

    Every case is a method named `test_<case>`; `skip` is keyed by `<case>`.
    """

    driver: Driver
    skip: dict[str, str] = {}  # noqa: RUF012

    @pytest.fixture(autouse=True)
    def _setup(self, harness: Harness, request):
        case = request.node.name.removeprefix("test_")
        if case in self.skip:
            pytest.skip(self.skip[case])
        self.h = harness

    # plumbing
    def call(self, scenario: Scenario, key: str | None = KEY) -> httpx.Response:
        return self.h.send(self.driver.authorize(self.driver.build(scenario), key))

    def ok(self, scenario: Scenario) -> Outcome:
        response = self.call(scenario)
        assert response.status_code == 200, response.text
        return self.driver.parse(response, scenario)

    def sent(self) -> dict:
        assert self.h.upstream.hits, "the request never reached the upstream"
        return self.h.upstream.hits[-1][2]

    # cases
    def test_plain_text(self):
        self.h.upstream.mode = "echo"
        out = self.ok(Scenario((HI,)))
        assert out.text == ECHO
        assert out.finish == "stop"
        messages = self.sent()["messages"]
        assert [(m["role"], text_of(m)) for m in messages] == [("user", "hi")]

    def test_system_prompt(self):
        self.h.upstream.mode = "echo"
        self.ok(Scenario((HI,), system="Be terse."))
        first = self.sent()["messages"][0]
        assert first["role"] == "system" and text_of(first) == "Be terse."

    def test_multi_turn(self):
        self.h.upstream.mode = "echo"
        history = (Message("user", "one"), Message("assistant", "two"), Message("user", "hi"))
        out = self.ok(Scenario(history))
        assert out.text == ECHO
        got = [(m["role"], text_of(m)) for m in self.sent()["messages"]]
        assert got == [("user", "one"), ("assistant", "two"), ("user", "hi")]

    def test_image_input(self):
        self.h.upstream.mode = "echo"
        self.ok(Scenario((Message("user", "hi", image_png_b64=PNG_B64),)))
        message = self.sent()["messages"][-1]
        assert text_of(message) == "hi"
        (url,) = images_of(message)
        assert url.startswith("data:image/png;base64,") and url.endswith(PNG_B64)

    def test_tool_call_response(self):
        self.h.upstream.mode = "tools"
        out = self.ok(Scenario((HI,), tools=(READ_FILE,)))
        assert out.tool_uses and out.tool_uses[0].name == "read_file"
        assert out.tool_uses[0].arguments == {"p": "a.txt"}
        assert out.tool_uses[0].id
        assert out.finish == "tool_calls"
        (tool,) = self.sent()["tools"]
        assert tool["function"]["name"] == "read_file"
        assert tool["function"]["parameters"]["required"] == ["p"]

    def test_tool_result_request(self):
        self.h.upstream.mode = "echo"
        use = ToolUse("call_1", "read_file", {"p": "a.txt"})
        history = (
            HI,
            Message("assistant", tool_uses=(use,)),
            Message("user", tool_results=(ToolResult("call_1", "file body"),)),
        )
        self.ok(Scenario(history, tools=(READ_FILE,)))
        messages = self.sent()["messages"]
        call = next(m for m in messages if m["role"] == "assistant")["tool_calls"][0]
        assert call["id"] == "call_1" and call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"p": "a.txt"}
        result = next(m for m in messages if m["role"] == "tool")
        assert result["tool_call_id"] == "call_1" and text_of(result) == "file body"

    def test_streaming_text(self):
        self.h.upstream.mode = "echo"
        out = self.ok(Scenario((HI,), stream=True))
        assert out.text == ECHO
        assert out.finish == "stop"
        assert self.sent()["stream"] is True

    def test_streaming_tool_call(self):
        self.h.upstream.mode = "tools"
        out = self.ok(Scenario((HI,), tools=(READ_FILE,), stream=True))
        assert [(u.name, u.arguments) for u in out.tool_uses] == [("read_file", {"p": "a.txt"})]
        assert out.tool_uses[0].id == "call_1"
        assert out.finish == "tool_calls"

    def test_usage(self):
        out = self.ok(Scenario((HI,)))
        assert (out.input_tokens, out.output_tokens) == (3, 3)

    def test_streaming_usage(self):
        out = self.ok(Scenario((HI,), stream=True))
        assert (out.input_tokens, out.output_tokens) == (3, 3)

    def test_decision_path_header(self):
        for stream in (False, True):
            response = self.call(Scenario((HI,), stream=stream))
            path = json.loads(response.headers["x-routellm-path"])
            assert isinstance(path, list) and path

    def test_upstream_error(self):
        # The router falls back to the sibling, so both must fail.
        self.h.upstream.mode = self.h.other.mode = "500"
        response = self.call(Scenario((HI,)))
        assert response.status_code == 502
        message = self.driver.parse_error(response)
        assert "Upstream request failed" in message
        assert "boom" not in response.text

    def test_unauthorized(self):
        for key in (None, "sk-not-a-key"):
            response = self.call(Scenario((HI,)), key=key)
            assert response.status_code == 401
            assert self.driver.parse_error(response)
        assert not self.h.upstream.hits

    def test_refuses_unsupported_field(self):
        http = self.driver.unsupported()
        if http is None:
            pytest.skip("driver names no unsupported field")
        response = self.h.send(self.driver.authorize(http, KEY))
        assert response.status_code == 400
        assert self.driver.parse_error(response)
        assert not self.h.upstream.hits
