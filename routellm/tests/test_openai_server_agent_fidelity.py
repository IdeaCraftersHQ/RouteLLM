"""Agent-client fidelity of the OpenAI server, against fake upstreams.

Agent clients reach this server through a translating proxy that speaks
OpenAI chat completions on its far side: nested `tools`, object `tool_choice`,
tool history with null content, image parts, and `stream: true`. Every
case here drives the real app and the real litellm against local fake
upstreams, so what is asserted is what an upstream would actually receive.

The server parses `sys.argv` at import, so each case runs in its own
subprocess and prints one JSON line.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow

_PREAMBLE = """
import json, sys
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

A = FakeUpstream("A")   # strong side: the picked leaf at threshold 0
B = FakeUpstream("B")   # weak side: its sibling
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{
        "a": {{"model": "openai/model-a", "api_base": A.base, "api_key_env": "FAKE_KEY"}},
        "b": {{"model": "openai/model-b", "api_base": B.base, "api_key_env": "FAKE_KEY"}},
    }},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}}}},
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient

TOOLS = [{{"type": "function", "function": {{
    "name": "read_file", "description": "d",
    "parameters": {{"type": "object", "properties": {{"p": {{"type": "string"}}}}, "required": ["p"]}},
}}}}]


def sse(response):
    return [line[6:] for line in response.iter_lines() if line.startswith("data: ")]


with TestClient(server.app) as client:
    # The completion cache is a repo-relative SQLite file shared by every
    # run, and a hit never reaches an upstream at all.
    server.CONTROLLER.cache.config.enabled = False
    server.CONTROLLER.resilience.config.max_retries = 0
"""


def _run(tmp_path, body):
    """Run `body` inside the `with` block and return its last JSON line."""
    indented = "\n".join("    " + line for line in body.strip("\n").splitlines())
    script = _PREAMBLE.format(config=str(tmp_path / "config.json")) + indented + "\n"
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


_USER = '[{"role": "user", "content": "hi"}]'


# ---------------------------------------------------------------------------
# Request shapes
# ---------------------------------------------------------------------------


def test_openai_tools_and_object_tool_choice_reach_the_upstream_unchanged(tmp_path):
    out = _run(
        tmp_path,
        f"""
choice = {{"type": "function", "function": {{"name": "read_file"}}}}
r = client.post("/v1/chat/completions", json={{
    "model": "default", "messages": {_USER}, "tools": TOOLS, "tool_choice": choice}})
print(json.dumps({{"status": r.status_code, "body": r.json(),
                  "seen": A.hits[-1][2] if A.hits else None, "choice": choice}}))
""",
    )

    assert out["status"] == 200, out["body"]
    assert out["seen"]["tools"][0]["function"]["name"] == "read_file"
    assert out["seen"]["tools"][0]["function"]["parameters"]["required"] == ["p"]
    assert out["seen"]["tool_choice"] == out["choice"]


def test_tool_history_with_null_and_absent_content_round_trips(tmp_path):
    out = _run(
        tmp_path,
        """
call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
messages = [
    {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]},
    {"role": "user", "content": "h"},
    {"role": "assistant", "content": None, "tool_calls": [call]},
    {"role": "assistant", "tool_calls": [{**call, "id": "call_2"}], "reasoning_content": "r"},
    {"role": "tool", "tool_call_id": "call_1", "content": "result", "name": "read_file"},
    {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "r2"}]},
]
r = client.post("/v1/chat/completions", json={"model": "default", "messages": messages, "tools": TOOLS})
print(json.dumps({"status": r.status_code, "body": r.json(), "sent": messages,
                  "seen": A.hits[-1][2]["messages"] if A.hits else None}))
""",
    )

    assert out["status"] == 200, out["body"]
    # litellm's provider layer omits a null `content` on a tool-calling
    # assistant turn, which is the same message on the wire. Nothing else
    # about the history changes.
    expected = json.loads(json.dumps(out["sent"]))
    expected[2].pop("content")
    assert out["seen"] == expected


def test_nulls_inside_tool_schemas_are_data_not_unset_fields(tmp_path):
    out = _run(
        tmp_path,
        f"""
schema = {{"type": "object", "properties": {{"p": {{"type": ["string", "null"], "default": None}}}}}}
tools = [{{"type": "function", "function": {{"name": "f", "parameters": schema}}}}]
r = client.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER}, "tools": tools}})
print(json.dumps({{"status": r.status_code, "schema": schema,
                  "seen": A.hits[-1][2]["tools"][0]["function"]["parameters"] if A.hits else None}}))
""",
    )

    assert out["status"] == 200
    assert out["seen"] == out["schema"]


def test_image_parts_and_json_schema_response_format_are_accepted(tmp_path):
    out = _run(
        tmp_path,
        """
img = [{"role": "user", "content": [
    {"type": "text", "text": "what"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}]}]
fmt = {"type": "json_schema", "json_schema": {"name": "s", "strict": True,
       "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}}
r = client.post("/v1/chat/completions", json={"model": "default", "messages": img, "response_format": fmt})
print(json.dumps({"status": r.status_code, "body": r.json(), "img": img, "fmt": fmt,
                  "seen": A.hits[-1][2] if A.hits else None}))
""",
    )

    assert out["status"] == 200, out["body"]
    assert out["seen"]["messages"] == out["img"]
    assert out["seen"]["response_format"] == out["fmt"]


def test_standard_params_outside_the_old_schema_are_passed_through(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json={{
    "model": "default", "messages": {_USER},
    "max_completion_tokens": 77, "reasoning_effort": "low", "parallel_tool_calls": False,
    "stream_options": {{"include_usage": True}}, "metadata": {{"user_id": "u"}}, "bogus": 1}})
print(json.dumps({{"status": r.status_code, "seen": A.hits[-1][2] if A.hits else None}}))
""",
    )

    assert out["status"] == 200
    seen = out["seen"]
    assert seen["max_completion_tokens"] == 77
    assert seen["reasoning_effort"] == "low"
    assert seen["parallel_tool_calls"] is False
    assert "metadata" not in seen
    assert "bogus" not in seen


def test_translated_agent_bodies_replay(tmp_path):
    """Chat bodies as a proxy translates them from Anthropic-Messages and Responses clients."""
    out = _run(
        tmp_path,
        """
bodies = {
 # Anthropic-Messages-translated: system as parts, tool round trip, streamed, usage requested.
 "messages": {
  "model": "default", "max_tokens": 256, "temperature": 1, "stream": True,
  "stream_options": {"include_usage": True}, "reasoning_effort": "medium",
  "tool_choice": "auto", "tools": TOOLS,
  "messages": [
   {"role": "system", "content": [{"type": "text", "text": "You are an agent"}]},
   {"role": "user", "content": [{"type": "text", "text": "read it"}]},
   {"role": "assistant", "content": [{"type": "text", "text": "ok"}],
    "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": "{\\"p\\":\\"a\\"}"}}]},
   {"role": "tool", "tool_call_id": "toolu_1", "content": "file text"}]},
 # Responses-translated: no content key on tool-call turns, reasoning_content,
 # parallel_tool_calls added by the translation.
 "responses": {
  "model": "default", "stream": True, "parallel_tool_calls": True, "reasoning_effort": "medium",
  "tools": TOOLS,
  "messages": [
   {"role": "system", "content": "instructions"},
   {"role": "user", "content": "run it"},
   {"role": "assistant", "reasoning_content": "thinking",
    "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
   {"role": "tool", "tool_call_id": "call_9", "content": "out"}]},
 # Anthropic-Messages-translated, with an image.
 "image": {
  "model": "default", "max_tokens": 64,
  "messages": [{"role": "user", "content": [
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
   {"type": "text", "text": "what is this"}]}]},
}
res = {}
for name, body in bodies.items():
    n = len(A.hits)
    if body.get("stream"):
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            events = sse(r)
            res[name] = {"status": r.status_code, "events": len(events), "last": events[-1]}
    else:
        r = client.post("/v1/chat/completions", json=body)
        res[name] = {"status": r.status_code}
    res[name]["hits"] = len(A.hits) - n
    res[name]["messages"] = A.hits[-1][2]["messages"] == body["messages"]
print(json.dumps(res))
""",
    )

    for name, got in out.items():
        assert got["status"] == 200, name
        assert got["hits"] == 1, name
        assert got["messages"], name
    assert out["messages"]["last"] == "[DONE]"
    assert out["responses"]["last"] == "[DONE]"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_yields_chunks_then_done_with_one_upstream_call(tmp_path):
    out = _run(
        tmp_path,
        f"""
with client.stream("POST", "/v1/chat/completions",
                   json={{"model": "default", "messages": {_USER}, "stream": True}}) as r:
    events = sse(r)
    ctype = r.headers["content-type"]
    status = r.status_code
print(json.dumps({{"status": status, "ctype": ctype, "events": events,
                  "a": len(A.hits), "b": len(B.hits)}}))
""",
    )

    assert out["status"] == 200
    assert out["ctype"].startswith("text/event-stream")
    assert out["events"][-1] == "[DONE]"
    text = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in map(json.loads, out["events"][:-1])
        if c["choices"]
    )
    assert text == "hello from A"
    assert (out["a"], out["b"]) == (1, 0)


def test_streamed_tool_call_deltas_arrive_intact(tmp_path):
    out = _run(
        tmp_path,
        f"""
A.mode = "tools"
with client.stream("POST", "/v1/chat/completions",
                   json={{"model": "default", "messages": {_USER}, "tools": TOOLS, "stream": True}}) as r:
    events = sse(r)
chunks = [json.loads(e) for e in events[:-1]]
args = "".join(
    tc["function"].get("arguments") or ""
    for c in chunks if c["choices"] for tc in (c["choices"][0]["delta"].get("tool_calls") or []))
names = [tc["function"].get("name") for c in chunks if c["choices"]
         for tc in (c["choices"][0]["delta"].get("tool_calls") or []) if tc["function"].get("name")]
finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
print(json.dumps({{"args": args, "names": names, "finish": finish, "last": events[-1]}}))
""",
    )

    assert json.loads(out["args"]) == {"p": "a.txt"}
    assert out["names"] == ["read_file"]
    assert out["finish"] == ["tool_calls"]
    assert out["last"] == "[DONE]"


def test_stream_path_is_logged_and_nothing_is_cached(tmp_path):
    out = _run(
        tmp_path,
        f"""
import logging
server.CONTROLLER.cache.config.enabled = True
class Grab(logging.Handler):
    def __init__(self): super().__init__(); self.lines = []
    def emit(self, record): self.lines.append(record.getMessage())
grab = Grab()
logging.getLogger().addHandler(grab); logging.getLogger().setLevel(logging.INFO)
puts = []
orig = server.CONTROLLER.cache.aput
async def spy(*a, **k): puts.append(a); return await orig(*a, **k)
server.CONTROLLER.cache.aput = spy
for _ in range(2):
    with client.stream("POST", "/v1/chat/completions",
                       json={{"model": "default", "messages": {_USER}, "stream": True}}) as r:
        sse(r)
print(json.dumps({{"puts": len(puts), "a": len(A.hits),
                  "logged": [l for l in grab.lines if "Routing path" in l or "routing path" in l]}}))
""",
    )

    assert out["puts"] == 0
    assert out["a"] == 2  # a stream is never served from, or stored in, the cache
    assert out["logged"]


def test_stream_trace_is_recorded_when_the_stream_ends(tmp_path):
    out = _run(
        tmp_path,
        f"""
traces = []
orig = server.CONTROLLER.quality_manager.record_trace
def spy(prompt, model, response, meta, **kw): traces.append((model, response, kw)); return orig(prompt, model, response, meta, **kw)
server.CONTROLLER.quality_manager.record_trace = spy
with client.stream("POST", "/v1/chat/completions",
                   json={{"model": "default", "messages": {_USER}, "stream": True}}) as r:
    sse(r)
t = traces[0] if traces else None
print(json.dumps({{"n": len(traces), "model": t and t[0],
                  "text": t and t[1]["choices"][0]["message"]["content"],
                  "path": t and t[2]["path"]}}))
""",
    )

    assert out["n"] == 1
    assert out["model"] == "a"
    assert out["text"] == "hello from A"
    assert out["path"]


def test_stream_options_reach_upstream_and_the_usage_chunk_comes_back(tmp_path):
    out = _run(
        tmp_path,
        f"""
with client.stream("POST", "/v1/chat/completions", json={{
        "model": "default", "messages": {_USER}, "stream": True,
        "stream_options": {{"include_usage": True}}}}) as r:
    events = sse(r)
usage = [json.loads(e)["usage"] for e in events[:-1] if json.loads(e).get("usage")]
print(json.dumps({{"sent": A.hits[-1][2].get("stream_options"), "usage": usage}}))
""",
    )

    assert out["sent"] == {"include_usage": True}
    assert out["usage"] and out["usage"][-1]["total_tokens"] == 6


def test_picked_leaf_failing_before_first_chunk_serves_the_sibling_stream(tmp_path):
    for mode in ("500", "die_first"):
        out = _run(
            tmp_path,
            f"""
A.mode = {mode!r}
with client.stream("POST", "/v1/chat/completions",
                   json={{"model": "default", "messages": {_USER}, "stream": True}}) as r:
    events = sse(r)
    status = r.status_code
text = "".join(c["choices"][0]["delta"].get("content") or ""
               for c in map(json.loads, events[:-1]) if c["choices"])
print(json.dumps({{"status": status, "text": text, "last": events[-1], "a": len(A.hits), "b": len(B.hits)}}))
""",
        )

        assert out["status"] == 200, mode
        assert out["text"] == "hello from B", mode
        assert out["last"] == "[DONE]", mode
        # The SDK may retry a 5xx on its own, so the picked leaf can be
        # hit more than once; the sibling is called exactly once.
        assert out["a"] >= 1 and out["b"] == 1, mode


def test_failure_after_first_chunk_is_an_error_event_and_sibling_is_not_called(tmp_path):
    out = _run(
        tmp_path,
        f"""
A.mode = "die_after"
with client.stream("POST", "/v1/chat/completions",
                   json={{"model": "default", "messages": {_USER}, "stream": True}}) as r:
    events = sse(r)
    status = r.status_code
print(json.dumps({{"status": status, "events": events, "a": len(A.hits), "b": len(B.hits)}}))
""",
    )

    assert out["status"] == 200
    assert (out["a"], out["b"]) == (1, 0)
    last = json.loads(out["events"][-1])
    assert "error" in last and last["error"]["message"]
    assert "[DONE]" not in out["events"]


def test_every_leaf_failing_a_stream_is_a_clean_error_not_a_hang(tmp_path):
    out = _run(
        tmp_path,
        f"""
A.mode = B.mode = "500"
quiet = TestClient(server.app, raise_server_exceptions=False)
r = quiet.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER}, "stream": True}})
print(json.dumps({{"status": r.status_code, "a": len(A.hits), "b": len(B.hits)}}))
""",
    )

    assert out["status"] >= 500
    assert out["a"] >= 1 and out["b"] >= 1


# ---------------------------------------------------------------------------
# Loop refusal
# ---------------------------------------------------------------------------


def test_hop_header_is_refused_with_508_and_no_upstream_call(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", headers={{"X-RouteLLM-Hop": "1"}},
                json={{"model": "default", "messages": {_USER}}})
s = client.post("/v1/chat/completions", headers={{"x-routellm-hop": "1"}},
                json={{"model": "default", "messages": {_USER}, "stream": True}})
print(json.dumps({{"status": r.status_code, "stream_status": s.status_code,
                  "body": r.json(), "a": len(A.hits), "b": len(B.hits)}}))
""",
    )

    assert out["status"] == 508
    assert out["stream_status"] == 508
    assert out["body"]["message"]
    assert (out["a"], out["b"]) == (0, 0)


def test_a_malformed_body_with_the_hop_header_is_still_a_508(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", headers={"x-routellm-hop": "1"},
                content=b"not json", )
s = client.post("/v1/chat/completions", headers={"x-routellm-hop": "1"},
                json={"messages": "no model field"})
print(json.dumps({"status": r.status_code, "other": s.status_code,
                  "body": r.json(), "a": len(A.hits), "b": len(B.hits)}))
""",
    )

    assert (out["status"], out["other"]) == (508, 508)
    assert out["body"]["message"]
    assert (out["a"], out["b"]) == (0, 0)


# ---------------------------------------------------------------------------
# Outbound hop header
# ---------------------------------------------------------------------------


def test_every_outbound_call_carries_the_hop_header(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER}}})
s = client.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER},
                                              "stream": True}})
sse(s)
seen = [{{k.lower(): v for k, v in h.items()}} for _, h, _ in A.hits]
print(json.dumps({{"status": [r.status_code, s.status_code], "headers": seen}}))
""",
    )

    assert out["status"] == [200, 200]
    assert len(out["headers"]) == 2
    assert all(h.get("x-routellm-hop") for h in out["headers"])


def test_endpoint_extra_headers_merge_with_the_hop_header(tmp_path):
    out = _run(
        tmp_path,
        f"""
server.CONTROLLER.endpoints.resolve("a").extra["extra_headers"] = {{"X-Tenant": "t1"}}
r = client.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER}}})
seen = {{k.lower(): v for k, v in A.hits[-1][1].items()}}
print(json.dumps({{"status": r.status_code, "headers": seen}}))
""",
    )

    assert out["status"] == 200, out
    assert out["headers"]["x-tenant"] == "t1"
    assert out["headers"].get("x-routellm-hop")


def test_the_hop_header_cannot_be_overridden_by_endpoint_config(tmp_path):
    out = _run(
        tmp_path,
        f"""
server.CONTROLLER.endpoints.resolve("a").extra["extra_headers"] = {{"x-routellm-hop": ""}}
r = client.post("/v1/chat/completions", json={{"model": "default", "messages": {_USER}}})
seen = {{k.lower(): v for k, v in A.hits[-1][1].items()}}
print(json.dumps({{"status": r.status_code, "headers": seen}}))
""",
    )

    assert out["status"] == 200, out
    assert out["headers"].get("x-routellm-hop")
