"""The harness's end-to-end world: the redact treatment behind a recording provider.

One subprocess runs the real server (policy, auth, audit, cache, traces,
the redact transform built from the `redact:` section under test) against
a fake provider on loopback that records every request and echoes the
last message. The harness drives it as a client would and hands back what
was sent upstream and where the run left files; `harness_checks` judges
them. The control request goes through a raw class, where traces and cache
do hold the text: the scan must find it there, or it is blind.

Nothing here leaves the loopback interface.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from routellm.redact.gate import REPORT_SCHEMA, REQUIRED_CHECKS, config_digest

__all__ = ["CONTROL", "build_messages", "run_world"]

CONTROL = "CTRL-CANARY-7731 zq"
KEY = "sk-harness-0000"
_TIMEOUT = 300


def build_messages(texts: list[str]) -> list[dict]:
    """One conversation that spreads `texts` over every place text travels.

    System, a multimodal text part, an assistant tool call's JSON
    arguments, a tool result, and the last user message (which the
    provider echoes back, so the client's restored reply can be checked).
    """
    n = len(texts)
    third = max(1, n // 4)
    system, parts, args, tool = (
        texts[:third],
        texts[third : 2 * third],
        texts[2 * third : 3 * third],
        texts[3 * third : n - 1] or texts[:1],
    )
    return [
        {"role": "system", "content": " ".join(system)},
        {"role": "user", "content": [{"type": "text", "text": t} for t in parts]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"notes": args})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": " ".join(tool)},
        {"role": "user", "content": texts[-1]},
    ]


_SCRIPT = r"""
import json, logging, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

work = {work!r}
hits = []

def last_text(body):
    c = (body.get("messages") or [{{}}])[-1].get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _chunk(self, data):
        self.wfile.write(f"{{len(data):x}}\r\n".encode() + data + b"\r\n"); self.wfile.flush()
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{{}}")
        hits.append(json.dumps(body, ensure_ascii=False))
        reply = "echo: " + last_text(body)
        base = {{"id": "c1", "created": 1, "model": body.get("model", "m")}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            pieces = [reply[i:i + 4] for i in range(0, len(reply), 4)]
            for i, p in enumerate(pieces):
                self._chunk(("data: " + json.dumps({{**base, "object": "chat.completion.chunk",
                    "choices": [{{"index": 0, "delta": {{"content": p}}, "finish_reason": None}}]}})
                    + "\n\n").encode())
            self._chunk(("data: " + json.dumps({{**base, "object": "chat.completion.chunk",
                "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "stop"}}],
                "usage": {{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}}})
                + "\n\ndata: [DONE]\n\n").encode())
            self.wfile.write(b"0\r\n\r\n")
            return
        data = json.dumps({{**base, "object": "chat.completion",
            "choices": [{{"index": 0, "finish_reason": "stop",
                         "message": {{"role": "assistant", "content": reply}}}}],
            "usage": {{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{{httpd.server_address[1]}}/v1"

cfg = json.loads(open(os.path.join(work, "world.json")).read())
cfg["endpoints"]["cloud"]["api_base"] = base
open(os.path.join(work, "config.json"), "w").write(json.dumps(cfg))
os.environ["ROUTELLM_SYSTEM_CONFIG"] = os.path.join(work, "config.json")
sys.argv = ["x", "--routers", "random"]

logs = []
class Keep(logging.Handler):
    def emit(self, record):
        logs.append(self.format(record))
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(Keep())
logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

import routellm.openai_server as server
from fastapi.testclient import TestClient

plan = json.loads(open(os.path.join(work, "plan.json")).read())
out = {{"scenarios": {{}}, "replies": {{}}}}
with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    for name, step in plan.items():
        before = len(hits)
        headers = {{"Authorization": "Bearer " + {key!r}, "X-Data-Class": step["class"]}}
        body = {{"model": "default", "messages": step["messages"]}}
        if step.get("stream"):
            body["stream"] = True
            with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                status = r.status_code
                events = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
            text = ""
            for e in events:
                if e != "[DONE]":
                    for ch in json.loads(e).get("choices") or []:
                        text += ch.get("delta", {{}}).get("content") or ""
        else:
            r = client.post("/v1/chat/completions", json=body, headers=headers)
            status = r.status_code
            text = r.json()["choices"][0]["message"]["content"] if status == 200 else ""
        out["scenarios"][name] = hits[before:]
        out["replies"][name] = {{"status": status, "text": text}}
open(os.path.join(work, "result.json"), "w").write(json.dumps(out))
open(os.path.join(work, "logs.txt"), "w").write("\n".join(logs))
"""


def _world_config(work: Path, redact: dict) -> dict:
    keyfile = work / "audit.key"
    keyfile.write_text("harness-audit-hmac-key-0123456789")
    os.chmod(keyfile, 0o600)
    (work / "keys.yaml").write_text(
        f'keys:\n  - key_sha256: "{hashlib.sha256(KEY.encode()).hexdigest()}"\n'
        "    principal: harness\n"
    )
    (work / "policy.json").write_text(
        json.dumps(
            {
                "classes": {
                    "pii": {
                        "treatment": "redact",
                        "reach": ["local", "cloud"],
                        "tiers": ["default"],
                    },
                    "open": {
                        "treatment": "raw",
                        "cache": "exact",
                        "reach": ["local", "cloud"],
                        "tiers": ["default"],
                    },
                },
                "principals": {"harness": {"classes": ["pii", "open"]}},
            }
        )
    )
    store = {**(redact.get("store") or {}), "path": str(work / "tokens.db")}
    section = {k: v for k, v in redact.items() if k != "harness_report"}
    # The server under measurement must load to be measured, so its gate is
    # met with a report stamped for the section and good for this run only.
    (work / "world-report.json").write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "digest": config_digest(section),
                "passed": True,
                "categories": {"world": {"passed": True}},
                "checks": {name: {"passed": True} for name in REQUIRED_CHECKS},
            }
        )
    )
    return {
        "endpoints": {
            "cloud": {
                "model": "openai/harness-cloud",
                "api_base": "",
                "api_key_env": "FAKE_KEY",
                "tags": ["cloud", "jur:ca"],
                "extra": {"max_retries": 0},
            }
        },
        "tiers": {
            "default": {"router": "random", "threshold": 0.0, "strong": "cloud", "weak": "cloud"}
        },
        "policy": str(work / "policy.json"),
        "auth": {"keys_file": str(work / "keys.yaml")},
        "audit": {
            "sink": "file",
            "path": str(work / "audit" / "audit.jsonl"),
            "hmac_key_file": str(keyfile),
        },
        "cache": {"enabled": True, "db_path": str(work / "cache.db")},
        "traces": {"enabled": True, "trace_dir": str(work / "traces"), "capture_content": True},
        "redact": {**section, "store": store, "harness_report": str(work / "world-report.json")},
    }


def run_world(
    redact: dict, plan: dict[str, dict[str, Any]], work: Path, *, timeout: float = _TIMEOUT
) -> dict[str, Any]:
    """Run `plan` (`{name: {class, messages, stream?}}`) through the server under `work`.

    Returns `{scenarios: {name: [outbound request JSON, ...]}, replies:
    {name: {status, text}}, files: {outbound?, logs, traces, cache, audit,
    token_store: path}, stdout, stderr}`. `redact` is the section under
    test; its store path is moved under `work` and its `harness_report`
    replaced by a throwaway stamp (the gate is for the server a policy loads).
    """
    work.mkdir(parents=True, exist_ok=True)
    (work / "world.json").write_text(json.dumps(_world_config(work, redact)))
    (work / "plan.json").write_text(json.dumps(plan))
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(work=str(work), key=KEY)],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[2]), *(p for p in sys.path if p)]
            ),
            "XDG_CONFIG_HOME": str(work / "xdg"),
            "XDG_STATE_HOME": str(work / "state"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "FAKE_KEY": "sk-fake",
        },
    )
    if proc.returncode != 0 or not (work / "result.json").exists():
        raise RuntimeError(f"harness world failed: {proc.stderr[-2000:]}")
    result = json.loads((work / "result.json").read_text())
    result["files"] = {
        "logs": work / "logs.txt",
        "traces": work / "traces",
        "cache": work / "cache.db",
        "audit": work / "audit",
        "token_store": work / "tokens.db",
    }
    result["stdout"], result["stderr"] = proc.stdout, proc.stderr
    return result
