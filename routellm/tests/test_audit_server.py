"""Audit events through the real app: one per request, never any content.

The app parses `sys.argv` at import, so each case runs in a subprocess
against local fake upstreams and prints one JSON line.
"""

import hashlib
import hmac
import json
import subprocess
import sys
from pathlib import Path

import pytest

from routellm.tests.audit_fixture import HMAC_KEY, audit_section
from routellm.tests.subproc import SUBPROCESS_TIMEOUT

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_A, KEY_B = "sk-test-aaaa", "sk-test-bbbb"
CANARY = "CANARY-7f3a91"

_PREAMBLE = """
import json, os, sys
from routellm.tests.fake_upstream import FakeUpstream
os.environ["ROUTELLM_SYSTEM_CONFIG"] = {config!r}
sys.argv = ["x", "--routers", "random"]

A, B = FakeUpstream("A"), FakeUpstream("B")
def ep(name, up):
    return {{"model": "openai/model-" + name, "api_base": up.base, "api_key_env": "FAKE_KEY",
            "tags": ["local", "jur:ca"], "extra": {{"max_retries": 0}}}}
cfg = {{
    "endpoints": {{"a": ep("a", A), "b": ep("b", B)}},
    "tiers": {{
        "default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}},
        "extra": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}},
    }},
    "comparison": {{"enabled": True, "compare_allowed_models": ["a", "b"]}},
}}
cfg.update({extra!r})
open({config!r}, "w").write(json.dumps(cfg))

import routellm.openai_server as server
from fastapi.testclient import TestClient

AUDIT = {audit_path!r}
PROMPT = "{canary} please summarise"
MESSAGES = [{{"role": "user", "content": PROMPT}}]
BODY = {{"model": "default", "messages": MESSAGES, "user": "{canary}-user"}}
def H(key, **h):
    return {{"Authorization": "Bearer " + key, **h}}
def events():
    try:
        return [json.loads(x) for x in open(AUDIT).read().splitlines()]
    except FileNotFoundError:
        return []
def emit(obj):
    print("@@RESULT@@" + json.dumps(obj))

with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000  # starved CPU is not an upstream timeout
"""


def sha(key):
    return hashlib.sha256(key.encode()).hexdigest()


def prompt_hmac(text):
    return hmac.new(HMAC_KEY.encode(), text.encode(), hashlib.sha256).hexdigest()


def _run(
    tmp_path,
    body,
    *,
    policy=True,
    auth=True,
    audit=True,
    policy_extra=None,
    config_extra=None,
    expect_ok=True,
    keys_text=None,
):
    extra = {}
    if auth:
        keys = tmp_path / "keys.yaml"
        keys.write_text(
            keys_text
            or f"""
keys:
  - key_sha256: "{sha(KEY_A)}"
    principal: team-a
    profile: prof-a
  - key_sha256: "{sha(KEY_B)}"
    principal: team-b
"""
        )
        extra["auth"] = {"keys_file": str(keys)}
    if policy:
        pol = {
            "classes": {
                "open": {
                    "treatment": "raw",
                    "cache": "exact",
                    "reach": ["local", "cloud"],
                    "tiers": ["default"],
                },
                "internal": {"treatment": "raw", "reach": ["local"], "tiers": ["default"]},
            },
            "principals": {
                "team-a": {"classes": ["open", "internal"]},
                "team-b": {"classes": ["open"]},
            },
            **(policy_extra or {}),
        }
        pfile = tmp_path / "policy.json"
        pfile.write_text(json.dumps(pol))
        extra["policy"] = str(pfile)
    audit_path = tmp_path / "audit" / "audit.jsonl"
    if audit:
        extra["audit"] = audit_section(tmp_path)
    extra["cache"] = {"enabled": True, "db_path": str(tmp_path / "cache.db")}
    extra.update(config_extra or {})
    preamble = _PREAMBLE.format(
        config=str(tmp_path / "config.json"),
        extra=extra,
        audit_path=str(audit_path),
        canary=CANARY,
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
    if not expect_ok:
        return result
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    return json.loads(line.split("@@RESULT@@", 1)[1])


def by_status(out):
    return {e["request_id"]: e for e in out["events"]}


@pytest.mark.slow
def test_served_request_writes_one_event_with_the_decision(tmp_path):
    out = _run(
        tmp_path,
        f"""
h = H("{KEY_A}", **{{"X-Data-Class": "open", "X-Operation": "ctx.summarize"}})
r1 = client.post("/v1/chat/completions", json=BODY, headers=h)
r2 = client.post("/v1/chat/completions", json=BODY, headers=h)
emit({{"s": [r1.status_code, r2.status_code], "events": events()}})
""",
    )
    assert out["s"] == [200, 200]
    first, second = out["events"]
    assert first["outcome"] == "served" and first["status"] == 200 and first["phase"] == "complete"
    assert first["principal"] == "team-a" and first["profile"] == "prof-a"
    assert first["data_class"] == "open" and first["effective_class"] == "open"
    assert first["operation"] == "ctx.summarize"
    assert first["entry_tier"] == "default" and first["route"] == "chat"
    assert first["endpoint"] in ("a", "b")
    assert first["reach"] == ["local"] and first["jur"] == "ca"
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert (first["tokens_in"], first["tokens_out"]) == (3, 3)
    assert first["latency_ms"] >= 0 and first["reason"] is None
    assert first["prompt_hmac"] == prompt_hmac(f"{CANARY} please summarise")
    assert first["request_id"] != second["request_id"]


@pytest.mark.slow
def test_every_refusal_path_writes_an_event(tmp_path):
    out = _run(
        tmp_path,
        f"""
def post(key, headers=None, body=None, url="/v1/chat/completions"):
    h = H(key, **(headers or {{}})) if key else (headers or {{}})
    return client.post(url, json=BODY if body is None else body, headers=h).status_code
cls = {{"X-Data-Class": "open"}}
sent = {{
  "unauthorized": post("sk-nope", cls),
  "unknown_class": post("{KEY_A}", {{"X-Data-Class": "nope"}}),
  "forbidden_class": post("{KEY_B}", {{"X-Data-Class": "internal"}}),
  "missing_class": post("{KEY_A}"),
  "tier_denied": post("{KEY_A}", cls, {{**BODY, "model": "extra"}}),
  "routing": post("{KEY_A}", cls, {{**BODY, "model": "no-such-tier:router-random-0.5"}}),
  "invalid_body": post("{KEY_A}", cls, {{"model": "default"}}),
  "loop": post("{KEY_A}", {{**cls, "X-RouteLLM-Hop": "1"}}),
  "compare_stream": post("{KEY_A}", cls, {{**BODY, "models": ["a"], "stream": True}}, "/v1/compare"),
}}
emit({{"sent": sent, "events": events(), "hits": len(A.hits + B.hits)}})
""",
    )
    assert out["sent"] == {
        "unauthorized": 401,
        "unknown_class": 400,
        "forbidden_class": 403,
        "missing_class": 400,
        "tier_denied": 403,
        "routing": 400,
        "invalid_body": 422,
        "loop": 508,
        "compare_stream": 400,
    }
    assert out["hits"] == 0
    events = out["events"]
    assert len(events) == len(out["sent"])
    assert all(e["outcome"] == "refused" and e["endpoint"] is None for e in events)
    assert [e["status"] for e in events] == list(out["sent"].values())
    assert all(e["reason"] for e in events)
    assert [e["reason"] for e in events][:1] == ["unauthorized"]
    assert {e["reason"] for e in events} >= {"policy_denied", "loop_detected", "invalid_request"}
    assert events[0]["principal"] is None
    assert events[1]["principal"] == "team-a"
    assert events[4]["entry_tier"] == "extra" and events[4]["data_class"] == "open"
    assert events[8]["route"] == "compare"


@pytest.mark.slow
def test_provider_failure_and_fallback_walk(tmp_path):
    out = _run(
        tmp_path,
        f"""
cls = H("{KEY_A}", **{{"X-Data-Class": "open"}})
A.mode = "500"
r1 = client.post("/v1/chat/completions", json=BODY, headers=cls)
A.mode = "500"; B.mode = "500"
r2 = client.post("/v1/chat/completions", json={{**BODY, "user": "x2", "messages":
    [{{"role": "user", "content": "other"}}]}}, headers=cls)
emit({{"s": [r1.status_code, r2.status_code], "events": events()}})
""",
    )
    assert out["s"] == [200, 502]
    served, failed = out["events"]
    # The event names the endpoint that answered, not the first pick.
    assert served["outcome"] == "served" and served["endpoint"] == "b"
    assert failed["outcome"] == "failed" and failed["status"] == 502
    assert failed["endpoint"] is None and failed["reason"] == "provider_error"
    assert failed["routing"]["walk"], failed


@pytest.mark.slow
def test_compare_writes_an_event_naming_its_targets(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/compare", json={{**BODY, "models": ["a", "b"], "tier": "default"}},
    headers=H("{KEY_A}", **{{"X-Data-Class": "open"}}))
emit({{"s": r.status_code, "events": events()}})
""",
    )
    assert out["s"] == 200
    (ev,) = out["events"]
    assert ev["route"] == "compare" and ev["outcome"] == "served"
    assert ev["routing"]["targets"] == ["a", "b"]


@pytest.mark.slow
def test_no_content_or_secret_reaches_the_file(tmp_path):
    out = _run(
        tmp_path,
        f"""
h = H("{KEY_A}", **{{"X-Data-Class": "open"}})
body = {{**BODY, "metadata": {{"note": "{CANARY}-meta"}}}}
client.post("/v1/chat/completions", json=body, headers=h)
client.post("/v1/chat/completions", json={{**body, "stream": True}}, headers=h)
client.post("/v1/chat/completions", json=body, headers=H("{KEY_B}", **{{"X-Data-Class": "internal"}}))
A.mode = "500"; B.mode = "500"
client.post("/v1/chat/completions", json=body, headers=h)
emit({{"raw": open(AUDIT).read(), "n": len(events())}})
""",
    )
    assert out["n"] == 5  # served, stream decision + outcome, refused, failed
    raw = out["raw"]
    for secret in (CANARY, KEY_A, KEY_B, sha(KEY_A), HMAC_KEY, "hello from"):
        assert secret not in raw, secret
    assert "Authorization" not in raw and "authorization" not in raw


@pytest.mark.slow
def test_stream_writes_the_decision_before_the_first_byte_then_the_outcome(tmp_path):
    out = _run(
        tmp_path,
        f"""
h = H("{KEY_A}", **{{"X-Data-Class": "open"}})
seen = []
with client.stream("POST", "/v1/chat/completions", json={{**BODY, "stream": True}}, headers=h) as r:
    first = True
    for line in r.iter_lines():
        if line and first:
            seen.append([e["phase"] for e in events()])
            first = False
final = events()
emit({{"seen": seen, "events": final}})
""",
    )
    assert out["seen"] == [["decision"]]
    decision, outcome = out["events"]
    assert (decision["phase"], outcome["phase"]) == ("decision", "outcome")
    assert decision["request_id"] == outcome["request_id"]
    assert decision["endpoint"] in ("a", "b") and decision["cache"] == "bypass"
    assert outcome["outcome"] == "served"


@pytest.mark.slow
def test_stream_that_fails_midway_records_a_failed_outcome(tmp_path):
    out = _run(
        tmp_path,
        f"""
A.mode = "die_after"; B.mode = "die_after"
h = H("{KEY_A}", **{{"X-Data-Class": "open"}})
with client.stream("POST", "/v1/chat/completions", json={{**BODY, "stream": True}}, headers=h) as r:
    list(r.iter_lines())
emit({{"events": events()}})
""",
    )
    decision, outcome = out["events"]
    assert decision["phase"] == "decision"
    assert outcome["phase"] == "outcome" and outcome["outcome"] == "failed"
    assert outcome["reason"] == "stream_error"


@pytest.mark.slow
def test_policy_loaded_sink_failure_refuses_503_and_withholds_the_response(tmp_path):
    out = _run(
        tmp_path,
        f"""
os.chmod(AUDIT, 0o400)  # the sink can no longer write
h = H("{KEY_A}", **{{"X-Data-Class": "open"}})
r = client.post("/v1/chat/completions", json=BODY, headers=h)
rs = client.post("/v1/chat/completions", json={{**BODY, "stream": True}}, headers=h)
rr = client.post("/v1/chat/completions", json=BODY, headers=H("{KEY_B}", **{{"X-Data-Class": "internal"}}))
os.chmod(AUDIT, 0o600)
emit({{"s": [r.status_code, rs.status_code, rr.status_code],
      "bodies": [r.text, rs.text, rr.text], "events": events()}})
""",
    )
    assert out["s"] == [503, 503, 503]
    assert out["events"] == []
    assert all("hello from" not in b for b in out["bodies"])
    assert all(CANARY not in b for b in out["bodies"])


@pytest.mark.slow
def test_no_policy_sink_failure_logs_and_serves(tmp_path):
    out = _run(
        tmp_path,
        f"""
os.chmod(AUDIT, 0o400)
r = client.post("/v1/chat/completions", json=BODY, headers=H("{KEY_A}"))
os.chmod(AUDIT, 0o600)
emit({{"s": r.status_code, "failures": server.AUDIT.failures, "events": events()}})
""",
        policy=False,
    )
    assert out["s"] == 200 and out["failures"] == 1 and out["events"] == []


@pytest.mark.slow
def test_no_policy_audit_writes_events_without_a_principal(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json=BODY)
emit({"s": r.status_code, "events": events()})
""",
        policy=False,
        auth=False,
    )
    (ev,) = out["events"]
    assert out["s"] == 200 and ev["principal"] is None and ev["effective_class"] is None


@pytest.mark.slow
def test_audit_off_without_policy_writes_nothing(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json=BODY)
emit({"s": r.status_code, "exists": os.path.exists(AUDIT)})
""",
        policy=False,
        auth=False,
        audit=False,
    )
    assert out == {"s": 200, "exists": False}


@pytest.mark.slow
@pytest.mark.parametrize("how", ["absent", "disabled"])
def test_policy_with_audit_off_stops_startup(tmp_path, how):
    out = _run(
        tmp_path,
        "emit({})",
        audit=False,
        config_extra={"audit": {"enabled": False}} if how == "disabled" else None,
        expect_ok=False,
    )
    assert out.returncode != 0 and "audit" in out.stderr


@pytest.mark.slow
def test_audit_sink_survives_logrotate_style_rename(tmp_path):
    out = _run(
        tmp_path,
        f"""
h = H("{KEY_A}", **{{"X-Data-Class": "open"}})
client.post("/v1/chat/completions", json=BODY, headers=h)
os.rename(AUDIT, AUDIT + ".1")
client.post("/v1/chat/completions", json={{**BODY, "messages": [{{"role": "user", "content": "2"}}]}}, headers=h)
emit({{"mode": oct(os.stat(AUDIT).st_mode & 0o777), "n": len(events())}})
""",
    )
    assert out == {"mode": "0o600", "n": 1}
