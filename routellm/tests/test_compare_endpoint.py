"""`POST /v1/compare` and the compare header, through the real app.

Each case runs in a subprocess (the server parses `sys.argv` at import)
against local fake upstreams, and prints one JSON line.
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
import json, sys, time
from routellm.tests.fake_upstream import FakeUpstream
from routellm.tests.subproc import WAIT_DEADLINE
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

A, B, C = FakeUpstream("A"), FakeUpstream("B"), FakeUpstream("C")
tags = {tags!r}
def ep(name, up):
    spec = {{"model": "openai/model-" + name, "api_base": up.base, "api_key_env": "FAKE_KEY"}}
    if name in tags:
        spec["tags"] = tags[name]
    return spec
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{"a": ep("a", A), "b": ep("b", B), "c": ep("c", C)}},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}}}},
    "comparison": {comparison!r},
    **({{"traces": {traces!r}}} if {traces!r} else {{}}),
}}))

import routellm.openai_server as server
from fastapi.testclient import TestClient

MESSAGES = [{{"role": "user", "content": "hello"}}]


def emit(obj):
    print("@@RESULT@@" + json.dumps(obj))


def wait_for(predicate, seconds=WAIT_DEADLINE):
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000  # starved CPU is not an upstream timeout
"""


def _run(tmp_path, body, comparison=None, tags=None, traces=None):
    comparison = (
        {"enabled": True, "compare_allowed_models": ["a", "b", "c"]}
        if comparison is None
        else comparison
    )
    preamble = _PREAMBLE.format(
        config=str(tmp_path / "config.json"),
        comparison=comparison,
        tags=tags or {},
        traces=traces,
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


def test_compare_returns_every_model_side_by_side(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES})
emit({"status": r.status_code, "body": r.json(), "hops": [h[1].get("x-routellm-hop") for h in A.hits + B.hits]})
""",
    )
    assert out["status"] == 200
    results = {row["endpoint"]: row for row in out["body"]["results"]}
    assert sorted(results) == ["a", "b"]
    assert results["a"]["output"] == "hello from A"
    assert results["b"]["output"] == "hello from B"
    metrics = results["a"]["metrics"]
    assert metrics["latency_ms"] >= 0 and metrics["completion_tokens"] == 3
    assert metrics["prompt_tokens"] == 3 and metrics["ttft_ms"] is None
    assert results["a"]["error"] is None
    assert out["hops"] == ["1", "1"]


def test_compare_reports_a_failing_model_without_failing_the_request(tmp_path):
    out = _run(
        tmp_path,
        """
B.mode = "500"
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES})
emit({"status": r.status_code, "body": r.json()})
""",
    )
    assert out["status"] == 200
    results = {row["endpoint"]: row for row in out["body"]["results"]}
    assert results["a"]["error"] is None
    assert results["b"]["error"] == "upstream_error"
    assert "boom" not in json.dumps(out["body"])


def test_compare_can_measure_ttft(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES,
                                      "measure_ttft": True})
emit({"body": r.json(), "streamed": [h[2].get("stream") for h in A.hits]})
""",
    )
    assert all(row["metrics"]["ttft_ms"] is not None for row in out["body"]["results"])
    assert out["streamed"] == [True]


def test_compare_accepts_a_profile_name(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": "duo", "messages": MESSAGES})
emit({"endpoints": sorted(x["endpoint"] for x in r.json()["results"])})
""",
        comparison={"enabled": True, "profiles": {"duo": ["b", "c"]}},
    )
    assert out["endpoints"] == ["b", "c"]


def test_compare_is_absent_unless_enabled(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES})
emit({"status": r.status_code, "hits": len(A.hits)})
""",
        comparison={"enabled": False},
    )
    assert out == {"status": 404, "hits": 0}


def test_compare_endpoint_can_be_switched_off(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES})
emit({"status": r.status_code})
""",
        comparison={"enabled": True, "endpoint": False, "compare_allowed_models": ["a", "b"]},
    )
    assert out["status"] == 404


@pytest.mark.parametrize(
    "payload",
    [
        '{"models": ["a"], "messages": MESSAGES}',
        '{"models": ["a", "ghost"], "messages": MESSAGES}',
        '{"models": ["a", "b"], "messages": MESSAGES, "stream": True}',
    ],
)
def test_compare_refuses_a_bad_request(tmp_path, payload):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/compare", json={payload})
emit({{"status": r.status_code, "hits": len(A.hits) + len(B.hits)}})
""",
    )
    assert out == {"status": 400, "hits": 0}


def test_compare_for_a_local_only_tier_refuses_a_cloud_model(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "c"], "messages": MESSAGES,
                                      "tier": "default"})
emit({"status": r.status_code, "hits": len(A.hits) + len(C.hits)})
""",
        tags={"a": ["local"], "b": ["local"], "c": ["cloud"]},
    )
    assert out == {"status": 400, "hits": 0}


def test_compare_refuses_a_request_that_already_carries_the_hop_header(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/compare", json={"models": ["a", "b"], "messages": MESSAGES},
                headers={"x-routellm-hop": "1"})
emit({"status": r.status_code})
""",
    )
    assert out["status"] == 508


def test_header_shadows_a_normal_request_without_changing_the_answer(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES},
                headers={"X-RouteLLM-Compare": "b,c"})
seen = wait_for(lambda: len(B.hits) == 1 and len(C.hits) == 1)
emit({"status": r.status_code, "text": r.json()["choices"][0]["message"]["content"],
      "keys": sorted(r.json()), "seen": seen})
""",
    )
    assert out["status"] == 200 and out["seen"]
    assert out["text"] == "hello from A"
    assert "comparison" not in " ".join(out["keys"])


def test_a_failing_shadow_is_invisible_to_the_client(tmp_path):
    out = _run(
        tmp_path,
        """
B.mode = "500"
r = client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES},
                headers={"X-RouteLLM-Compare": "b"})
wait_for(lambda: len(B.hits) == 1)
emit({"status": r.status_code, "text": r.json()["choices"][0]["message"]["content"]})
""",
    )
    assert out == {"status": 200, "text": "hello from A"}


def test_header_is_ignored_when_comparison_is_disabled(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json={"model": "default", "messages": MESSAGES},
                headers={"X-RouteLLM-Compare": "b"})
time.sleep(0.5)
emit({"status": r.status_code, "b": len(B.hits)})
""",
        comparison={"enabled": False},
    )
    assert out == {"status": 200, "b": 0}


def test_streaming_primary_is_shadowed_without_streaming_the_shadow(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions",
                json={"model": "default", "messages": MESSAGES, "stream": True},
                headers={"X-RouteLLM-Compare": "b"})
lines = [l for l in r.iter_lines() if l.startswith("data: ")]
wait_for(lambda: len(B.hits) == 1)
emit({"done": lines[-1], "shadow_stream": B.hits[0][2].get("stream")})
""",
    )
    assert out["done"] == "data: [DONE]"
    assert not out["shadow_stream"]


def test_server_records_comparison_traces_when_the_config_turns_traces_on(tmp_path):
    out = _run(
        tmp_path,
        f"""
import glob
client.post("/v1/chat/completions", json={{"model": "default", "messages": MESSAGES}},
            headers={{"X-RouteLLM-Compare": "b"}})
wait_for(lambda: len(glob.glob({str(tmp_path / "traces")!r} + "/*.json")) == 2)
rows = [json.load(open(p)) for p in glob.glob({str(tmp_path / "traces")!r} + "/*.json")]
emit(sorted((r["routellm"]["endpoint"], r["routellm"]["comparison"]["role"]) for r in rows))
""",
        traces={"enabled": True, "trace_dir": str(tmp_path / "traces")},
    )
    assert out == [["a", "primary"], ["b", "shadow"]]


def test_a_mistyped_traces_key_is_refused(tmp_path):
    from routellm.config import ConfigError
    from routellm.server_config import build_quality_manager

    with pytest.raises(ConfigError, match="traces"):
        build_quality_manager({"traces": {"enabld": True}}, None)
    assert build_quality_manager({}, None) is None
    assert build_quality_manager({"traces": {"enabled": False}}, None) is None


def test_a_relative_trace_dir_resolves_against_the_config_directory(tmp_path):
    from routellm.server_config import build_quality_manager

    manager = build_quality_manager(
        {"traces": {"enabled": True, "trace_dir": "t"}}, str(tmp_path / "c.yaml")
    )
    assert manager.fine_tune_config.trace_dir == str(tmp_path / "t")


def _compare(tmp_path, models, comparison, **body):
    extra = "".join(f", {k!r}: {v!r}" for k, v in body.items())
    return _run(
        tmp_path,
        f"""
r = client.post("/v1/compare", json={{"models": {models!r}, "messages": MESSAGES{extra}}})
emit({{"status": r.status_code, "hits": len(A.hits) + len(B.hits) + len(C.hits)}})
""",
        comparison=comparison,
        tags=body_tags(body),
    )


def body_tags(body):
    return {"a": ["local"], "b": ["local"], "c": ["cloud"]} if body.get("tier") else None


def test_untiered_compare_with_an_unlisted_model_is_refused(tmp_path):
    out = _compare(tmp_path, ["a", "c"], {"enabled": True, "compare_allowed_models": ["a", "b"]})
    assert out == {"status": 400, "hits": 0}


def test_untiered_compare_with_no_allow_list_is_refused(tmp_path):
    assert _compare(tmp_path, ["a", "b"], {"enabled": True}) == {"status": 400, "hits": 0}


def test_untiered_compare_with_listed_models_is_served(tmp_path):
    out = _compare(tmp_path, ["a", "b"], {"enabled": True, "compare_allowed_models": ["a", "b"]})
    assert out == {"status": 200, "hits": 2}


def test_untiered_compare_accepts_models_from_a_profile(tmp_path):
    out = _compare(tmp_path, "duo", {"enabled": True, "profiles": {"duo": ["b", "c"]}})
    assert out == {"status": 200, "hits": 2}


def test_tiered_compare_needs_no_allow_list(tmp_path):
    out = _compare(tmp_path, ["a", "b"], {"enabled": True}, tier="default")
    assert out == {"status": 200, "hits": 2}


def test_opt_out_allows_any_endpoint_untiered(tmp_path):
    out = _compare(tmp_path, ["a", "c"], {"enabled": True, "allow_untiered_compare": True})
    assert out == {"status": 200, "hits": 2}
