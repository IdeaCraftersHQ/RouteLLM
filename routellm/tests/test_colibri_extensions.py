"""colibri's request extensions and session-to-KV-slot pinning.

`cache_slot` and `x_colibri_ignore_leading_stop` reach an endpoint that
lists them (`kind: colibri`, or `extensions:`) and no other. A named
session is pinned to one slot of the endpoint's `kv_slots:` budget. The
upstream's `x-colibri-queue-wait-ms` comes back to the caller.

The integration cases drive the real app and the real litellm against a
fake OpenAI-shaped upstream standing in for colibri, each in its own
subprocess because the server parses `sys.argv` at import.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from routellm.endpoints import Endpoint
from routellm.kv_slots import Extensions, SlotMap
from routellm.tests.subproc import SUBPROCESS_TIMEOUT

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- units ----------------------------------------------------------------


def test_slot_is_stable_per_session():
    slots = SlotMap(4)
    first = slots.slot_for("s1")
    assert [slots.slot_for("s1") for _ in range(5)] == [first] * 5


def test_live_sessions_get_distinct_slots_up_to_the_budget():
    slots = SlotMap(3)
    assert sorted(slots.slot_for(s) for s in ("a", "b", "c")) == [0, 1, 2]


def test_slots_stay_inside_the_budget_and_evict_least_recently_used():
    slots = SlotMap(2)
    a, b = slots.slot_for("a"), slots.slot_for("b")
    slots.slot_for("a")  # "b" is now the coldest
    assert slots.slot_for("c") == b
    assert slots.slot_for("a") == a
    assert {slots.slot_for(f"s{i}") for i in range(50)} <= {0, 1}


def test_a_slot_map_refuses_a_budget_colibri_cannot_serve():
    for bad in (0, 17):
        with pytest.raises(ValueError):
            SlotMap(bad)


def test_colibri_kind_forwards_both_hints():
    ep = Endpoint(name="c", model="openai/m", api_base="http://127.0.0.1:1/v1", kind="colibri")
    assert set(ep.forwarded_extensions) == {"cache_slot", "x_colibri_ignore_leading_stop"}


def test_a_plain_endpoint_forwards_no_hint():
    ep = Endpoint(name="p", model="openai/m", api_base="http://127.0.0.1:1/v1")
    assert ep.forwarded_extensions == ()


def test_extensions_list_overrides_the_kind_preset():
    ep = Endpoint(name="c", model="openai/m", kind="colibri", extensions=["cache_slot"])
    assert ep.forwarded_extensions == ("cache_slot",)


@pytest.mark.parametrize(
    "fields",
    [
        {"extensions": ["temperature"]},  # not a hint
        {"kind": "vllm"},  # unknown preset
        {"kv_slots": 4},  # nothing forwards cache_slot
        {"kind": "colibri", "kv_slots": 0},
        {"kind": "colibri", "kv_slots": 17},
        {"extensions": ["x_colibri_ignore_leading_stop"], "kv_slots": 4},
    ],
)
def test_a_malformed_extension_config_is_refused_at_load(fields):
    with pytest.raises(ValidationError):
        Endpoint(name="c", model="openai/m", **fields)


def _colibri():
    return Endpoint(name="c", model="openai/m", api_base="http://127.0.0.1:1/v1", kind="colibri")


def test_extensions_body_carries_well_typed_hints():
    body = Extensions().body(
        _colibri(), {"cache_slot": 0, "x_colibri_ignore_leading_stop": False}, None
    )
    assert body == {"cache_slot": 0, "x_colibri_ignore_leading_stop": False}


@pytest.mark.parametrize(
    "hints",
    [
        {"cache_slot": "SECRET-PROMPT"},  # text must never ride extra_body
        {"cache_slot": True},  # bool is not a slot
        {"cache_slot": 1.5},
        {"cache_slot": -1},
        {"cache_slot": 16},
        {"x_colibri_ignore_leading_stop": "yes"},
        {"x_colibri_ignore_leading_stop": 1},
    ],
)
def test_extensions_body_refuses_a_mistyped_hint(hints):
    with pytest.raises(TypeError) as caught:
        Extensions().body(_colibri(), hints, None)
    assert "SECRET-PROMPT" not in str(caught.value)  # names the hint, never the value


def test_a_mistyped_hint_for_an_endpoint_that_drops_it_is_not_inspected():
    plain = Endpoint(name="p", model="openai/m", api_base="http://127.0.0.1:1/v1")
    assert Extensions().body(plain, {"cache_slot": "x"}, None) == {}


# ---- integration ----------------------------------------------------------

_PREAMBLE = """
import json, logging, sys
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

C = FakeUpstream("C")   # stands in for colibri
A = FakeUpstream("A")   # a plain OpenAI-compatible leaf
C.headers = {{"x-colibri-queue-wait-ms": "42"}}
open({config!r}, "w").write(json.dumps({{
    "endpoints": {{
        "c": {{"model": "openai/model-c", "api_base": C.base, "api_key_env": "FAKE_KEY",
              "kind": "colibri", "kv_slots": 3}},
        "a": {{"model": "openai/model-a", "api_base": A.base, "api_key_env": "FAKE_KEY"}},
    }},
    "tiers": {{
        "to_c": {{"router": "random", "threshold": 0.0, "strong": "c", "weak": "a"}},
        "to_a": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "c"}},
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

def post(model, headers=None, **body):
    return client.post("/v1/chat/completions", headers=headers or {{}}, json={{
        "model": model, "messages": [{{"role": "user", "content": "SECRET-PROMPT"}}], **body}})

with TestClient(server.app) as client:
    server.CONTROLLER.cache.config.enabled = False
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000
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
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.parametrize("stream", ["False", "True"])
def test_hints_reach_colibri_only(tmp_path, stream):
    out = _run(
        tmp_path,
        f"""
r1 = post("to_c", cache_slot=2, x_colibri_ignore_leading_stop=True, stream={stream})
r2 = post("to_a", cache_slot=2, x_colibri_ignore_leading_stop=True, stream={stream})
print(json.dumps({{"s": [r1.status_code, r2.status_code],
                  "c": C.hits[-1][2], "a": A.hits[-1][2], "logs": logs.lines}}))
""",
    )

    assert out["s"] == [200, 200], out
    assert out["c"]["cache_slot"] == 2
    assert out["c"]["x_colibri_ignore_leading_stop"] is True
    assert "cache_slot" not in out["a"]
    assert "x_colibri_ignore_leading_stop" not in out["a"]


@pytest.mark.slow
def test_hints_dropped_for_other_endpoints_are_logged_by_name_only(tmp_path):
    out = _run(
        tmp_path,
        """
post("to_a", cache_slot=2)
print(json.dumps({"logs": logs.lines}))
""",
    )

    dropped = [line for line in out["logs"] if "cache_slot" in line]
    assert dropped, out["logs"]
    assert not any("SECRET-PROMPT" in line for line in out["logs"])


@pytest.mark.slow
def test_a_wrong_typed_hint_is_refused_not_forwarded(tmp_path):
    out = _run(
        tmp_path,
        """
r = post("to_c", cache_slot="two")
print(json.dumps({"status": r.status_code, "hits": len(C.hits)}))
""",
    )

    assert out["status"] == 422, out
    assert out["hits"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("stream", ["False", "True"])
def test_a_session_is_pinned_to_one_slot_within_the_budget(tmp_path, stream):
    out = _run(
        tmp_path,
        f"""
seen = {{}}
for turn in range(3):
    for sid in ["s1", "s2", "s3"]:
        post("to_c", headers={{"x-routellm-session": sid}}, stream={stream})
        seen.setdefault(sid, []).append(C.hits[-1][2].get("cache_slot"))
over = []
for sid in ["s4", "s5", "s6", "s7"]:  # more live sessions than slots
    post("to_c", headers={{"x-routellm-session": sid}}, stream={stream})
    over.append(C.hits[-1][2].get("cache_slot"))
print(json.dumps({{"seen": seen, "over": over}}))
""",
    )

    seen = out["seen"]
    for sid, slots in seen.items():
        assert len(set(slots)) == 1, (sid, out)
    # While the sessions fit the budget each owns a slot.
    assert sorted(v[0] for v in seen.values()) == [0, 1, 2], out
    # Past the budget a slot is reused, never a fourth one made.
    assert set(out["over"]) <= {0, 1, 2}, out


@pytest.mark.slow
def test_metadata_session_id_pins_like_the_header(tmp_path):
    out = _run(
        tmp_path,
        """
post("to_c", headers={"x-routellm-session": "h"})
post("to_c", metadata={"session_id": "m"})
post("to_c", headers={"x-routellm-session": "h"})
slots = [h[2].get("cache_slot") for h in C.hits]
print(json.dumps(slots))
""",
    )

    assert out[0] == out[2] and out[1] != out[0], out


@pytest.mark.slow
def test_an_explicit_slot_wins_over_the_session(tmp_path):
    out = _run(
        tmp_path,
        """
post("to_c", headers={"x-routellm-session": "s1"}, cache_slot=9)
post("to_c", headers={"x-routellm-session": "s1"})
print(json.dumps([h[2].get("cache_slot") for h in C.hits]))
""",
    )

    assert out[0] == 9
    assert out[1] in (0, 1, 2)


@pytest.mark.slow
def test_no_session_and_no_slot_sends_no_slot(tmp_path):
    out = _run(
        tmp_path,
        """
post("to_c")
print(json.dumps(C.hits[-1][2]))
""",
    )

    assert "cache_slot" not in out


@pytest.mark.slow
def test_the_session_header_is_not_forwarded_upstream(tmp_path):
    out = _run(
        tmp_path,
        """
post("to_c", headers={"x-routellm-session": "s1"})
print(json.dumps({k.lower(): v for k, v in C.hits[-1][1].items()}))
""",
    )

    assert "x-routellm-session" not in out


@pytest.mark.slow
@pytest.mark.parametrize("stream", ["False", "True"])
def test_queue_wait_header_is_surfaced(tmp_path, stream):
    out = _run(
        tmp_path,
        f"""
r = post("to_c", stream={stream})
r2 = post("to_a", stream={stream})
print(json.dumps({{"c": r.headers.get("x-colibri-queue-wait-ms"),
                  "a": r2.headers.get("x-colibri-queue-wait-ms")}}))
""",
    )

    assert out["c"] == "42", out
    assert out["a"] is None


@pytest.mark.slow
def test_a_malformed_queue_wait_is_not_forwarded(tmp_path):
    out = _run(
        tmp_path,
        """
C.headers = {"x-colibri-queue-wait-ms": "soon"}
r = post("to_c")
print(json.dumps({"status": r.status_code, "h": r.headers.get("x-colibri-queue-wait-ms")}))
""",
    )

    assert out["status"] == 200 and out["h"] is None, out


def test_queue_wait_is_recorded_on_the_audit_event():
    from routellm.audit import Auditor, AuditRecord

    class Mem:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(dict(event))

        def close(self):
            pass

    sink = Mem()
    record = AuditRecord(request_id="r1", route="chat")
    record.queue_wait_ms = 42
    Auditor(sink, b"k" * 32, enforce=True).emit(record, "served", status=200)

    assert sink.events[0]["routing"]["queue_wait_ms"] == 42


def test_queue_wait_reads_only_a_plain_integer():
    from routellm.kv_slots import queue_wait_ms

    class Res:
        def __init__(self, value):
            self._hidden_params = {
                "additional_headers": {"llm_provider-x-colibri-queue-wait-ms": value}
            }

    assert queue_wait_ms(Res("42")) == 42
    for bad in ("-1", "4.2", "soon", "", "9" * 40, "4\r\nSet-Cookie: x"):
        assert queue_wait_ms(Res(bad)) is None
