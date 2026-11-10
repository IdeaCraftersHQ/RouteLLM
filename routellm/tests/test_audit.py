"""Audit events: the file sink, the HMAC key, config, and failure policy.

The server-level guarantees (one event per request, none leaking content)
live in `test_audit_server.py`.
"""

import hashlib
import hmac
import json
import logging
import os
import stat

import pytest

from routellm import audit
from routellm.audit import AuditError, Auditor, AuditRecord, AuditUnavailable, FileSink
from routellm.config import ConfigError

KEY = b"k" * 32


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


# --- file sink ---------------------------------------------------------------


def test_sink_creates_a_private_dir_and_file(tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    FileSink(path).emit({"a": 1})
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o700


def test_sink_appends_one_json_line_per_event(tmp_path):
    path = tmp_path / "audit.jsonl"
    sink = FileSink(path)
    sink.emit({"n": 1})
    sink.emit({"n": 2})
    assert lines(path) == [{"n": 1}, {"n": 2}]


def test_sink_opens_with_o_append_and_never_truncates(tmp_path, monkeypatch):
    flags = []
    real = os.open

    def spy(p, f, *a, **kw):
        flags.append(f)
        return real(p, f, *a, **kw)

    monkeypatch.setattr(audit.os, "open", spy)
    path = tmp_path / "audit.jsonl"
    path.write_text('{"old": 1}\n')
    os.chmod(path, 0o600)
    FileSink(path).emit({"n": 1})
    assert flags and all(f & os.O_APPEND for f in flags)
    assert not any(f & os.O_TRUNC for f in flags)
    assert lines(path) == [{"old": 1}, {"n": 1}]


def test_sink_reopens_by_path_so_rotation_works(tmp_path):
    path = tmp_path / "audit.jsonl"
    sink = FileSink(path)
    sink.emit({"n": 1})
    os.rename(path, tmp_path / "audit.jsonl.1")
    sink.emit({"n": 2})
    assert lines(tmp_path / "audit.jsonl.1") == [{"n": 1}]
    assert lines(path) == [{"n": 2}]
    assert mode(path) == 0o600


def test_sink_refuses_a_file_wider_than_0600(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    os.chmod(path, 0o644)
    with pytest.raises(AuditError, match="0600"):
        FileSink(path).emit({"n": 1})
    assert path.read_text() == ""


def test_sink_refuses_a_dir_wider_than_0700(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    os.chmod(d, 0o755)
    with pytest.raises(AuditError, match="0700"):
        FileSink(d / "audit.jsonl")


def test_sink_refuses_a_symlink(tmp_path):
    target = tmp_path / "t"
    target.write_text("")
    os.chmod(target, 0o600)
    link = tmp_path / "audit.jsonl"
    link.symlink_to(target)
    with pytest.raises(AuditError):
        FileSink(link).emit({"n": 1})


def test_sink_failure_raises_audit_error(tmp_path):
    path = tmp_path / "audit.jsonl"
    sink = FileSink(path)
    sink.emit({"n": 1})
    os.chmod(path, 0o400)  # not writable: the open fails
    with pytest.raises(AuditError):
        sink.emit({"n": 2})
    os.chmod(path, 0o600)
    assert lines(path) == [{"n": 1}]


def test_sink_is_a_protocol_a_custom_sink_satisfies(tmp_path):
    class Mem:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(dict(event))

        def close(self):
            pass

    assert isinstance(Mem(), audit.AuditSink)
    assert isinstance(FileSink(tmp_path / "a.jsonl"), audit.AuditSink)


# --- failure policy ----------------------------------------------------------


class Broken:
    def emit(self, event):
        raise AuditError("disk full")

    def close(self):
        pass


def rec(**kw):
    return AuditRecord(request_id="r1", **kw)


def test_enforced_failure_raises_unavailable():
    a = Auditor(Broken(), KEY, enforce=True)
    with pytest.raises(AuditUnavailable):
        a.emit(rec(), "served")


def test_unenforced_failure_logs_counts_and_continues(caplog):
    a = Auditor(Broken(), KEY, enforce=False)
    with caplog.at_level(logging.ERROR):
        a.emit(rec(), "served")
        a.emit(rec(), "refused", reason="x")
    assert a.failures == 2
    assert "audit" in caplog.text and "disk full" in caplog.text


def test_failure_log_never_carries_the_key(caplog):
    a = Auditor(Broken(), KEY, enforce=False)
    with caplog.at_level(logging.DEBUG):
        a.emit(rec(), "served")
    assert KEY.decode() not in caplog.text


# --- digest ------------------------------------------------------------------


def test_digest_is_keyed_hmac_sha256():
    a = Auditor(Broken(), KEY, enforce=False)
    want = hmac.new(KEY, b"hello", hashlib.sha256).hexdigest()
    assert a.digest("hello") == want
    assert a.digest("hello") != hashlib.sha256(b"hello").hexdigest()
    assert Auditor(Broken(), b"z" * 32, enforce=False).digest("hello") != want


# --- event shape -------------------------------------------------------------


class Mem:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(dict(event))

    def close(self):
        pass


def test_event_carries_the_decision_fields_and_nothing_else():
    sink = Mem()
    a = Auditor(sink, KEY, enforce=True)
    r = AuditRecord(
        request_id="r1",
        route="chat",
        principal="suite",
        profile="p",
        data_class="private",
        operation="ctx.step",
        prompt_hmac="ab" * 32,
        effective_class="private",
        policy="default",
        entry_tier="private",
        cache="miss",
    )
    a.emit(r, "served", status=200, tokens_in=3, tokens_out=4)
    (ev,) = sink.events
    assert set(ev) == {
        "v",
        "ts",
        "request_id",
        "phase",
        "route",
        "principal",
        "profile",
        "data_class",
        "operation",
        "effective_class",
        "policy",
        "entry_tier",
        "endpoint",
        "reach",
        "jur",
        "treatment",
        "outcome",
        "status",
        "reason",
        "cache",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "prompt_hmac",
        "routing",
    }
    assert ev["v"] == 1 and ev["phase"] == "complete" and ev["outcome"] == "served"
    assert ev["tokens_in"] == 3 and ev["status"] == 200


def test_endpoint_reach_and_jur_are_resolved():
    sink = Mem()
    a = Auditor(sink, KEY, enforce=True, resolve=lambda n: (["local"], "ca"))

    class Res:
        _hidden_params = {"routellm_endpoint": "leaf", "routellm_path": [{"tier": "t"}]}

    a.emit(rec(), "served", res=Res())
    ev = sink.events[0]
    assert (ev["endpoint"], ev["reach"], ev["jur"]) == ("leaf", ["local"], "ca")


def test_decision_detail_and_error_text_are_dropped():
    sink = Mem()
    a = Auditor(sink, KEY, enforce=True)

    class Res:
        _hidden_params = {
            "routellm_endpoint": "leaf",
            "routellm_path": [
                {
                    "tier": "t",
                    "decision": {
                        "data_class": "c",
                        "source": "model",
                        "error": "CANARY echoed",
                        "detail": {"match": "CANARY"},
                    },
                }
            ],
        }

    a.emit(rec(), "served", res=Res())
    assert "CANARY" not in json.dumps(sink.events)


def test_phases_are_decision_then_outcome_for_a_stream():
    sink = Mem()
    a = Auditor(sink, KEY, enforce=True)
    a.emit(rec(streaming=True), "served", phase="decision")
    a.emit(rec(streaming=True), "served", phase="outcome")
    assert [e["phase"] for e in sink.events] == ["decision", "outcome"]


# --- build_audit -------------------------------------------------------------


class Pol:  # any non-None policy
    pass


def cfg(**over):
    return {"audit": {"sink": "file", "path": "A", "hmac_key_env": "AK", **over}}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("AK", "s" * 32)
    return os.environ


def test_no_audit_and_no_policy_is_off(env):
    assert audit.build_audit({}, None, env=env) is None
    assert audit.build_audit(None, None, env=env) is None


def test_policy_without_audit_stops_startup(env):
    with pytest.raises(ConfigError, match="audit"):
        audit.build_audit({}, Pol(), env=env)


def test_policy_with_audit_disabled_stops_startup(env, tmp_path):
    c = cfg(path=str(tmp_path / "a"), enabled=False)
    with pytest.raises(ConfigError, match="audit"):
        audit.build_audit(c, Pol(), env=env)


def test_disabled_without_policy_is_off(env, tmp_path):
    assert audit.build_audit(cfg(path=str(tmp_path / "a"), enabled=False), None, env=env) is None


def test_policy_enforces_failures(env, tmp_path):
    a = audit.build_audit(cfg(path=str(tmp_path / "a.jsonl")), Pol(), env=env)
    assert a.enforce is True
    a = audit.build_audit(cfg(path=str(tmp_path / "b.jsonl")), None, env=env)
    assert a.enforce is False


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"sink": "kafka"}, "sink"),
        ({"path": None}, "path"),
        ({"path": ""}, "path"),
        ({"bogus": 1}, "bogus"),
    ],
)
def test_bad_audit_config_stops_startup(env, tmp_path, over, needle):
    c = cfg(path=str(tmp_path / "a"))
    c["audit"].update(over)
    with pytest.raises(ConfigError, match=needle):
        audit.build_audit(c, None, env=env)


def test_key_missing_stops_startup(tmp_path):
    c = cfg(path=str(tmp_path / "a"))
    with pytest.raises(ConfigError, match="AK"):
        audit.build_audit(c, None, env={})


def test_key_empty_stops_startup(tmp_path):
    c = cfg(path=str(tmp_path / "a"))
    with pytest.raises(ConfigError, match="AK"):
        audit.build_audit(c, None, env={"AK": ""})


def test_no_key_source_stops_startup(tmp_path):
    c = {"audit": {"sink": "file", "path": str(tmp_path / "a")}}
    with pytest.raises(ConfigError, match="hmac_key"):
        audit.build_audit(c, None, env={})


def test_both_key_sources_stop_startup(tmp_path):
    c = cfg(path=str(tmp_path / "a"), hmac_key_file=str(tmp_path / "k"))
    with pytest.raises(ConfigError, match="one of"):
        audit.build_audit(c, None, env={"AK": "x" * 32})


def test_key_file_must_be_0600(tmp_path):
    k = tmp_path / "k"
    k.write_text("s" * 32)
    os.chmod(k, 0o644)
    c = {"audit": {"sink": "file", "path": str(tmp_path / "a"), "hmac_key_file": str(k)}}
    with pytest.raises(ConfigError, match="0600"):
        audit.build_audit(c, None, env={})


def test_key_file_is_read_and_stripped(tmp_path):
    k = tmp_path / "k"
    k.write_text("topsecretkey\n")
    os.chmod(k, 0o600)
    c = {"audit": {"sink": "file", "path": str(tmp_path / "a"), "hmac_key_file": str(k)}}
    a = audit.build_audit(c, None, env={})
    assert a.digest("x") == hmac.new(b"topsecretkey", b"x", hashlib.sha256).hexdigest()


def test_empty_key_file_stops_startup(tmp_path):
    k = tmp_path / "k"
    k.write_text("\n")
    os.chmod(k, 0o600)
    c = {"audit": {"sink": "file", "path": str(tmp_path / "a"), "hmac_key_file": str(k)}}
    with pytest.raises(ConfigError, match="empty"):
        audit.build_audit(c, None, env={})


def test_the_key_never_appears_in_an_error(tmp_path):
    k = tmp_path / "k"
    k.write_text("SECRETVALUE")
    os.chmod(k, 0o644)
    c = {"audit": {"sink": "file", "path": str(tmp_path / "a"), "hmac_key_file": str(k)}}
    with pytest.raises(ConfigError) as e:
        audit.build_audit(c, None, env={})
    assert "SECRETVALUE" not in str(e.value)


def test_a_key_in_the_config_itself_is_refused(tmp_path):
    c = cfg(path=str(tmp_path / "a"), hmac_key="inline")
    with pytest.raises(ConfigError, match="hmac_key"):
        audit.build_audit(c, None, env={"AK": "x" * 32})


def test_note_fills_the_current_record_and_is_a_noop_without_one():
    audit.note(cache="hit")  # no record: nothing happens
    r = rec()
    token = audit.current_audit.set(r)
    try:
        audit.note(cache="hit", entry_tier="t", bogus=1)
    finally:
        audit.current_audit.reset(token)
    assert r.cache == "hit" and r.entry_tier == "t"
