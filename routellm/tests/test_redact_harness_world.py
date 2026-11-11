"""The harness's end-to-end checks: judged on staged worlds, then on the real one."""

import json

import pytest

from routellm.redact import harness
from routellm.redact.harness import end_to_end, load_canaries, run
from routellm.redact.harness_world import CONTROL

RAW = {"engine": "patterns", "client_id_patterns": [r"ACME-\d{5}"]}
CANARIES = load_canaries()
VALUE = "046 454 286"
ARTIFACTS = ("logs", "traces", "cache", "audit", "token_store", "outbound")


class Staged:
    """A `run_world` stand-in: files and outbound text the test chooses."""

    def __init__(self, tmp_path, monkeypatch, **plant):
        self.root = tmp_path
        self.plant = plant
        monkeypatch.setattr(harness, "run_world", self)

    def __call__(self, redact, plan, work, **kw):
        def f(name, text=""):
            path = self.root / name
            path.mkdir(exist_ok=True)
            (path / "x.txt").write_text(self.plant.get(name, "") + text)
            return path

        (self.root / "logs.txt").write_text(self.plant.get("logs", ""))
        last = plan["leak"]["messages"][-1]["content"]
        sent = {name: ["{}"] for name in plan}
        sent["control"] = [json.dumps(CONTROL)]
        sent["leak"] = [json.dumps({"m": "[[SIN_1]] [[PERSON_1]]"})]
        sent["link_a"], sent["link_b"] = ["[[DATE_OF_BIRTH_1]]"], ["[[DATE_OF_BIRTH_2]]"]
        sent.update(self.plant.get("sent", {}))
        if "outbound" in self.plant:
            sent["leak_stream"] = [self.plant["outbound"]]
        reply = {"status": 200, "text": self.plant.get("reply", f"echo: {last}")}
        return {
            "scenarios": sent,
            "replies": {"leak": reply, "leak_stream": reply},
            "files": {
                "logs": self.root / "logs.txt",
                "traces": f("traces", self.plant.get("trace_extra", CONTROL)),
                "cache": f("cache", CONTROL),
                "audit": f("audit"),
                "token_store": f("token_store"),
            },
            "stdout": self.plant.get("stdout", ""),
            "stderr": self.plant.get("stderr", ""),
        }


def staged(tmp_path, monkeypatch, redact=RAW, **plant):
    Staged(tmp_path, monkeypatch, **plant)
    return end_to_end(redact, "patterns", CANARIES, tmp_path / "w")


def test_a_clean_staged_run_passes_every_check(tmp_path, monkeypatch):
    got = staged(tmp_path, monkeypatch)
    assert all(c["passed"] for c in got.values()), got


@pytest.mark.parametrize("where", ["traces", "cache", "audit", "outbound", "stdout", "stderr"])
def test_a_value_in_any_artifact_fails_the_leak_check(tmp_path, monkeypatch, where):
    plant = {where: VALUE}
    if where == "traces":
        plant = {"trace_extra": f"{CONTROL} {VALUE}"}
    got = staged(tmp_path, monkeypatch, **plant)["leak"]
    assert got["passed"] is False and got["found"], got


def test_a_value_in_the_cache_or_audit_files_fails_the_leak_check(tmp_path, monkeypatch):
    class Planted(Staged):
        def __call__(self, *a, **kw):
            world = super().__call__(*a, **kw)
            for name in self.plant["into"]:
                (world["files"][name] / "x.txt").write_text(CONTROL + VALUE)
            return world

    for names in (["cache"], ["audit"], ["traces"]):
        Planted(tmp_path, monkeypatch, into=names)
        got = end_to_end(RAW, "patterns", CANARIES, tmp_path / "w")["leak"]
        assert got["passed"] is False and set(got["found"]) == {names[0]}, got


def test_a_value_in_the_log_file_fails_the_leak_check(tmp_path, monkeypatch):
    class Logged(Staged):
        def __call__(self, *a, **kw):
            world = super().__call__(*a, **kw)
            world["files"]["logs"].write_text(f"DEBUG body {VALUE}")
            return world

    Logged(tmp_path, monkeypatch)
    got = end_to_end(RAW, "patterns", CANARIES, tmp_path / "w")["leak"]
    assert got["passed"] is False and set(got["found"]) == {"logs"}


def test_the_plaintext_token_store_is_not_scanned_but_an_encrypted_one_is(tmp_path, monkeypatch):
    class Stored(Staged):
        def __call__(self, *a, **kw):
            world = super().__call__(*a, **kw)
            (world["files"]["token_store"] / "x.txt").write_text(VALUE)
            return world

    Stored(tmp_path, monkeypatch)
    plain = end_to_end(RAW, "patterns", CANARIES, tmp_path / "w")["leak"]
    assert plain["passed"] and "token_store" not in plain["scanned"]
    sealed = {**RAW, "store": {"encryption": "aes-gcm"}}
    got = end_to_end(sealed, "patterns", CANARIES, tmp_path / "w2")["leak"]
    assert got["passed"] is False and set(got["found"]) == {"token_store"}


def test_a_scan_that_cannot_see_the_control_is_blind_and_fails(tmp_path, monkeypatch):
    got = staged(tmp_path, monkeypatch, trace_extra="")["leak"]
    assert got["passed"] is False and got["blind"] == ["traces"]


def test_a_reply_the_client_cannot_read_fails(tmp_path, monkeypatch):
    got = staged(tmp_path, monkeypatch, reply="echo: [[PERSON_1]]")["leak"]
    assert got["passed"] is False and got["client_restored"] is False


def test_a_dob_and_postal_pair_in_clear_fails_reidentification(tmp_path, monkeypatch):
    sent = {"reid_0": ["Cliente née le 12 mars 1985, code postal H2X 3K8"]}
    got = staged(tmp_path, monkeypatch, sent=sent)["reidentification"]
    assert got["passed"] is False and got["flagged"] == 1


def test_a_shared_placeholder_fails_linkability(tmp_path, monkeypatch):
    sent = {"link_a": ["[[DATE_OF_BIRTH_1]]"], "link_b": ["[[DATE_OF_BIRTH_1]]"]}
    got = staged(tmp_path, monkeypatch, sent=sent)["linkability"]
    assert got["passed"] is False and got["shared"] == ["[[DATE_OF_BIRTH_1]]"]


def test_no_placeholders_at_all_fails_linkability(tmp_path, monkeypatch):
    sent = {"link_a": ["raw"], "link_b": ["raw"]}
    assert staged(tmp_path, monkeypatch, sent=sent)["linkability"]["passed"] is False


# --- the real world ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return run(RAW, work=tmp_path_factory.mktemp("harness-world"))


@pytest.mark.slow
def test_the_pattern_engine_passes_against_the_real_server(report):
    assert report["passed"] is True, report["failed"]
    assert set(report["checks"]) == {"leak", "reidentification", "linkability"}
    leak = report["checks"]["leak"]
    assert leak["found"] == {} and leak["blind"] == [] and leak["client_restored"]
    assert {"outbound", "logs", "traces", "cache", "audit"} <= set(leak["scanned"])
    assert "token_store" not in leak["scanned"]


@pytest.mark.slow
def test_the_report_is_stamped_and_opens_the_gate(report, tmp_path):
    from routellm.redact.gate import config_digest, verify_report

    assert report["digest"] == config_digest(RAW)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    verify_report(str(path), RAW)


@pytest.mark.slow
def test_the_store_is_scanned_under_aes_gcm(tmp_path):
    pytest.importorskip("cryptography")
    key = tmp_path / "k"
    import base64
    import os

    key.write_text(base64.b64encode(os.urandom(32)).decode())
    key.chmod(0o600)
    sealed = {**RAW, "store": {"encryption": "aes-gcm", "key_file": str(key)}}
    got = run(sealed, work=tmp_path / "w")
    assert got["passed"] is True and "token_store" in got["checks"]["leak"]["scanned"]


def test_the_cli_writes_the_report_and_exits_by_the_verdict(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("redact:\n  engine: patterns\n  client_id_patterns: ['ACME-\\d{5}']\n")
    out = tmp_path / "sub" / "r.json"
    good = {k: {"passed": True} for k in ("leak", "reidentification", "linkability")}
    monkeypatch.setattr(harness, "end_to_end", lambda *a, **k: good)
    assert harness.main(["--config", str(cfg), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setattr(harness, "end_to_end", lambda *a, **k: {**good, "leak": {"passed": False}})
    assert harness.main(["--config", str(cfg), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["failed"] == ["leak"]
    cfg.write_text("other: 1\n")
    assert harness.main(["--config", str(cfg), "--out", str(out)]) == 2


def test_a_dropped_recognizer_fails_the_cli_report(tmp_path, monkeypatch):
    from routellm.redact import recognizers

    monkeypatch.setattr(recognizers, "find_sin", lambda text: [])
    good = {k: {"passed": True} for k in ("leak", "reidentification", "linkability")}
    monkeypatch.setattr(harness, "end_to_end", lambda *a, **k: good)
    report = run(RAW)
    assert report["passed"] is False and report["failed"] == ["sin"]


@pytest.mark.redact
@pytest.mark.slow
def test_the_presidio_engine_meets_its_table(tmp_path):
    got = run({"engine": "presidio", "client_id_patterns": [r"ACME-\d{5}"]}, work=tmp_path / "w")
    assert got["passed"] is True, got["failed"]
    assert got["categories"]["person"]["recall"] >= 0.9
