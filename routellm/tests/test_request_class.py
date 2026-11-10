"""Request data_class and operation: carriage, validation, never forwarded.

Unit cases call `routellm.request_class.declare`. Server cases run the
real app in a subprocess (it parses `sys.argv` at import) against local
fake upstreams and print one JSON line.
"""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from routellm.auth import Principal
from routellm.policy import PolicyError, parse_policy
from routellm.request_class import (
    RequestClass,
    RequestClassRefusal,
    check_policy_principals,
    declare,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

POLICY = parse_policy(
    {
        "default_class": None,
        "classes": {
            "open": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
            "internal": {"treatment": "block", "reach": ["local"], "tiers": ["default"]},
            "sealed": {"treatment": "vault"},
        },
        "principals": {
            "team-a": {"classes": ["open", "internal"]},
            "team-b": {"classes": ["open"]},
        },
    },
    "p.yaml",
)
A = Principal("team-a")
B = Principal("team-b")


def refusal(*args, **kw):
    with pytest.raises(RequestClassRefusal) as exc:
        declare(*args, **kw)
    return exc.value


# --- policy mode -----------------------------------------------------------


def test_header_carries_class_and_operation():
    rc = declare({"x-data-class": "open", "x-operation": "ctx.summarize"}, None, A, POLICY)
    assert rc == RequestClass("open", "ctx.summarize")


def test_metadata_carries_class_and_operation():
    meta = {"data_class": "open", "operation": "ctx.summarize", "other": 1}
    assert declare({}, meta, A, POLICY) == RequestClass("open", "ctx.summarize")


def test_same_value_in_header_and_metadata_is_fine():
    assert declare({"x-data-class": "open"}, {"data_class": "open"}, A, POLICY).data_class == "open"


def test_header_and_metadata_conflict_is_400():
    e = refusal({"x-data-class": "open"}, {"data_class": "internal"}, A, POLICY)
    assert e.status == 400
    e = refusal({"x-operation": "a.b"}, {"operation": "a.c"}, A, POLICY)
    assert e.status == 400


def test_unknown_class_is_400_naming_it():
    e = refusal({"x-data-class": "nope"}, None, A, POLICY)
    assert e.status == 400 and "nope" in str(e)


def test_class_not_allowed_for_principal_is_403():
    assert refusal({"x-data-class": "internal"}, None, B, POLICY).status == 403


def test_unknown_principal_and_no_principal_are_403():
    assert refusal({"x-data-class": "open"}, None, Principal("stranger"), POLICY).status == 403
    assert refusal({"x-data-class": "open"}, None, None, POLICY).status == 403


def test_policy_wins_over_keys_file_allowed_classes():
    # The keys file grants `internal`; the policy does not: refused.
    p = Principal("team-b", allowed_classes=("open", "internal"))
    assert refusal({"x-data-class": "internal"}, None, p, POLICY).status == 403
    # The keys file grants nothing; the policy does: allowed.
    assert declare({"x-data-class": "open"}, None, Principal("team-b"), POLICY).data_class == "open"


def test_missing_class_with_null_default_is_400():
    assert refusal({}, None, A, POLICY).status == 400


def test_missing_class_uses_default_class():
    raw = {
        "default_class": "open",
        "classes": POLICY.model_dump()["classes"],
        "principals": {"team-a": {"classes": ["open"]}},
    }
    pol = parse_policy(raw, "p.yaml")
    assert declare({}, None, A, pol).data_class == "open"


def test_vaulted_class_is_403():
    p = Principal("team-a")
    pol = parse_policy(
        {
            **POLICY.model_dump(exclude={"source"}),
            "principals": {"team-a": {"classes": ["open", "sealed"]}},
        },
        "p.yaml",
    )
    assert refusal({"x-data-class": "sealed"}, None, p, pol).status == 403


@pytest.mark.parametrize("op", ["", "a" * 129, "a b", "a\nb", "é.x"])
def test_malformed_operation_is_400(op):
    assert refusal({"x-operation": op}, {"data_class": "open"}, A, POLICY).status == 400
    assert refusal({}, {"data_class": "open", "operation": op}, A, POLICY).status == 400


def test_operation_of_128_chars_is_kept():
    op = "a" * 128
    assert declare({"x-data-class": "open", "x-operation": op}, None, A, POLICY).operation == op


@pytest.mark.parametrize("bad", ["", "a b", "x" * 80, "../x", "a\n"])
def test_malformed_class_is_400_and_not_echoed(bad):
    e = refusal({"x-data-class": bad}, None, A, POLICY)
    assert e.status == 400 and (bad.strip() == "" or bad not in str(e))


def test_non_string_metadata_is_400():
    assert refusal({}, {"data_class": 3}, A, POLICY).status == 400
    assert refusal({"x-data-class": "open"}, {"operation": 3}, A, POLICY).status == 400


# --- no policy -------------------------------------------------------------


def test_no_policy_nothing_declared_is_nothing():
    assert declare({}, None, None, None) == RequestClass(None, None)


def test_no_policy_class_reaches_classifier_only_if_in_its_label_set():
    h = {"x-data-class": "public", "x-operation": "t.s"}
    assert declare(h, None, None, None, ("public", "secret")) == RequestClass("public", "t.s")
    assert declare(h, None, None, None, ("secret",)) == RequestClass(None, "t.s")
    assert declare(h, None, None, None) == RequestClass(None, "t.s")


def test_no_policy_keys_file_allowed_classes_is_enforced():
    p = Principal("t", allowed_classes=("public",))
    assert declare({"x-data-class": "public"}, None, p, None, ("public",)).data_class == "public"
    assert refusal({"x-data-class": "secret"}, None, p, None, ("secret",)).status == 403
    assert refusal({"x-data-class": "public"}, None, Principal("t"), None).status == 403


def test_no_policy_conflict_is_still_400():
    assert refusal({"x-data-class": "a"}, {"data_class": "b"}, None, None).status == 400


# --- startup ---------------------------------------------------------------


class Resolver:
    def __init__(self, *principals):
        self._p = list(principals)

    def principals(self):
        return self._p


def test_policy_principal_without_a_key_stops_startup():
    with pytest.raises(PolicyError) as exc:
        check_policy_principals(POLICY, Resolver(Principal("team-a")))
    assert "team-b" in str(exc.value)


def test_keys_file_allowed_classes_with_policy_warns_once(caplog):
    r = Resolver(
        Principal("team-a", allowed_classes=("open",)),
        Principal("team-b", allowed_classes=("open",)),
    )
    with caplog.at_level(logging.WARNING):
        check_policy_principals(POLICY, r)
    assert len([m for m in caplog.records if "allowed_classes" in m.getMessage()]) == 1


def test_no_warning_without_allowed_classes(caplog):
    with caplog.at_level(logging.WARNING):
        check_policy_principals(POLICY, Resolver(Principal("team-a"), Principal("team-b")))
    assert not caplog.records


def test_policy_principals_with_auth_off_warns(caplog):
    with caplog.at_level(logging.WARNING):
        check_policy_principals(POLICY, None)
    assert any("auth" in m.getMessage() for m in caplog.records)


# --- server ----------------------------------------------------------------

KEY_A, KEY_B, KEY_C = "sk-test-aaaa", "sk-test-bbbb", "sk-test-cccc"


def sha(key):
    return hashlib.sha256(key.encode()).hexdigest()


_PREAMBLE = """
import json, logging, sys
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

A, B = FakeUpstream("A"), FakeUpstream("B")
def ep(name, up):
    return {{"model": "openai/model-" + name, "api_base": up.base, "api_key_env": "FAKE_KEY", "tags": ["local", "jur:ca"]}}
cfg = {{
    "endpoints": {{"a": ep("a", A), "b": ep("b", B)}},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}}}},
    "comparison": {{"enabled": True, "compare_allowed_models": ["a", "b"]}},
}}
cfg.update({extra!r})
open({config!r}, "w").write(json.dumps(cfg))

import routellm.openai_server as server
from fastapi.testclient import TestClient

MESSAGES = [{{"role": "user", "content": "hello"}}]
BODY = {{"model": "default", "messages": MESSAGES}}
def H(key, **h):
    return {{"Authorization": "Bearer " + key, **h}}


def emit(obj):
    print("@@RESULT@@" + json.dumps(obj))

seen_rc = []
@server.app.middleware("http")
async def probe(request, call_next):
    resp = await call_next(request)
    rc = getattr(request.state, "request_class", None)
    seen_rc.append([rc.data_class, rc.operation] if rc else None)
    return resp

def hits():
    return len(A.hits + B.hits)

with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    declared = []
    real = server.CONTROLLER.acompletion
    async def spy(**kw):
        declared.append(kw.get("declared_class"))
        return await real(**kw)
    server.CONTROLLER.acompletion = spy
"""


def _run(
    tmp_path,
    body,
    *,
    policy=True,
    auth=True,
    keys_extra="",
    policy_extra=None,
    config_extra=None,
    expect_ok=True,
):
    extra = {}
    if auth:
        keys = tmp_path / "keys.yaml"
        keys.write_text(
            f"""
keys:
  - key_sha256: "{sha(KEY_A)}"
    principal: team-a
    allowed_classes: [public]
  - key_sha256: "{sha(KEY_B)}"
    principal: team-b
{keys_extra}
"""
        )
        extra["auth"] = {"keys_file": str(keys)}
    if policy:
        pol = {
            "classes": {
                "open": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
                "internal": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
                "sealed": {"treatment": "vault"},
            },
            "principals": {
                "team-a": {"classes": ["open", "internal", "sealed"]},
                "team-b": {"classes": ["open"]},
            },
            **(policy_extra or {}),
        }
        pfile = tmp_path / "policy.json"
        pfile.write_text(json.dumps(pol))
        extra["policy"] = str(pfile)
    extra.update(config_extra or {})
    preamble = _PREAMBLE.format(config=str(tmp_path / "config.json"), extra=extra)
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
        timeout=120,
    )
    if not expect_ok:
        return result
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    return json.loads(line.split("@@RESULT@@", 1)[1])


@pytest.mark.slow
def test_refusals_map_to_status_and_never_reach_upstream(tmp_path):
    out = _run(
        tmp_path,
        f"""
def post(key, headers=None, body=None):
    r = client.post("/v1/chat/completions", json=body or BODY, headers=H(key, **(headers or {{}})))
    return [r.status_code, r.text]
rs = {{
  "unknown": post("{KEY_A}", {{"X-Data-Class": "nope"}}),
  "forbidden": post("{KEY_B}", {{"X-Data-Class": "internal"}}),
  "missing": post("{KEY_A}"),
  "conflict": post("{KEY_A}", {{"X-Data-Class": "open"}}, {{**BODY, "metadata": {{"data_class": "internal"}}}}),
  "vault": post("{KEY_A}", {{"X-Data-Class": "sealed"}}),
  "long_op": post("{KEY_A}", {{"X-Data-Class": "open", "X-Operation": "a" * 129}}),
}}
emit({{"rs": rs, "hits": hits()}})
""",
    )
    got = {k: v[0] for k, v in out["rs"].items()}
    assert got == {
        "unknown": 400,
        "forbidden": 403,
        "missing": 400,
        "conflict": 400,
        "vault": 403,
        "long_op": 400,
    }
    assert "nope" in out["rs"]["unknown"][1]
    assert json.loads(out["rs"]["forbidden"][1])["object"] == "error"
    assert KEY_A not in json.dumps(out["rs"]) and KEY_B not in json.dumps(out["rs"])
    assert out["hits"] == 0


@pytest.mark.slow
def test_carriage_is_never_forwarded_upstream(tmp_path):
    out = _run(
        tmp_path,
        f"""
r1 = client.post("/v1/chat/completions", json=BODY,
    headers=H("{KEY_A}", **{{"X-Data-Class": "open", "X-Operation": "ctx.summarize"}}))
r2 = client.post("/v1/chat/completions",
    json={{**BODY, "metadata": {{"data_class": "internal", "operation": "ctx.step", "keep": "x"}}}},
    headers=H("{KEY_A}"))
sent = A.hits + B.hits
emit({{
  "status": [r1.status_code, r2.status_code],
  "declared": declared,
  "rc": seen_rc,
  "n": len(sent),
  "headers": [sorted(k.lower() for k in h) for _, h, _ in sent],
  "bodies": [b for _, _, b in sent],
}})
""",
    )
    assert out["status"] == [200, 200]
    assert out["declared"] == ["open", "internal"]
    assert out["rc"] == [["open", "ctx.summarize"], ["internal", "ctx.step"]]
    assert out["n"] == 2
    for names in out["headers"]:
        assert "x-data-class" not in names and "x-operation" not in names
    blob = json.dumps(out["bodies"])
    assert "metadata" not in blob and "data_class" not in blob and "ctx.step" not in blob


@pytest.mark.slow
def test_default_class_serves_a_request_with_none(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json=BODY, headers=H("{KEY_A}"))
emit({{"status": r.status_code, "declared": declared}})
""",
        policy_extra={"default_class": "open"},
    )
    assert out == {"status": 200, "declared": ["open"]}


@pytest.mark.slow
def test_policy_wins_over_keys_file_allowed_classes(tmp_path):
    # team-a's keys-file entry lists `public` (not a policy class); the policy grants `open`.
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json=BODY, headers=H("{KEY_A}", **{{"X-Data-Class": "open"}}))
emit({{"status": r.status_code}})
""",
    )
    assert out == {"status": 200}


@pytest.mark.slow
def test_policy_without_auth_refuses_every_class(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json=BODY, headers={"X-Data-Class": "open"})
emit({"status": r.status_code, "hits": hits()})
""",
        auth=False,
        policy_extra={"principals": {}},
    )
    assert out == {"status": 403, "hits": 0}


@pytest.mark.slow
def test_policy_principal_without_key_stops_startup(tmp_path):
    out = _run(
        tmp_path,
        "emit({})",
        policy_extra={"principals": {"ghost": {"classes": ["open"]}}},
        expect_ok=False,
    )
    assert out.returncode != 0 and "ghost" in out.stderr


@pytest.mark.slow
def test_no_policy_keys_file_allowed_classes_is_checked(tmp_path):
    out = _run(
        tmp_path,
        f"""
def post(key, cls=None):
    h = {{"X-Data-Class": cls}} if cls else {{}}
    return client.post("/v1/chat/completions", json=BODY, headers=H(key, **h)).status_code
emit({{
  "allowed": post("{KEY_A}", "public"),
  "denied": post("{KEY_A}", "secret"),
  "none_allowed": post("{KEY_B}", "public"),
  "undeclared": post("{KEY_B}"),
  "declared": declared,
}})
""",
        policy=False,
    )
    # Not a classifier label (no decision section): reaches the controller as nothing.
    assert out == {
        "allowed": 200,
        "denied": 403,
        "none_allowed": 403,
        "undeclared": 200,
        "declared": [None, None],
    }


@pytest.mark.slow
def test_no_policy_carriage_is_stripped_and_conflict_is_400(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions",
    json={**BODY, "metadata": {"data_class": "x", "operation": "t.s"}},
    headers={"X-Data-Class": "x", "X-Operation": "t.s"})
c = client.post("/v1/chat/completions", json={**BODY, "metadata": {"data_class": "y"}},
    headers={"X-Data-Class": "x"})
sent = A.hits + B.hits
emit({"status": [r.status_code, c.status_code],
      "headers": [sorted(k.lower() for k in h) for _, h, _ in sent],
      "bodies": [b for _, _, b in sent]})
""",
        policy=False,
        auth=False,
    )
    assert out["status"] == [200, 400]
    assert all("x-data-class" not in h and "x-operation" not in h for h in out["headers"])
    assert "metadata" not in json.dumps(out["bodies"])


@pytest.mark.slow
def test_compare_obeys_the_same_check_and_strips_carriage(tmp_path):
    out = _run(
        tmp_path,
        f"""
cmp = {{"models": ["a", "b"], "messages": MESSAGES}}
bad = client.post("/v1/compare", json=cmp, headers=H("{KEY_B}", **{{"X-Data-Class": "internal"}}))
unk = client.post("/v1/compare", json=cmp, headers=H("{KEY_A}", **{{"X-Data-Class": "nope"}}))
none = client.post("/v1/compare", json=cmp, headers=H("{KEY_A}"))
n0 = hits()
ok = client.post("/v1/compare", json={{**cmp, "metadata": {{"data_class": "open", "operation": "t.s"}}}},
    headers=H("{KEY_A}", **{{"X-Operation": "t.s"}}))
sent = A.hits + B.hits
emit({{"codes": [bad.status_code, unk.status_code, none.status_code, ok.status_code],
      "before": n0, "after": len(sent),
      "headers": [sorted(k.lower() for k in h) for _, h, _ in sent],
      "bodies": [b for _, _, b in sent]}})
""",
    )
    assert out["codes"] == [403, 400, 400, 200]
    assert out["before"] == 0 and out["after"] >= 2
    assert all("x-data-class" not in h and "x-operation" not in h for h in out["headers"])
    assert "metadata" not in json.dumps(out["bodies"])
