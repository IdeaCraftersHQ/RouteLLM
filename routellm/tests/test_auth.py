"""Caller identity: bearer key to principal.

Unit cases read `routellm.auth` directly. The server cases run the real
app in a subprocess (it parses `sys.argv` at import) against a local
fake upstream and print one JSON line.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from routellm import auth
from routellm.config import ConfigError
from routellm.tests.subproc import SUBPROCESS_TIMEOUT

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_A = "sk-test-aaaa-1111"
KEY_B = "sk-test-bbbb-2222"


def sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def write_keys(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def good_keys(path: Path) -> Path:
    return write_keys(
        path,
        f"""
keys:
  - key_sha256: "{sha(KEY_A)}"
    principal: team-a
    profile: default
    allowed_classes: [public, internal]
  - key_sha256: "{sha(KEY_B)}"
    principal: team-b
""",
    )


# --- resolver ---------------------------------------------------------


def test_known_key_resolves_to_its_principal(tmp_path):
    resolver = auth.load_keys_file(good_keys(tmp_path / "k.yaml"))
    p = resolver.resolve(KEY_A)
    assert p == auth.Principal("team-a", "default", ("public", "internal"))
    assert p.allows("public") and not p.allows("secret")
    assert resolver.resolve(KEY_B).allowed_classes == ()


def test_unknown_key_resolves_to_none(tmp_path):
    resolver = auth.load_keys_file(good_keys(tmp_path / "k.yaml"))
    assert resolver.resolve("sk-nope") is None
    assert resolver.resolve("") is None
    assert resolver.resolve(sha(KEY_A)) is None  # the hash is not the key


def test_key_compare_is_constant_time(tmp_path, monkeypatch):
    """Every stored hash goes through hmac.compare_digest; no early exit."""
    calls = []
    real = auth.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    resolver = auth.load_keys_file(good_keys(tmp_path / "k.yaml"))
    assert resolver.resolve(KEY_A).principal == "team-a"  # the first entry
    assert len(calls) == 2


@pytest.mark.parametrize("field", ["key", "api_key", "token", "secret"])
def test_clear_text_key_in_file_is_a_startup_error(tmp_path, field):
    path = write_keys(
        tmp_path / "k.yaml",
        f"keys:\n  - {field}: {KEY_A}\n    key_sha256: {sha(KEY_A)}\n    principal: x\n",
    )
    with pytest.raises(ConfigError) as exc:
        auth.load_keys_file(path)
    assert "clear-text" in str(exc.value)
    assert KEY_A not in str(exc.value)


def test_clear_text_in_hash_field_is_a_startup_error(tmp_path):
    path = write_keys(tmp_path / "k.yaml", f"keys:\n  - key_sha256: {KEY_A}\n    principal: x\n")
    with pytest.raises(ConfigError) as exc:
        auth.load_keys_file(path)
    assert KEY_A not in str(exc.value)


@pytest.mark.parametrize(
    "body",
    [
        "keys: []\n",
        "",
        "keys:\n  - principal: x\n",
        f"keys:\n  - key_sha256: {sha('a')}\n    principal: x\n    extra: 1\n",
        f"keys:\n  - {{key_sha256: {sha('a')}, principal: x}}\n"
        f"  - {{key_sha256: {sha('a')}, principal: y}}\n",
    ],
)
def test_malformed_keys_files_are_refused(tmp_path, body):
    with pytest.raises(ConfigError):
        auth.load_keys_file(write_keys(tmp_path / "k.yaml", body))


def test_auth_section_is_off_when_absent_and_strict_when_present(tmp_path):
    assert auth.build_auth({}, None) is None
    assert auth.build_auth(None, None) is None
    with pytest.raises(ConfigError):
        auth.build_auth({"auth": {}}, None)
    with pytest.raises(ConfigError):
        auth.build_auth({"auth": {"keys_file": "k", "keys": []}}, None)


def test_relative_keys_file_resolves_against_the_config(tmp_path):
    good_keys(tmp_path / "k.yaml")
    resolver = auth.build_auth({"auth": {"keys_file": "k.yaml"}}, str(tmp_path / "c.yaml"))
    assert resolver.resolve(KEY_A).principal == "team-a"


@pytest.mark.parametrize(
    "header,token",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer  ", None),
        ("", None),
        (None, None),
    ],
)
def test_bearer_token_parsing(header, token):
    assert auth.bearer_token(header) == token


# --- server -----------------------------------------------------------

_PREAMBLE = """
import json, logging, sys
from routellm.tests.fake_upstream import FakeUpstream
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]

A, B = FakeUpstream("A"), FakeUpstream("B")
def ep(name, up):
    return {{"model": "openai/model-" + name, "api_base": up.base, "api_key_env": "FAKE_KEY"}}
cfg = {{
    "endpoints": {{"a": ep("a", A), "b": ep("b", B)}},
    "tiers": {{"default": {{"router": "random", "threshold": 0.0, "strong": "a", "weak": "b"}}}},
}}
if {keys_file!r}:
    cfg["auth"] = {{"keys_file": {keys_file!r}}}
open({config!r}, "w").write(json.dumps(cfg))

import routellm.openai_server as server
from fastapi.testclient import TestClient

MESSAGES = [{{"role": "user", "content": "hello"}}]
BODY = {{"model": "default", "messages": MESSAGES}}


def emit(obj):
    print("@@RESULT@@" + json.dumps(obj))


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []
    def emit(self, record):
        self.lines.append(record.getMessage())

cap = Capture()
logging.getLogger().addHandler(cap)
logging.getLogger().setLevel(logging.DEBUG)

mw = []
@server.app.middleware("http")
async def probe(request, call_next):
    resp = await call_next(request)
    p = getattr(request.state, "principal", None)
    mw.append(p.principal if p else None)
    return resp

with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    server.CONTROLLER.resilience.config.timeout_ms = 600_000  # starved CPU is not an upstream timeout
    seen = []
    real = server.CONTROLLER.acompletion
    async def spy(**kw):
        seen.append(kw.get("principal"))
        return await real(**kw)
    server.CONTROLLER.acompletion = spy
"""


def _run(tmp_path, body, keys=True):
    keys_file = str(good_keys(tmp_path / "keys.yaml")) if keys else ""
    preamble = _PREAMBLE.format(config=str(tmp_path / "config.json"), keys_file=keys_file)
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


@pytest.mark.slow
def test_no_bearer_is_401_and_never_reaches_upstream(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json=BODY)
emit({"status": r.status_code, "www": r.headers.get("www-authenticate"), "hits": len(A.hits + B.hits)})
""",
    )
    assert out == {"status": 401, "www": "Bearer", "hits": 0}


@pytest.mark.slow
def test_unknown_key_is_401_on_every_protected_route(tmp_path):
    out = _run(
        tmp_path,
        """
h = {"Authorization": "Bearer sk-wrong"}
emit({
  "chat": client.post("/v1/chat/completions", json=BODY, headers=h).status_code,
  "compare": client.post("/v1/compare", json={"models": ["a"], "messages": MESSAGES}, headers=h).status_code,
  "models": client.get("/v1/models", headers=h).status_code,
  "basic": client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Basic " + "x"}).status_code,
  "hits": len(A.hits + B.hits),
})
""",
    )
    assert out == {"chat": 401, "compare": 401, "models": 401, "basic": 401, "hits": 0}


@pytest.mark.slow
def test_known_key_passes_and_principal_reaches_the_controller_and_a_middleware(tmp_path):
    out = _run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json=BODY, headers={{"Authorization": "Bearer {KEY_A}"}})
emit({{
  "status": r.status_code,
  "seen": [[p.principal, p.profile, list(p.allowed_classes)] for p in seen],
  "mw": mw,
  "upstream_auth_forwarded": any("{KEY_A}" in json.dumps(h) for _, h, _ in A.hits + B.hits),
}})
""",
    )
    assert out["status"] == 200
    assert out["seen"] == [["team-a", "default", ["public", "internal"]]]
    assert out["mw"] == ["team-a"]
    # The caller's key is not the upstream's key and must not travel there.
    assert out["upstream_auth_forwarded"] is False


@pytest.mark.slow
def test_health_needs_no_token(tmp_path):
    out = _run(
        tmp_path,
        """
emit({"status": client.get("/health").status_code})
""",
    )
    assert out["status"] == 200


@pytest.mark.slow
def test_key_never_appears_in_logs_or_error_bodies(tmp_path):
    out = _run(
        tmp_path,
        """
bad = "sk-leakcheck-9999"
rs = [
  client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer " + bad}),
  client.post("/v1/chat/completions", json={"nope": 1}, headers={"Authorization": "Bearer " + bad}),
]
emit({"bodies": [r.text for r in rs], "headers": [dict(r.headers) for r in rs],
      "logs": cap.lines, "bad": bad})
""",
    )
    blob = json.dumps([out["bodies"], out["headers"], out["logs"]])
    assert out["bad"] not in blob


@pytest.mark.slow
def test_auth_off_keeps_todays_behaviour(tmp_path):
    out = _run(
        tmp_path,
        """
r = client.post("/v1/chat/completions", json=BODY)
emit({"status": r.status_code, "seen": [p for p in seen]})
""",
        keys=False,
    )
    assert out == {"status": 200, "seen": [None]}


@pytest.mark.slow
def test_clear_text_key_file_stops_startup_without_echoing_it(tmp_path):
    bad = tmp_path / "keys.yaml"
    bad.write_text(f"keys:\n  - key: {KEY_A}\n    principal: x\n")
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"auth": {"keys_file": str(bad)}}))
    code = (
        f"import sys; sys.argv=['x','--config',{str(config)!r}]\n"
        "import routellm.openai_server as s\n"
        "from fastapi.testclient import TestClient\n"
        "with TestClient(s.app): pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode != 0
    assert "clear-text" in result.stderr
    assert KEY_A not in result.stderr + result.stdout
