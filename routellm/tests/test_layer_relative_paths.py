"""Every relative path a config file sets resolves against that file's directory.

A deployment keeps `policy:`, `auth.keys_file:` and friends in the system
file and points `ROUTELLM_CONFIG` at a file elsewhere; the system file's
paths must not follow the env layer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from routellm import config as cfg
from routellm.auth import build_auth
from routellm.config import ConfigError
from routellm.policy import build_policy
from routellm.redact.config import build_redact_transform
from routellm.redact.gate import Gate
from routellm.server_config import (
    build_cache_config,
    build_quality_manager,
)

SYSTEM_SUBDIR = "sys"
ENV_SUBDIR = "envdir"
FLAG_SUBDIR = "flagdir"


def _write(path: Path, mapping) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    return path


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = home / "repo"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.delenv("ROUTELLM_NO_USER_CONFIG", raising=False)

    class D:
        system = tmp_path / SYSTEM_SUBDIR
        env = tmp_path / ENV_SUBDIR
        flag = tmp_path / FLAG_SUBDIR
        cwd = work

    system_file = D.system / "config.yaml"
    monkeypatch.setenv("ROUTELLM_SYSTEM_CONFIG", str(system_file))
    D.system_file = system_file
    D.env_file = D.env / "config.yaml"
    D.flag_file = D.flag / "config.yaml"
    return D


def _build_all(loaded):
    origin = str(loaded.layers[-1].path)
    return origin, loaded.resolve_path


# A harmless env layer in a different directory than the system file.
ENV_LAYER = {"embeddings": {"endpoint": "embedding"}}


def _env(monkeypatch, dirs):
    monkeypatch.setenv("ROUTELLM_CONFIG", str(dirs.env_file))


def _resolved(loaded, key, value):
    return loaded.resolve_path(key, value)


# (builder name, system-layer mapping, how to observe the resolved path)
def _obs_cache(loaded):
    origin, resolve = _build_all(loaded)
    return Path(build_cache_config(loaded.data, origin, resolve).db_path)


def _obs_trace(loaded):
    origin, resolve = _build_all(loaded)
    qm = build_quality_manager(loaded.data, origin, resolve)
    return Path(qm.fine_tune_config.trace_dir)


def _obs_redact_store(loaded):
    origin, resolve = _build_all(loaded)
    import routellm.redact.config as rc

    seen = {}
    orig = rc.TokenStore

    class Spy:
        def __init__(self, path, **kw):
            seen["p"] = Path(path)
            self.path = path

    rc.TokenStore = Spy
    try:
        try:
            build_redact_transform(loaded.data["redact"], origin, resolve)
        except Exception:
            pass
    finally:
        rc.TokenStore = orig
    return seen["p"]


def _obs_gate(loaded):
    origin, resolve = _build_all(loaded)
    return Path(Gate(loaded.data["redact"], origin, resolve).report)


def _obs_audit(loaded):
    # audit.path is observed through the resolver, as build_audit uses it
    return loaded.resolve_path("audit.path", loaded.data["audit"]["path"])


def _obs_audit_key(loaded):
    return loaded.resolve_path("audit.hmac_key_file", loaded.data["audit"]["hmac_key_file"])


def _obs_quality_from(loaded):
    origin, resolve = _build_all(loaded)
    seen = {}

    from routellm import endpoints as ep

    real = ep.EndpointRegistry
    import routellm.quality_scores as qs

    orig = qs.load_sidecar

    def spy(path):
        seen["p"] = Path(path)
        raise ConfigError("stop")

    qs.load_sidecar = spy
    try:
        with pytest.raises(ConfigError):
            real.from_config(loaded.data, config_path=origin, resolve_path=resolve)
    finally:
        qs.load_sidecar = orig
    return seen["p"]


def _obs_policy(loaded):
    seen = {}
    import routellm.policy as pol

    orig = pol.load_policy
    pol.load_policy = lambda p: seen.setdefault("p", Path(p)) and None
    try:
        build_policy(dict(loaded.data), loaded.resolve_path)
    finally:
        pol.load_policy = orig
    return seen["p"]


def _obs_keys_file(loaded):
    import routellm.auth as au

    seen = {}
    orig = au.load_keys_file
    au.load_keys_file = lambda path: seen.setdefault("p", Path(path))
    try:
        origin, resolve = _build_all(loaded)
        build_auth(loaded.data, origin, resolve)
    finally:
        au.load_keys_file = orig
    return seen["p"]


CASES = [
    ("auth.keys_file", {"auth": {"keys_file": "keys.yaml"}}, _obs_keys_file, "keys.yaml"),
    ("policy", {"policy": "p/pol.yaml"}, _obs_policy, "p/pol.yaml"),
    ("cache.db_path", {"cache": {"enabled": True, "db_path": "c.db"}}, _obs_cache, "c.db"),
    (
        "traces.trace_dir",
        {"traces": {"enabled": True, "trace_dir": "tr"}},
        _obs_trace,
        "tr",
    ),
    (
        "redact.store.path",
        {"redact": {"engine": "patterns", "store": {"path": "tok.db"}}},
        _obs_redact_store,
        "tok.db",
    ),
    (
        "redact.harness_report",
        {"redact": {"engine": "patterns", "harness_report": "rep.json"}},
        _obs_gate,
        "rep.json",
    ),
    ("audit.path", {"audit": {"path": "a.log", "hmac_key_file": "k"}}, _obs_audit, "a.log"),
    (
        "audit.hmac_key_file",
        {"audit": {"path": "a.log", "hmac_key_file": "k.key"}},
        _obs_audit_key,
        "k.key",
    ),
    ("quality_from", {"quality_from": "q.json"}, _obs_quality_from, "q.json"),
]


@pytest.mark.parametrize("key,system,observe,rel", CASES, ids=[c[0] for c in CASES])
def test_system_relative_path_ignores_env_layer_dir(dirs, monkeypatch, key, system, observe, rel):
    _write(dirs.env_file, ENV_LAYER)
    _env(monkeypatch, dirs)
    _write(dirs.system_file, system)
    loaded = cfg.load_config(interpolate=False)
    assert [e.source for e in loaded.layers][-1] == "env"
    assert observe(loaded) == dirs.system / rel


# `policy:` is pinned to the system layer, so no other layer may set it.
NON_PINNED = [c for c in CASES if c[0] != "policy"]


@pytest.mark.parametrize("key,system,observe,rel", NON_PINNED, ids=[c[0] for c in NON_PINNED])
def test_env_layer_relative_path_resolves_beside_env_file(
    dirs, monkeypatch, key, system, observe, rel
):
    _write(dirs.system_file, ENV_LAYER)
    _write(dirs.env_file, system)
    _env(monkeypatch, dirs)
    loaded = cfg.load_config(interpolate=False)
    assert observe(loaded) == dirs.env / rel


def test_flag_layer_override_resolves_beside_flag_file(dirs, monkeypatch):
    _write(dirs.system_file, {"cache": {"enabled": True, "db_path": "sys.db"}})
    _write(dirs.env_file, ENV_LAYER)
    _env(monkeypatch, dirs)
    _write(dirs.flag_file, {"cache": {"db_path": "flag.db"}})
    loaded = cfg.load_config(explicit=dirs.flag_file, interpolate=False)
    assert _obs_cache(loaded) == dirs.flag / "flag.db"


def test_absolute_path_unchanged(dirs, monkeypatch, tmp_path):
    _write(dirs.system_file, {"cache": {"enabled": True, "db_path": str(tmp_path / "abs.db")}})
    _write(dirs.env_file, ENV_LAYER)
    _env(monkeypatch, dirs)
    loaded = cfg.load_config(interpolate=False)
    assert _obs_cache(loaded) == tmp_path / "abs.db"


def test_tilde_expands_for_every_key(dirs, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    _write(dirs.system_file, {"policy": "~/pol.yaml"})
    loaded = cfg.load_config(interpolate=False)
    assert loaded.resolve_path("policy", "~/pol.yaml") == tmp_path / "h" / "pol.yaml"


def test_unrecorded_origin_resolves_against_cwd(dirs):
    _write(dirs.system_file, ENV_LAYER)
    loaded = cfg.load_config(interpolate=False)
    assert loaded.resolve_path("cache.db_path", "x.db") == Path("x.db")


def test_pinned_policy_still_refused_from_env_layer(dirs, monkeypatch):
    _write(dirs.system_file, {"policy": "pol.yaml"})
    _write(dirs.env_file, {"policy": "other.yaml"})
    _env(monkeypatch, dirs)
    with pytest.raises(Exception, match="policy"):
        cfg.load_config(interpolate=False)


def test_keys_file_unpinned_override_by_flag_resolves_beside_flag(dirs, monkeypatch):
    digest = hashlib.sha256(b"k").hexdigest()
    for d in (dirs.system, dirs.flag):
        d.mkdir(parents=True, exist_ok=True)
        (d / "keys.yaml").write_text(
            yaml.safe_dump({"keys": [{"key_sha256": digest, "principal": d.name, "profile": "b"}]})
        )
    _write(dirs.system_file, {"auth": {"keys_file": "keys.yaml"}})
    _write(dirs.env_file, ENV_LAYER)
    _env(monkeypatch, dirs)
    _write(dirs.flag_file, {"auth": {"keys_file": "keys.yaml"}})
    loaded = cfg.load_config(explicit=dirs.flag_file, interpolate=False)
    origin, resolve = _build_all(loaded)
    resolver = build_auth(loaded.data, origin, resolve)
    assert resolver.resolve("k", {}).principal == FLAG_SUBDIR
