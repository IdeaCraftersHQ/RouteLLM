"""Governance pinned to the system layer: tags, `policy:` and friends.

Everything lives under `tmp_path`: the autouse conftest fixture moves
HOME, XDG_CONFIG_HOME and `SYSTEM_PATH` there, so the real `/etc` and
`~/.config` are never read.
"""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from routellm import config as cfg

SYSTEM_ENDPOINTS = {
    "local_a": {"model": "ollama/a", "tags": ["local", "jur:ca", "fast"]},
    "cloud_b": {"model": "gpt-x", "tags": ["cloud", "jur:us"]},
}


@pytest.fixture
def layers(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = home / "repo"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    monkeypatch.delenv("ROUTELLM_NO_USER_CONFIG", raising=False)

    class L:
        system = Path(cfg.SYSTEM_PATH)
        user = Path(cfg._user_config_dir()) / "routellm" / "config.yaml"
        project = work / ".routellm.yaml"
        env = tmp_path / "env.yaml"
        flag = tmp_path / "flag.yaml"

        @staticmethod
        def write(path, mapping):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(mapping, sort_keys=False))
            return path

    L.write(L.system, {"policy": "/etc/routellm/policy.yaml", "endpoints": SYSTEM_ENDPOINTS})
    return L


def _load(layers, source=None):
    return cfg.load_config(
        explicit=layers.flag if layers.flag.exists() else None, interpolate=False
    )


def _higher(layers, source, mapping):
    path = getattr(layers, source)
    layers.write(path, mapping)
    return path


SOURCES = ["user", "project", "env", "flag"]


@pytest.fixture(autouse=True)
def _env_layer(layers, monkeypatch):
    # the env layer is only read when ROUTELLM_CONFIG names it
    def point():
        if layers.env.exists():
            monkeypatch.setenv("ROUTELLM_CONFIG", str(layers.env))

    return point


def _go(layers, _env_layer):
    _env_layer()
    return _load(layers)


# --- retag / add / remove on endpoints --------------------------------------


@pytest.mark.parametrize("source", SOURCES)
def test_higher_layer_retagging_an_endpoint_stops_startup(layers, _env_layer, source):
    path = _higher(
        layers, source, {"endpoints": {"cloud_b": {"tags": ["cloud", "local", "jur:us"]}}}
    )
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    text = str(exc.value)
    assert str(path) in text and source in text
    assert "cloud_b" in text and "tags" in text and str(layers.system) in text


@pytest.mark.parametrize("source", SOURCES)
def test_higher_layer_removing_governance_tags_stops_startup(layers, _env_layer, source):
    _higher(layers, source, {"endpoints": {"local_a": {"tags": ["fast"]}}})
    with pytest.raises(cfg.ConfigError, match="local_a"):
        _go(layers, _env_layer)


def test_higher_layer_nulling_tags_stops_startup(layers, _env_layer):
    _higher(layers, "project", {"endpoints": {"local_a": {"tags": None}}})
    with pytest.raises(cfg.ConfigError, match="local_a"):
        _go(layers, _env_layer)


def test_higher_layer_changing_jur_stops_startup(layers, _env_layer):
    _higher(layers, "user", {"endpoints": {"local_a": {"tags": ["local", "jur:fr", "fast"]}}})
    with pytest.raises(cfg.ConfigError, match="jur"):
        _go(layers, _env_layer)


@pytest.mark.parametrize("source", SOURCES)
def test_new_endpoint_tagged_in_a_higher_layer_stops_startup(layers, _env_layer, source):
    path = _higher(
        layers, source, {"endpoints": {"extra": {"model": "m", "tags": ["local", "jur:ca"]}}}
    )
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    assert str(path) in str(exc.value) and "extra" in str(exc.value)


def test_new_namespaced_tag_in_a_higher_layer_stops_startup(layers, _env_layer):
    _higher(layers, "project", {"endpoints": {"extra": {"model": "m", "tags": ["team:x"]}}})
    with pytest.raises(cfg.ConfigError, match="extra"):
        _go(layers, _env_layer)


def test_new_untagged_endpoint_in_a_higher_layer_is_allowed(layers, _env_layer):
    _higher(layers, "project", {"endpoints": {"extra": {"model": "m", "tags": ["fast"]}}})
    loaded = _go(layers, _env_layer)
    assert "extra" in loaded.data["endpoints"]


def test_non_governance_edits_to_a_tagged_endpoint_are_allowed(layers, _env_layer):
    _higher(
        layers,
        "user",
        {"endpoints": {"local_a": {"model": "ollama/b", "tags": ["jur:ca", "local", "slow"]}}},
    )
    loaded = _go(layers, _env_layer)
    assert loaded.data["endpoints"]["local_a"]["model"] == "ollama/b"


def test_deleting_an_endpoint_in_a_higher_layer_is_allowed(layers, _env_layer):
    _higher(layers, "project", {"endpoints": {"cloud_b": None}})
    assert "cloud_b" not in _go(layers, _env_layer).data["endpoints"]


def test_deleting_then_readding_with_tags_stops_startup(layers, _env_layer):
    _higher(layers, "user", {"endpoints": {"cloud_b": None}})
    _higher(layers, "project", {"endpoints": {"cloud_b": {"tags": ["local"]}}})
    with pytest.raises(cfg.ConfigError, match="cloud_b"):
        _go(layers, _env_layer)


# --- pinned top-level keys ---------------------------------------------------


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    "key,value",
    [
        ("policy", "elsewhere.yaml"),
        ("governance", False),
        ("strict_registry", False),
        ("tag_namespaces", ["team"]),
        ("egress_gates", {"gated": {"allow": ["x"]}}),
        ("auth", {"keys_file": "k.yaml"}),
        ("decision", {"endpoint": "local_a"}),
        ("audit", {"sink": "file", "path": "x.jsonl"}),
    ],
)
def test_pinned_key_in_a_higher_layer_stops_startup(layers, _env_layer, source, key, value):
    path = _higher(layers, source, {key: value})
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    text = str(exc.value)
    assert str(path) in text and f"`{key}`" in text and source in text


def test_nulling_a_pinned_key_stops_startup(layers, _env_layer):
    _higher(layers, "project", {"policy": None})
    with pytest.raises(cfg.ConfigError, match="`policy`"):
        _go(layers, _env_layer)


def test_system_layer_sets_every_pinned_key(layers, _env_layer):
    layers.write(
        layers.system,
        {
            "policy": "p.yaml",
            "governance": True,
            "strict_registry": True,
            "tag_namespaces": ["team"],
            "endpoints": {"a": {"tags": ["team:x", "local"]}},
        },
    )
    assert _go(layers, _env_layer).data["policy"] == "p.yaml"


def test_policy_from_a_non_system_layer_is_refused_even_without_a_system_policy(layers, _env_layer):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    path = _higher(layers, "project", {"policy": "p.yaml"})
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    assert str(path) in str(exc.value) and "system" in str(exc.value)


def test_policy_from_the_flag_layer_is_refused_without_any_system_layer(layers, _env_layer):
    layers.system.unlink()
    _higher(layers, "flag", {"policy": "p.yaml"})
    with pytest.raises(cfg.ConfigError, match="`policy`"):
        _go(layers, _env_layer)


# --- no policy: today's merge ------------------------------------------------


def test_without_a_policy_higher_layers_merge_as_before(layers, _env_layer):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    _higher(layers, "project", {"endpoints": {"cloud_b": {"tags": ["cloud", "local"]}}})
    _higher(layers, "user", {"governance": True, "strict_registry": True, "egress_gates": {}})
    loaded = _go(layers, _env_layer)
    assert "local" in loaded.data["endpoints"]["cloud_b"]["tags"]
    assert loaded.data["governance"] is True


def test_the_system_pin_never_applies_with_no_system_layer(layers, _env_layer):
    layers.system.unlink()
    _higher(layers, "project", {"endpoints": {"x": {"tags": ["local"]}}, "governance": True})
    assert _go(layers, _env_layer).data["governance"] is True


# --- inspection ----------------------------------------------------------------


def test_loaded_config_records_what_is_pinned_and_by_which_file(layers, _env_layer):
    loaded = _go(layers, _env_layer)
    assert loaded.pinned_by == layers.system
    assert "policy" in loaded.pinned_keys and "endpoints.*.tags" in loaded.pinned_keys


def test_unpinned_config_records_nothing(layers, _env_layer):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    loaded = _go(layers, _env_layer)
    assert loaded.pinned_by is None and loaded.pinned_keys == ()


def test_show_marks_pinned_keys_with_their_source(layers, _env_layer, capsys):
    _higher(layers, "project", {"cache": {"enabled": False}})
    assert cfg.main(["show"]) == 0
    out = capsys.readouterr().out
    policy_line = next(line for line in out.splitlines() if line.startswith("policy:"))
    assert f"from {layers.system}" in policy_line and "[pinned]" in policy_line
    cache_line = next(line for line in out.splitlines() if line.startswith("cache:"))
    assert "[pinned]" not in cache_line
    assert f"pinned to the system layer ({layers.system})" in out
    assert "endpoints.*.tags" in out


def test_paths_marks_the_system_layer_as_pinning(layers, capsys):
    assert cfg.main(["paths"]) == 0
    system_line = next(line for line in capsys.readouterr().out.splitlines() if " system " in line)
    assert "pins governance" in system_line


def test_paths_json_carries_the_pin_flag(layers, capsys):
    assert cfg.main(["paths", "--format", "json"]) == 0
    rows = {row["source"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["system"]["pins_governance"] is True
    assert rows["user"]["pins_governance"] is False


def test_paths_without_a_system_policy_marks_nothing(layers, capsys):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    assert cfg.main(["paths"]) == 0
    assert "pins governance" not in capsys.readouterr().out


def test_a_pin_violation_makes_the_cli_exit_1(layers, capsys):
    _higher(layers, "project", {"endpoints": {"cloud_b": {"tags": ["local"]}}})
    assert cfg.main(["show"]) == 1
    assert "cloud_b" in capsys.readouterr().err


# --- system path override ------------------------------------------------------


def test_env_overrides_the_system_path(layers, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / "sys.yaml"
    layers.write(other, {"marker": 1})
    monkeypatch.setenv("ROUTELLM_SYSTEM_CONFIG", str(other))
    chain = cfg.config_paths()
    assert chain[0].source == "system" and chain[0].path == other
    assert cfg.load_config().data["marker"] == 1


def test_default_system_path_is_etc():
    assert cfg.DEFAULT_SYSTEM_PATH == "/etc/routellm/config.yaml"


# --- server startup ------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def _boot(server, monkeypatch):
    monkeypatch.setattr(
        server,
        "args",
        Namespace(config=None, no_user_config=False, strict_registry=False),
    )
    with TestClient(server.app):
        pass


def test_startup_refuses_a_second_layer_adding_a_cloud_leaf(server, layers, monkeypatch, caplog):
    path = _higher(
        layers,
        "project",
        {
            "endpoints": {"sneaky": {"model": "gpt-x", "tags": ["cloud"]}},
            "tiers": {"private": {"router": "random", "strong": "local_a", "weak": "sneaky"}},
        },
    )
    seen = []
    monkeypatch.setattr(server, "build_registry", lambda *a, **k: seen.append(1))
    with pytest.raises(cfg.ConfigError) as exc:
        _boot(server, monkeypatch)
    assert str(path) in str(exc.value) and "sneaky" in str(exc.value)
    assert not seen


def test_startup_without_a_policy_keeps_the_old_merge(server, layers, monkeypatch):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    _higher(layers, "project", {"endpoints": {"cloud_b": {"tags": ["local"]}}})
    seen = {}

    def stop(file_config, *a, **k):
        seen["tags"] = file_config["endpoints"]["cloud_b"]["tags"]
        raise RuntimeError("stop")

    monkeypatch.setattr(server, "build_registry", stop)
    with pytest.raises(RuntimeError, match="stop"):
        _boot(server, monkeypatch)
    assert seen["tags"] == ["local"]


def test_layer_summary_still_logs_paths_only(layers, caplog):
    caplog.set_level(logging.INFO)
    text = cfg.layer_summary(_load(layers))
    assert "system=" in text


# --- allow_public_base ---------------------------------------------------------


@pytest.mark.parametrize("source", SOURCES)
def test_setting_allow_public_base_on_an_endpoint_stops_startup(layers, _env_layer, source):
    path = _higher(layers, source, {"endpoints": {"local_a": {"allow_public_base": True}}})
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    text = str(exc.value)
    assert str(path) in text and "allow_public_base" in text and "local_a" in text


def test_allow_public_base_on_a_new_endpoint_stops_startup(layers, _env_layer):
    _higher(layers, "project", {"endpoints": {"extra": {"allow_public_base": False}}})
    with pytest.raises(cfg.ConfigError, match="extra"):
        _go(layers, _env_layer)


def test_changing_or_nulling_allow_public_base_stops_startup(layers, _env_layer):
    layers.write(
        layers.system,
        {
            "policy": "p.yaml",
            "endpoints": {"a": {"tags": ["local"], "allow_public_base": True}},
        },
    )
    _higher(layers, "user", {"endpoints": {"a": {"allow_public_base": None}}})
    with pytest.raises(cfg.ConfigError, match="allow_public_base"):
        _go(layers, _env_layer)


def test_system_layer_may_set_allow_public_base(layers, _env_layer):
    layers.write(
        layers.system,
        {"policy": "p.yaml", "endpoints": {"a": {"allow_public_base": True}}},
    )
    assert _go(layers, _env_layer).data["endpoints"]["a"]["allow_public_base"] is True


def test_allow_public_base_merges_freely_without_a_system_policy(layers, _env_layer):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    _higher(layers, "project", {"endpoints": {"local_a": {"allow_public_base": True}}})
    assert _go(layers, _env_layer).data["endpoints"]["local_a"]["allow_public_base"] is True


def test_show_lists_the_new_pins(layers, capsys):
    assert cfg.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "auth" in out.split("pinned to the system layer", 1)[1]
    assert "decision" in out.split("pinned to the system layer", 1)[1]
    assert "endpoints.*.allow_public_base" in out


MEDIA_SECTIONS = {
    "transcriber": {"mode": "transcribe", "api_base": "http://127.0.0.1:9000"},
    "video": {"api_base": "http://127.0.0.1:9100"},
}


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_media_section_in_a_higher_layer_stops_startup(layers, _env_layer, source, key):
    path = _higher(layers, source, {key: MEDIA_SECTIONS[key]})
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    text = str(exc.value)
    assert str(path) in text and f"`{key}`" in text and "system" in text


@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_system_layer_may_set_media_sections(layers, _env_layer, key):
    layers.write(layers.system, {"policy": "p.yaml", key: MEDIA_SECTIONS[key]})
    assert _go(layers, _env_layer).data[key] == MEDIA_SECTIONS[key]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_tier_media_override_from_a_higher_layer_stops_startup(layers, _env_layer, source, key):
    path = _higher(layers, source, {"tiers": {"fast": {key: {"timeout_ms": 5}}}})
    with pytest.raises(cfg.ConfigError) as exc:
        _go(layers, _env_layer)
    text = str(exc.value)
    assert str(path) in text and f"tiers.fast.{key}" in text and "system" in text


@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_tier_media_change_over_a_system_value_stops_startup(layers, _env_layer, key):
    layers.write(
        layers.system,
        {"policy": "p.yaml", "tiers": {"fast": {key: {"timeout_ms": 1}}}},
    )
    _higher(layers, "project", {"tiers": {"fast": {key: {"timeout_ms": 2}}}})
    with pytest.raises(cfg.ConfigError, match=f"tiers.fast.{key}"):
        _go(layers, _env_layer)


@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_tier_media_override_from_the_system_layer_loads(layers, _env_layer, key):
    layers.write(
        layers.system,
        {"policy": "p.yaml", "tiers": {"fast": {key: {"timeout_ms": 1}}}},
    )
    _higher(layers, "project", {"tiers": {"fast": {"models": ["x"]}}})
    assert _go(layers, _env_layer).data["tiers"]["fast"][key] == {"timeout_ms": 1}


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("key", sorted(MEDIA_SECTIONS))
def test_media_sections_merge_freely_without_a_system_policy(layers, _env_layer, source, key):
    layers.write(layers.system, {"endpoints": SYSTEM_ENDPOINTS})
    _higher(layers, source, {key: MEDIA_SECTIONS[key], "tiers": {"fast": {key: {"a": 1}}}})
    data = _go(layers, _env_layer).data
    assert data[key] == MEDIA_SECTIONS[key]
    assert data["tiers"]["fast"][key] == {"a": 1}


def test_show_lists_the_media_pins(layers, capsys):
    assert cfg.main(["show"]) == 0
    out = capsys.readouterr().out.split("pinned to the system layer", 1)[1]
    assert "transcriber" in out and "video" in out and "tiers.*.transcriber" in out
