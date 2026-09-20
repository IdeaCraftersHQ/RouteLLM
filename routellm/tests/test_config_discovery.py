"""Layered config discovery: the precedence chain, deep merge, and origins.

Every path in this file lives under `tmp_path`: `HOME` and
`XDG_CONFIG_HOME` are set with `monkeypatch.setenv`, the CWD is moved
with `monkeypatch.chdir`, and `config.SYSTEM_PATH` is monkeypatched to a
temp file so nothing ever reads the real `/etc`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from routellm import config as cfg


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate HOME, XDG_CONFIG_HOME, SYSTEM_PATH and the CWD under tmp_path.

    Returns
    -------
    SimpleNamespace
        Attributes `home`, `xdg`, `system`, `project` (a nested dir that
        is the default CWD), plus `write(path, mapping)`.
    """
    home = tmp_path / "home"
    xdg = tmp_path / "xdgconf"
    system = tmp_path / "etc" / "routellm" / "config.yaml"
    project = home / "work" / "repo" / "pkg"
    for d in (home, xdg, system.parent, project):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("ROUTELLM_CONFIG", raising=False)
    monkeypatch.setattr(cfg, "SYSTEM_PATH", str(system))
    monkeypatch.chdir(project)

    e = SimpleNamespace()
    e.home = home
    e.xdg = xdg
    e.system = system
    e.project = project
    e.user = xdg / "routellm" / "config.yaml"

    def write(path, mapping):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(mapping, sort_keys=False))
        return path

    e.write = write
    return e


def sources(paths):
    return [p.source for p in paths]


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def test_chain_order_lowest_first(env):
    env.write(env.system, {"a": 1})
    env.write(env.user, {"a": 2})
    env.write(env.project / ".routellm.yaml", {"a": 3})
    env.write(env.project / "explicit.yaml", {"a": 4})

    paths = cfg.config_paths(explicit=str(env.project / "explicit.yaml"))

    assert sources(paths) == ["system", "user", "project", "flag"]
    assert all(p.exists for p in paths)


def test_absent_layers_are_skipped(env):
    env.write(env.user, {"a": 2})

    loaded = cfg.load_config()

    assert [layer.source for layer in loaded.layers] == ["user"]
    assert loaded.data["a"] == 2
    # The chain still reports the locations it looked at.
    assert sources(cfg.config_paths()) == ["system", "user"]
    assert [p.exists for p in cfg.config_paths()] == [False, True]


def test_user_layer_overrides_system(env):
    env.write(env.system, {"a": "system", "only_system": 1})
    env.write(env.user, {"a": "user"})

    loaded = cfg.load_config()

    assert loaded.data["a"] == "user"
    assert loaded.data["only_system"] == 1
    # The defaults layer is still underneath both files.
    assert loaded.data["mf"] == {"checkpoint_path": "routellm/mf_gpt4_augmented"}


def test_project_marker_in_cwd(env):
    env.write(env.project / "routellm.yaml", {"a": "project"})

    loaded = cfg.load_config()

    assert [layer.source for layer in loaded.layers] == ["project"]
    assert loaded.layers[0].path == env.project / "routellm.yaml"
    assert loaded.data["a"] == "project"


def test_project_marker_found_by_walk_up(env):
    marker = env.write(env.home / "work" / "repo" / ".routellm.yaml", {"a": "up"})

    loaded = cfg.load_config()

    assert [layer.path for layer in loaded.layers] == [marker]
    assert loaded.data["a"] == "up"


def test_walk_up_stops_at_home(env):
    # A marker sitting in $HOME itself must never be read as a project file.
    env.write(env.home / ".routellm.yaml", {"a": "home"})

    loaded = cfg.load_config()

    assert loaded.layers == []
    assert "a" not in loaded.data
    assert "project" not in sources(cfg.config_paths())


def test_walk_up_stops_at_a_symlinked_home(env):
    """A symlinked `$HOME` is still the boundary, resolved on both sides.

    macOS ships `/tmp` as a symlink to `/private/tmp`, so an unresolved
    comparison silently reads `$HOME/.routellm.yaml` as a project file.
    """
    import os

    link = env.home.parent / "home-link"
    link.symlink_to(env.home)
    env.write(env.home / ".routellm.yaml", {"a": "home"})
    marker = env.write(env.project / ".routellm.yaml", {"a": "project"})

    os.environ["HOME"] = str(link)

    found = cfg._project_file(env.project.resolve())

    assert found == marker
    assert cfg._project_file(env.home.resolve()) is None


def test_env_overrides_project(env, monkeypatch):
    env.write(env.project / ".routellm.yaml", {"a": "project"})
    env_file = env.write(env.project / "from-env.yaml", {"a": "env"})
    monkeypatch.setenv("ROUTELLM_CONFIG", str(env_file))

    loaded = cfg.load_config()

    assert [layer.source for layer in loaded.layers] == ["project", "env"]
    assert loaded.data["a"] == "env"


def test_flag_overrides_env(env, monkeypatch):
    env_file = env.write(env.project / "from-env.yaml", {"a": "env"})
    flag_file = env.write(env.project / "from-flag.yaml", {"a": "flag"})
    monkeypatch.setenv("ROUTELLM_CONFIG", str(env_file))

    loaded = cfg.load_config(explicit=str(flag_file))

    assert [layer.source for layer in loaded.layers] == ["env", "flag"]
    assert loaded.data["a"] == "flag"


def test_missing_env_file_names_the_variable(env, monkeypatch):
    monkeypatch.setenv("ROUTELLM_CONFIG", str(env.project / "nope.yaml"))

    with pytest.raises(FileNotFoundError) as excinfo:
        cfg.load_config()

    assert "ROUTELLM_CONFIG" in str(excinfo.value)
    assert "nope.yaml" in str(excinfo.value)


def test_missing_flag_file_names_the_flag(env):
    with pytest.raises(FileNotFoundError) as excinfo:
        cfg.load_config(explicit=str(env.project / "nope.yaml"))

    assert "--config" in str(excinfo.value)
    assert "nope.yaml" in str(excinfo.value)


# --------------------------------------------------------------------------
# Merge semantics
# --------------------------------------------------------------------------


def test_mappings_merge_by_key_across_layers(env):
    env.write(
        env.user,
        {
            "endpoints": {
                "cloud_strong": {"model": "gpt-4", "tags": ["cloud"]},
                "local_weak": {"model": "llama"},
            }
        },
    )
    env.write(
        env.project / ".routellm.yaml",
        {
            "endpoints": {
                "cloud_strong": {"tags": ["cloud", "eu"]},
                "extra": {"model": "mistral"},
            }
        },
    )

    loaded = cfg.load_config()

    assert loaded.data["endpoints"] == {
        "cloud_strong": {"model": "gpt-4", "tags": ["cloud", "eu"]},
        "local_weak": {"model": "llama"},
        "extra": {"model": "mistral"},
    }


def test_null_deletes_a_key_from_a_lower_layer(env):
    env.write(env.user, {"endpoints": {"keep": {"model": "a"}, "drop": {"model": "b"}}})
    env.write(env.project / ".routellm.yaml", {"endpoints": {"drop": None}})

    loaded = cfg.load_config()

    assert loaded.data["endpoints"] == {"keep": {"model": "a"}}


def test_null_deletes_under_a_parent_no_lower_layer_set(env):
    """A null leaf is dropped even when its parent is new in this layer."""
    env.write(env.user, {"endpoints": {"keep": {"model": "a"}}})
    env.write(env.project / ".routellm.yaml", {"tiers": {"gone": None}})

    loaded = cfg.load_config()

    assert loaded.data.get("tiers", {}) == {}
    assert cfg.deep_merge({}, {"tiers": {"t": None}}) == {"tiers": {}}


def test_missing_env_file_errors_even_with_a_valid_flag(env, monkeypatch):
    """An explicitly named file must exist; a good `--config` excuses nothing."""
    good = env.write(env.project / "good.yaml", {"a": "flag"})
    monkeypatch.setenv("ROUTELLM_CONFIG", str(env.project / "gone.yaml"))

    with pytest.raises(FileNotFoundError) as excinfo:
        cfg.load_config(explicit=str(good))

    assert "ROUTELLM_CONFIG" in str(excinfo.value)


def test_lists_replace_not_concatenate(env):
    env.write(env.user, {"endpoints": {"e": {"tags": ["a", "b"]}}})
    env.write(env.project / ".routellm.yaml", {"endpoints": {"e": {"tags": ["c"]}}})

    loaded = cfg.load_config()

    assert loaded.data["endpoints"]["e"]["tags"] == ["c"]


def test_origins_track_the_last_layer_that_set_a_key(env):
    user = env.write(env.user, {"endpoints": {"e": {"model": "a", "tags": ["x"]}}})
    project = env.write(env.project / ".routellm.yaml", {"endpoints": {"e": {"tags": ["y"]}}})

    loaded = cfg.load_config()

    assert loaded.origins["endpoints.e.model"] == user
    assert loaded.origins["endpoints.e.tags"] == project


def test_relative_prompt_file_resolves_against_its_own_file(env):
    env.write(env.user, {"prompt_file": "prompts/router.yaml"})

    loaded = cfg.load_config()

    assert loaded.resolve_path("prompt_file", loaded.data["prompt_file"]) == (
        env.user.parent / "prompts" / "router.yaml"
    )
    # An absolute value is handed back untouched.
    absolute = env.project / "elsewhere.yaml"
    assert loaded.resolve_path("prompt_file", str(absolute)) == absolute


def test_yaml_error_names_file_and_layer(env):
    env.user.parent.mkdir(parents=True, exist_ok=True)
    env.user.write_text("endpoints: [unclosed\n")

    with pytest.raises(cfg.ConfigError) as excinfo:
        cfg.load_config()

    message = str(excinfo.value)
    assert str(env.user) in message
    assert "user" in message


def test_explain_lists_every_location_and_marks_used(env):
    env.write(env.system, {"a": "system"})
    env.write(env.user, {"a": "user", "b": 1})

    text = cfg.explain(cfg.load_config())

    assert f"[used]   system  {env.system}" in text
    assert f"[used]   user    {env.user}" in text
    # The project layer was searched and found nothing.
    assert "[absent] project" in text
    # The effective config follows, each top-level key naming its origin.
    assert f"a: user  # from {env.user}" in text
    assert f"b: 1  # from {env.user}" in text
