"""Tests for `python -m routellm.config`: `path`, `paths`, and `show`.

Every location the chain searches lives under `tmp_path`: `HOME` and
`XDG_CONFIG_HOME` are set with `monkeypatch.setenv`, the CWD is moved
with `monkeypatch.chdir`, and `config.SYSTEM_PATH` is pointed at a temp
file so nothing ever reads the real `/etc`.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from routellm import config as cfg


@pytest.fixture
def env(tmp_path, monkeypatch, capsys):
    """Isolate HOME, XDG_CONFIG_HOME, SYSTEM_PATH and the CWD under tmp_path.

    Returns
    -------
    SimpleNamespace
        Attributes `home`, `xdg`, `system`, `user`, `project` (the CWD),
        `write(path, mapping)`, and `run(*argv)` which invokes
        `config.main` and returns `(exit_code, stdout, stderr)`.
    """
    home = tmp_path / "home"
    xdg = tmp_path / "xdgconf"
    system = tmp_path / "etc" / "routellm" / "config.yaml"
    project = home / "work" / "repo"
    for directory in (home, xdg, system.parent, project):
        directory.mkdir(parents=True, exist_ok=True)

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

    def run(*argv):
        code = cfg.main(list(argv))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    e.write = write
    e.run = run
    return e


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------


def test_path_prints_the_winner(env):
    """`path` names the highest-precedence file that exists, and only it."""
    env.write(env.system, {"a": 1})
    env.write(env.user, {"a": 2})
    marker = env.write(env.project / ".routellm.yaml", {"a": 3})

    code, out, _ = env.run("path")

    assert code == 0
    assert out.strip() == str(marker)


def test_no_config_anywhere_path_exits_1(env):
    """Nothing on the chain: exit 1 and print nothing to stdout."""
    code, out, err = env.run("path")

    assert code == 1
    assert out.strip() == ""
    assert "no routellm config" in err.lower()


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_paths_marks_used_and_absent(env):
    """Every searched location is listed, lowest first, with its state."""
    env.write(env.user, {"a": 2})
    env.write(env.project / "routellm.yaml", {"a": 3})

    code, out, _ = env.run("paths")

    assert code == 0
    lines = out.strip().splitlines()
    assert str(env.system) in lines[0]
    assert lines[0].startswith("[absent]")
    assert str(env.user) in lines[1]
    assert lines[1].startswith("[used]")
    assert lines[2].startswith("[used]")
    assert str(env.project / "routellm.yaml") in lines[2]


def test_paths_json_carries_source_and_exists(env):
    """`--format json` gives the same chain as objects."""
    env.write(env.user, {"a": 2})

    code, out, _ = env.run("paths", "--format", "json")

    assert code == 0
    chain = json.loads(out)
    assert [entry["source"] for entry in chain] == ["system", "user"]
    assert [entry["exists"] for entry in chain] == [False, True]
    assert chain[1]["path"] == str(env.user)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_carries_origin_comments(env):
    """Merged YAML, each top-level key commented with the file that set it."""
    env.write(env.user, {"endpoints": {"big": {"model": "m_big"}}})
    marker = env.write(
        env.project / ".routellm.yaml", {"tiers": {"default": {"strong": "big"}}}
    )

    code, out, _ = env.run("show")

    assert code == 0
    endpoints_line = next(
        line for line in out.splitlines() if line.startswith("endpoints:")
    )
    tiers_line = next(line for line in out.splitlines() if line.startswith("tiers:"))
    assert str(env.user) in endpoints_line
    assert str(marker) in tiers_line


def test_show_json_is_the_merged_dict(env):
    """`--format json` is the merged mapping, with no origin annotation."""
    env.write(env.user, {"endpoints": {"big": {"model": "m_big"}}})
    env.write(env.project / ".routellm.yaml", {"endpoints": {"small": {"model": "m_s"}}})

    code, out, _ = env.run("show", "--format", "json")

    assert code == 0
    data = json.loads(out)
    assert data["endpoints"] == {
        "big": {"model": "m_big"},
        "small": {"model": "m_s"},
    }


# ---------------------------------------------------------------------------
# The explicit layers
# ---------------------------------------------------------------------------


def test_flag_wins_over_the_discovered_chain(env):
    """`--config` is appended last, so `path` names it."""
    env.write(env.user, {"a": 2})
    explicit = env.write(env.project / "explicit.yaml", {"a": 9})

    code, out, _ = env.run("path", "--config", str(explicit))

    assert code == 0
    assert out.strip() == str(explicit)


def test_env_var_joins_the_chain(env, monkeypatch):
    """`ROUTELLM_CONFIG` is honoured without a flag."""
    explicit = env.write(env.project / "from-env.yaml", {"a": 9})
    monkeypatch.setenv("ROUTELLM_CONFIG", str(explicit))

    code, out, _ = env.run("path")

    assert code == 0
    assert out.strip() == str(explicit)


def test_missing_flag_file_is_reported_not_traced(env):
    """A named file that is not there exits 1 with the flag in the message."""
    code, out, err = env.run("show", "--config", str(env.project / "gone.yaml"))

    assert code == 1
    assert out.strip() == ""
    assert "--config" in err


def test_leading_flag_existing_file_wins_in_path(env):
    """`--config` before the subcommand still names the explicit file."""
    env.write(env.user, {"a": 2})
    explicit = env.write(env.project / "explicit.yaml", {"a": 9})

    code, out, _ = env.run("--config", str(explicit), "path")

    assert code == 0
    assert out.strip() == str(explicit)


def test_trailing_flag_existing_file_wins_in_path(env):
    """`--config` after the subcommand also names the explicit file."""
    env.write(env.user, {"a": 2})
    explicit = env.write(env.project / "explicit.yaml", {"a": 9})

    code, out, _ = env.run("path", "--config", str(explicit))

    assert code == 0
    assert out.strip() == str(explicit)


def test_leading_flag_missing_file_exits_1(env):
    """A missing file named before the subcommand is reported, not discarded."""
    missing = env.project / "gone.yaml"

    code, out, err = env.run("--config", str(missing), "show")

    assert code == 1
    assert out.strip() == ""
    assert "--config" in err
    assert str(missing) in err


def test_trailing_flag_missing_file_exits_1(env):
    """A missing file named after the subcommand is reported, not discarded."""
    missing = env.project / "gone.yaml"

    code, out, err = env.run("show", "--config", str(missing))

    assert code == 1
    assert out.strip() == ""
    assert "--config" in err
    assert str(missing) in err


def test_path_leading_flag_missing_file_exits_1(env):
    """`path` refuses a missing explicit file named before the subcommand.

    `config_paths()`/`winning_path()` never validate the explicit layer,
    only rank it; a missing file there used to fall through silently to
    whatever exists lower on the chain instead of being reported.
    """
    missing = env.project / "gone.yaml"

    code, out, err = env.run("--config", str(missing), "path")

    assert code == 1
    assert out.strip() == ""
    assert "--config" in err
    assert str(missing) in err


def test_path_trailing_flag_missing_file_exits_1(env):
    """`path` refuses a missing explicit file named after the subcommand."""
    missing = env.project / "gone.yaml"

    code, out, err = env.run("path", "--config", str(missing))

    assert code == 1
    assert out.strip() == ""
    assert "--config" in err
    assert str(missing) in err


def test_path_prints_the_explicit_file_when_it_exists_leading(env):
    """An existing explicit file, named before the subcommand, still wins."""
    env.write(env.user, {"a": 2})
    explicit = env.write(env.project / "explicit.yaml", {"a": 9})

    code, out, _ = env.run("--config", str(explicit), "path")

    assert code == 0
    assert out.strip() == str(explicit)


def test_path_prints_the_explicit_file_when_it_exists_trailing(env):
    """An existing explicit file, named after the subcommand, still wins."""
    env.write(env.user, {"a": 2})
    explicit = env.write(env.project / "explicit.yaml", {"a": 9})

    code, out, _ = env.run("path", "--config", str(explicit))

    assert code == 0
    assert out.strip() == str(explicit)


def test_paths_and_show_agree_when_no_project_marker_exists(env):
    """Both surfaces report the project layer, absent, on the same line.

    `config_paths` omits the project entry entirely when no marker was
    found — there is no path to name — so both commands synthesize the
    same `[absent] project` line rather than one of them staying silent.
    """
    env.write(env.user, {"a": 2})

    _, paths_out, _ = env.run("paths")
    _, show_out, _ = env.run("show")

    project_lines = [
        line for line in paths_out.splitlines() if "project" in line
    ]
    assert len(project_lines) == 1
    assert project_lines[0].startswith("[absent]")
    assert project_lines[0] in show_out
