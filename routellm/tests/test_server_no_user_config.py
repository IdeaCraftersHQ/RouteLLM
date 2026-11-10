"""Server startup honours `--no-user-config` and logs which layers loaded."""

from __future__ import annotations

import logging
import sys
from argparse import Namespace

import pytest
import yaml
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def server():
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def _boot(server, monkeypatch, tmp_path, no_user_config):
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    (xdg / "routellm").mkdir(parents=True)
    (xdg / "routellm" / "config.yaml").write_text(yaml.safe_dump({"marker_user": "u-secret"}))
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("ROUTELLM_CONFIG", raising=False)
    monkeypatch.delenv("ROUTELLM_NO_USER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def stop(file_config, *a, **kw):
        seen["data"] = dict(file_config)
        raise RuntimeError("stop")

    monkeypatch.setattr(server, "build_registry", stop)
    monkeypatch.setattr(
        server,
        "args",
        Namespace(config=None, no_user_config=no_user_config, strict_registry=False),
    )
    with pytest.raises(RuntimeError, match="stop"):
        with TestClient(server.app):
            pass
    return seen["data"]


def test_flag_skips_user_layer_at_startup(server, monkeypatch, tmp_path):
    data = _boot(server, monkeypatch, tmp_path, True)
    assert "marker_user" not in data


def test_without_flag_user_layer_loads(server, monkeypatch, tmp_path):
    data = _boot(server, monkeypatch, tmp_path, False)
    assert data["marker_user"] == "u-secret"


def test_startup_logs_loaded_and_skipped_paths_only(server, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _boot(server, monkeypatch, tmp_path, True)
    text = caplog.text
    assert "config layers loaded:" in text
    assert "skipped: user=" in text
    assert "u-secret" not in text
