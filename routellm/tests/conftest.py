"""Shared test setup.

Isolates every test from the developer's own machine configuration.
Without this, `load_config` walks the real discovery chain and merges
`$XDG_CONFIG_HOME/routellm/config.yaml` over whatever a test wrote,
so a test that passes an explicit `--config` still sees the endpoints
and tiers of whoever is running it. That is a real merge — the chain
is working as designed — but it makes results depend on the machine,
which is exactly what a test must not do.

Tests that want to exercise discovery itself set these variables
themselves (see `test_config_discovery.py` and `test_config_cli.py`);
`monkeypatch.setenv` inside a test overrides the autouse fixture, so
those keep working unchanged.
"""


import pytest

import routellm.config as cfg


@pytest.fixture(autouse=True)
def _isolate_ambient_config(tmp_path_factory, monkeypatch):
    """Point HOME, XDG_CONFIG_HOME and SYSTEM_PATH at empty temp dirs.

    Autouse, so a test never reads the developer's real config by
    accident. `ROUTELLM_CONFIG` is cleared for the same reason.
    """
    base = tmp_path_factory.mktemp("ambient")
    home = base / "home"
    xdg = base / "xdgconf"
    system = base / "etc" / "routellm" / "config.yaml"
    for directory in (home, xdg, system.parent):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("ROUTELLM_CONFIG", raising=False)
    monkeypatch.setattr(cfg, "SYSTEM_PATH", str(system), raising=False)

    # Subprocesses inherit HOME and XDG_CONFIG_HOME, which covers the
    # user and project layers. SYSTEM_PATH is a module constant, so
    # the monkeypatch above reaches in-process callers only; a
    # subprocess still stats the real /etc path, which is absent on
    # every machine this runs on and is skipped as a missing layer.
    yield
