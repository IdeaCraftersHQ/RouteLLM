"""Registration, auto-discovery and mounting: adding a dialect edits nothing shared."""

import importlib
import sys
import textwrap

import pytest
from fastapi import FastAPI

from routellm.adapters import Adapter, Route, adapters, discover, mount_adapters, register
from routellm.adapters import registry as registry_module
from routellm.adapters.registry import MODULES_ENV


class Dummy(Adapter):
    name = "dummy"
    routes = (Route("POST", "/dummy/chat"),)


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTRY", {})


def test_register_adds_an_instance_by_name(clean_registry):
    assert register(Dummy) is Dummy
    assert [type(a) for a in adapters()] == [Dummy]
    assert adapters()[0].name == "dummy"


def test_registering_the_same_class_twice_is_idempotent(clean_registry):
    register(Dummy)
    register(Dummy)
    assert len(adapters()) == 1


def test_a_second_class_under_a_taken_name_is_refused(clean_registry):
    register(Dummy)

    class Other(Adapter):
        name = "dummy"
        routes = (Route("GET", "/other"),)

    with pytest.raises(ValueError, match="dummy"):
        register(Other)


@pytest.mark.parametrize("attrs", [{}, {"name": "x"}, {"routes": (Route("GET", "/x"),)}])
def test_an_adapter_needs_a_name_and_routes(clean_registry, attrs):
    with pytest.raises(TypeError, match="name"):
        register(type("Bad", (Adapter,), attrs))


def test_routes_mount_and_a_claimed_path_is_an_error(clean_registry):
    class Clash(Adapter):
        name = "clash"
        routes = (Route("POST", "/dummy/chat"),)

    app = FastAPI()
    with pytest.raises(ValueError, match=r"dummy.*clash|clash.*dummy"):
        mount_adapters(app, object(), [Dummy(), Clash()])
    app = FastAPI()
    mount_adapters(app, object(), [Dummy()])
    assert "/dummy/chat" in {r.path for r in app.routes}


def test_same_path_with_another_method_is_not_a_clash(clean_registry):
    class Get(Adapter):
        name = "get"
        routes = (Route("GET", "/dummy/chat", operation="x", body="none"),)

    mount_adapters(FastAPI(), object(), [Dummy(), Get()])


def test_discover_imports_every_module_of_the_package_and_the_env_modules(
    clean_registry, tmp_path, monkeypatch
):
    (tmp_path / "extra_dialect.py").write_text(
        textwrap.dedent(
            """
            from routellm.adapters import Adapter, Route, register

            @register
            class Extra(Adapter):
                name = "extra"
                routes = (Route("POST", "/extra"),)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(MODULES_ENV, "extra_dialect")
    try:
        assert "extra" in [a.name for a in discover()]
    finally:
        sys.modules.pop("extra_dialect", None)


def test_a_module_that_fails_to_import_stops_discovery_naming_it(
    clean_registry, tmp_path, monkeypatch
):
    (tmp_path / "broken_dialect.py").write_text("raise ImportError('no sdk')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(MODULES_ENV, "broken_dialect")
    importlib.invalidate_caches()
    with pytest.raises(ImportError, match="no sdk"):
        discover()


def test_importing_the_package_does_not_import_the_server():
    # The server parses sys.argv at import; adapters must be importable without it.
    code = (
        "import sys; sys.argv=['x','--bogus']; import routellm.adapters;"
        "assert 'routellm.openai_server' not in sys.modules"
    )
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "."},
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, result.stderr[-500:]
