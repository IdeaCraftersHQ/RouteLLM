"""Environment interpolation in config values, and the layering it rides on.

Syntax: `${VAR}`, `${VAR:-default}`, `${VAR:?message}`, `$$` for a
literal dollar; a bare `$NAME` is left alone. Interpolation runs on the
merged mapping, string scalars only, never on keys, and never on a
secret-by-name field.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from routellm import config as cfg
from routellm.config import ConfigError
from routellm.endpoints import EndpointRegistry
from routellm.tests.test_config_discovery import env  # noqa: F401  (fixture)


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _base(loaded, name="a"):
    return loaded.data["endpoints"][name]["api_base"]


def _endpoint_file(env, value, name="a"):
    return _write(
        env.user,
        f"endpoints:\n  {name}:\n    model: openai/m\n    api_base: {value!r}\n",
    )


# --- syntax ---------------------------------------------------------------


def test_set_variable_is_substituted(env, monkeypatch):
    monkeypatch.setenv("ROUTE_X_BASE", "http://127.0.0.1:9/v1")
    _endpoint_file(env, "${ROUTE_X_BASE}")
    assert _base(cfg.load_config()) == "http://127.0.0.1:9/v1"


def test_variable_inside_a_longer_string(env, monkeypatch):
    monkeypatch.setenv("ROUTE_HOST", "gpu1")
    _endpoint_file(env, "http://${ROUTE_HOST}:11434/v1")
    assert _base(cfg.load_config()) == "http://gpu1:11434/v1"


def test_unset_variable_is_a_load_error_naming_file_key_and_variable(env, monkeypatch):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    path = _endpoint_file(env, "${ROUTE_NOPE}")
    with pytest.raises(ConfigError) as err:
        cfg.load_config()
    msg = str(err.value)
    assert str(path) in msg
    assert "endpoints.a.api_base" in msg
    assert "ROUTE_NOPE" in msg


def test_default_is_used_when_unset(env, monkeypatch):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    _endpoint_file(env, "${ROUTE_NOPE:-http://127.0.0.1:1/v1}")
    assert _base(cfg.load_config()) == "http://127.0.0.1:1/v1"


def test_default_is_used_when_empty_and_ignored_when_set(env, monkeypatch):
    _endpoint_file(env, "${ROUTE_V:-d}")
    monkeypatch.setenv("ROUTE_V", "")
    assert _base(cfg.load_config()) == "d"
    monkeypatch.setenv("ROUTE_V", "real")
    assert _base(cfg.load_config()) == "real"


def test_required_form_errors_with_the_message(env, monkeypatch):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    _endpoint_file(env, "${ROUTE_NOPE:?set ROUTE_NOPE to the box url}")
    with pytest.raises(ConfigError) as err:
        cfg.load_config()
    assert "set ROUTE_NOPE to the box url" in str(err.value)
    assert "endpoints.a.api_base" in str(err.value)


def test_double_dollar_is_a_literal_dollar(env, monkeypatch):
    monkeypatch.setenv("ROUTE_V", "x")
    _endpoint_file(env, "a$$b $${ROUTE_V} ${ROUTE_V}")
    assert _base(cfg.load_config()) == "a$b ${ROUTE_V} x"


def test_bare_dollar_name_is_untouched(env, monkeypatch):
    monkeypatch.setenv("HOME_X", "nope")
    _endpoint_file(env, "Bearer $HOME_X and $5")
    assert _base(cfg.load_config()) == "Bearer $HOME_X and $5"


def test_malformed_reference_is_an_error(env):
    _endpoint_file(env, "http://${unterminated")
    with pytest.raises(ConfigError, match="endpoints.a.api_base"):
        cfg.load_config()


def test_keys_are_never_interpolated(env, monkeypatch):
    monkeypatch.setenv("ROUTE_K", "renamed")
    _write(env.user, "weird:\n  '${ROUTE_K}': 1\n")
    assert cfg.load_config().data["weird"] == {"${ROUTE_K}": 1}


def test_list_items_are_interpolated(env, monkeypatch):
    monkeypatch.setenv("ROUTE_T", "fast")
    _write(env.user, "endpoints:\n  a:\n    model: m\n    tags: ['${ROUTE_T}', plain]\n")
    assert cfg.load_config().data["endpoints"]["a"]["tags"] == ["fast", "plain"]


def test_non_strings_pass_through(env):
    _write(env.user, "endpoints:\n  a:\n    model: m\n    quality: 50\n    strict: true\n")
    a = cfg.load_config().data["endpoints"]["a"]
    assert a["quality"] == 50 and a["strict"] is True


def test_secret_by_name_fields_refuse_interpolation(env, monkeypatch):
    monkeypatch.setenv("ROUTE_S", "sk-1")
    _write(
        env.user,
        "endpoints:\n  a:\n    model: m\n    extra:\n      api_key: '${ROUTE_S}'\n",
    )
    with pytest.raises(ConfigError, match=r"api_key_env|secret"):
        cfg.load_config()


def test_api_key_env_is_a_plain_name_not_interpolated(env):
    _write(env.user, "endpoints:\n  a:\n    model: m\n    api_key_env: MY_KEY\n")
    assert cfg.load_config().data["endpoints"]["a"]["api_key_env"] == "MY_KEY"


# --- layering -------------------------------------------------------------


def test_higher_layer_literal_overrides_lower_layer_variable(env, monkeypatch):
    """The overridden variable is never read, so it need not be set."""
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    _endpoint_file(env, "${ROUTE_NOPE}")
    flag = _write(
        env.project / "flag.yaml",
        "endpoints:\n  a:\n    api_base: http://127.0.0.1:2/v1\n",
    )
    loaded = cfg.load_config(explicit=flag)
    assert _base(loaded) == "http://127.0.0.1:2/v1"


def test_higher_layer_variable_overrides_lower_literal(env, monkeypatch):
    monkeypatch.setenv("ROUTE_HI", "http://hi/v1")
    _endpoint_file(env, "http://lo/v1")
    flag = _write(env.project / "flag.yaml", "endpoints:\n  a:\n    api_base: ${ROUTE_HI}\n")
    assert _base(cfg.load_config(explicit=flag)) == "http://hi/v1"


def test_env_layer_and_flag_layer_still_overlay(env, monkeypatch):
    monkeypatch.setenv("ROUTE_ENVL", "http://env/v1")
    monkeypatch.setenv("ROUTE_FLAGL", "http://flag/v1")
    _endpoint_file(env, "http://user/v1")
    layer = _write(env.project / "env.yaml", "endpoints:\n  a:\n    api_base: ${ROUTE_ENVL}\n")
    monkeypatch.setenv("ROUTELLM_CONFIG", str(layer))
    assert _base(cfg.load_config()) == "http://env/v1"
    flag = _write(env.project / "flag.yaml", "endpoints:\n  a:\n    api_base: ${ROUTE_FLAGL}\n")
    assert _base(cfg.load_config(explicit=flag)) == "http://flag/v1"


def test_error_names_the_layer_file_that_set_the_value(env, monkeypatch):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    _endpoint_file(env, "http://ok/v1")
    flag = _write(env.project / "flag.yaml", "endpoints:\n  a:\n    api_base: ${ROUTE_NOPE}\n")
    with pytest.raises(ConfigError) as err:
        cfg.load_config(explicit=flag)
    assert str(flag) in str(err.value)
    assert str(env.user) not in str(err.value)


# --- inspection CLI -------------------------------------------------------


def _cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "routellm.config", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_show_prints_uninterpolated_text_and_the_variables_read(env, monkeypatch):
    monkeypatch.setenv("ROUTE_SHOWN", "http://resolved/v1")
    flag = _write(
        env.project / "flag.yaml", "endpoints:\n  a:\n    model: m\n    api_base: ${ROUTE_SHOWN}\n"
    )
    code, out = _run_main(["show", "--config", str(flag)])
    assert code == 0
    assert "${ROUTE_SHOWN}" in out
    assert "http://resolved/v1" not in out
    assert "endpoints.a.api_base" in out and "ROUTE_SHOWN" in out


def test_show_does_not_need_the_variables_set(env, monkeypatch):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    flag = _write(
        env.project / "flag.yaml", "endpoints:\n  a:\n    model: m\n    api_base: ${ROUTE_NOPE}\n"
    )
    code, out = _run_main(["show", "--config", str(flag)])
    assert code == 0 and "${ROUTE_NOPE}" in out


def test_show_resolve_prints_resolved_values(env, monkeypatch):
    monkeypatch.setenv("ROUTE_SHOWN", "http://resolved/v1")
    flag = _write(
        env.project / "flag.yaml", "endpoints:\n  a:\n    model: m\n    api_base: ${ROUTE_SHOWN}\n"
    )
    code, out = _run_main(["show", "--resolve", "--config", str(flag)])
    assert code == 0
    assert "http://resolved/v1" in out


def test_show_resolve_unset_variable_exits_1(env, monkeypatch, capsys):
    monkeypatch.delenv("ROUTE_NOPE", raising=False)
    flag = _write(
        env.project / "flag.yaml", "endpoints:\n  a:\n    model: m\n    api_base: ${ROUTE_NOPE}\n"
    )
    code, _ = _run_main(["show", "--resolve", "--config", str(flag)])
    assert code == 1
    assert "ROUTE_NOPE" in capsys.readouterr().err


def _run_main(argv):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cfg.main(argv)
    return code, buf.getvalue()


# --- the example config still loads --------------------------------------


def test_example_config_loads_unchanged(env, monkeypatch):
    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    for name in ("ROUTE_LOCAL_BASE",):
        monkeypatch.delenv(name, raising=False)
    loaded = cfg.load_config(explicit=example)
    EndpointRegistry.from_config(loaded.data)


# --- on the wire ----------------------------------------------------------

_WIRE = """
import json, os, sys
from routellm.tests.fake_upstream import FakeUpstream
A = FakeUpstream("A")
B = FakeUpstream("B")
os.environ["ROUTE_A_BASE"] = A.base
open({config!r}, "w").write(
    "endpoints:\\n"
    "  a: {{model: openai/model-a, api_base: '${{ROUTE_A_BASE}}', api_key_env: FAKE_KEY}}\\n"
    "  b: {{model: openai/model-b, api_base: '" + B.base + "', api_key_env: FAKE_KEY}}\\n"
    "tiers:\\n"
    "  default: {{router: random, threshold: 0.0, strong: a, weak: b}}\\n"
)
sys.argv = ["x", "--config", {config!r}, "--routers", "random"]
import routellm.openai_server as server
from fastapi.testclient import TestClient
with TestClient(server.app, raise_server_exceptions=False) as client:
    server.CONTROLLER.resilience.config.max_retries = 0
    for _ in range(12):
        client.post("/v1/chat/completions", json={{"model": "default",
            "messages": [{{"role": "user", "content": "hi"}}]}})
print("@@RESULT@@" + json.dumps({{"a": len(A.hits), "b": len(B.hits), "base": A.base}}))
"""


@pytest.mark.slow
def test_interpolated_base_is_the_one_hit_on_the_wire(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _WIRE.format(config=str(tmp_path / "c.yaml"))],
        cwd=repo,
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
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(ln for ln in reversed(result.stdout.splitlines()) if "@@RESULT@@" in ln)
    out = json.loads(line.split("@@RESULT@@", 1)[1])
    assert out["a"] > 0, out
