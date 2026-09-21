"""Tests for measured quality: the sidecar a config references.

Covers `quality_from:` on the config, its resolution relative to the
config file rather than the CWD, the precedence that lets the measured
sidecar beat a hand-set `quality:` unless `quality_from_override:
false` says otherwise, the warning for a sidecar naming an endpoint the
registry does not carry, the error for a missing file, and the shape
the evals harness writes.

The merge runs inside `EndpointRegistry.from_config`, so these build a
registry from a config dict rather than calling any one consumer of it.

Not one of these makes a network call: the harness's benchmark and its
completion are patched.
"""

import logging
from pathlib import Path

import pytest
import yaml

import routellm.pairing as pairing
from routellm.endpoints import EndpointRegistry, Selector
from routellm.pairing import rank_candidates
from routellm.quality_scores import load_sidecar

REPO_ROOT = Path(__file__).resolve().parents[2]

SIDECAR = {
    "version": 1,
    "generated_at": "2026-09-21T10:00:00Z",
    "min_samples": 50,
    "transform": "linear",
    "source": "evals",
    "endpoints": {
        "cloud_strong": {"quality": 91, "n": 50, "by_area": {}},
        "local_server": {"quality": 78, "n": 50, "by_area": {}},
    },
}

CONFIG = {
    "endpoints": {
        "cloud_strong": {"model": "gpt-4o"},
        "local_server": {"model": "ollama_chat/qwen3:8b"},
    }
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    """Keep every test off the network and out of the real cache."""
    monkeypatch.setenv(pairing.CATALOG_CACHE_ENV, str(tmp_path / "catalog.json"))
    monkeypatch.setattr(pairing, "_fetch_catalog", lambda: [])


def _write(tmp_path, config, sidecar=SIDECAR, sidecar_name="quality.yaml"):
    """Write a config and its sidecar, returning the config path."""
    if sidecar is not None:
        (tmp_path / sidecar_name).write_text(yaml.safe_dump(sidecar))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_sidecar_fills_missing_quality(tmp_path):
    config = {**CONFIG, "quality_from": "quality.yaml"}
    path = _write(tmp_path, config)

    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality == 91
    assert registry.get("local_server").quality == 78


def test_sidecar_beats_an_explicit_quality_by_default(tmp_path):
    """The default: a measurement replaces a hand-written guess.

    No `quality_from_override:` key at all, which is the same thing as
    setting it true.
    """
    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "quality": 42},
            "local_server": {"model": "ollama_chat/qwen3:8b"},
        },
        "quality_from": "quality.yaml",
    }
    path = _write(tmp_path, config)

    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality == 91
    assert registry.get("local_server").quality == 78


def test_quality_from_override_true_beats_an_explicit_quality(tmp_path):
    """Spelling the default out explicitly reads the same."""
    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "quality": 42},
            "local_server": {"model": "ollama_chat/qwen3:8b"},
        },
        "quality_from": "quality.yaml",
        "quality_from_override": True,
    }
    path = _write(tmp_path, config)

    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality == 91


def test_explicit_quality_wins_when_override_is_false(tmp_path):
    """`quality_from_override: false` flips the precedence back.

    An endpoint that set its own number keeps it; one that set none is
    still filled from the sidecar, so turning the flip on does not turn
    the measurement off.
    """
    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "quality": 42},
            "local_server": {"model": "ollama_chat/qwen3:8b"},
        },
        "quality_from": "quality.yaml",
        "quality_from_override": False,
    }
    path = _write(tmp_path, config)

    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality == 42
    assert registry.get("local_server").quality == 78


def test_quality_from_resolves_relative_to_the_config_file(tmp_path):
    nested = tmp_path / "conf"
    nested.mkdir()
    (nested / "quality.yaml").write_text(yaml.safe_dump(SIDECAR))
    path = nested / "config.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, "quality_from": "quality.yaml"}))

    # Deliberately NOT chdir'ing into `nested`: the path is relative to
    # the config file, never to the CWD.
    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality == 91


def test_missing_sidecar_file_raises_naming_the_path(tmp_path):
    """A named sidecar that is not there is fatal, not a shrug.

    `load_sidecar` reports it as FileNotFoundError, and `from_config`
    lets that out unwrapped: a config naming a file it cannot read is
    an operator error, and the message names both the path and the key
    that asked for it.
    """
    config = {**CONFIG, "quality_from": "absent.yaml"}
    path = _write(tmp_path, config, sidecar=None)

    with pytest.raises(FileNotFoundError) as excinfo:
        EndpointRegistry.from_config(
            yaml.safe_load(path.read_text()), config_path=path
        )

    message = str(excinfo.value)
    assert str(tmp_path / "absent.yaml") in message
    assert "quality_from" in message


def test_unknown_endpoint_in_the_sidecar_warns_and_is_ignored(tmp_path, caplog):
    sidecar = {
        **SIDECAR,
        "endpoints": {
            **SIDECAR["endpoints"],
            "retired_model": {"quality": 55, "n": 50, "by_area": {}},
        },
    }
    path = _write(tmp_path, {**CONFIG, "quality_from": "quality.yaml"}, sidecar)

    with caplog.at_level(logging.WARNING, logger="routellm.quality_scores"):
        registry = EndpointRegistry.from_config(
            yaml.safe_load(path.read_text()), config_path=path
        )

    assert "retired_model" not in registry.names()
    assert any("retired_model" in record.message for record in caplog.records)


def test_sidecar_quality_orders_a_quality_desc_selector(tmp_path):
    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "tags": ["pool"]},
            "local_server": {"model": "ollama_chat/qwen3:8b", "tags": ["pool"]},
        },
        "quality_from": "quality.yaml",
    }
    path = _write(tmp_path, config)
    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    ranked = [
        name
        for name, _ in rank_candidates(
            registry, Selector(select="tag:pool", order="quality_desc")
        )
    ]

    assert ranked == ["cloud_strong", "local_server"]


def test_a_config_without_quality_from_is_untouched(tmp_path):
    path = _write(tmp_path, CONFIG)

    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    assert registry.get("cloud_strong").quality is None


def test_load_sidecar_reads_the_documented_shape(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(SIDECAR))

    sidecar = load_sidecar(path)

    assert sidecar.version == 1
    assert sidecar.generated_at == "2026-09-21T10:00:00Z"
    assert sidecar.min_samples == 50
    assert sidecar.transform == "linear"
    assert sidecar.source == "evals"
    assert set(sidecar.endpoints) == {"cloud_strong", "local_server"}
    assert sidecar.endpoints["cloud_strong"].quality == 91
    assert sidecar.endpoints["cloud_strong"].n == 50
    assert sidecar.endpoints["cloud_strong"].by_area == {}
    assert sidecar.endpoints["local_server"].quality == 78
    assert sidecar.endpoints["local_server"].n == 50


def test_harness_writes_the_sidecar_shape(tmp_path, monkeypatch):
    from routellm.evals import endpoint_quality

    monkeypatch.setattr(
        endpoint_quality,
        "_score_endpoint",
        lambda controller, name, limit: (0.91, 50)
        if name == "cloud_strong"
        else (0.78, 50),
    )
    monkeypatch.setattr(
        endpoint_quality, "_build_controller", lambda config, name, path: object()
    )

    out = tmp_path / "quality.yaml"
    config = _write(tmp_path, CONFIG)

    code = endpoint_quality.main(
        [
            "--config",
            str(config),
            "--endpoints",
            "cloud_strong,local_server",
            "--limit",
            "50",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    written = yaml.safe_load(out.read_text())

    assert written["version"] == 1
    assert written["source"] == "evals"
    assert written["min_samples"] == 50
    assert written["transform"] == "linear"
    assert written["endpoints"] == {
        "cloud_strong": {"quality": 91, "n": 50, "by_area": {}},
        "local_server": {"quality": 78, "n": 50, "by_area": {}},
    }
    assert written["generated_at"].endswith("Z")


def test_harness_rejects_an_endpoint_the_config_does_not_carry(tmp_path):
    from routellm.evals import endpoint_quality

    config = _write(tmp_path, CONFIG)

    code = endpoint_quality.main(
        [
            "--config",
            str(config),
            "--endpoints",
            "nope",
            "--out",
            str(tmp_path / "q.yaml"),
        ]
    )

    assert code == 1


def test_matrix_marks_quality_measured_or_manual(tmp_path):
    """The matrix says where each number came from.

    `quality_from_override: false` is what makes a table carrying both
    kinds at once: `cloud_strong` keeps the 42 an operator typed, and
    `local_server`, which typed none, is filled from the sidecar. Under
    the default the sidecar would win both cells and there would be no
    manual number left to label.
    """
    from routellm.pairing import _capability_matrix

    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "quality": 42},
            "local_server": {"model": "ollama_chat/qwen3:8b"},
        },
        "quality_from": "quality.yaml",
        "quality_from_override": False,
    }
    path = _write(tmp_path, config)
    registry = EndpointRegistry.from_config(
        yaml.safe_load(path.read_text()), config_path=path
    )

    rendered = _capability_matrix(registry, [])
    cloud = next(
        line for line in rendered.splitlines() if line.startswith("cloud_strong")
    )
    local = next(
        line for line in rendered.splitlines() if line.startswith("local_server")
    )

    assert "42 (manual)" in cloud
    assert "78 (measured)" in local


def test_the_explain_cli_resolves_quality_from_against_the_config(
    tmp_path, capsys, monkeypatch
):
    """The CLI must not resolve a relative sidecar against the CWD.

    Regression: `_explain` and the matrix both built the registry
    without passing the config path, so `quality_from: quality.yaml`
    was looked for next to wherever the operator happened to stand.
    """
    from routellm.pairing import main

    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "quality": 42},
            "local_server": {"model": "ollama_chat/qwen3:8b"},
        },
        "quality_from": "quality.yaml",
        # As in the matrix test: keeps one manual number next to one
        # measured one, so the table proves the sidecar was read at all.
        "quality_from_override": False,
    }
    path = _write(tmp_path, config)

    # Stand somewhere the sidecar is NOT, which is the whole point.
    monkeypatch.chdir(tmp_path.parent)

    assert main(["--config", str(path), "--capabilities"]) == 0

    out = capsys.readouterr().out
    assert "78 (measured)" in out
    assert "42 (manual)" in out


def test_the_explain_cli_without_the_matrix_also_resolves_it(
    tmp_path, capsys, monkeypatch
):
    from routellm.pairing import main

    config = {
        "endpoints": {
            "cloud_strong": {"model": "gpt-4o", "tags": ["pool"]},
            "local_server": {"model": "ollama_chat/qwen3:8b", "tags": ["pool"]},
        },
        "quality_from": "quality.yaml",
        "tiers": {
            "default": {
                "router": "random",
                "threshold": 0.5,
                "strong": {"select": "tag:pool", "order": "quality_desc"},
                "weak": "local_server",
            }
        },
    }
    path = _write(tmp_path, config)
    monkeypatch.chdir(tmp_path.parent)

    assert main(["--config", str(path)]) == 0

    out = capsys.readouterr().out
    # The sidecar's 91 orders cloud_strong first; without it both are
    # unrated and the tie breaks on name, putting cloud_strong first
    # anyway -- so assert the score itself is what the table printed.
    assert "quality=91" in out


@pytest.mark.slow
def test_the_server_registry_resolves_quality_from_against_the_config(tmp_path):
    """`build_registry` must pass the config path through too.

    In a subprocess: `routellm.openai_server` parses `sys.argv` at
    import, so it cannot be imported under pytest's own argv. The
    subprocess also stands OUTSIDE the config's directory, which is
    what makes the CWD-versus-config-file distinction observable.
    """
    import subprocess
    import sys

    path = _write(tmp_path, {**CONFIG, "quality_from": "quality.yaml"})
    script = (
        "import sys, yaml\n"
        f"sys.argv = ['openai_server', '--config', {str(path)!r}, "
        "'--routers', 'random']\n"
        "import routellm.openai_server as server\n"
        f"cfg = yaml.safe_load(open({str(path)!r}))\n"
        f"r = server.build_registry(cfg, {str(path)!r})\n"
        "print(r.get('local_server').quality)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": ".",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ROUTELLM_CATALOG_CACHE": str(tmp_path / "catalog.json"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "78"
