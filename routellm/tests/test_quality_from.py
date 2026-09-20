"""Reading the measured quality sidecar from config: `quality_from`.

The sidecar wins over a hand-written YAML `quality` by default: it is
measured, and the YAML number is a guess someone typed once and never
revisited. `quality_from_override: false` flips that. Either way an
endpoint the sidecar does not name is never touched.
`routellm.openai_server` parses `sys.argv` at import, so the two tests
that touch it import it behind a neutral argv, as the other server
tests do.
"""

import logging
import sys

import pytest
import yaml

from routellm.endpoints import Endpoint, EndpointRegistry, Selector, Tier
from routellm.pairing import resolve_pairing
from routellm.quality_scores import apply_sidecar, load_sidecar


def _sidecar(endpoints=None, **kwargs):
    body = {
        "version": 1,
        "generated_at": "2026-09-20T18:03:11Z",
        "min_samples": 30,
        "transform": "linear",
        "source": "traces",
        "endpoints": endpoints
        if endpoints is not None
        else {"local_fast": {"quality": 72, "n": 412, "by_area": {}}},
    }
    body.update(kwargs)
    return body


def _write(tmp_path, name="quality.yaml", **kwargs):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(_sidecar(**kwargs), sort_keys=False))
    return path


def _registry(**qualities):
    return EndpointRegistry(
        endpoints={
            name: Endpoint(name=name, model=f"m/{name}", quality=quality)
            for name, quality in qualities.items()
        }
    )


def _server(monkeypatch):
    """Import `openai_server` behind a neutral argv."""
    monkeypatch.setattr(sys, "argv", ["routellm.openai_server"])
    import routellm.openai_server as server

    return server


def test_no_quality_from_leaves_every_endpoint_untouched(tmp_path, monkeypatch):
    """The backward-compatible default: no key, no change."""
    apply_quality_from = _server(monkeypatch).apply_quality_from

    registry = _registry(local_fast=60, cloud_strong=88)
    changed = apply_quality_from(registry, {"endpoints": {}}, None)

    assert changed == 0
    assert registry.get("local_fast").quality == 60
    assert registry.get("cloud_strong").quality == 88


def test_sidecar_overrides_the_yaml_quality_and_logs_it(tmp_path, caplog):
    registry = _registry(local_fast=60)
    sidecar = load_sidecar(str(_write(tmp_path)))

    with caplog.at_level(logging.INFO, logger="routellm.quality_scores"):
        changed = apply_sidecar(registry, sidecar, override=True)

    assert changed == 1
    assert registry.get("local_fast").quality == 72
    lines = [r.getMessage() for r in caplog.records if "local_fast" in r.getMessage()]
    assert len(lines) == 1
    assert "60" in lines[0] and "72" in lines[0] and "412" in lines[0]


def test_override_false_keeps_an_explicit_yaml_quality(caplog, tmp_path):
    registry = _registry(local_fast=60)
    sidecar = load_sidecar(str(_write(tmp_path)))

    with caplog.at_level(logging.INFO, logger="routellm.quality_scores"):
        changed = apply_sidecar(registry, sidecar, override=False)

    assert changed == 0
    assert registry.get("local_fast").quality == 60
    assert [r for r in caplog.records if "declined" in r.getMessage()]


def test_override_false_still_fills_an_endpoint_with_no_quality(tmp_path):
    registry = _registry(local_fast=None)
    sidecar = load_sidecar(str(_write(tmp_path)))

    assert apply_sidecar(registry, sidecar, override=False) == 1
    assert registry.get("local_fast").quality == 72


def test_an_endpoint_absent_from_the_sidecar_is_untouched(tmp_path):
    registry = _registry(local_fast=60, cloud_strong=88, unmeasured=None)
    sidecar = load_sidecar(str(_write(tmp_path)))

    apply_sidecar(registry, sidecar, override=True)

    assert registry.get("cloud_strong").quality == 88
    assert registry.get("unmeasured").quality is None


def test_wrong_version_names_both_versions(tmp_path):
    path = _write(tmp_path, version=7)

    with pytest.raises(ValueError) as excinfo:
        load_sidecar(str(path))

    message = str(excinfo.value)
    assert "7" in message and "1" in message


def test_missing_file_names_the_path_and_the_key(tmp_path):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_sidecar(str(missing))

    message = str(excinfo.value)
    assert str(missing) in message
    assert "quality_from" in message


def test_malformed_yaml_names_the_path(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text("version: 1\n  bad: [indent\n")

    with pytest.raises(ValueError) as excinfo:
        load_sidecar(str(path))

    assert str(path) in str(excinfo.value)


def test_relative_path_resolves_against_the_config_file(tmp_path, monkeypatch):
    apply_quality_from = _server(monkeypatch).apply_quality_from

    config_dir = tmp_path / "etc"
    config_dir.mkdir()
    _write(config_dir)
    config_path = config_dir / "routellm.yaml"
    config_path.write_text("quality_from: ./quality.yaml\n")

    registry = _registry(local_fast=60)
    changed = apply_quality_from(
        registry, {"quality_from": "./quality.yaml"}, str(config_path)
    )

    assert changed == 1
    assert registry.get("local_fast").quality == 72


def test_stale_sidecar_warns_but_loads(tmp_path, caplog):
    path = _write(tmp_path, generated_at="2020-01-01T00:00:00Z")

    with caplog.at_level(logging.WARNING, logger="routellm.quality_scores"):
        sidecar = load_sidecar(str(path))

    # An old measurement still beats no measurement, so this is never
    # an error.
    assert sidecar.endpoints["local_fast"].quality == 72
    assert [r for r in caplog.records if "stale" in r.getMessage().lower()]


def test_selector_ordering_uses_the_merged_quality(tmp_path):
    """Two endpoints whose YAML order reverses their measured order."""
    registry = EndpointRegistry(
        endpoints={
            "alpha": Endpoint(name="alpha", model="m/alpha", quality=90, tags=["chat"]),
            "beta": Endpoint(name="beta", model="m/beta", quality=10, tags=["chat"]),
        }
    )
    selector = Selector(select="tag:chat", order="quality_desc")
    assert resolve_pairing(registry, selector) == "alpha"

    sidecar = load_sidecar(
        str(
            _write(
                tmp_path,
                endpoints={
                    "alpha": {"quality": 10, "n": 400, "by_area": {}},
                    "beta": {"quality": 90, "n": 400, "by_area": {}},
                },
            )
        )
    )
    apply_sidecar(registry, sidecar, override=True)

    assert resolve_pairing(registry, selector) == "beta"


def test_server_pops_quality_keys_out_of_router_config(tmp_path, monkeypatch):
    """An unknown key handed to every router is a config error."""
    openai_server = _server(monkeypatch)

    _write(tmp_path)
    config = tmp_path / "routellm.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "endpoints": {"local_fast": {"model": "m/local", "quality": 60}},
                "quality_from": "./quality.yaml",
                "quality_from_override": True,
                "mf": {"checkpoint_path": "x"},
            },
            sort_keys=False,
        )
    )

    router_config = openai_server.build_router_config(
        yaml.safe_load(config.read_text())
    )

    assert "quality_from" not in router_config
    assert "quality_from_override" not in router_config
    assert "endpoints" not in router_config
    assert router_config["mf"] == {"checkpoint_path": "x"}


def test_area_quality_lands_on_the_registry(tmp_path):
    sidecar = load_sidecar(
        str(
            _write(
                tmp_path,
                endpoints={
                    "local_fast": {
                        "quality": 72,
                        "n": 412,
                        "by_area": {
                            "coding": {"quality": 81, "n": 260},
                            "copywriting": {"quality": 57, "n": 152},
                        },
                    }
                },
            )
        )
    )
    registry = _registry(local_fast=60)
    apply_sidecar(registry, sidecar, override=True)

    assert registry.area_quality["local_fast"]["coding"] == 81
    assert registry.area_quality["local_fast"]["copywriting"] == 57
