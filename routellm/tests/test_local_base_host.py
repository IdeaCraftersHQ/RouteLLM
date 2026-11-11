"""A `local` endpoint's `api_base` must name a loopback/private/internal host."""

import pytest

from routellm.endpoints import EndpointRegistry
from routellm.tests.local_hosts import LOCAL_BASES, NON_LOCAL_BASES


def _load(base, **extra):
    spec = {"model": "openai/m", "tags": ["local"], "api_base": base, **extra}
    return EndpointRegistry.from_config({"endpoints": {"leaf": spec}})


@pytest.mark.parametrize("base", LOCAL_BASES)
def test_local_hosts_load(base):
    assert _load(base).get("leaf").api_base == base


@pytest.mark.parametrize("base", NON_LOCAL_BASES)
def test_public_or_hostless_base_is_refused(base):
    with pytest.raises(ValueError) as err:
        _load(base)
    msg = str(err.value)
    assert "leaf" in msg and "api_base" in msg and "allow_public_base" in msg


def test_opt_out_allows_a_public_base():
    ep = _load("https://api.example.com/v1", allow_public_base=True).get("leaf")
    assert ep.api_base == "https://api.example.com/v1"


def test_opt_out_must_be_a_real_bool():
    with pytest.raises(ValueError):
        _load("https://api.example.com/v1", allow_public_base="yes")


def test_untagged_endpoint_may_use_a_public_base():
    spec = {"model": "gpt-4o", "api_base": "https://api.example.com/v1"}
    EndpointRegistry.from_config({"endpoints": {"leaf": spec}})


def test_interpolated_base_is_checked_after_expansion(monkeypatch, tmp_path):
    from routellm.config import load_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "endpoints:\n  leaf:\n    model: openai/m\n    tags: [local]\n"
        "    api_base: ${LB_TEST_BASE}\n"
    )
    monkeypatch.setenv("LB_TEST_BASE", "https://api.example.com/v1")
    with pytest.raises(ValueError, match="allow_public_base"):
        EndpointRegistry.from_config(load_config(explicit=str(cfg)).data)
    monkeypatch.setenv("LB_TEST_BASE", "http://10.0.0.2/v1")
    EndpointRegistry.from_config(load_config(explicit=str(cfg)).data)


def test_empty_base_is_still_refused_as_missing():
    with pytest.raises(ValueError, match="api_base"):
        _load("")
