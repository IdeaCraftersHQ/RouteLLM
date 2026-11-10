"""A `local` endpoint's `api_base` must name a loopback/private/internal host."""

import pytest

from routellm.endpoints import EndpointRegistry


def _load(base, **extra):
    spec = {"model": "openai/m", "tags": ["local"], "api_base": base, **extra}
    return EndpointRegistry.from_config({"endpoints": {"leaf": spec}})


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:11434/v1",
        "http://127.8.9.1/v1",
        "http://localhost:8000",
        "http://LOCALHOST:8000",
        "http://[::1]:8000/v1",
        "http://10.1.2.3/v1",
        "http://172.16.0.5:80",
        "http://172.31.255.1",
        "http://192.168.1.20:8080/v1",
        "http://169.254.10.10",
        "http://[fd12:3456::1]:80",
        "http://[fe80::1]",
        "http://box.local:80",
        "http://box.local./v1",
        "http://gpu.internal/v1",
        "http://app.localhost:3000",
        "localhost:11434",  # scheme-less
        "http://user:pw@127.0.0.1:1/v1",
    ],
)
def test_local_hosts_load(base):
    assert _load(base).get("leaf").api_base == base


@pytest.mark.parametrize(
    "base",
    [
        "https://api.openai.com/v1",
        "https://api.openai.com",
        "http://8.8.8.8/v1",
        "http://172.32.0.1",  # just outside RFC1918 172.16/12
        "http://100.100.1.1:80",  # shared/CGNAT space: not this network
        "http://gpu.lan/v1",  # .lan: unreserved, resolver-defined
        "http://ollama:11434",  # single-label: search domains may resolve it anywhere
        "http://0.0.0.0:11434",  # unspecified
        "http://192.0.2.1",  # documentation range, not private
        "http://[::ffff:8.8.8.8]",  # mapped public v4
        "http://192.169.0.1",
        "http://[2001:4860:4860::8888]",
        "https://evil.example.com/v1",
        "https://localhost.evil.com/v1",
        "https://box.local.evil.com/v1",
        "https://notinternal.com",
        "http://127.0.0.1.evil.com",
        "https://evil.com/x@127.0.0.1",
        "http://",
        "http://:8080",
    ],
)
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
