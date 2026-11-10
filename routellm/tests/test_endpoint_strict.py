"""Endpoint config is strict: unknown keys refused, `local` needs a base."""

import pytest

from routellm.endpoints import EndpointRegistry


def _load(spec):
    return EndpointRegistry.from_config({"endpoints": {"leaf": spec}})


def test_unknown_key_is_refused_naming_endpoint_and_key():
    with pytest.raises(ValueError) as err:
        _load({"model": "m", "api_base_env": "X"})
    msg = str(err.value)
    assert "leaf" in msg and "api_base_env" in msg


def test_near_miss_gets_a_did_you_mean():
    with pytest.raises(ValueError, match=r"did you mean 'api_base'"):
        _load({"model": "m", "api_bse": "http://x"})


def test_far_miss_has_no_suggestion():
    with pytest.raises(ValueError) as err:
        _load({"model": "m", "zzzzzzzz": 1})
    assert "did you mean" not in str(err.value)


def test_known_keys_still_load():
    _load({"model": "m", "api_base": "http://x", "tags": ["a"], "extra": {"anything": 1}})


def test_local_endpoint_requires_an_explicit_api_base():
    with pytest.raises(ValueError) as err:
        _load({"model": "openai/m", "tags": ["local"]})
    msg = str(err.value)
    assert "leaf" in msg and "api_base" in msg and "local" in msg


def test_local_endpoint_with_a_base_loads():
    registry = _load({"model": "openai/m", "tags": ["local"], "api_base": "http://127.0.0.1:1/v1"})
    assert registry.get("leaf").api_base == "http://127.0.0.1:1/v1"


def test_untagged_endpoint_without_base_still_loads():
    _load({"model": "gpt-4o"})
