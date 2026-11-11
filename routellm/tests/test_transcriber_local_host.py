"""A `local` transcriber's `api_base` must name a loopback/private/internal host."""

import pytest

from routellm.endpoints import EndpointRegistry
from routellm.tests.local_hosts import LOCAL_BASES, NON_LOCAL_BASES

_HTTP_LOCAL = [b for b in LOCAL_BASES if "://" in b]
_SCHEMELESS = [b for b in LOCAL_BASES if "://" not in b]
_HOSTLESS = [b for b in NON_LOCAL_BASES if b == "http://"]
_PUBLIC = [b for b in NON_LOCAL_BASES if b not in _HOSTLESS]


def _load(base, where="global", **extra):
    spec = {"mode": "transcribe", "api_base": base, "tags": ["local"], **extra}
    cfg = {
        "endpoints": {"a": {"model": "openai/a", "api_base": "http://127.0.0.1:1/v1"}},
        "tiers": {"default": {"router": "random", "threshold": 0.5, "strong": "a", "weak": "a"}},
    }
    if where == "global":
        cfg["transcriber"] = spec
    else:
        cfg["tiers"]["default"]["transcriber"] = spec
        cfg["transcriber"] = {"mode": "passthrough"}
    return EndpointRegistry.from_config(cfg)


SCOPES = ["global", "tier"]


def _scope_words(where):
    return ("transcriber",) if where == "global" else ("transcriber", "default")


@pytest.mark.parametrize("where", SCOPES)
@pytest.mark.parametrize("base", _HTTP_LOCAL)
def test_local_hosts_load(base, where):
    reg = _load(base, where)
    assert reg.transcriber_for("default" if where == "tier" else None).api_base == base


@pytest.mark.parametrize("where", SCOPES)
@pytest.mark.parametrize("base", _SCHEMELESS)
def test_schemeless_base_is_still_refused(base, where):
    with pytest.raises(ValueError, match="api_base"):
        _load(base, where)


@pytest.mark.parametrize("where", SCOPES)
@pytest.mark.parametrize("base", _PUBLIC)
def test_public_base_is_refused(base, where):
    with pytest.raises(ValueError) as err:
        _load(base, where)
    msg = str(err.value)
    assert all(w in msg for w in _scope_words(where))
    assert "api_base" in msg and "allow_public_base" in msg


@pytest.mark.parametrize("where", SCOPES)
@pytest.mark.parametrize("base", _HOSTLESS)
def test_hostless_base_is_refused(base, where):
    with pytest.raises(ValueError, match="api_base"):
        _load(base, where)


@pytest.mark.parametrize("where", SCOPES)
def test_opt_out_allows_a_public_base(where):
    base = "https://api.example.com/v1"
    reg = _load(base, where, allow_public_base=True)
    assert reg.transcriber_for("default" if where == "tier" else None).api_base == base


@pytest.mark.parametrize("where", SCOPES)
@pytest.mark.parametrize("flag", ["yes", 1, "true"])
def test_opt_out_must_be_a_real_bool(where, flag):
    with pytest.raises(ValueError, match="allow_public_base"):
        _load("https://api.example.com/v1", where, allow_public_base=flag)


@pytest.mark.parametrize("where", SCOPES)
def test_untagged_transcriber_may_use_a_public_base(where):
    reg = _load("https://api.example.com/v1", where, tags=[])
    assert reg.transcriber_for("default" if where == "tier" else None).api_base


def test_tier_inherits_a_global_local_tag_and_is_checked():
    cfg = {
        "endpoints": {"a": {"model": "openai/a", "api_base": "http://127.0.0.1:1/v1"}},
        "transcriber": {"mode": "transcribe", "api_base": "http://10.0.0.1/v1", "tags": ["local"]},
        "tiers": {
            "default": {
                "router": "random",
                "threshold": 0.5,
                "strong": "a",
                "weak": "a",
                "transcriber": {"api_base": "https://stt.example.com/v1"},
            }
        },
    }
    with pytest.raises(ValueError, match="allow_public_base"):
        EndpointRegistry.from_config(cfg)


def test_interpolated_base_is_checked_after_expansion(monkeypatch, tmp_path):
    from routellm.config import load_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "endpoints:\n  a:\n    model: openai/a\n    api_base: http://127.0.0.1:1/v1\n"
        "tiers:\n  default: {router: random, threshold: 0.5, strong: a, weak: a}\n"
        "transcriber:\n  mode: transcribe\n  tags: [local]\n  api_base: ${TR_TEST_BASE}\n"
    )
    monkeypatch.setenv("TR_TEST_BASE", "https://api.example.com/v1")
    with pytest.raises(ValueError, match="allow_public_base"):
        EndpointRegistry.from_config(load_config(explicit=str(cfg)).data)
    monkeypatch.setenv("TR_TEST_BASE", "http://10.0.0.2/v1")
    EndpointRegistry.from_config(load_config(explicit=str(cfg)).data)
