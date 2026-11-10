"""The server's implicit `default` tier keeps every registry setting.

With no `default` tier in the config, `--strong-model`/`--weak-model`
add one. That path used to rebuild the registry from a hand-picked set
of constructor arguments, so a setting not on the list (`governance`,
`tag_namespaces`, `areas`) silently reverted to its default there. These
tests build the registry both ways, from the same config, and require the
same settings.
"""

import sys
from argparse import Namespace

import pytest

CONFIG = {
    "governance": True,
    "tag_namespaces": ["team"],
    "endpoints": {
        "big": {"model": "openai/big-1", "tags": ["cloud", "jur:us", "team:red"]},
        "small": {
            "model": "openai/small-1",
            "api_base": "http://127.0.0.1:9/v1",
            "tags": ["local", "jur:ca", "team:blue"],
        },
    },
    "tiers": {"review": {"router": "random", "strong": "big", "weak": "small"}},
    "areas": {"coding": ["review"]},
}


@pytest.fixture(scope="module")
def server():
    argv = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as module
    finally:
        sys.argv = argv
    return module


def _args(**kw):
    return Namespace(
        **{
            "strong_model": "big",
            "weak_model": "small",
            "routers": ["random"],
            "default_threshold": 0.5,
            **kw,
        }
    )


@pytest.fixture
def registries(server, monkeypatch):
    """The registry from the config alone, and with the flags' implicit tier."""
    monkeypatch.setattr(server, "args", _args(strong_model=None, weak_model=None))
    plain = server.build_registry(CONFIG)
    monkeypatch.setattr(server, "args", _args())
    return plain, server.build_registry(CONFIG)


def test_the_flags_add_the_default_tier(registries):
    plain, rebuilt = registries
    assert not plain.has_tier("default")
    assert rebuilt.get_tier("default").strong == "big"


def test_governance_survives_the_implicit_default_tier(registries):
    assert registries[1].governance is True


def test_tag_namespaces_survive_the_implicit_default_tier(registries):
    plain, rebuilt = registries
    assert "team" in plain.known_namespaces
    assert rebuilt.known_namespaces == plain.known_namespaces


def test_areas_survive_the_implicit_default_tier(registries):
    assert registries[1].area_of("review") == "coding"


def test_a_raw_flag_model_is_not_a_governed_endpoint(server, monkeypatch):
    # A raw model has no reach or jurisdiction to vouch for, so under
    # `governance: true` it is refused rather than slipped past the grammar.
    monkeypatch.setattr(server, "args", _args(strong_model="gpt-4o"))
    with pytest.raises(ValueError, match="reach tag"):
        server.build_registry(CONFIG)
