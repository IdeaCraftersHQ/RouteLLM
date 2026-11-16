"""Embeddings resolve through the registry; there is no implicit default.

Covers `routellm.routers.embeddings` (endpoint choice, refusal, the opt-in
environment fallback), the controller's startup refusal for the routers
and the semantic cache that embed, and that both really call the
configured endpoint (a loopback fake; nothing leaves the host).
"""

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from routellm.caching import Cache, CacheConfig
from routellm.controller import Controller
from routellm.endpoints import Endpoint, EndpointRegistry
from routellm.routers.base import Router
from routellm.routers.embeddings import (
    configure_embeddings,
    embed,
    embedding_endpoint_name,
    require_embeddings,
    reset_embedding_client,
)
from routellm.routers.routers import ROUTER_CLS
from routellm.tests.fake_embedding_server import FakeEmbeddingServer

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "EMBED_KEY"):
        monkeypatch.delenv(var, raising=False)
    reset_embedding_client()
    yield
    reset_embedding_client()


@pytest.fixture
def fake():
    server = FakeEmbeddingServer()
    yield server
    server.close()


def _registry(base=None, name="embedding", **spec):
    spec.setdefault("model", "openai/embed-model")
    spec.setdefault("api_key_env", "EMBED_KEY")
    spec.setdefault("tags", ["local"])
    endpoints = {
        "strong": Endpoint(name="strong", model="openai/s", api_base="http://127.0.0.1:9/v1"),
        "weak": Endpoint(name="weak", model="openai/w", api_base="http://127.0.0.1:9/v1"),
    }
    if base is not None:
        endpoints[name] = Endpoint(name=name, api_base=base, **spec)
    return EndpointRegistry(endpoints)


class _EmbeddingRouter(Router):
    requires_embedding = True

    def calculate_strong_win_rate(self, prompt):
        return 1.0 if embed(prompt) else 0.0


@pytest.fixture
def embedding_router(monkeypatch):
    monkeypatch.setitem(ROUTER_CLS, "emb", _EmbeddingRouter)


def _controller(registry, **kw):
    return Controller(
        routers=kw.pop("routers", ["random"]),
        strong_model="strong",
        weak_model="weak",
        endpoints=registry,
        **kw,
    )


# -- refusal ---------------------------------------------------------------


def test_router_that_embeds_refuses_to_start_without_an_endpoint(embedding_router):
    with pytest.raises(ValueError) as excinfo:
        _controller(_registry(), routers=["emb"])

    message = str(excinfo.value)
    assert "router 'emb'" in message
    assert "embeddings.endpoint" in message


def test_semantic_cache_refuses_to_start_without_an_endpoint(tmp_path):
    cache = CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"), semantic_enabled=True)

    with pytest.raises(ValueError) as excinfo:
        _controller(_registry(), cache_config=cache)

    message = str(excinfo.value)
    assert "semantic cache" in message
    assert "embeddings.endpoint" in message


def test_exact_cache_and_plain_routers_need_no_endpoint(tmp_path):
    cache = CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"))

    _controller(_registry(), cache_config=cache)


def test_named_endpoint_must_be_in_the_registry():
    with pytest.raises(ValueError, match="not a registry endpoint"):
        _controller(_registry(), embedding_endpoint="nope")


def test_the_environment_alone_never_supplies_an_endpoint(embedding_router, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    with pytest.raises(ValueError, match=r"embeddings.endpoint"):
        _controller(_registry(), routers=["emb"])


def test_embed_without_an_endpoint_raises():
    with pytest.raises(RuntimeError, match="No embedding endpoint"):
        embed("hello")


def test_cache_config_refuses_the_removed_model_setting():
    with pytest.raises(ValueError, match=r"embeddings.endpoint"):
        CacheConfig(embedding_model="text-embedding-3-small")


# -- opt-in fallback --------------------------------------------------------


def test_allow_remote_embeddings_warns_naming_the_host(embedding_router, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embeddings.example.test/v1")

    with caplog.at_level(logging.WARNING):
        _controller(_registry(), routers=["emb"], allow_remote_embeddings=True)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("embeddings.example.test" in w for w in warnings)


def _strict_registry():
    registry = _registry()
    registry.strict_registry = True
    return registry


def _policy():
    from routellm.policy import parse_policy

    return parse_policy(
        {
            "default_class": "public",
            "classes": {
                "public": {"treatment": "raw", "reach": ["local"], "tiers": ["default"]},
            },
        },
        "p.yaml",
    )


def test_strict_registry_refuses_allow_remote_embeddings(embedding_router, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embeddings.example.test/v1")

    with pytest.raises(ValueError) as excinfo:
        _controller(_strict_registry(), routers=["emb"], allow_remote_embeddings=True)

    message = str(excinfo.value)
    assert "allow-remote-embeddings" in message
    assert "strict_registry" in message
    assert "embeddings.endpoint" in message


def test_strict_flag_on_the_controller_refuses_allow_remote_embeddings():
    with pytest.raises(ValueError, match="allow-remote-embeddings"):
        _controller(_registry(), strict_registry=True, allow_remote_embeddings=True)


def test_a_policy_refuses_allow_remote_embeddings():
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {
                name: {"model": "openai/m", "api_base": "http://127.0.0.1:9/v1", "tags": ["local"]}
                for name in ("strong", "weak")
            },
            "tiers": {
                "default": {
                    "router": "random",
                    "threshold": 0.5,
                    "strong": "strong",
                    "weak": "weak",
                }
            },
        }
    )
    with pytest.raises(ValueError, match="allow-remote-embeddings"):
        _controller(registry, policy=_policy(), allow_remote_embeddings=True)


def test_strict_with_an_endpoint_still_refuses_the_flag(fake, monkeypatch):
    monkeypatch.setenv("EMBED_KEY", "k")
    registry = _registry(fake.base)
    registry.strict_registry = True

    with pytest.raises(ValueError, match="allow-remote-embeddings"):
        _controller(registry, allow_remote_embeddings=True)


def test_strict_without_the_flag_still_starts(fake, monkeypatch):
    monkeypatch.setenv("EMBED_KEY", "k")
    registry = _registry(fake.base)
    registry.strict_registry = True

    _controller(registry)


def test_allow_remote_embeddings_uses_the_environment(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", fake.base)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    configure_embeddings(_registry(), allow_remote=True)

    assert embed("hello") == [1.0] * 8
    assert len(fake.hits) == 1


# -- the configured endpoint is the one called ------------------------------


def test_embed_calls_the_endpoint_with_its_model(fake, monkeypatch):
    monkeypatch.setenv("EMBED_KEY", "k")
    configure_embeddings(_registry(fake.base))

    assert embed("hello") == [1.0] * 8

    assert [h["model"] for h in fake.hits] == ["embed-model"]
    assert fake.hits[0]["path"].endswith("/embeddings")


def test_a_named_endpoint_wins_over_the_default_name(fake, monkeypatch):
    monkeypatch.setenv("EMBED_KEY", "k")
    configure_embeddings(_registry(fake.base, name="vectors"), endpoint="vectors")

    assert embedding_endpoint_name() == "vectors"
    embed("hello")
    assert len(fake.hits) == 1


def test_semantic_cache_embeds_through_the_endpoint(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_KEY", "k")
    cache = CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"), semantic_enabled=True)
    _controller(_registry(fake.base), cache_config=cache)
    store = Cache(cache)

    store.put("what is up", "strong", {}, {"id": "r"})
    assert store.get("what is up!", "strong", {}) == {"id": "r"}

    assert [h["input"] for h in fake.hits] == [["what is up"], ["what is up!"]]


def test_router_embeds_through_the_endpoint(fake, monkeypatch, embedding_router):
    monkeypatch.setenv("EMBED_KEY", "k")
    controller = _controller(_registry(fake.base), routers=["emb"])

    assert controller.routers["emb"].calculate_strong_win_rate("hi") == 1.0
    assert len(fake.hits) == 1


def test_startup_logs_the_endpoint_and_its_reach(fake, caplog):
    configure_embeddings(_registry(fake.base))

    with caplog.at_level(logging.INFO):
        assert require_embeddings([]) == "embedding"

    assert any("embedding" in r.getMessage() and "local" in r.getMessage() for r in caplog.records)


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="router module imports torch",
)
def test_router_and_server_import_without_openai_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["PYTHONPATH"] = "."
    snippet = (
        "import sys; sys.argv = ['x']; import routellm.routers.routers, routellm.openai_server"
    )

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
