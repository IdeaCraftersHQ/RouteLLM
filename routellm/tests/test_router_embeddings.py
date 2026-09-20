"""Tests for the lazily built embedding client used by routers.

Covers the source order of `get_embedding_client` (configured
`embedding` endpoint, then environment), its caching and reset, the
error raised when neither source is present, and the guarantee that
importing the router module and the server builds no client.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from routellm.endpoints import Endpoint, EndpointRegistry
from routellm.routers.embeddings import (
    configure_embeddings,
    get_embedding_client,
    reset_embedding_client,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_embedding_state(monkeypatch):
    """Reset module state and embedding env around every test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_KEY", raising=False)
    reset_embedding_client()
    yield
    reset_embedding_client()


@pytest.fixture
def fake_openai(monkeypatch):
    """Patch `openai.OpenAI` and hand back the constructor mock."""
    import openai

    factory = MagicMock(name="OpenAI", return_value=MagicMock(name="client"))
    monkeypatch.setattr(openai, "OpenAI", factory)
    return factory


def _registry(**spec):
    """Build a registry holding one endpoint named `embedding`."""
    return EndpointRegistry({"embedding": Endpoint(name="embedding", **spec)})


def test_client_built_from_embedding_endpoint(fake_openai, monkeypatch):
    monkeypatch.setenv("EMBEDDING_KEY", "endpoint-key")
    configure_embeddings(
        _registry(
            model="text-embedding-3-small",
            api_base="http://127.0.0.1:11500/v1",
            api_key_env="EMBEDDING_KEY",
        )
    )

    client = get_embedding_client()

    assert client is fake_openai.return_value
    fake_openai.assert_called_once_with(
        base_url="http://127.0.0.1:11500/v1", api_key="endpoint-key"
    )


def test_endpoint_wins_over_environment(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("EMBEDDING_KEY", "endpoint-key")
    configure_embeddings(
        _registry(model="text-embedding-3-small", api_key_env="EMBEDDING_KEY")
    )

    get_embedding_client()

    fake_openai.assert_called_once_with(base_url=None, api_key="endpoint-key")


def test_endpoint_missing_env_variable_raises(fake_openai, monkeypatch):
    configure_embeddings(
        _registry(model="text-embedding-3-small", api_key_env="EMBEDDING_KEY")
    )

    with pytest.raises(ValueError) as excinfo:
        get_embedding_client()

    assert "embedding" in str(excinfo.value)
    assert "EMBEDDING_KEY" in str(excinfo.value)


def test_keyless_endpoint_keeps_its_base_with_the_environment_key(
    fake_openai, monkeypatch
):
    """A base without a key must not send embeddings to OpenAI.

    Base URL and credential resolve independently: the endpoint's
    `api_base` stands even when only the environment supplies the key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    configure_embeddings(
        _registry(
            model="text-embedding-3-small",
            api_base="http://127.0.0.1:11500/v1",
        )
    )

    get_embedding_client()

    fake_openai.assert_called_once_with(
        base_url="http://127.0.0.1:11500/v1", api_key="env-key"
    )


def test_keyless_endpoint_without_a_base_still_uses_the_environment_base(
    fake_openai, monkeypatch
):
    """An endpoint setting neither leaves both to the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    configure_embeddings(_registry(model="text-embedding-3-small"))

    get_embedding_client()

    fake_openai.assert_called_once_with(
        base_url="http://localhost:9999/v1", api_key="env-key"
    )


def test_keyless_endpoint_with_a_base_and_no_key_anywhere_raises(fake_openai):
    """A base alone is not credentials; the error still names both."""
    configure_embeddings(
        _registry(
            model="text-embedding-3-small",
            api_base="http://127.0.0.1:11500/v1",
        )
    )

    with pytest.raises(RuntimeError) as excinfo:
        get_embedding_client()

    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_registry_without_embedding_endpoint_falls_back_to_env(
    fake_openai, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    configure_embeddings(EndpointRegistry({"other": Endpoint(name="other", model="m")}))

    get_embedding_client()

    fake_openai.assert_called_once_with(base_url=None, api_key="env-key")


def test_environment_fallback_uses_base_url(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")

    get_embedding_client()

    fake_openai.assert_called_once_with(
        base_url="http://localhost:9999/v1", api_key="env-key"
    )


def test_no_endpoint_and_no_env_raises_naming_both(fake_openai):
    with pytest.raises(RuntimeError) as excinfo:
        get_embedding_client()

    message = str(excinfo.value)
    assert "embedding" in message
    assert "OPENAI_API_KEY" in message


def test_client_is_cached_across_calls(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    first = get_embedding_client()
    second = get_embedding_client()

    assert first is second
    assert fake_openai.call_count == 1


def test_reset_clears_client_and_registry(fake_openai, monkeypatch):
    monkeypatch.setenv("EMBEDDING_KEY", "endpoint-key")
    configure_embeddings(
        _registry(model="text-embedding-3-small", api_key_env="EMBEDDING_KEY")
    )
    get_embedding_client()

    reset_embedding_client()
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    get_embedding_client()

    assert fake_openai.call_count == 2
    assert fake_openai.call_args_list[-1].kwargs == {
        "base_url": None,
        "api_key": "env-key",
    }


def test_configure_with_none_clears_registry(fake_openai, monkeypatch):
    monkeypatch.setenv("EMBEDDING_KEY", "endpoint-key")
    configure_embeddings(
        _registry(model="text-embedding-3-small", api_key_env="EMBEDDING_KEY")
    )
    configure_embeddings(None)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    get_embedding_client()

    fake_openai.assert_called_once_with(base_url=None, api_key="env-key")


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="router module imports torch",
)
def test_router_and_server_import_without_openai_key():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["PYTHONPATH"] = "."
    snippet = (
        "import sys; sys.argv = ['x']; "
        "import routellm.routers.routers, routellm.openai_server"
    )

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
