"""Lazily built OpenAI client for the routers that embed prompts.

Only two routers embed at route time, and both do so inside a method.
Building the client at import made `import routellm.routers.routers`
fail without `OPENAI_API_KEY`, even for a server whose routers never
embed. The client is therefore built on first use and cached here.

The controller calls `configure_embeddings` with its endpoint registry
before constructing routers, so an endpoint named `embedding` can point
the embedding calls at a different provider than the completions. No
registry, or no such endpoint, falls back to the `OPENAI_API_KEY` /
`OPENAI_BASE_URL` environment variables.
"""

import logging
import os
from typing import Optional

import openai

from routellm.endpoints import EndpointRegistry

logger = logging.getLogger(__name__)

EMBEDDING_ENDPOINT = "embedding"

_registry: Optional[EndpointRegistry] = None
_client: Optional["openai.OpenAI"] = None


def configure_embeddings(registry: Optional[EndpointRegistry]) -> None:
    """Set the registry consulted for the `embedding` endpoint.

    Called by the controller before it constructs its routers. Does not
    build a client; the next `get_embedding_client` call does.

    Parameters
    ----------
    registry : EndpointRegistry, optional
        Registry to search for an endpoint named `embedding`. None
        clears any previously configured registry, leaving the
        environment as the only source.
    """
    global _registry, _client

    _registry = registry
    _client = None


def reset_embedding_client() -> None:
    """Drop the cached client and the configured registry.

    Intended for tests, which need each case to build its own client.
    """
    global _registry, _client

    _registry = None
    _client = None


def get_embedding_client() -> "openai.OpenAI":
    """Return the shared embedding client, building it on first use.

    Sources, in order: an endpoint named `embedding` in the configured
    registry, then the `OPENAI_BASE_URL` / `OPENAI_API_KEY` environment
    variables. The result is cached until `configure_embeddings` or
    `reset_embedding_client` is called.

    Returns
    -------
    openai.OpenAI
        Client to use for embedding calls.

    Raises
    ------
    ValueError
        If the `embedding` endpoint names an environment variable that
        is not set.
    RuntimeError
        If neither source supplies a key.
    """
    global _client

    if _client is not None:
        return _client

    base_url, api_key = _resolve_credentials()
    if api_key is None:
        raise RuntimeError(
            "No credentials for prompt embedding. Configure an endpoint "
            f"named {EMBEDDING_ENDPOINT!r} in the `endpoints:` section of "
            "the config, or set OPENAI_API_KEY (with OPENAI_BASE_URL for a "
            "non-OpenAI provider)."
        )

    _client = openai.OpenAI(base_url=base_url, api_key=api_key)
    return _client


def _resolve_credentials() -> tuple[Optional[str], Optional[str]]:
    """Return the `(base_url, api_key)` pair for the embedding client."""
    if _registry is not None and EMBEDDING_ENDPOINT in _registry.names():
        endpoint = _registry.get(EMBEDDING_ENDPOINT)
        # Defaults stay None so a configured endpoint without a key
        # falls through to the environment below rather than silently
        # inheriting the completion credentials.
        base_url, api_key = endpoint.credentials(None, None)
        if api_key is not None:
            return base_url, api_key
        logger.debug(
            "endpoint %s carries no credential; falling back to the "
            "environment",
            EMBEDDING_ENDPOINT,
        )

    return os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
