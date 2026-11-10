"""Prompt embeddings, resolved through the endpoint registry.

Three things embed prompt text at request time: the `mf` and
`sw_ranking` routers, and the semantic cache. All three call `embed`,
which sends the text to one registry endpoint, so the call carries that
endpoint's reach and jurisdiction tags like any model call and a class
policy or egress gate can name it.

There is no default endpoint and no default model. The endpoint is the
one named by the config's `embeddings.endpoint`, else the one named
`embedding`; its `model`, `api_base`, `api_key_env` and `extra` are used
as written. A server that needs embeddings and has neither refuses to
start (`require_embeddings`).

`--allow-remote-embeddings` restores the legacy path for a server with no
such endpoint: the `OPENAI_BASE_URL` / `OPENAI_API_KEY` environment and
the embedding model that path always used. It logs a WARNING naming the
host the prompts go to, and no tag describes that call, so no policy can
gate it.

The controller calls `configure_embeddings` before building its routers
and its cache. Importing this module builds nothing.
"""

import logging
import os
from urllib.parse import urlparse

from routellm.endpoints import EndpointRegistry

logger = logging.getLogger(__name__)

EMBEDDING_ENDPOINT = "embedding"

#: Model sent on the opt-in environment path only; a registry endpoint
#: always supplies its own.
LEGACY_ENV_MODEL = "openai/text-embedding-3-small"
LEGACY_ENV_HOST = "api.openai.com"

_registry: EndpointRegistry | None = None
_endpoint: str | None = None
_allow_remote: bool = False


def configure_embeddings(
    registry: EndpointRegistry | None,
    endpoint: str | None = None,
    allow_remote: bool = False,
    strict: bool = False,
) -> None:
    """Set where embeddings go. Builds and checks nothing.

    Parameters
    ----------
    registry : EndpointRegistry, optional
        Registry searched for the embedding endpoint. None clears it.
    endpoint : str, optional
        Name of the registry endpoint to use; `embedding` when None.
    allow_remote : bool
        Permit the environment fallback when no endpoint exists.
    strict : bool
        Whether `strict_registry` is on (a loaded policy forces it).

    Raises
    ------
    ValueError
        If `allow_remote` is set under `strict`: the fallback sends
        prompts to a raw model through the environment, which a strict
        registry exists to prevent.
    """
    global _registry, _endpoint, _allow_remote

    if allow_remote and (strict or (registry is not None and registry.strict_registry)):
        raise ValueError(
            "--allow-remote-embeddings is refused under strict_registry (a loaded "
            "`policy:` forces it): it sends prompts to OPENAI_BASE_URL with a raw "
            f"{LEGACY_ENV_MODEL!r} model that no tag describes. Define an "
            f"endpoint named {EMBEDDING_ENDPOINT!r} or name one with "
            "`embeddings.endpoint: <endpoint>`, and drop the flag."
        )
    _registry = registry
    _endpoint = endpoint
    _allow_remote = allow_remote


def reset_embedding_client() -> None:
    """Forget the registry, endpoint name and fallback flag. For tests."""
    configure_embeddings(None)


def embedding_endpoint_name() -> str | None:
    """Return the registry endpoint embeddings go to, or None when there is none."""
    if _registry is None:
        return None
    name = _endpoint or EMBEDDING_ENDPOINT
    return name if name in _registry.names() else None


def require_embeddings(consumers: list[str]) -> str | None:
    """Fail startup unless embeddings can be sent somewhere explicit.

    Parameters
    ----------
    consumers : list[str]
        What needs embeddings (`"router 'mf'"`, `"the semantic cache"`).
        Empty means nothing does; a configured name is still checked.

    Returns
    -------
    str or None
        The registry endpoint in use, or None on the opt-in environment
        path.

    Raises
    ------
    ValueError
        If the configured endpoint is not in the registry, or a
        consumer needs one and there is none and the fallback is off.
    """
    if _registry is not None and _endpoint and _endpoint not in _registry.names():
        raise ValueError(
            f"`embeddings.endpoint: {_endpoint}` is not a registry endpoint. "
            f"Configured endpoints: {', '.join(_registry.names()) or '<none>'}"
        )
    name = embedding_endpoint_name()
    if name is not None:
        endpoint = _registry.get(name)
        logger.info(
            "embeddings via endpoint %s (reach: %s)",
            name,
            ", ".join(sorted(endpoint.reach)) or "untagged",
        )
        return name
    if not consumers:
        return None
    if not _allow_remote:
        raise ValueError(
            f"{', '.join(consumers)} embed(s) prompts and no embedding endpoint is "
            f"configured. Define an endpoint named {EMBEDDING_ENDPOINT!r} under "
            "`endpoints:`, or name one with `embeddings.endpoint: <endpoint>`. "
            "`--allow-remote-embeddings` falls back to OPENAI_BASE_URL / OPENAI_API_KEY."
        )
    logger.warning(
        "--allow-remote-embeddings: %s embed(s) prompts through the environment, "
        "sending them to %s; no tag describes this call, so no policy can gate it",
        ", ".join(consumers),
        _env_host(),
    )
    return None


def _env_host() -> str:
    base = os.environ.get("OPENAI_BASE_URL")
    return (urlparse(base).hostname if base else None) or LEGACY_ENV_HOST


def _call_params() -> tuple[str, str | None, str | None, dict]:
    """Return `(model, api_base, api_key, extra)` for one embedding call."""
    name = embedding_endpoint_name()
    if name is not None:
        endpoint = _registry.get(name)
        base, key = endpoint.credentials(None, None)
        return endpoint.model, base, key, dict(endpoint.extra)
    if _allow_remote:
        return (
            LEGACY_ENV_MODEL,
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_API_KEY"),
            {},
        )
    raise RuntimeError(
        "No embedding endpoint is configured: define an endpoint named "
        f"{EMBEDDING_ENDPOINT!r} or set `embeddings.endpoint`."
    )


def embed(text: str) -> list[float]:
    """Embed one text through the configured endpoint.

    Raises
    ------
    RuntimeError
        If no endpoint is configured and the fallback is off.
    """
    from litellm import embedding

    model, base, key, extra = _call_params()
    res = embedding(model=model, input=[text], api_base=base, api_key=key, **extra)
    return list(res.data[0]["embedding"])
