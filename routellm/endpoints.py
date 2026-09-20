"""Named model endpoints with their own base URL and credential.

An endpoint gives a reachable model a stable name, so routing,
caching, and traces refer to "cloud_strong" rather than to whatever
litellm model string happens to sit behind it. The registry is built
from the `endpoints:` key of the YAML passed to `--config`::

    endpoints:
      cloud_strong:
        model: gpt-4o
        api_key_env: OPENAI_API_KEY
        tags: [tools, vision]
        quality: 90
        extra: {timeout: 60}
      local_fast:
        model: ollama_chat/qwen3:8b
        api_base: http://127.0.0.1:11500
        tags: [local]

Credentials are never stored: `api_key_env` names an environment
variable that is read at call time, so a config loads on a machine
that holds none of the keys.

Names are restricted to `[A-Za-z0-9_]+`; hyphens are reserved for the
model-name grammar that splits on '-'.
"""

import logging
import os
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class Endpoint(BaseModel):
    """A named, individually addressable model endpoint.

    Attributes
    ----------
    name : str
        Registry name, matching `[A-Za-z0-9_]+`.
    model : str
        Non-empty litellm model name to send to the provider.
    api_base : str, optional
        Base URL for this endpoint; falls back to the controller default.
    api_key_env : str, optional
        Name of the environment variable holding this endpoint's key.
        Never the key itself.
    tags : list[str]
        Free-form labels for selection by later routing layers.
    quality : int, optional
        Manual quality score in [0, 100].
    extra : dict
        Extra keyword arguments passed to litellm. Request kwargs win.
    """

    name: str
    model: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    quality: Optional[int] = Field(default=None, ge=0, le=100)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(
                f"Invalid endpoint name: {value!r}. "
                "Names must match [A-Za-z0-9_]+ (no hyphens)."
            )
        return value

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Endpoint model must be a non-empty string")
        return value

    def credentials(
        self,
        default_base: Optional[str],
        default_key: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve the base URL and API key for a call to this endpoint.

        The environment is read here rather than at config load, so a
        config that names variables loads on a machine that holds none
        of them.

        Parameters
        ----------
        default_base : str, optional
            Base URL to use when this endpoint sets none.
        default_key : str, optional
            API key to use when this endpoint names no environment
            variable.

        Returns
        -------
        tuple[str or None, str or None]
            The `(api_base, api_key)` pair for this call.

        Raises
        ------
        ValueError
            If `api_key_env` is set but that variable is not in the
            environment.
        """
        api_base = self.api_base or default_base

        if self.api_key_env is None:
            return api_base, default_key

        api_key = os.environ.get(self.api_key_env)
        if api_key is None:
            raise ValueError(
                f"Endpoint {self.name!r} needs environment variable "
                f"{self.api_key_env!r}, which is not set."
            )
        return api_base, api_key


class EndpointRegistry:
    """Lookup of endpoints by name, with raw-model passthrough.

    A name the registry does not know is treated as a raw litellm model
    name and wrapped in an anonymous endpoint carrying no base URL and
    no credential, so calls written against model names keep working.
    """

    def __init__(self, endpoints: Optional[dict[str, Endpoint]] = None):
        """Initialize the registry.

        Parameters
        ----------
        endpoints : dict[str, Endpoint], optional
            Endpoints keyed by name. Empty when None.
        """
        self._endpoints: dict[str, Endpoint] = dict(endpoints or {})
        self._warned: set[str] = set()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EndpointRegistry":
        """Build a registry from a loaded config dict.

        Reads only the `endpoints:` key; every other top-level key is
        left to its own owner.

        Parameters
        ----------
        config : dict
            Loaded YAML config. An absent `endpoints:` key yields an
            empty registry.

        Returns
        -------
        EndpointRegistry
            Registry holding one `Endpoint` per configured name.
        """
        raw = (config or {}).get("endpoints") or {}
        endpoints = {
            name: Endpoint(name=name, **(spec or {})) for name, spec in raw.items()
        }
        return cls(endpoints)

    def names(self) -> list[str]:
        """Return the configured endpoint names, sorted."""
        return sorted(self._endpoints)

    def get(self, name: str) -> Endpoint:
        """Return the endpoint registered under `name`.

        Parameters
        ----------
        name : str
            Endpoint name.

        Returns
        -------
        Endpoint
            The registered endpoint.

        Raises
        ------
        KeyError
            If no endpoint carries that name. The message lists the
            known names.
        """
        try:
            return self._endpoints[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown endpoint: {name}. Configured endpoints: {known}"
            ) from None

    def resolve(self, name_or_model: str) -> Endpoint:
        """Return an endpoint for a name or a raw litellm model name.

        A configured name returns its endpoint. Anything else becomes an
        anonymous endpoint whose model is the given string and which
        carries no base URL and no credential, so the controller
        defaults apply. The first anonymous resolution of each distinct
        string logs one warning; repeats stay quiet.

        Parameters
        ----------
        name_or_model : str
            Endpoint name, or a litellm model name.

        Returns
        -------
        Endpoint
            The configured or anonymous endpoint.
        """
        endpoint = self._endpoints.get(name_or_model)
        if endpoint is not None:
            return endpoint

        if name_or_model not in self._warned:
            self._warned.add(name_or_model)
            logger.warning(
                "no endpoint named %s; using it as a raw model with the "
                "controller defaults",
                name_or_model,
            )
        return Endpoint.model_construct(
            name=name_or_model,
            model=name_or_model,
            api_base=None,
            api_key_env=None,
            tags=[],
            quality=None,
            extra={},
        )
