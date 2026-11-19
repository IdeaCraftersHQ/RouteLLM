"""Adapter registry: `@register` collects, `discover` imports, `mount_adapters` serves.

Discovery imports every module of `routellm.adapters` (and each module
named in `ROUTELLM_ADAPTER_MODULES`, comma separated, for adapters kept
outside this package); a module's `@register` runs at import. So adding a
dialect is adding a module: no list to edit. A module that fails to
import stops startup, naming the module: a dialect that silently
vanishes is worse than one that refuses to start.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil

from fastapi import Request

import routellm.adapters as package
from routellm.adapters.base import Adapter
from routellm.adapters.dispatch import serve

__all__ = ["MODULES_ENV", "adapters", "discover", "mount_adapters", "register"]

logger = logging.getLogger(__name__)

#: Extra modules to import for their `@register`, comma separated.
MODULES_ENV = "ROUTELLM_ADAPTER_MODULES"

_REGISTRY: dict[str, Adapter] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    """Class decorator: instantiate `cls` (no arguments) and add it by `name`."""
    name = getattr(cls, "name", None)
    if not name or not getattr(cls, "routes", None):
        raise TypeError(f"{cls.__qualname__}: an adapter needs `name` and `routes`")
    existing = _REGISTRY.get(name)
    if existing is not None and type(existing).__qualname__ != cls.__qualname__:
        raise ValueError(f"adapter name {name!r} is taken by {type(existing).__module__}")
    _REGISTRY[name] = cls()
    return cls


def adapters() -> list[Adapter]:
    """The registered adapters, by name. Call `discover` first."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def discover() -> list[Adapter]:
    """Import every adapter module, then return the registered adapters."""
    for module in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module.name}")
    for name in filter(None, (m.strip() for m in os.environ.get(MODULES_ENV, "").split(","))):
        importlib.import_module(name)
    return adapters()


def mount_adapters(app, pipeline, found: list[Adapter] | None = None) -> list[Adapter]:
    """Add every adapter's routes to `app`.

    `pipeline` is the module (or object) that owns the chat pipeline:
    it must provide `authenticate`, `refuse_loops`, `run_chat_completion`,
    `list_models`, `CONTROLLER`, `AUDIT`, `Unauthorized` and `LoopDetected`.
    Routes are hidden from the OpenAPI schema. A method and path claimed
    twice is an error naming both adapters.
    """
    found = discover() if found is None else found
    taken: dict[tuple[str, str], str] = {}
    for adapter in found:
        for route in adapter.routes:
            key = (route.method.upper(), route.path)
            if key in taken:
                raise ValueError(
                    f"{route.method} {route.path} is claimed by adapters "
                    f"{taken[key]!r} and {adapter.name!r}"
                )
            taken[key] = adapter.name
            app.add_api_route(
                route.path,
                _endpoint(adapter, route, pipeline),
                methods=[route.method.upper()],
                include_in_schema=False,
                name=f"adapter_{adapter.name}_{route.operation}",
            )
    logger.info("adapters mounted: %s", ", ".join(a.name for a in found) or "none")
    return found


def _endpoint(adapter: Adapter, route, pipeline):
    async def endpoint(request: Request):
        return await serve(adapter, route, request, pipeline)

    return endpoint
