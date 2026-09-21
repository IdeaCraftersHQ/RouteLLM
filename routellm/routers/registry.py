"""Registry of router classes keyed by their CLI name.

`ROUTER_CLS` is the single backing dict; `routellm.routers.routers`
re-exports it, so every reader of `ROUTER_CLS` sees registrations made
after import, including those from installed extension packages.

An extension package publishes a router through an entry point in the
`routellm.routers` group, which `discover_routers` loads::

    [project.entry-points."routellm.routers"]
    myrouter = "pkg.mod:Cls"

Deliberately free of torch, the TypeSafe SDK, and of any import of
`routellm.routers.routers`, so it stays importable in environments
without the heavy ML stack.
"""

import inspect
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from routellm.routers.base import Router

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "routellm.routers"

ROUTER_CLS: dict[str, type] = {}

discovery_failures: dict[str, str] = {}


def register_router(name, cls=None, *, replace=False):
    """Register a router class under a name.

    Supports both a direct call, `register_router("x", Cls)`, and
    decorator use, `@register_router("x")`, which returns a decorator
    when `cls` is omitted.

    Parameters
    ----------
    name : str
        Non-empty CLI name for the router.
    cls : type, optional
        Router class to register. Omit to get a decorator back.
    replace : bool, optional
        Allow overwriting an existing registration (default False).

    Returns
    -------
    type
        The registered class, so the call doubles as a decorator.

    Raises
    ------
    ValueError
        If `name` is not a non-empty str, `cls` is not a class, or the
        name is already registered and `replace` is False.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"router name must be a non-empty str, got {name!r}")

    if cls is None:

        def decorator(decorated):
            register_router(name, decorated, replace=replace)
            return decorated

        return decorator

    if not inspect.isclass(cls):
        raise ValueError(f"router for {name!r} must be a class, got {cls!r}")

    if not replace and name in ROUTER_CLS:
        raise ValueError(
            f"router {name!r} already registered as "
            f"{ROUTER_CLS[name].__name__}; pass replace=True to override"
        )

    ROUTER_CLS[name] = cls
    discovery_failures.pop(name, None)

    return cls


def get_router_class(name):
    """Look up a registered router class by name.

    Parameters
    ----------
    name : str
        Registered router name.

    Returns
    -------
    type
        The router class.

    Raises
    ------
    KeyError
        If the name is unknown. The message lists the registered names
        and any entry points that failed to load.
    """
    if name in ROUTER_CLS:
        return ROUTER_CLS[name]

    message = f"unknown router {name!r}; registered: {router_names()}"
    if discovery_failures:
        failures = ", ".join(
            f"{failed}: {reason}" for failed, reason in sorted(discovery_failures.items())
        )
        message += f"; failed to load: {failures}"

    raise KeyError(message)


def router_names():
    """Return the registered router names, sorted.

    Returns
    -------
    list[str]
        Sorted registered names.
    """
    return sorted(ROUTER_CLS)


def name_for(cls):
    """Look up the name a router class is registered under.

    Reverse lookup done live against the registry, so classes
    registered after import resolve too.

    Parameters
    ----------
    cls : type
        Router class.

    Returns
    -------
    str
        Registered name.

    Raises
    ------
    KeyError
        If the class is not registered.
    """
    for name, registered in ROUTER_CLS.items():
        if registered is cls:
            return name

    raise KeyError(f"router class {cls!r} is not registered")


def discover_routers(group=ENTRY_POINT_GROUP):
    """Register routers published by installed packages as entry points.

    Idempotent: names already registered are skipped, so repeated calls
    add nothing and raise nothing. A plugin whose `load()` or
    registration raises is recorded in `discovery_failures` and logged
    at WARNING; it never propagates.

    Parameters
    ----------
    group : str, optional
        Entry point group to read (default "routellm.routers").

    Returns
    -------
    list[str]
        Names registered by this call, in entry point order.
    """
    registered = []

    for entry_point in entry_points(group=group):
        # Only a successful registration is skipped; a name still in
        # discovery_failures is retried and re-warned every call, so a
        # plugin fixed at runtime is picked up without a restart.
        if entry_point.name in ROUTER_CLS:
            continue

        try:
            register_router(entry_point.name, entry_point.load())
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            discovery_failures[entry_point.name] = reason
            logger.warning("failed to load router entry point %r: %s", entry_point.name, reason)
            continue

        registered.append(entry_point.name)

    return registered


def reset_registry():
    """Clear the registry and recorded discovery failures.

    For tests only: production code registers at import and never
    unregisters.
    """
    ROUTER_CLS.clear()
    discovery_failures.clear()
