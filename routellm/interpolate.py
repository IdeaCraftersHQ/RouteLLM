"""Environment interpolation for config string values.

One config file has to serve hosts whose endpoint URLs differ, so a
string value may read the environment:

``${VAR}``
    The variable's value; an unset variable is a load error.
``${VAR:-default}``
    The default when the variable is unset or empty.
``${VAR:?message}``
    A load error carrying `message` when the variable is unset or empty.
``$$``
    A literal ``$``.

A bare ``$NAME`` is left alone: header templates and model ids may
contain a dollar. Only string scalars are expanded (inside mappings and
lists); mapping keys never are. This is the POSIX shell parameter
syntax, the same dialect Docker Compose uses.

Secrets do not go through here. A field whose name says it holds one
(`api_key`, `secret`, `token`, `password`, `authorization`) refuses a
reference: name the environment variable in `api_key_env` instead, so a
key is read at call time and never lands in a merged config dump.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from routellm.config import ConfigError

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_TOKEN = re.compile(rf"\$\$|\$\{{(?P<name>{_NAME})(?:(?P<op>:-|:\?)(?P<arg>[^}}]*))?\}}|\$\{{")
_SECRET_KEY = re.compile(r"(api_key|secret|token|password|authorization)$", re.IGNORECASE)


def is_secret_key(key: str) -> bool:
    """Whether a config key names a secret, and so refuses interpolation.

    A key ending `_env` names the *variable* holding a secret, which is
    the sanctioned path, so it is not a secret itself.
    """
    return not key.lower().endswith("_env") and bool(_SECRET_KEY.search(key))


def variables_in(value: str) -> list[str]:
    """Names of the variables `value` reads, in order, without resolving."""
    return [m["name"] for m in _TOKEN.finditer(value) if m["name"]]


def expand(value: str, env: Mapping[str, str], fail: Callable[[str], Exception]) -> str:
    """Expand every reference in one string.

    Parameters
    ----------
    value : str
        The raw text.
    env : Mapping
        Where variables are read from.
    fail : callable
        Builds the exception to raise from a problem description.
    """

    def sub(match: re.Match) -> str:
        if match.group(0) == "$$":
            return "$"
        name = match["name"]
        if name is None:
            raise fail(
                "malformed `${...}` reference "
                "(expected ${NAME}, ${NAME:-default}, ${NAME:?message})"
            )
        raw = env.get(name)
        if match["op"] is None:
            if raw is None:
                raise fail(
                    f"environment variable {name} is not set "
                    f"(write ${{{name}:-default}} to allow it to be absent)"
                )
            return raw
        if raw:
            return raw
        if match["op"] == ":-":
            return match["arg"]
        raise fail(f"environment variable {name} is not set: {match['arg'] or 'required'}")

    return _TOKEN.sub(sub, value)


def interpolate_tree(
    data: Any,
    *,
    origin_of: Callable[[str], str | None],
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return `data` with every string scalar expanded.

    Parameters
    ----------
    data : Any
        The merged config (not mutated).
    origin_of : callable
        Maps a dotted key to the file that set it, for error messages.
    env : Mapping, optional
        Defaults to `os.environ`.

    Raises
    ------
    ConfigError
        On an unset variable, a `:?` failure, a malformed reference, or a
        reference inside a secret-by-name field. The message names the
        file, the dotted key and the variable.
    """
    environ = os.environ if env is None else env

    def walk(node: Any, dotted: str, secret: bool) -> Any:
        if isinstance(node, Mapping):
            return {
                k: walk(v, f"{dotted}.{k}" if dotted else str(k), is_secret_key(str(k)))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [walk(v, f"{dotted}[{i}]", secret) for i, v in enumerate(node)]
        if not isinstance(node, str):
            return node

        def fail(problem: str) -> ConfigError:
            where = origin_of(dotted)
            return ConfigError(f"{where + ': ' if where else ''}{dotted}: {problem}")

        if secret and "${" in node:
            raise fail(
                "a secret-by-name field cannot read the environment; "
                "put the variable's name in `api_key_env` instead"
            )
        return expand(node, environ, fail)

    return walk(data, "", False)
