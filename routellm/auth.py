"""Caller identity: a bearer key mapped to a principal.

Off unless the config has an `auth:` section, which keeps the server as
it has always been: unauthenticated. With `auth.keys_file` set, every
route except `/health` needs `Authorization: Bearer <key>`; a missing or
unknown key is a 401 and the matched caller is exposed as a `Principal`
on `request.state.principal`.

The keys file holds hashes, never keys:

    keys:
      - key_sha256: "<64 hex chars: sha256 of the key>"
        principal: team-a
        profile: default            # optional
        allowed_classes: [public]   # optional

A key never sits in the config or on a command line; the operator
hashes it (`printf %s "$KEY" | shasum -a 256`) and keeps the key
wherever the caller does. An entry carrying a clear-text key
(`key:`, `api_key:`, ...) is refused at startup.

`PrincipalResolver` is the seam: anything with `resolve(bearer)` can
replace the static map.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from routellm.config import ConfigError

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_CLEARTEXT_FIELDS = ("key", "api_key", "secret", "token", "bearer", "password")


@dataclass(frozen=True)
class Principal:
    """Who is calling.

    Parameters
    ----------
    principal : str
        Stable caller name; what audit and cache isolation key on.
    profile : str, optional
        Caller profile name, if the key map carries one.
    allowed_classes : tuple of str
        Data classes this caller may send. Empty means none.
    """

    principal: str
    profile: str | None = None
    allowed_classes: tuple[str, ...] = ()

    def allows(self, data_class: str) -> bool:
        return data_class in self.allowed_classes


#: The caller of the request being served, for code that has no
#: `Request` in hand. Set by the auth dependency; None when auth is off.
current_principal: ContextVar[Principal | None] = ContextVar("current_principal", default=None)


class PrincipalResolver(Protocol):
    """Maps a presented bearer token to a principal, or None if unknown."""

    def resolve(self, bearer: str) -> Principal | None: ...


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_sha256: str
    principal: str
    profile: str | None = None
    allowed_classes: list[str] = []


class _KeysFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[_Entry] = []


class StaticKeyResolver:
    """Resolve against a fixed list of (sha256, principal) pairs.

    The presented key is hashed and compared with every entry using a
    constant-time comparison, with no early exit, so response time does
    not say which entry (if any) was close.
    """

    def __init__(self, entries: list[tuple[bytes, Principal]]):
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, bearer: str) -> Principal | None:
        digest = hashlib.sha256(bearer.encode("utf-8")).digest()
        found: Principal | None = None
        for known, principal in self._entries:
            if hmac.compare_digest(digest, known):
                found = principal
        return found


def load_keys_file(path: Path) -> StaticKeyResolver:
    """Read a keys file into a resolver, refusing anything that holds a key.

    Raises
    ------
    ConfigError
        If the file is unreadable or malformed, an entry carries a
        clear-text key, a hash is not 64 hex chars, or two entries share
        a hash. Messages name the entry's position, never its content.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"auth.keys_file {path}: cannot read: {type(exc).__name__}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"auth.keys_file {path}: top level must be a mapping with `keys:`")

    items = raw.get("keys") or []
    if not isinstance(items, list):
        raise ConfigError(f"auth.keys_file {path}: `keys:` must be a list")
    for i, item in enumerate(items):
        if isinstance(item, dict):
            bad = [f for f in _CLEARTEXT_FIELDS if f in item]
            if bad:
                raise ConfigError(
                    f"auth.keys_file {path}: entry {i} has a clear-text "
                    f"`{bad[0]}`; store only `key_sha256` (sha256 of the key)"
                )
    try:
        parsed = _KeysFile(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        # Pydantic echoes input values only in `input`, which is not used.
        raise ConfigError(f"auth.keys_file {path}: {problems}") from None

    entries: list[tuple[bytes, Principal]] = []
    seen: set[bytes] = set()
    for i, e in enumerate(parsed.keys):
        if not _HEX64.match(e.key_sha256):
            raise ConfigError(
                f"auth.keys_file {path}: entry {i} `key_sha256` is not a "
                "64-character hex sha256; a clear-text key does not belong here"
            )
        digest = bytes.fromhex(e.key_sha256)
        if digest in seen:
            raise ConfigError(f"auth.keys_file {path}: entry {i} repeats an earlier key_sha256")
        seen.add(digest)
        entries.append((digest, Principal(e.principal, e.profile, tuple(e.allowed_classes))))
    if not entries:
        raise ConfigError(f"auth.keys_file {path}: no keys; auth on with no keys admits nobody")
    return StaticKeyResolver(entries)


def build_auth(file_config: dict[str, Any] | None, config_path: str | None):
    """Return the resolver the `auth:` section asks for, else None (auth off)."""
    raw = (file_config or {}).get("auth")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"`auth:` must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"keys_file"}
    if unknown:
        raise ConfigError(f"invalid `auth:` section: unknown key(s) {sorted(unknown)}")
    keys_file = raw.get("keys_file")
    if not isinstance(keys_file, str) or not keys_file:
        raise ConfigError("invalid `auth:` section: `keys_file` is required")
    path = Path(keys_file).expanduser()
    if not path.is_absolute() and config_path:
        path = Path(config_path).resolve().parent / path
    return load_keys_file(path)


def bearer_token(authorization: str | None) -> str | None:
    """The token of an `Authorization: Bearer` header, or None."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token
