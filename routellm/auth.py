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

A key held by a trusted proxy may carry several clients, named per
request by a header. Opt in per entry; entries without it are
untouched and the header is ignored for them:

        client_header: X-Route-Client
        clients:
          web: {profile: web, allowed_classes: [public]}   # both optional
          batch: {}

With the header present and listed, the principal is `<principal>/<client>`
(profile and classes from the client, else the entry's); present but not
listed is a 403 (`ClientNotListed`); absent is the entry's own principal.
Only a key that is already valid reaches this step, so use it only for a
proxy that sets the header itself and strips any the caller sent.

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
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from routellm.config import ConfigError

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEADER_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_CLIENT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
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


class ClientNotListed(Exception):
    """A valid key whose `client_header` names a client the entry does not list.

    The value sent is deliberately not kept: it is caller-controlled and
    unbounded, and nothing downstream needs it.
    """

    def __init__(self, header: str):
        super().__init__(f"client named by header {header} is not listed for this key")
        self.header = header


class PrincipalResolver(Protocol):
    """Maps a presented bearer token to a principal, or None if unknown.

    `headers` are the request headers, for a resolver that lets a key
    name a client; one that does not may ignore them. It may raise
    `ClientNotListed` for a valid key naming an unlisted client.
    """

    def resolve(
        self, bearer: str, headers: Mapping[str, str] | None = None
    ) -> Principal | None: ...


@dataclass(frozen=True)
class TrustedProxy:
    """An entry's opt-in: the header naming a client, and the clients it may name."""

    header: str
    clients: dict[str, Principal]


class _Client(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    allowed_classes: list[str] | None = None


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_sha256: str
    principal: str
    profile: str | None = None
    allowed_classes: list[str] = []
    client_header: str | None = None
    clients: dict[str, _Client] | None = None

    @model_validator(mode="after")
    def _trusted_proxy(self) -> _Entry:
        if (self.client_header is None) != (self.clients is None):
            raise ValueError("`client_header` and `clients` go together")
        if self.client_header is None:
            return self
        if not _HEADER_TOKEN.match(self.client_header):
            raise ValueError("`client_header` is not a valid header name")
        if not self.clients:
            raise ValueError("`clients` must name at least one client")
        for name in self.clients:
            if not _CLIENT_NAME.match(name):
                raise ValueError("client names are 1-64 chars of letters, digits, `.`, `_`, `-`")
        return self


class _KeysFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[_Entry] = []


class StaticKeyResolver:
    """Resolve against a fixed list of (sha256, principal[, proxy]) entries.

    The presented key is hashed and compared with every entry using a
    constant-time comparison, with no early exit, so response time does
    not say which entry (if any) was close. Only after a key matches is
    the client header read, and only for an entry that opted in.
    """

    def __init__(self, entries: list[tuple]):
        self._entries = [(e[0], e[1], e[2] if len(e) > 2 else None) for e in entries]

    def __len__(self) -> int:
        return len(self._entries)

    def principals(self) -> list[Principal]:
        """Every principal the map can yield, client ones included, for startup checks."""
        out: list[Principal] = []
        for _, principal, proxy in self._entries:
            out.append(principal)
            if proxy is not None:
                out.extend(proxy.clients.values())
        return out

    def resolve(self, bearer: str, headers: Mapping[str, str] | None = None) -> Principal | None:
        digest = hashlib.sha256(bearer.encode("utf-8")).digest()
        found: tuple[Principal, TrustedProxy | None] | None = None
        for known, principal, proxy in self._entries:
            if hmac.compare_digest(digest, known):
                found = (principal, proxy)
        if found is None:
            return None
        principal, proxy = found
        if proxy is None or headers is None:
            return principal
        client = headers.get(proxy.header)
        if client is None:
            return principal
        resolved = proxy.clients.get(client)
        if resolved is None:
            raise ClientNotListed(proxy.header)
        return resolved


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

    entries: list[tuple[bytes, Principal, TrustedProxy | None]] = []
    seen: set[bytes] = set()
    names: set[str] = set()
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
        principal = Principal(e.principal, e.profile, tuple(e.allowed_classes))
        proxy = None
        if e.client_header is not None and e.clients is not None:
            proxy = TrustedProxy(
                e.client_header,
                {
                    name: Principal(
                        f"{e.principal}/{name}",
                        c.profile if c.profile is not None else e.profile,
                        tuple(
                            c.allowed_classes
                            if c.allowed_classes is not None
                            else e.allowed_classes
                        ),
                    )
                    for name, c in e.clients.items()
                },
            )
        for who in [principal, *(proxy.clients.values() if proxy else ())]:
            if who.principal in names:
                raise ConfigError(
                    f"auth.keys_file {path}: entry {i} yields principal "
                    f"`{who.principal}`, already taken by another entry or client"
                )
            names.add(who.principal)
        entries.append((digest, principal, proxy))
    if not entries:
        raise ConfigError(f"auth.keys_file {path}: no keys; auth on with no keys admits nobody")
    return StaticKeyResolver(entries)


def build_auth(file_config: dict[str, Any] | None, config_path: str | None, resolve_path=None):
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
    from routellm.config import path_resolver

    return load_keys_file(path_resolver(config_path, resolve_path)("auth.keys_file", keys_file))


def bearer_token(authorization: str | None) -> str | None:
    """The token of an `Authorization: Bearer` header, or None."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token
