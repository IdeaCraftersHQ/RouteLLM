"""Local token store for reversible placeholders.

SQLite, owner-only. Lives on the gateway host and is read by nothing but
`Redactor.restore`: its contents never reach a log, a trace, the response
cache, an audit event or an upstream call.

Layout of the directory holding the database: mode 0700; database and
key file: mode 0600, tightened if an older file is looser.

A value is looked up by `HMAC-SHA256(gateway_key, scope || value)`, so
the lookup column cannot be matched against a guessed value without the
gateway key and cannot be linked across scopes. The value itself is kept
so a reply can be restored; how that column is protected is the store's
`ValueCipher` (`routellm.redact.cipher`): plaintext with the file modes
above by default, like the response cache, or AES-GCM when configured.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable

from routellm.redact.cipher import PlainCipher, ValueCipher

__all__ = [
    "DEFAULT_TTL_S",
    "MappingError",
    "MappingExpired",
    "RedactError",
    "TokenStore",
    "format_placeholder",
]

DEFAULT_TTL_S = 24 * 3600


class RedactError(Exception):
    """A redaction step failed; callers must refuse the request."""


class MappingError(RedactError):
    """No usable mapping: unknown, expired, or not this caller's."""


class MappingExpired(MappingError):
    """The mapping exists but its time is up."""


def format_placeholder(category: str, n: int) -> str:
    """`[[CATEGORY_n]]`, the category upper-cased."""
    return f"[[{category.upper()}_{n}]]"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mappings (
    id TEXT PRIMARY KEY, principal TEXT NOT NULL, scope TEXT NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tokens (
    scope TEXT NOT NULL, lookup TEXT NOT NULL, placeholder TEXT NOT NULL,
    category TEXT NOT NULL, value TEXT NOT NULL, expires REAL NOT NULL,
    PRIMARY KEY (scope, lookup), UNIQUE (scope, placeholder));
CREATE TABLE IF NOT EXISTS counters (
    ckey TEXT NOT NULL, category TEXT NOT NULL, n INTEGER NOT NULL, expires REAL,
    PRIMARY KEY (ckey, category));
"""


def _private_file(path: str, create: bool = True) -> None:
    if create:
        os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))
    os.chmod(path, 0o600)


class TokenStore:
    """Mappings from placeholders back to values, with a TTL.

    `path`: database file. `key`: gateway key (32 bytes); when omitted it
    is read from, or created in, `<path>.key`. `ttl_s`: lifetime of a
    mapping and of each token, counted from the last use of the token.
    `cipher`: protection of the stored values (default: none, file modes
    only).
    """

    def __init__(
        self,
        path: str,
        *,
        key: bytes | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        cipher: ValueCipher | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.path = path
        self.key_path = path + ".key"
        self.ttl_s = ttl_s
        self.cipher = cipher if cipher is not None else PlainCipher()
        self._clock = clock
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, mode=0o700, exist_ok=True)
            os.chmod(parent, 0o700)
        _private_file(path)
        self._key = key if key is not None else self._load_key()
        if len(self._key) < 16:
            raise ValueError("gateway key too short")
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # -- plumbing ---------------------------------------------------------

    def _load_key(self) -> bytes:
        try:
            fd = os.open(self.key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            _private_file(self.key_path, create=False)
            with open(self.key_path, "rb") as f:
                return f.read()
        with os.fdopen(fd, "wb") as f:
            key = secrets.token_bytes(32)
            f.write(key)
        os.chmod(self.key_path, 0o600)
        return key

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        return _Closing(conn)

    def _mac(self, *parts: str) -> str:
        msg = "\x00".join(parts).encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def _context(self, scope: str, placeholder: str) -> bytes:
        return f"{scope}\x00{placeholder}".encode()

    # -- keys -------------------------------------------------------------

    def lookup_key(self, scope: str, value: str) -> str:
        """HMAC-SHA256 of `value` under the gateway key, bound to `scope`."""
        return self._mac("value", scope, value)

    def scope_id(self, kind: str, principal: str, ident: str) -> str:
        """Opaque scope for a principal's request or conversation."""
        return self._mac("scope", kind, principal, ident)

    def principal_id(self, principal: str) -> str:
        return self._mac("principal", principal)

    # -- writes -----------------------------------------------------------

    def new_mapping(self, principal: str, scope: str) -> str:
        """A fresh unguessable mapping id bound to `principal` and `scope`."""
        mid = secrets.token_urlsafe(16)
        now = self._clock()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            self._purge(c, now)
            c.execute(
                "INSERT INTO mappings VALUES (?,?,?,?)",
                (mid, self.principal_id(principal), scope, now + self.ttl_s),
            )
            c.execute("COMMIT")
        return mid

    def assign(
        self,
        scope: str,
        counter_key: str,
        items: Iterable[tuple[str, str]],
        reserved: frozenset[str],
        *,
        counter_expires: bool,
    ) -> list[str]:
        """Placeholders for `(category, value)` items, minted as needed.

        An existing value keeps its placeholder; a new one takes the next
        number for its category under `counter_key`, skipping any
        placeholder in `reserved` (text already in the prompt). An
        existing placeholder that is reserved raises `RedactError`.
        """
        now = self._clock()
        exp = now + self.ttl_s
        out: list[str] = []
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                for category, value in items:
                    lk = self.lookup_key(scope, value)
                    row = c.execute(
                        "SELECT placeholder FROM tokens WHERE scope=? AND lookup=? AND expires>?",
                        (scope, lk, now),
                    ).fetchone()
                    if row:
                        if row[0] in reserved:
                            raise RedactError("placeholder collides with text in the prompt")
                        c.execute(
                            "UPDATE tokens SET expires=? WHERE scope=? AND lookup=?",
                            (exp, scope, lk),
                        )
                        out.append(row[0])
                        continue
                    ph = self._next(
                        c, counter_key, category, reserved, exp if counter_expires else None
                    )
                    c.execute(
                        "INSERT OR REPLACE INTO tokens VALUES (?,?,?,?,?,?)",
                        (
                            scope,
                            lk,
                            ph,
                            category,
                            self.cipher.protect(value, self._context(scope, ph)),
                            exp,
                        ),
                    )
                    out.append(ph)
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return out

    def _next(self, c, ckey: str, category: str, reserved: frozenset[str], expires) -> str:
        row = c.execute(
            "SELECT n FROM counters WHERE ckey=? AND category=?", (ckey, category)
        ).fetchone()
        n = row[0] if row else 0
        while True:
            n += 1
            ph = format_placeholder(category, n)
            if ph not in reserved:
                break
        c.execute("INSERT OR REPLACE INTO counters VALUES (?,?,?,?)", (ckey, category, n, expires))
        return ph

    # -- reads ------------------------------------------------------------

    def resolve(self, mapping_id: str, principal: str, scope: str) -> dict[str, str]:
        """`{placeholder: value}` for a live mapping this caller owns.

        Raises `MappingExpired` past the TTL and `MappingError` for an
        unknown mapping or one bound to another principal or scope; the
        two look alike so ids cannot be probed.
        """
        now = self._clock()
        with self._conn() as c:
            row = c.execute(
                "SELECT principal, scope, expires FROM mappings WHERE id=?", (mapping_id,)
            ).fetchone()
            if row is None or not (
                hmac.compare_digest(row[0], self.principal_id(principal))
                and hmac.compare_digest(row[1], scope)
            ):
                raise MappingError("no such mapping")
            if row[2] <= now:
                raise MappingExpired("mapping expired")
            rows = c.execute(
                "SELECT placeholder, value FROM tokens WHERE scope=? AND expires>?", (scope, now)
            ).fetchall()
        try:
            return {ph: self.cipher.reveal(v, self._context(scope, ph)) for ph, v in rows}
        except ValueError as exc:
            raise MappingError("a stored value cannot be read") from exc

    # -- upkeep and inspection ---------------------------------------------

    def _purge(self, c, now: float) -> int:
        n = c.execute("DELETE FROM mappings WHERE expires<=?", (now,)).rowcount
        n += c.execute("DELETE FROM tokens WHERE expires<=?", (now,)).rowcount
        c.execute("DELETE FROM counters WHERE expires IS NOT NULL AND expires<=?", (now,))
        return n

    def purge_expired(self) -> int:
        """Delete expired rows; returns how many mappings and tokens went."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            n = self._purge(c, self._clock())
            c.execute("COMMIT")
            return n

    def row_counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("mappings", "tokens")
            }

    def dump_lookup_keys(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT lookup FROM tokens")]


class _Closing:
    """sqlite3 connection that closes on `with` exit (the stock one only commits)."""

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def __enter__(self):
        return self._c

    def __exit__(self, *exc):
        self._c.close()
        return False
