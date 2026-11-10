"""Protection of the values the token store holds.

The store keeps each redacted value so a reply can be restored. How that
column is protected is a `ValueCipher`:

- `PlainCipher` (`encryption: none`, the default): the value is stored
  as it is. The protection is the file mode (database and key file 0600,
  their directory 0700), like the response cache. No dependency.
- `AesGcmCipher` (`encryption: aes-gcm`): AES-256-GCM with a fresh nonce
  per value, bound to the value's scope and placeholder, so a row cannot
  be moved or edited undetected. Needs `pip install 'routellm[redact-encrypt]'`
  and a 32-byte key from a 0600 file or an environment variable.

Config (`redact.store`)::

    encryption: aes-gcm
    key_file: /etc/routellm/redact.key   # or: key_env: ROUTELLM_REDACT_KEY
                                         # base64 (or hex) of 32 random bytes

Whichever is chosen, a row written by the other is refused on read: the
store never returns ciphertext as a value, or a value it cannot verify.
"""

import base64
import binascii
import os
import secrets
import stat
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from routellm.config import ConfigError

__all__ = ["AesGcmCipher", "PlainCipher", "ValueCipher", "build_cipher"]

_PREFIX = "aesgcm1:"
_KEYS = {"encryption", "key_file", "key_env"}


@runtime_checkable
class ValueCipher(Protocol):
    """Protects one stored value; `context` binds it to its row."""

    def protect(self, value: str, context: bytes) -> str:
        """The text to store for `value`."""
        ...

    def reveal(self, stored: str, context: bytes) -> str:
        """The value for a stored text; `ValueError` when it cannot be trusted."""
        ...


class PlainCipher:
    """Store values as they are; owner-only file modes are the protection."""

    def protect(self, value: str, context: bytes) -> str:
        return value

    def reveal(self, stored: str, context: bytes) -> str:
        if stored.startswith(_PREFIX):
            raise ValueError("row is encrypted but the store is not configured to decrypt")
        return stored


class AesGcmCipher:
    """AES-256-GCM, a random 96-bit nonce per value, the row as associated data."""

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("aes-gcm key must be 32 bytes")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._aead = AESGCM(key)

    def protect(self, value: str, context: bytes) -> str:
        nonce = secrets.token_bytes(12)
        blob = nonce + self._aead.encrypt(nonce, value.encode(), context)
        return _PREFIX + base64.b64encode(blob).decode()

    def reveal(self, stored: str, context: bytes) -> str:
        if not stored.startswith(_PREFIX):
            raise ValueError("row is not encrypted")
        try:
            blob = base64.b64decode(stored[len(_PREFIX) :], validate=True)
            return self._aead.decrypt(blob[:12], blob[12:], context).decode()
        except Exception as exc:  # bad base64, bad tag, wrong key, wrong row
            raise ValueError("value cannot be decrypted") from exc


def _decode_key(text: str, where: str) -> bytes:
    text = text.strip()
    for decode in (
        bytes.fromhex,
        lambda t: base64.b64decode(t, validate=True),
        lambda t: base64.urlsafe_b64decode(t + "=" * (-len(t) % 4)),
    ):
        try:
            key = decode(text)
        except (ValueError, binascii.Error):
            continue
        if len(key) == 32:
            return key
    raise ConfigError(f"redact.store: {where} must be base64 or hex of exactly 32 bytes")


def _read_key_file(path: str) -> bytes:
    try:
        mode = os.stat(path).st_mode
        with open(path, encoding="ascii") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"redact.store: cannot read key file {path!r}: {exc}") from exc
    if stat.S_IMODE(mode) & 0o077:
        raise ConfigError(f"redact.store: key file {path!r} must be mode 0600 (chmod 600)")
    return _decode_key(text, f"key file {path!r}")


def build_cipher(store: Mapping[str, Any] | None) -> ValueCipher:
    """The cipher a `redact.store` section asks for.

    Raises
    ------
    ConfigError
        Unknown key or encryption, `aes-gcm` without the optional extra,
        without a key, with a key file that is missing or readable by
        others, or with a key that is not 32 bytes. Startup stops.
    """
    store = dict(store or {})
    unknown = sorted(set(store) - _KEYS)
    if unknown:
        raise ConfigError(f"redact.store: unknown key(s) {', '.join(unknown)}")
    mode = store.get("encryption", "none")
    if mode == "none":
        if "key_file" in store or "key_env" in store:
            raise ConfigError("redact.store: a key is only used with encryption: aes-gcm")
        return PlainCipher()
    if mode != "aes-gcm":
        raise ConfigError(f"redact.store: encryption must be none or aes-gcm, got {mode!r}")
    key_file, key_env = store.get("key_file"), store.get("key_env")
    if key_file and key_env:
        raise ConfigError("redact.store: give only one of key_file and key_env")
    if not key_file and not key_env:
        raise ConfigError("redact.store: aes-gcm needs key_file or key_env")
    try:
        import cryptography  # noqa: F401
    except ImportError as exc:
        raise ConfigError(
            "redact.store: encryption: aes-gcm needs the optional extra "
            "(pip install 'routellm[redact-encrypt]')"
        ) from exc
    if key_file:
        key = _read_key_file(str(key_file))
    else:
        value = os.environ.get(str(key_env))
        if not value:
            raise ConfigError(f"redact.store: environment variable {key_env} is not set")
        key = _decode_key(value, f"environment variable {key_env}")
    return AesGcmCipher(key)
