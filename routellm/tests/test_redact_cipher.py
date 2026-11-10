"""Protection of the values held in the token store: plaintext by default, AES-GCM on request."""

import base64
import builtins
import importlib.util
import os
import sqlite3
import stat

import pytest

from routellm.config import ConfigError
from routellm.redact.cipher import AesGcmCipher, PlainCipher, build_cipher
from routellm.redact.placeholders import MappingError, RedactContext, Redactor
from routellm.redact.tokens import TokenStore

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cryptography") is None,
    reason="the redact-encrypt extra is not installed",
)

VALUE = "Jean Tremblay"
CTX = RedactContext(principal="team-a", request_id="r1", language="fr")


def person(text, language):
    start = text.find(VALUE)
    return (
        []
        if start < 0
        else [{"start": start, "end": start + len(VALUE), "category": "person", "score": 0.9}]
    )


def raw_db(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def rows(path):
    with sqlite3.connect(path) as c:
        return c.execute("SELECT scope, placeholder, value FROM tokens").fetchall()


def key_bytes(n=32):
    return bytes(range(n))


# --- default: plaintext, owner-only files --------------------------------------------


def test_default_store_holds_plaintext_in_owner_only_files(tmp_path):
    path = tmp_path / "state" / "tokens.db"
    store = TokenStore(str(path))
    assert isinstance(store.cipher, PlainCipher)
    Redactor(store, analyze=person).redact(f"hi {VALUE}", CTX)
    assert VALUE.encode() in raw_db(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700


# --- aes-gcm -------------------------------------------------------------------------


def test_aes_gcm_keeps_values_out_of_the_database_and_still_restores(tmp_path):
    path = tmp_path / "t.db"
    store = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()))
    redactor = Redactor(store, analyze=person)
    text, mid = redactor.redact(f"hi {VALUE}", CTX)
    assert VALUE.encode() not in raw_db(path)
    assert VALUE not in "".join(r[2] for r in rows(path))
    assert redactor.restore(text, mid, CTX) == f"hi {VALUE}"


def test_each_value_gets_its_own_nonce(tmp_path):
    path = tmp_path / "t.db"
    store = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()))
    redactor = Redactor(store, analyze=person)
    redactor.redact(f"hi {VALUE}", CTX)
    redactor.redact(f"hi {VALUE}", RedactContext("team-a", "r2", language="fr"))
    blobs = [r[2] for r in rows(path)]
    assert len(blobs) == len(set(blobs)) == 2


def test_a_tampered_value_is_an_error_never_text(tmp_path):
    path = tmp_path / "t.db"
    store = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()))
    redactor = Redactor(store, analyze=person)
    text, mid = redactor.redact(f"hi {VALUE}", CTX)
    with sqlite3.connect(path) as c:
        blob = c.execute("SELECT value FROM tokens").fetchone()[0]
        flipped = blob[:-4] + ("AAAA" if not blob.endswith("AAAA") else "BBBB")
        c.execute("UPDATE tokens SET value=?", (flipped,))
    with pytest.raises(MappingError):
        redactor.restore(text, mid, CTX)


def test_a_value_moved_to_another_placeholder_does_not_decrypt(tmp_path):
    path = tmp_path / "t.db"
    store = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()))
    redactor = Redactor(store, analyze=person)
    text, mid = redactor.redact(f"{VALUE}", CTX)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE tokens SET placeholder='[[PERSON_9]]'")
    with pytest.raises(MappingError):
        redactor.restore("[[PERSON_9]]", mid, CTX)


def test_the_wrong_key_cannot_read_what_another_wrote(tmp_path):
    path = tmp_path / "t.db"
    one = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()), key=b"k" * 32)
    text, mid = Redactor(one, analyze=person).redact(f"hi {VALUE}", CTX)
    two = TokenStore(str(path), cipher=AesGcmCipher(bytes(reversed(key_bytes()))), key=b"k" * 32)
    with pytest.raises(MappingError):
        Redactor(two, analyze=person).restore(text, mid, CTX)


def test_a_plaintext_row_is_not_accepted_by_an_encrypting_store(tmp_path):
    path = tmp_path / "t.db"
    plain = TokenStore(str(path), key=b"k" * 32)
    text, mid = Redactor(plain, analyze=person).redact(f"hi {VALUE}", CTX)
    enc = TokenStore(str(path), cipher=AesGcmCipher(key_bytes()), key=b"k" * 32)
    with pytest.raises(MappingError):
        Redactor(enc, analyze=person).restore(text, mid, CTX)


def test_the_key_must_be_32_bytes():
    with pytest.raises(ValueError):
        AesGcmCipher(b"short")


# --- config --------------------------------------------------------------------------


def write_key(path, mode=0o600, raw=None):
    path.write_text(base64.b64encode(raw or key_bytes()).decode())
    os.chmod(path, mode)
    return str(path)


def test_none_is_the_default():
    assert isinstance(build_cipher(None), PlainCipher)
    assert isinstance(build_cipher({"encryption": "none"}), PlainCipher)


def test_aes_gcm_from_a_key_file(tmp_path):
    cipher = build_cipher({"encryption": "aes-gcm", "key_file": write_key(tmp_path / "k")})
    assert isinstance(cipher, AesGcmCipher)


def test_aes_gcm_from_an_environment_variable(monkeypatch):
    monkeypatch.setenv("REDACT_KEY", base64.b64encode(key_bytes()).decode())
    assert isinstance(
        build_cipher({"encryption": "aes-gcm", "key_env": "REDACT_KEY"}), AesGcmCipher
    )


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"encryption": "rot13"}, "encryption"),
        ({"encryption": "aes-gcm"}, "key_file or key_env"),
        ({"encryption": "aes-gcm", "key_file": "/nonexistent/k"}, "key file"),
        ({"encryption": "aes-gcm", "key_env": "NO_SUCH_VAR_XYZ"}, "NO_SUCH_VAR_XYZ"),
        ({"encryption": "aes-gcm", "key_file": "x", "key_env": "y"}, "only one"),
        ({"encryption": "none", "key_file": "x"}, "aes-gcm"),
        ({"bogus": 1}, "unknown"),
    ],
)
def test_bad_encryption_config_stops_startup(cfg, match):
    with pytest.raises(ConfigError, match=match):
        build_cipher(cfg)


def test_a_key_file_others_can_read_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="0600"):
        build_cipher({"encryption": "aes-gcm", "key_file": write_key(tmp_path / "k", 0o644)})


def test_a_key_of_the_wrong_length_is_refused(tmp_path):
    key = write_key(tmp_path / "k", raw=b"short")
    with pytest.raises(ConfigError, match="32 bytes"):
        build_cipher({"encryption": "aes-gcm", "key_file": key})


def test_aes_gcm_without_the_extra_stops_startup(tmp_path, monkeypatch):
    real = builtins.__import__

    def deny(name, *a, **k):
        if name.startswith("cryptography"):
            raise ImportError("no cryptography")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(ConfigError, match="redact-encrypt"):
        build_cipher({"encryption": "aes-gcm", "key_file": write_key(tmp_path / "k")})
