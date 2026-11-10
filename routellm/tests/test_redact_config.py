"""The `redact:` config section: what it builds and what stops startup."""

import importlib.util
import os
import stat

import pytest

from routellm.config import ConfigError
from routellm.redact.config import build_redact_transform
from routellm.redact.transform import RedactTransform
from routellm.server_config import SERVER_SECTIONS, build_transforms
from routellm.transform import TransformContext

CTX = TransformContext(endpoint="e", reach=frozenset({"cloud"}), treatment="redact", data_class="p")
MSG = [{"role": "user", "content": "client ACME-12345 wrote to a@example.com"}]


def cfg(tmp_path, **over):
    return {
        "engine": "patterns",
        "client_id_patterns": [r"ACME-\d{5}"],
        "store": {"path": str(tmp_path / "state" / "tokens.db")},
        **over,
    }


def test_the_section_is_a_known_server_section():
    assert "redact" in SERVER_SECTIONS


def test_no_section_no_transform():
    assert build_redact_transform(None, None) is None
    assert build_transforms({}) == []


def test_the_section_builds_a_working_transform_and_a_private_store(tmp_path):
    transform = build_redact_transform(cfg(tmp_path), None)
    assert isinstance(transform, RedactTransform)
    out, state = transform.forward(MSG, CTX)
    assert "ACME-12345" not in out[0]["content"] and "a@example.com" not in out[0]["content"]
    assert transform.inverse(out[0]["content"], state) == MSG[0]["content"]
    db = tmp_path / "state" / "tokens.db"
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(db.parent).st_mode) == 0o700


def test_the_store_path_is_resolved_against_the_config_file(tmp_path):
    raw = cfg(tmp_path, store={"path": "state/tokens.db"})
    build_redact_transform(raw, str(tmp_path / "conf" / "routellm.yaml"))
    assert (tmp_path / "conf" / "state" / "tokens.db").exists()


def test_it_runs_before_the_user_s_transforms(tmp_path):
    built = build_transforms(
        {"redact": cfg(tmp_path), "transforms": ["identity"]}, str(tmp_path / "c.yaml")
    )
    assert isinstance(built[0], RedactTransform)


@pytest.mark.skipif(
    importlib.util.find_spec("cryptography") is None, reason="the redact-encrypt extra is absent"
)
def test_aes_gcm_is_selected_by_config(tmp_path):
    import base64

    key = tmp_path / "k"
    key.write_text(base64.b64encode(bytes(range(32))).decode())
    key.chmod(0o600)
    raw = cfg(
        tmp_path, store={**cfg(tmp_path)["store"], "encryption": "aes-gcm", "key_file": str(key)}
    )
    transform = build_redact_transform(raw, None)
    out, _ = transform.forward(MSG, CTX)
    assert b"ACME-12345" not in (tmp_path / "state" / "tokens.db").read_bytes()
    assert "ACME" not in out[0]["content"]


@pytest.mark.parametrize(
    "over,match",
    [
        ({"engine": "magic"}, "engine"),
        ({"bogus": 1}, "bogus"),
        ({"languages": ["de"]}, "languages"),
        ({"languages": []}, "languages"),
        ({"non_text": "maybe"}, "non_text"),
        ({"store": {"ttl_s": -1}}, "ttl_s"),
        ({"store": {"nope": 1}}, "nope"),
        ({"client_id_patterns": ["("]}, "client_id_patterns"),
        ({"stable_across_requests": True}, "stable_across_requests"),
    ],
)
def test_a_bad_section_stops_startup(tmp_path, over, match):
    raw = {**cfg(tmp_path), **over}
    if "store" in over:
        raw["store"] = {"path": str(tmp_path / "t.db"), **over["store"]}
    with pytest.raises(ConfigError, match=match):
        build_redact_transform(raw, None)


def test_a_missing_presidio_stops_startup_not_the_first_request(tmp_path, monkeypatch):
    import routellm.redact as redact_mod

    def unavailable(config=None):
        from routellm.redact.presidio_adapter import RedactUnavailable

        raise RedactUnavailable("spaCy model missing")

    monkeypatch.setattr(redact_mod, "build_analyzer", unavailable)
    with pytest.raises(ConfigError, match="spaCy model missing"):
        build_redact_transform(cfg(tmp_path, engine="presidio"), None)


def test_presidio_is_the_default_engine(tmp_path, monkeypatch):
    import routellm.redact as redact_mod

    seen = []

    class Fake:
        def analyze(self, text, language, threshold=0.4):
            seen.append(language)
            return []

    monkeypatch.setattr(redact_mod, "build_analyzer", lambda config=None: Fake())
    raw = cfg(tmp_path)
    raw.pop("engine")
    transform = build_redact_transform(raw, None)
    transform.forward(MSG, CTX)
    assert {"fr", "en"} <= set(seen)
