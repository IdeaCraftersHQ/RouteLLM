"""The Presidio tier, and the guarantees around it.

Tests marked `redact` need the extra and both spaCy models and skip
without them. The rest run on a base install.
"""

import itertools
import socket
import subprocess
import sys

import pytest

from routellm import redact
from routellm.redact.recognizers import analyze_patterns


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open or resolve a connection raises."""

    def deny(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args[:2]}")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "gethostbyname", deny)


FR = (
    "Jean Dupont, né le 12 mars 1985, habite à Montréal. NAS 046 454 286, "
    "RAMQ DUPJ 8503 1234, H2X 1Y4, facture 1 234,56 $, jean@exemple.fr, "
    "IBAN FR7630006000011234567890189, carte 4111 1111 1111 1111, "
    "tel +1 514 555 0199, dossier CLI-123456."
)
EN = (
    "John Smith, born on March 12, 1985, lives in Toronto. SIN 046 454 286, "
    "postal M5V 2T6, balance 1 234,56 $, john@example.com, phone 416-555-0199."
)


# -- base install ------------------------------------------------------


def test_core_import_does_not_load_optional_deps():
    code = (
        "import sys, routellm.redact;"
        "from routellm.redact import analyze_patterns;"
        "analyze_patterns('NAS 046 454 286', 'fr');"
        "bad = [m for m in ('presidio_analyzer', 'spacy') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_patterns_tier_runs_offline(no_network):
    spans = analyze_patterns(EN, "en")
    cats = {s.category for s in spans}
    assert {"sin", "postal_code", "amount", "email", "date_of_birth"} <= cats


def test_missing_extra_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    redact._cache.clear()
    with pytest.raises(redact.RedactUnavailable, match=r"routellm\[redact\]"):
        redact.analyze("x", "en")


def test_missing_model_is_an_error_not_a_download(monkeypatch):
    pytest.importorskip("presidio_analyzer")
    spacy_util = pytest.importorskip("spacy.util")
    monkeypatch.setattr(spacy_util, "is_package", lambda name: False)
    redact._cache.clear()
    with pytest.raises(redact.RedactUnavailable, match="never downloads"):
        redact.analyze("x", "en")


def test_unknown_config_key_rejected():
    pytest.importorskip("presidio_analyzer")
    with pytest.raises(ValueError, match="unknown key"):
        redact.build_analyzer({"client_ids": []})


# -- Presidio tier -----------------------------------------------------


@pytest.mark.redact
def test_french_message_every_category(no_network):
    redact._cache.clear()
    spans = redact.analyze(FR, "fr", client_id_patterns=[r"CLI-\d{6}"])
    got = {}
    for s in spans:
        got.setdefault(s["category"], []).append(FR[s["start"] : s["end"]])
    assert got["person"] == ["Jean Dupont"]
    assert "Montréal" in got["location"]
    assert got["date_of_birth"] == ["12 mars 1985"]
    assert got["sin"] == ["046 454 286"]
    assert got["ramq"] == ["DUPJ 8503 1234"]
    assert got["postal_code"] == ["H2X 1Y4"]
    assert got["amount"] == ["1 234,56 $"]
    assert got["email"] == ["jean@exemple.fr"]
    assert got["iban"] == ["FR7630006000011234567890189"]
    assert got["card"] == ["4111 1111 1111 1111"]
    assert "+1 514 555 0199" in got["phone"]
    assert got["client_id"] == ["CLI-123456"]


@pytest.mark.redact
def test_english_message(no_network):
    spans = redact.analyze(EN, "en")
    got = {s["category"]: EN[s["start"] : s["end"]] for s in spans}
    assert got["person"] == "John Smith"
    assert got["location"] == "Toronto"
    assert got["date_of_birth"] == "March 12, 1985"
    assert got["sin"] == "046 454 286"
    assert got["email"] == "john@example.com"
    assert got["phone"] == "416-555-0199"


@pytest.mark.redact
def test_invalid_sin_and_card_not_reported(no_network):
    text = "NAS 046 454 287 et carte 4111 1111 1111 1112"
    cats = {s["category"] for s in redact.analyze(text, "fr")}
    assert "sin" not in cats
    assert "card" not in cats


@pytest.mark.redact
def test_spans_are_non_overlapping_and_ordered(no_network):
    spans = redact.analyze(FR, "fr", client_id_patterns=[r"CLI-\d{6}"])
    for a, b in itertools.pairwise(spans):
        assert a["end"] <= b["start"]
    assert all(set(s) == {"start", "end", "category", "score"} for s in spans)


@pytest.mark.redact
def test_unconfigured_language_rejected():
    with pytest.raises(ValueError, match="not configured"):
        redact.analyze("hola", "es")


@pytest.mark.redact
def test_no_network_at_runtime_across_every_recognizer(no_network):
    """Engine build, model load and a full analysis, sockets blocked.

    Regression guard for two silent fetches: Presidio's email
    validator pulls the public-suffix list, and its spaCy engine
    downloads a model it cannot find.
    """
    redact._cache.clear()
    text = "écrire à jean@exemple.fr ou john@example.com, https://example.org"
    cats = {s["category"] for s in redact.analyze(text, "fr")}
    assert "email" in cats
