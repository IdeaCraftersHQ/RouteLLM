"""Tests for the domain intent detector's cache loading.

The cache is best-effort: a missing, unreadable or malformed file is a
cold start, not a failure. What it must NOT do is swallow a programming
error and leave the caller with a silently empty cache.
"""

import json

import pytest

from routellm.middleware.domain_intent_detector import DomainIntentDetector


def _detector(cache_path):
    return DomainIntentDetector(intent_mappings=[], cache_path=str(cache_path))


def test_malformed_cache_is_tolerated_as_a_cold_start(tmp_path, capsys):
    path = tmp_path / "cache.json"
    path.write_text("{not json at all")

    detector = _detector(path)

    assert detector.embedding_cache == {}
    assert "Error loading cache" in capsys.readouterr().out


def test_a_cache_of_the_wrong_shape_is_tolerated(tmp_path):
    path = tmp_path / "cache.json"
    # A list where a mapping is expected: .items() raises AttributeError,
    # which is a bug in the reader, not a corrupt-file case.
    path.write_text(json.dumps({"hello": [0.1, 0.2]}))

    detector = _detector(path)

    assert "hello" in detector.embedding_cache


def test_an_unexpected_error_while_loading_the_cache_propagates(tmp_path, monkeypatch):
    """A bug in the reader must not be hidden by the tolerant branch.

    The except list names the file/decode failures a cache load can
    legitimately hit. A broad `except Exception` would also absorb
    this AttributeError and report a cold start for a real defect.
    """
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"hello": [0.1, 0.2]}))

    def boom(*_args, **_kwargs):
        raise AttributeError("reader bug")

    monkeypatch.setattr(json, "load", boom)

    with pytest.raises(AttributeError, match="reader bug"):
        _detector(path)
