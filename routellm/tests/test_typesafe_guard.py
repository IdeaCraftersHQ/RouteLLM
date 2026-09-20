"""Tests for the typesafe-sdk lazy import guard."""
import sys

import pytest

from routellm.routers.typesafe import require_typesafe_sdk


def test_require_typesafe_sdk_returns_module_when_importable():
    module = require_typesafe_sdk()
    assert module.__name__ == "typesafe_sdk"


def test_require_typesafe_sdk_raises_with_extra_hint_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    with pytest.raises(ImportError, match=r"routellm\[typesafe\]"):
        require_typesafe_sdk()
