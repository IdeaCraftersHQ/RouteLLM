"""Tests for the torch-free router registry.

The root conftest stubs `routellm.routers.routers` (it imports torch at
module level), so these tests import `routellm.routers.registry`
directly: that module is real, torch-free, and holds the backing dict
the stubbed module re-exports.
"""
import logging

import pytest

import routellm.routers.registry as registry
from routellm.routers.base import Router
from routellm.routers.registry import (
    ROUTER_CLS,
    discovery_failures,
    get_router_class,
    name_for,
    register_router,
    reset_registry,
    router_names,
)


class DummyRouter:
    """Minimal stand-in class; registration never instantiates."""


class OtherRouter:
    """Second stand-in, for replace and reverse-lookup tests."""


class LateRouter(Router):
    """Real Router subclass registered after import, for __str__."""

    def calculate_strong_win_rate(self, prompt):
        return 0.5


class FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


def _broken_load():
    raise ImportError("no module named 'nope'")


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot registry + failures, run on a clean slate, restore after."""
    saved_cls = dict(ROUTER_CLS)
    saved_failures = dict(discovery_failures)
    reset_registry()
    yield
    reset_registry()
    ROUTER_CLS.update(saved_cls)
    discovery_failures.update(saved_failures)


@pytest.fixture
def fake_entry_points(monkeypatch):
    """Patch entry_points in the registry namespace with given endpoints."""

    def install(*eps):
        def fake(group=None):
            assert group == "routellm.routers"
            return list(eps)

        monkeypatch.setattr(registry, "entry_points", fake)

    return install


def test_register_get_names_round_trip():
    assert register_router("dummy", DummyRouter) is DummyRouter

    assert get_router_class("dummy") is DummyRouter
    assert router_names() == ["dummy"]
    assert name_for(DummyRouter) == "dummy"


def test_router_names_sorted():
    register_router("zeta", DummyRouter)
    register_router("alpha", OtherRouter)

    assert router_names() == ["alpha", "zeta"]


def test_duplicate_name_raises():
    register_router("dummy", DummyRouter)

    with pytest.raises(ValueError, match="dummy"):
        register_router("dummy", OtherRouter)

    assert get_router_class("dummy") is DummyRouter


def test_replace_overrides():
    register_router("dummy", DummyRouter)

    register_router("dummy", OtherRouter, replace=True)

    assert get_router_class("dummy") is OtherRouter


def test_empty_name_raises():
    with pytest.raises(ValueError):
        register_router("", DummyRouter)


def test_non_class_raises():
    with pytest.raises(ValueError):
        register_router("dummy", lambda: None)


def test_decorator_form_registers():
    @register_router("decorated")
    class Decorated:
        pass

    assert get_router_class("decorated") is Decorated
    assert name_for(Decorated) == "decorated"


def test_name_for_unregistered_raises():
    with pytest.raises(KeyError):
        name_for(DummyRouter)


def test_unknown_name_error_lists_registered():
    register_router("dummy", DummyRouter)

    with pytest.raises(KeyError) as excinfo:
        get_router_class("missing")

    assert "dummy" in str(excinfo.value)


def test_discover_registers_good_and_records_broken(fake_entry_points, caplog):
    fake_entry_points(
        FakeEntryPoint("good", lambda: DummyRouter),
        FakeEntryPoint("broken", _broken_load),
    )
    caplog.set_level(logging.WARNING, logger="routellm.routers.registry")

    registered = registry.discover_routers()

    assert registered == ["good"]
    assert get_router_class("good") is DummyRouter
    assert "broken" in discovery_failures
    assert "ImportError" in discovery_failures["broken"]
    assert any("broken" in record.message for record in caplog.records)


def test_discover_twice_is_idempotent(fake_entry_points):
    fake_entry_points(FakeEntryPoint("good", lambda: DummyRouter))

    first = registry.discover_routers()
    second = registry.discover_routers()

    assert first == ["good"]
    assert second == []
    assert router_names() == ["good"]


def test_unknown_name_error_lists_discovery_failure(fake_entry_points):
    fake_entry_points(FakeEntryPoint("broken", _broken_load))
    registry.discover_routers()

    with pytest.raises(KeyError) as excinfo:
        get_router_class("broken")

    assert "no module named" in str(excinfo.value)


def test_router_cls_is_backing_dict():
    from routellm.routers.registry import ROUTER_CLS as reimported

    assert reimported is ROUTER_CLS

    register_router("late", DummyRouter)

    assert ROUTER_CLS["late"] is DummyRouter


def test_str_of_late_registered_instance():
    register_router("late", LateRouter)

    assert str(LateRouter()) == "late"
