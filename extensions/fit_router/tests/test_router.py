"""The `fit` router: a trained advisor's confidence as the win rate."""

import pytest

from routellm.routers.base import Router
from routellm_fit.router import FitRouter


class _Advice:
    def __init__(self, confidence, domain="routellm"):
        self.confidence = confidence
        self.domain = domain
        self.steering_text = ""


class _Advisor:
    def __init__(self, confidence=0.8, raises=None):
        self._confidence = confidence
        self._raises = raises
        self.seen = []

    def generate_advice(self, context):
        if self._raises is not None:
            raise self._raises
        self.seen.append(context)
        return _Advice(self._confidence)

    def model_id(self):
        return "stub"


def test_it_is_a_router():
    assert issubclass(FitRouter, Router)


def test_confidence_becomes_the_win_rate():
    router = FitRouter(advisor=_Advisor(confidence=0.73))

    assert router.calculate_strong_win_rate("write me a parser") == pytest.approx(0.73)


def test_the_prompt_reaches_the_advisor():
    advisor = _Advisor()
    FitRouter(advisor=advisor).calculate_strong_win_rate("hello there")

    assert advisor.seen[0]["prompt"] == "hello there"


def test_the_domain_is_ignored():
    """Only the confidence is used; a foreign domain changes nothing."""
    advisor = _Advisor(confidence=0.4)
    advisor.generate_advice = lambda context: _Advice(0.4, domain="something_else")

    assert FitRouter(advisor=advisor).calculate_strong_win_rate("x") == pytest.approx(0.4)


@pytest.mark.parametrize(
    "confidence, expected", [(-0.5, 0.0), (0.0, 0.0), (1.0, 1.0), (4.2, 1.0)]
)
def test_a_confidence_outside_the_range_is_clamped(confidence, expected):
    """A win rate is compared against a threshold in [0, 1]."""
    router = FitRouter(advisor=_Advisor(confidence=confidence))

    assert router.calculate_strong_win_rate("x") == pytest.approx(expected)


def test_an_unreachable_advisor_raises_naming_the_endpoint():
    router = FitRouter(
        advisor=_Advisor(raises=OSError("connection refused")),
        endpoint="http://localhost:9999",
    )

    with pytest.raises(RuntimeError) as excinfo:
        router.calculate_strong_win_rate("x")

    message = str(excinfo.value)
    assert "http://localhost:9999" in message
    assert "connection refused" in message


def test_a_non_numeric_confidence_raises_naming_the_endpoint():
    advisor = _Advisor()
    advisor.generate_advice = lambda context: _Advice("not a number")

    router = FitRouter(advisor=advisor, endpoint="http://advisor:8080")

    with pytest.raises(RuntimeError) as excinfo:
        router.calculate_strong_win_rate("x")

    assert "http://advisor:8080" in str(excinfo.value)


class _FakeEntryPoint:
    """Stand-in for the entry point this package's pyproject declares."""

    name = "fit"

    def load(self):
        return FitRouter


# The real entry point only resolves once the package is pip-installed,
# which a standalone test run does not require. Patching `entry_points`
# in the registry namespace exercises the same code path with the same
# name and target class the pyproject declares.
def test_registered_through_entry_point(monkeypatch):
    from routellm.routers import registry

    def fake_entry_points(group=None):
        assert group == registry.ENTRY_POINT_GROUP
        return [_FakeEntryPoint()]

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)

    saved = dict(registry.ROUTER_CLS)
    saved_failures = dict(registry.discovery_failures)
    registry.reset_registry()
    try:
        assert registry.discover_routers() == ["fit"]
        assert registry.get_router_class("fit") is FitRouter
    finally:
        registry.reset_registry()
        registry.ROUTER_CLS.update(saved)
        registry.discovery_failures.update(saved_failures)
