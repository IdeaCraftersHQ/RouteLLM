"""Request-scoped access to the Decision, for policy routing and egress gates."""

from contextvars import ContextVar

from routellm.decision.models import Decision

_current: ContextVar[Decision | None] = ContextVar("routellm_decision", default=None)


def current_decision() -> Decision | None:
    """Return the Decision of the request being routed, or None."""
    return _current.get()


def set_decision(decision: Decision | None):
    """Set the request's Decision; returns the reset token."""
    return _current.set(decision)
