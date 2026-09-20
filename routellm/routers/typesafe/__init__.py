"""Lazy import guard for the optional typesafe-sdk dependency.

This module must not import `.router`: routellm/routers/ has no
`__init__.py`, so importing `routellm.routers.typesafe` alone must not
pull in `routers.py` (and therefore torch) as a side effect.
"""

def require_typesafe_sdk():
    """Import and return the typesafe_sdk module, or raise if unavailable.

    Re-imports on every call (no caching) so callers always see the
    current state of `sys.modules`.

    Returns
    -------
    module
        The imported `typesafe_sdk` module.

    Raises
    ------
    ImportError
        If `typesafe_sdk` cannot be imported. Message points to the
        optional extra that provides it.
    """
    try:
        import typesafe_sdk
    except ImportError as exc:
        raise ImportError(
            "TypeSafe router requires the typesafe extra: "
            "pip install 'routellm[typesafe]'"
        ) from exc
    return typesafe_sdk
