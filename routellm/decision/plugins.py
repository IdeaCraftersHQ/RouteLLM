"""Entry-point lookup for providers and recognizer types."""

from importlib.metadata import entry_points

PROVIDER_GROUP = "routellm.decision_providers"
RECOGNIZER_GROUP = "routellm.decision_recognizers"


def load_entry_point(group: str, name: str):
    """Return the object registered under `group` as `name`, or None.

    Raises
    ------
    ValueError
        If the entry point exists but fails to load; the message names
        the entry point, its group and the underlying error.
    """
    for ep in entry_points(group=group):
        if ep.name == name:
            try:
                return ep.load()
            except Exception as exc:
                raise ValueError(
                    f"entry point {name!r} in {group!r} failed to load: {type(exc).__name__}: {exc}"
                ) from exc
    return None
