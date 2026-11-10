"""Entry-point lookup for providers and recognizer types."""

from importlib.metadata import entry_points

PROVIDER_GROUP = "routellm.decision_providers"
RECOGNIZER_GROUP = "routellm.decision_recognizers"


def load_entry_point(group: str, name: str):
    """Return the object registered under `group` as `name`, or None."""
    for ep in entry_points(group=group):
        if ep.name == name:
            return ep.load()
    return None
