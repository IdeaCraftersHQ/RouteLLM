"""TypeSafe-backed routing for routellm.

Exports the `jev` router and the Jev intent detector. The router
registers itself with routellm through the `routellm.routers` entry
point declared in this package's pyproject, so installing the package
is the whole wiring; no import of this module is needed to make `jev`
resolvable.
"""

from routellm_typesafe.intent_detector import JevIntentDetector
from routellm_typesafe.router import JevRouter

__all__ = ["JevRouter", "JevIntentDetector"]
