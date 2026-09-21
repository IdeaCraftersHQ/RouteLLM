"""A routellm router trained on scored traces.

Exports the `fit` router. It registers itself with routellm through
the `routellm.routers` entry point declared in this package's
pyproject, so installing the package is the whole wiring; no import of
this module is needed to make `fit` resolvable.
"""

from routellm_fit.router import FitRouter

__all__ = ["FitRouter"]
