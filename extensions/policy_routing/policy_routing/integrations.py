"""Wire-up helpers for routellm's cache hook (US-0102).

Until US-0102 lands in routellm core, this module ships a thin
adapter that stages the integration: a builder returns a callable
matching the planned hook signature, so consumers can wire it now
and flip the import path once core ships.

Planned routellm contract (US-0102, line 32):

    cache_policy: Callable[[Request, ClassificationResult], CachePolicy]

The classifier is injected as a separate callable so the hook itself
stays pure — easier to unit-test and swap.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from policy_routing.policy import (
    DEFAULT_POLICY,
    CachePolicy,
    classify_to_policy,
)

# Type aliases for readability. `Any` because c12n bindings + routellm
# Request types may not be importable yet at adoption time.
Request = Any
ClassificationResult = Any
Classifier = Callable[[Request], ClassificationResult]
CacheHook = Callable[[Request, ClassificationResult], CachePolicy]


def build_cache_hook(
    classifier: Optional[Classifier] = None,
) -> CacheHook:
    """Return a cache_policy hook for routellm's controller.

    If `classifier` is provided AND the caller invokes the hook with
    `classification=None`, the hook will run the classifier on the
    request first. Otherwise the caller is expected to pass an
    already-classified result (the common case — classification
    middleware sits upstream of the cache hook).

    Returns a callable matching US-0102 contract.
    """

    def hook(
        request: Request,
        classification: ClassificationResult = None,
    ) -> CachePolicy:
        if classification is None and classifier is not None:
            classification = classifier(request)
        if classification is None:
            return DEFAULT_POLICY
        return classify_to_policy(request, classification)

    return hook


__all__ = [
    "CacheHook",
    "Classifier",
    "build_cache_hook",
]
