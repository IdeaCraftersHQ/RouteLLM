"""routellm.extensions.policy_routing.

Class-aware cache + routing policy mapping. Translates c12n
ClassificationResult labels into a routellm CachePolicy per the
showcase scenario 3 spec table.

Public API:
    CachePolicy            — dataclass mirroring routellm US-0102 spec
    CacheAction            — enum of legal action values
    MatchStrategy          — enum of legal match values
    classify_to_policy     — pure function: (request, classification) -> CachePolicy
    DEFAULT_NAMESPACE_KEY  — sentinel meaning "use workspace_id default"
"""

from policy_routing.policy import (
    DEFAULT_NAMESPACE_KEY,
    CacheAction,
    CachePolicy,
    MatchStrategy,
    classify_to_policy,
)

__all__ = [
    "CacheAction",
    "CachePolicy",
    "DEFAULT_NAMESPACE_KEY",
    "MatchStrategy",
    "classify_to_policy",
]
