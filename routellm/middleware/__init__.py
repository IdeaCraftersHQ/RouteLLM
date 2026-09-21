"""Middleware for intent-based model routing.

Provides intent detection and domain-aware model selection for advanced
routing strategies.
"""

from routellm.middleware.domain_intent_detector import DomainIntentDetector
from routellm.middleware.intent_config import (
    create_example_config,
    load_intent_config,
    save_intent_config,
)
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector

__all__ = [
    "DomainIntentDetector",
    "IntentModelMapping",
    "IntentModelSelector",
    "create_example_config",
    "load_intent_config",
    "save_intent_config",
]
