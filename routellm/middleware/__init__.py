"""Middleware for intent-based model routing.

Provides intent detection and domain-aware model selection for advanced
routing strategies.
"""

from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector
from routellm.middleware.domain_intent_detector import DomainIntentDetector
from routellm.middleware.intent_config import load_intent_config, save_intent_config, create_example_config

__all__ = [
    "IntentModelSelector",
    "IntentModelMapping",
    "DomainIntentDetector",
    "load_intent_config",
    "save_intent_config",
    "create_example_config"
]
