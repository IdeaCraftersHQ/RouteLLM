"""
Configuration utilities for intent-based routing.

This module provides utilities for loading and saving intent configurations
from YAML files, making it easy to define and manage intent mappings.
"""

import os
import yaml
from typing import Dict, List, Optional, Any

from routellm.types import ModelPair
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector


def load_intent_config(config_path: str) -> IntentModelSelector:
    """Load intent configuration from a YAML file.

    A file written for a tier-only selector carries no `default_models`
    and no per-intent `models`; both load as None, and `intent_tiers`
    is restored when the file names it.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    IntentModelSelector
        Configured selector with mappings from the file.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # A file naming no default pair belongs to a tier-only selector;
    # inventing one here would let it bypass the tier tree.
    raw_default = config.get("default_models")
    default_model_pair = (
        ModelPair(
            strong=raw_default.get("strong", "gpt-4"),
            weak=raw_default.get("weak", "gpt-3.5-turbo"),
        )
        if raw_default
        else None
    )

    # Parse intent mappings
    intent_mappings = []
    for intent_name, intent_config in (config.get("intents") or {}).items():
        intent_config = intent_config or {}
        raw_models = intent_config.get("models")
        if raw_models:
            model_pair = ModelPair(
                strong=raw_models.get(
                    "strong",
                    default_model_pair.strong if default_model_pair else None,
                ),
                weak=raw_models.get(
                    "weak",
                    default_model_pair.weak if default_model_pair else None,
                ),
            )
        else:
            model_pair = None

        mapping = IntentModelMapping(
            intent=intent_name,
            model_pair=model_pair,
            description=intent_config.get("description", "")
        )
        
        intent_mappings.append(mapping)
    
    # Create and return the selector
    return IntentModelSelector(
        intent_mappings=intent_mappings,
        default_model_pair=default_model_pair,
        intent_tiers=config.get("intent_tiers"),
        intent_detection_model=config.get("intent_detection_model", "gpt-3.5-turbo")
    )


def save_intent_config(selector: IntentModelSelector, config_path: str) -> None:
    """Save intent configuration to a YAML file.

    Parameters
    ----------
    selector : IntentModelSelector
        The IntentModelSelector instance to save.
    config_path : str
        Path to save the YAML configuration file to.
    """
    # A pair that is None is left out rather than written as null, so a
    # tier-only selector round-trips through `load_intent_config`.
    config = {
        "intent_detection_model": selector.intent_detection_model,
        "intents": {}
    }
    if selector.default_model_pair is not None:
        config["default_models"] = {
            "strong": selector.default_model_pair.strong,
            "weak": selector.default_model_pair.weak
        }
    if selector.intent_tiers:
        config["intent_tiers"] = dict(selector.intent_tiers)

    # Add intent mappings
    for mapping in selector.intent_mappings:
        entry = {"description": mapping.description}
        if mapping.model_pair is not None:
            entry["models"] = {
                "strong": mapping.model_pair.strong,
                "weak": mapping.model_pair.weak
            }
        config["intents"][mapping.intent] = entry
    
    # Save to file
    # Handle the case where config_path has no directory component
    directory = os.path.dirname(config_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_example_config(config_path: str) -> None:
    """Create an example intent configuration file.

    Generates a sample YAML configuration file with example intents,
    model mappings, and settings.

    Parameters
    ----------
    config_path : str
        Path to save the example configuration file to.
    """
    example_config = {
        "intent_detection_model": "gpt-3.5-turbo",
        "default_models": {
            "strong": "gpt-4",
            "weak": "gpt-3.5-turbo"
        },
        "intents": {
            "marketing": {
                "description": "Marketing content creation for digital products",
                "models": {
                    "strong": "gpt-4-turbo",
                    "weak": "gpt-3.5-turbo"
                }
            },
            "copywriting": {
                "description": "Ghost copywriting for blogs and articles",
                "models": {
                    "strong": "claude-3-opus",
                    "weak": "claude-3-sonnet"
                }
            },
            "technical": {
                "description": "Technical documentation and code explanation",
                "models": {
                    "strong": "gpt-4-turbo",
                    "weak": "mistralai/Mixtral-8x7B-Instruct-v0.1"
                }
            }
        }
    }
    
    # Save to file
    # Handle the case where config_path has no directory component
    directory = os.path.dirname(config_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
