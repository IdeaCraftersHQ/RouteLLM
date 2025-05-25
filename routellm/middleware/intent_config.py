"""
Configuration utilities for intent-based routing.

This module provides utilities for loading and saving intent configurations
from YAML files, making it easy to define and manage intent mappings.
"""

import os
import yaml
from typing import Dict, List, Optional, Any

from routellm.controller import ModelPair
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector


def load_intent_config(config_path: str) -> IntentModelSelector:
    """
    Load intent configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        An IntentModelSelector configured with the mappings from the file
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse default model pair
    default_model_pair = ModelPair(
        strong=config.get("default_models", {}).get("strong", "gpt-4"),
        weak=config.get("default_models", {}).get("weak", "gpt-3.5-turbo")
    )
    
    # Parse intent mappings
    intent_mappings = []
    for intent_name, intent_config in config.get("intents", {}).items():
        model_pair = ModelPair(
            strong=intent_config.get("models", {}).get("strong", default_model_pair.strong),
            weak=intent_config.get("models", {}).get("weak", default_model_pair.weak)
        )
        
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
        intent_detection_model=config.get("intent_detection_model", "gpt-3.5-turbo")
    )


def save_intent_config(selector: IntentModelSelector, config_path: str) -> None:
    """
    Save intent configuration to a YAML file.
    
    Args:
        selector: The IntentModelSelector to save
        config_path: Path to save the YAML configuration file
    """
    # Create config dictionary
    config = {
        "intent_detection_model": selector.intent_detection_model,
        "default_models": {
            "strong": selector.default_model_pair.strong,
            "weak": selector.default_model_pair.weak
        },
        "intents": {}
    }
    
    # Add intent mappings
    for mapping in selector.intent_mappings:
        config["intents"][mapping.intent] = {
            "description": mapping.description,
            "models": {
                "strong": mapping.model_pair.strong,
                "weak": mapping.model_pair.weak
            }
        }
    
    # Save to file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_example_config(config_path: str) -> None:
    """
    Create an example intent configuration file.
    
    Args:
        config_path: Path to save the example configuration
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
