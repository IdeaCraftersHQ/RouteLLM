"""Intent-based model selection middleware.

Provides middleware for selecting model pairs based on detected user intent.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

from routellm.types import ModelPair


@dataclass
class IntentModelMapping:
    """Maps intents to specific model pairs.

    Attributes
    ----------
    intent : str
        Intent label (e.g., "math", "code", "general").
    model_pair : ModelPair
        Model pair for this intent.
    description : str, optional
        Human-readable description of intent mapping (default "").
    """

    intent: str
    model_pair: ModelPair
    description: str = ""


class IntentModelSelector:
    """Middleware for intent-based model pair selection.

    Detects user intent and routes to appropriate model pair.
    """
    
    def __init__(
        self,
        intent_mappings: List[IntentModelMapping],
        default_model_pair: ModelPair,
        intent_detection_model: str = "gpt-3.5-turbo",
    ):
        """Initialize the intent-based model selector.

        Parameters
        ----------
        intent_mappings : list[IntentModelMapping]
            List of mappings from intents to model pairs.
        default_model_pair : ModelPair
            Default model pair to use when no intent matches.
        intent_detection_model : str, optional
            Model to use for intent detection (default "gpt-3.5-turbo").
        """
        self.intent_mappings = intent_mappings
        self.default_model_pair = default_model_pair
        self.intent_detection_model = intent_detection_model
        self.intent_cache = {}  # Cache detected intents
        
        # Create a lookup dictionary for faster access
        self.intent_lookup = {mapping.intent: mapping.model_pair for mapping in intent_mappings}
    
    def detect_intent(self, prompt: str) -> str:
        """Detect the intent of a prompt using an LLM.

        Caches results to avoid repeated LLM calls for identical prompts.

        Parameters
        ----------
        prompt : str
            The user prompt to analyze.

        Returns
        -------
        str
            The detected intent as a string. Returns "general" if detection
            fails.
        """
        # Check cache first
        if prompt in self.intent_cache:
            return self.intent_cache[prompt]
        
        # Get available intents and their descriptions
        intents = [mapping.intent for mapping in self.intent_mappings]
        intent_descriptions = {
            mapping.intent: mapping.description for mapping in self.intent_mappings
        }
        
        # Create a formatted list of intents with descriptions for the prompt
        intent_options = "\n".join([
            f"- {intent}: {intent_descriptions[intent]}"
            for intent in intents
        ])
        
        # Create the classification prompt with more detailed instructions
        classification_prompt = f"""
You are an expert intent classifier for a language model router system. Your task is to analyze the following user message and determine which category it best fits into.

Available categories:
{intent_options}

Guidelines for classification:
1. Consider the primary purpose of the message, not just keywords
2. Look for indicators of the user's underlying goal or need
3. Consider the domain expertise required to answer effectively
4. If multiple categories could apply, choose the one that would require the most specialized knowledge

User message: "{prompt}"

First, analyze the message and identify key indicators of intent.
Then, match these indicators to the most appropriate category.

Respond with ONLY the category name in lowercase, nothing else. If none of the categories fit well, respond with "general".
"""
        
        try:
            try:
                # Import here to avoid circular imports
                from litellm import completion
            except ImportError:
                print("Error: litellm is not installed.")
                print("This middleware requires litellm. Please install it in a virtual environment:")
                print("\npython3 -m venv venv")
                print("source venv/bin/activate")
                print("pip install litellm\n")
                print("Then run your script again from the activated environment.")
                raise
            
            # Call the LLM to classify the intent
            response = completion(
                model=self.intent_detection_model,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            
            # Extract the detected intent from the response
            detected_intent = response.choices[0].message.content.strip().lower()
            
            # Validate that the detected intent is in our list or "general"
            if detected_intent not in intents and detected_intent != "general":
                detected_intent = "general"
                
        except Exception as e:
            # If there's an error, fall back to "general"
            print(f"Error detecting intent: {e}")
            detected_intent = "general"
        
        # Cache the result
        self.intent_cache[prompt] = detected_intent
        return detected_intent
    
    def analyze_intent_confidence(self, prompt: str) -> dict:
        """Analyze the confidence of intent classification for a prompt.

        Uses a more detailed LLM prompt to get confidence scores
        for each possible intent category.

        Parameters
        ----------
        prompt : str
            The user prompt to analyze.

        Returns
        -------
        dict
            Dictionary with keys "analysis", "scores", and "best_match".
            Returns {"error": str, "best_match": "general"} on failure.
        """
        # Get available intents and their descriptions
        intents = [mapping.intent for mapping in self.intent_mappings]
        intent_descriptions = {
            mapping.intent: mapping.description for mapping in self.intent_mappings
        }
        
        # Create a formatted list of intents with descriptions
        intent_options = "\n".join([
            f"- {intent}: {intent_descriptions[intent]}"
            for intent in intents
        ])
        
        # Create the analysis prompt
        analysis_prompt = f"""
You are an expert intent classifier for a language model router system. Analyze the following user message and determine how well it fits into each available category.

Available categories:
{intent_options}

User message: "{prompt}"

For each category, provide a confidence score from 0-100 where:
- 0-20: Very poor fit
- 21-40: Poor fit
- 41-60: Moderate fit
- 61-80: Good fit
- 81-100: Excellent fit

Respond in JSON format like this:
{{
  "analysis": "Brief explanation of your reasoning",
  "scores": {{
    "category1": score,
    "category2": score,
    ...
  }},
  "best_match": "category_name"
}}
"""
        
        try:
            try:
                from litellm import completion
            except ImportError:
                print("Error: litellm is not installed.")
                return {"error": "litellm not installed", "best_match": "general"}
            
            # Call the LLM for analysis
            response = completion(
                model=self.intent_detection_model,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            # Extract and parse the JSON response
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response", "best_match": "general"}
                
        except Exception as e:
            print(f"Error analyzing intent confidence: {e}")
            return {"error": str(e), "best_match": "general"}

    def get_model_pair(self, prompt: str) -> ModelPair:
        """Get the appropriate model pair based on the detected intent.

        Parameters
        ----------
        prompt : str
            The user prompt to analyze.

        Returns
        -------
        ModelPair
            Model pair appropriate for the detected intent. Returns
            default_model_pair if intent not found.
        """
        intent = self.detect_intent(prompt)
        return self.intent_lookup.get(intent, self.default_model_pair)
        
    def get_available_intents(self) -> List[str]:
        """Get a list of all available intents.

        Returns
        -------
        list[str]
            List of intent names configured in this selector.
        """
        return [mapping.intent for mapping in self.intent_mappings]
    
    def save_mappings(self, filepath: str) -> None:
        """Save intent mappings to a JSON file.

        Serializes all intent mappings and configuration to JSON format.

        Parameters
        ----------
        filepath : str
            Path to save the mappings to.
        """
        # Convert mappings to a serializable format
        serializable_mappings = []
        for mapping in self.intent_mappings:
            mapping_dict = {
                "intent": mapping.intent,
                "description": mapping.description,
                "model_pair": {
                    "strong": mapping.model_pair.strong,
                    "weak": mapping.model_pair.weak
                }
            }
            serializable_mappings.append(mapping_dict)
            
        # Add default model pair
        config = {
            "intent_mappings": serializable_mappings,
            "default_model_pair": {
                "strong": self.default_model_pair.strong,
                "weak": self.default_model_pair.weak
            },
            "intent_detection_model": self.intent_detection_model
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_mappings(cls, filepath: str) -> 'IntentModelSelector':
        """Load intent mappings from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the mappings file.

        Returns
        -------
        IntentModelSelector
            New instance with loaded mappings and configuration.

        Raises
        ------
        FileNotFoundError
            If mappings file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mappings file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        # Convert to IntentModelMapping objects
        intent_mappings = []
        for mapping_dict in config.get("intent_mappings", []):
            model_pair = ModelPair(
                strong=mapping_dict["model_pair"]["strong"],
                weak=mapping_dict["model_pair"]["weak"]
            )
            mapping = IntentModelMapping(
                intent=mapping_dict["intent"],
                description=mapping_dict["description"],
                model_pair=model_pair
            )
            intent_mappings.append(mapping)
            
        # Create default model pair
        default_model_pair = ModelPair(
            strong=config["default_model_pair"]["strong"],
            weak=config["default_model_pair"]["weak"]
        )
        
        # Create and return the selector
        return cls(
            intent_mappings=intent_mappings,
            default_model_pair=default_model_pair,
            intent_detection_model=config.get("intent_detection_model", "gpt-3.5-turbo")
        )
