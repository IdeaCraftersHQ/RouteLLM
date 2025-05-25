"""
Domain-specific intent detector for RouteLLM.

This module provides a specialized intent detector that can be fine-tuned
for specific domains or use cases.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from routellm.middleware.intent_model_selector import IntentModelMapping


class DomainIntentDetector:
    """
    Domain-specific intent detector that uses embeddings to classify intents.
    
    This detector can be fine-tuned with domain-specific examples to improve
    intent detection accuracy for specific use cases.
    """
    
    def __init__(
        self,
        intent_mappings: List[IntentModelMapping],
        embedding_model: str = "text-embedding-ada-002",
        examples_per_intent: int = 5,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize the domain-specific intent detector.
        
        Args:
            intent_mappings: List of intent mappings
            embedding_model: Model to use for generating embeddings
            examples_per_intent: Number of examples to use per intent for fine-tuning
            cache_path: Path to cache embeddings
        """
        self.intent_mappings = intent_mappings
        self.embedding_model = embedding_model
        self.examples_per_intent = examples_per_intent
        self.cache_path = cache_path
        self.intent_examples = {}
        self.intent_embeddings = {}
        self.embedding_cache = {}
        
        # Load cache if available
        if cache_path and os.path.exists(cache_path):
            self._load_cache()
    
    def add_examples(self, intent: str, examples: List[str]) -> None:
        """
        Add examples for a specific intent.
        
        Args:
            intent: The intent to add examples for
            examples: List of example prompts for the intent
        """
        if intent not in self.intent_examples:
            self.intent_examples[intent] = []
        
        self.intent_examples[intent].extend(examples)
        
        # Update embeddings for this intent
        self._update_intent_embeddings(intent)
        
        # Save cache if path is specified
        if self.cache_path:
            self._save_cache()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using the specified embedding model.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding as numpy array
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Import here to avoid circular imports
            from litellm import embedding
            
            # Get embedding
            response = embedding(
                model=self.embedding_model,
                input=text
            )
            
            # Extract embedding
            embedding_vector = np.array(response['data'][0]['embedding'])
            
            # Cache embedding
            self.embedding_cache[text] = embedding_vector
            
            return embedding_vector
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Default size for text-embedding-ada-002
    
    def _update_intent_embeddings(self, intent: str) -> None:
        """
        Update embeddings for a specific intent.
        
        Args:
            intent: The intent to update embeddings for
        """
        if intent not in self.intent_examples or not self.intent_examples[intent]:
            return
        
        # Get embeddings for examples
        examples = self.intent_examples[intent]
        embeddings = [self._get_embedding(example) for example in examples]
        
        # Store embeddings
        self.intent_embeddings[intent] = embeddings
    
    def _save_cache(self) -> None:
        """Save embeddings cache to file."""
        if not self.cache_path:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_cache = {
            text: embedding.tolist() 
            for text, embedding in self.embedding_cache.items()
        }
        
        # Save to file
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(serializable_cache, f)
    
    def _load_cache(self) -> None:
        """Load embeddings cache from file."""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        
        try:
            with open(self.cache_path, 'r') as f:
                serializable_cache = json.load(f)
            
            # Convert lists back to numpy arrays
            self.embedding_cache = {
                text: np.array(embedding) 
                for text, embedding in serializable_cache.items()
            }
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def detect_intent(self, prompt: str) -> str:
        """
        Detect the intent of a prompt using embeddings similarity.
        
        Args:
            prompt: The prompt to detect intent for
            
        Returns:
            The detected intent
        """
        # If no intents have examples, return "general"
        if not self.intent_embeddings:
            return "general"
        
        # Get embedding for prompt
        prompt_embedding = self._get_embedding(prompt)
        
        # Calculate similarity with each intent
        intent_scores = {}
        for intent, embeddings in self.intent_embeddings.items():
            if not embeddings:
                continue
                
            # Calculate similarity with each example
            similarities = [
                cosine_similarity(
                    prompt_embedding.reshape(1, -1), 
                    example_embedding.reshape(1, -1)
                )[0][0]
                for example_embedding in embeddings
            ]
            
            # Use average similarity as score
            intent_scores[intent] = sum(similarities) / len(similarities)
        
        # Return intent with highest score, or "general" if no scores
        if not intent_scores:
            return "general"
            
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def get_intent_confidence(self, prompt: str) -> Dict[str, float]:
        """
        Get confidence scores for each intent.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dictionary mapping intents to confidence scores
        """
        # If no intents have examples, return empty dict
        if not self.intent_embeddings:
            return {}
        
        # Get embedding for prompt
        prompt_embedding = self._get_embedding(prompt)
        
        # Calculate similarity with each intent
        intent_scores = {}
        for intent, embeddings in self.intent_embeddings.items():
            if not embeddings:
                continue
                
            # Calculate similarity with each example
            similarities = [
                cosine_similarity(
                    prompt_embedding.reshape(1, -1), 
                    example_embedding.reshape(1, -1)
                )[0][0]
                for example_embedding in embeddings
            ]
            
            # Use average similarity as score
            intent_scores[intent] = sum(similarities) / len(similarities)
        
        # Normalize scores to sum to 1
        total_score = sum(intent_scores.values())
        if total_score > 0:
            intent_scores = {
                intent: score / total_score 
                for intent, score in intent_scores.items()
            }
        
        return intent_scores
    
    def export_examples(self, filepath: str) -> None:
        """
        Export intent examples to a JSON file.
        
        Args:
            filepath: Path to save examples to
        """
        with open(filepath, 'w') as f:
            json.dump(self.intent_examples, f, indent=2)
    
    def import_examples(self, filepath: str) -> None:
        """
        Import intent examples from a JSON file.
        
        Args:
            filepath: Path to load examples from
        """
        with open(filepath, 'r') as f:
            examples = json.load(f)
        
        for intent, intent_examples in examples.items():
            self.add_examples(intent, intent_examples)
