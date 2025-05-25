import unittest
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

# Add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

from routellm.controller import ModelPair
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector


class TestIntentModelSelector(unittest.TestCase):
    def setUp(self):
        # Define test model pairs
        self.marketing_pair = ModelPair(strong="gpt-4", weak="gpt-3.5-turbo")
        self.technical_pair = ModelPair(strong="claude-3-opus", weak="mistral-medium")
        self.default_pair = ModelPair(strong="default-strong", weak="default-weak")
        
        # Create intent mappings
        self.intent_mappings = [
            IntentModelMapping(
                intent="marketing",
                model_pair=self.marketing_pair,
                description="Marketing content"
            ),
            IntentModelMapping(
                intent="technical",
                model_pair=self.technical_pair,
                description="Technical content"
            )
        ]
        
        # Create the selector
        self.selector = IntentModelSelector(
            intent_mappings=self.intent_mappings,
            default_model_pair=self.default_pair
        )
        
        # Mock the detect_intent method
        self.selector.detect_intent = MagicMock()
    
    def test_get_model_pair_marketing(self):
        # Set up the mock to return "marketing"
        self.selector.detect_intent.return_value = "marketing"
        
        # Test with a marketing prompt
        prompt = "Create a Facebook ad for our product"
        model_pair = self.selector.get_model_pair(prompt)
        
        # Verify the correct model pair was returned
        self.assertEqual(model_pair, self.marketing_pair)
        self.selector.detect_intent.assert_called_once_with(prompt)
    
    def test_get_model_pair_technical(self):
        # Set up the mock to return "technical"
        self.selector.detect_intent.return_value = "technical"
        
        # Test with a technical prompt
        prompt = "Explain how to implement a binary search tree"
        model_pair = self.selector.get_model_pair(prompt)
        
        # Verify the correct model pair was returned
        self.assertEqual(model_pair, self.technical_pair)
        self.selector.detect_intent.assert_called_once_with(prompt)
    
    def test_get_model_pair_default(self):
        # Set up the mock to return an unknown intent
        self.selector.detect_intent.return_value = "unknown"
        
        # Test with an unknown intent
        prompt = "Something completely different"
        model_pair = self.selector.get_model_pair(prompt)
        
        # Verify the default model pair was returned
        self.assertEqual(model_pair, self.default_pair)
        self.selector.detect_intent.assert_called_once_with(prompt)


    def test_save_load_mappings(self):
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp:
            temp_path = temp.name
        
        try:
            # Save the mappings
            self.selector.save_mappings(temp_path)
            
            # Load the mappings
            loaded_selector = IntentModelSelector.load_mappings(temp_path)
            
            # Check that the loaded selector has the correct mappings
            self.assertEqual(len(loaded_selector.intent_mappings), len(self.intent_mappings))
            
            # Check that the intents are the same
            loaded_intents = [m.intent for m in loaded_selector.intent_mappings]
            original_intents = [m.intent for m in self.intent_mappings]
            self.assertEqual(set(loaded_intents), set(original_intents))
            
            # Check that the default model pair is the same
            self.assertEqual(loaded_selector.default_model_pair.strong, self.default_pair.strong)
            self.assertEqual(loaded_selector.default_model_pair.weak, self.default_pair.weak)
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Import sys and os at the top of the file to ensure they're available
import sys
import os

# Add the repository root to the Python path at import time
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

if __name__ == "__main__":
    # Print helpful information
    print(f"Python path: {sys.path[0]}")
    print(f"Current directory: {os.getcwd()}")
    
    # Try to import the module to verify it works
    try:
        from routellm.controller import ModelPair
        from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector
        print("Successfully imported routellm modules")
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Available files in middleware directory: {os.listdir(os.path.join(repo_root, 'routellm', 'middleware'))}")
    
    # Run the tests
    unittest.main()
