"""
Example of using the IntentModelSelector middleware with the Controller.

This script demonstrates how to use intent-based routing to dynamically select
appropriate model pairs based on the detected intent of user prompts.

Usage:
  python intent_based_routing.py [--test] [--analyze PROMPT] [--save CONFIG] [--load CONFIG]
"""

import sys
import os
import argparse
import json
from typing import List, Dict, Optional

# Add the parent directory to sys.path to allow imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)
print(f"Added to Python path: {repo_root}")

# Verify the path is correct
try:
    from routellm.controller import Controller, ModelPair
    print("Successfully imported routellm modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Available files in parent directory:", os.listdir(os.path.join(os.path.dirname(__file__), '..')))

# Check for required dependencies
required_packages = ["pandas", "litellm"]
optional_packages = ["matplotlib"]
missing_required = []
missing_optional = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_required.append(package)

for package in optional_packages:
    try:
        __import__(package)
    except ImportError:
        missing_optional.append(package)

if missing_required:
    print(f"Error: Missing required packages: {', '.join(missing_required)}")
    print("Please install them in a virtual environment:")
    print("\npython3 -m venv venv")
    print("source venv/bin/activate")
    print(f"pip install {' '.join(required_packages)}\n")
    print("Then run this script again from the activated environment.")
    sys.exit(1)

if missing_optional:
    print(f"Warning: Missing optional packages: {', '.join(missing_optional)}")
    print("For full functionality, install these packages:")
    print(f"pip install {' '.join(optional_packages)}")
    print("Continuing without visualization capabilities...\n")
    has_visualization = False
else:
    has_visualization = True
    import matplotlib.pyplot as plt
    
import pandas as pd

from routellm.controller import Controller, ModelPair
from routellm.middleware import IntentModelMapping, IntentModelSelector

# Define model pairs for different intents
marketing_model_pair = ModelPair(
    strong="gpt-4-turbo",
    weak="gpt-3.5-turbo"
)

copywriting_model_pair = ModelPair(
    strong="claude-3-opus",
    weak="claude-3-sonnet"
)

technical_model_pair = ModelPair(
    strong="gpt-4-turbo",
    weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
)

# Define intent mappings
intent_mappings = [
    IntentModelMapping(
        intent="marketing",
        model_pair=marketing_model_pair,
        description="Marketing content creation for digital products"
    ),
    IntentModelMapping(
        intent="copywriting",
        model_pair=copywriting_model_pair,
        description="Ghost copywriting for blogs and articles"
    ),
    IntentModelMapping(
        intent="technical",
        model_pair=technical_model_pair,
        description="Technical documentation and code explanation"
    )
]

# Create the intent model selector middleware
intent_selector = IntentModelSelector(
    intent_mappings=intent_mappings,
    default_model_pair=ModelPair(strong="gpt-4", weak="gpt-3.5-turbo"),
    intent_detection_model="gpt-3.5-turbo"
)

def setup_default_intent_mappings():
    """Set up default intent mappings for testing."""
    # Define model pairs for different intents
    marketing_model_pair = ModelPair(
        strong="gpt-4-turbo",
        weak="gpt-3.5-turbo"
    )

    copywriting_model_pair = ModelPair(
        strong="claude-3-opus",
        weak="claude-3-sonnet"
    )

    technical_model_pair = ModelPair(
        strong="gpt-4-turbo",
        weak="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    # Define intent mappings
    intent_mappings = [
        IntentModelMapping(
            intent="marketing",
            model_pair=marketing_model_pair,
            description="Marketing content creation for digital products"
        ),
        IntentModelMapping(
            intent="copywriting",
            model_pair=copywriting_model_pair,
            description="Ghost copywriting for blogs and articles"
        ),
        IntentModelMapping(
            intent="technical",
            model_pair=technical_model_pair,
            description="Technical documentation and code explanation"
        )
    ]
    
    return intent_mappings, ModelPair(strong="gpt-4", weak="gpt-3.5-turbo")

def get_test_prompts():
    """Get test prompts for different intents."""
    marketing_prompts = [
        "Create a Facebook ad campaign for our new accounting software targeting small businesses",
        "I need to market my digital product to accountants, what's the best approach?",
        "How can I improve the conversion rate for my SaaS landing page?",
        "Design a marketing strategy for our fintech app aimed at freelancers"
    ]

    copywriting_prompts = [
        "Write a blog post about the latest design trends for local agencies",
        "I need ghost writing for our agency's design blog",
        "Create compelling copy for our email newsletter about interior design",
        "Write an article about sustainable design practices for our company blog"
    ]

    technical_prompts = [
        "Explain how to implement a binary search tree in Python",
        "What's the difference between a mutex and a semaphore?",
        "Help me debug this React component that's not rendering properly",
        "How do I optimize database queries for a high-traffic website?"
    ]
    
    return {
        "marketing": marketing_prompts,
        "copywriting": copywriting_prompts,
        "technical": technical_prompts
    }

def analyze_prompt_confidence(prompt, visualize=False):
    """
    Analyze and print confidence scores for a prompt.
    
    Args:
        prompt: The prompt to analyze
        visualize: Whether to create a visualization of the scores
    
    Returns:
        The analysis result dictionary
    """
    print(f"\nAnalyzing confidence for: \"{prompt[:50]}...\"")
    try:
        result = intent_selector.analyze_intent_confidence(prompt)
        if "error" in result:
            print(f"Error: {result['error']}")
            return result
            
        print(f"Analysis: {result.get('analysis', 'No analysis provided')}")
        print("Confidence scores:")
        
        # Sort scores by value for better readability
        scores = result.get("scores", {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for intent, score in sorted_scores:
            print(f"  - {intent}: {score}")
        print(f"Best match: {result.get('best_match', 'general')}")
        
        # Create visualization if requested and available
        if visualize and has_visualization and scores:
            visualize_confidence_scores(scores, result.get("best_match", "general"))
            
        return result
    except Exception as e:
        print(f"Error during confidence analysis: {e}")
        return {"error": str(e)}

def visualize_confidence_scores(scores, best_match):
    """
    Create a bar chart visualization of confidence scores.
    
    Args:
        scores: Dictionary of intent:score pairs
        best_match: The best matching intent
    """
    # Sort scores for better visualization
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    intents = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create colors list with the best match highlighted
    colors = ['#1f77b4' if intent != best_match else '#ff7f0e' for intent in intents]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(intents, values, color=colors)
    
    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                 ha='left', va='center')
    
    plt.xlabel('Confidence Score')
    plt.title('Intent Classification Confidence Scores')
    plt.xlim(0, 105)  # Leave some space for the text labels
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f0e', label='Best Match'),
        Patch(facecolor='#1f77b4', label='Other Intents')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the intent-based routing example."""
    parser = argparse.ArgumentParser(description="Intent-based routing example for RouteLLM")
    parser.add_argument("--test", action="store_true", help="Run the test suite with example prompts")
    parser.add_argument("--analyze", type=str, help="Analyze the intent of a specific prompt")
    parser.add_argument("--visualize", action="store_true", help="Visualize confidence scores (requires matplotlib)")
    parser.add_argument("--save", type=str, help="Save intent mappings to a config file")
    parser.add_argument("--load", type=str, help="Load intent mappings from a config file")
    parser.add_argument("--router", type=str, default="random", help="Router to use (default: random)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Routing threshold (default: 0.7)")
    parser.add_argument("--web", action="store_true", help="Launch web UI for testing")
    parser.add_argument("--port", type=int, default=7860, help="Port for web UI")
    parser.add_argument("--create-config", nargs='?', const="intent_config.yaml", 
                        help="Create an example YAML config file (optionally specify path)")
    
    args = parser.parse_args()
    
    # Create example config if requested
    if args.create_config:
        try:
            from routellm.middleware import create_example_config
            # If args.create_config is True (boolean flag), use a default filename
            config_path = args.create_config if isinstance(args.create_config, str) else "intent_config.yaml"
            create_example_config(config_path)
            print(f"Created example config at {config_path}")
            return
        except Exception as e:
            print(f"Error creating example config: {e}")
            return
    
    # Set up intent mappings
    if args.load:
        try:
            print(f"Loading intent mappings from {args.load}")
            # Check if it's a YAML file
            if args.load.endswith(('.yaml', '.yml')):
                from routellm.middleware import load_intent_config
                intent_selector = load_intent_config(args.load)
            else:
                # Assume it's a JSON file
                intent_selector = IntentModelSelector.load_mappings(args.load)
            print(f"Loaded {len(intent_selector.intent_mappings)} intent mappings")
        except Exception as e:
            print(f"Error loading mappings: {e}")
            print("Using default mappings instead")
            intent_mappings, default_model_pair = setup_default_intent_mappings()
            intent_selector = IntentModelSelector(
                intent_mappings=intent_mappings,
                default_model_pair=default_model_pair,
                intent_detection_model="gpt-3.5-turbo"
            )
    else:
        intent_mappings, default_model_pair = setup_default_intent_mappings()
        intent_selector = IntentModelSelector(
            intent_mappings=intent_mappings,
            default_model_pair=default_model_pair,
            intent_detection_model="gpt-3.5-turbo"
        )
    
    # Save mappings if requested
    if args.save:
        try:
            intent_selector.save_mappings(args.save)
            print(f"Saved intent mappings to {args.save}")
        except Exception as e:
            print(f"Error saving mappings: {e}")
    
    # Initialize controller if needed
    controller_available = False
    if args.test or args.analyze:
        try:
            # Check if torch is available before trying to initialize the controller
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False
                print("Warning: PyTorch is not installed. Some routers may not work.")
                print("To install PyTorch, run: pip install torch")
            
            if torch_available or args.router == "random":
                controller = Controller(
                    routers=[args.router],
                    strong_model=default_model_pair.strong,
                    weak_model=default_model_pair.weak,
                    middleware=[intent_selector],
                )
                controller_available = True
            else:
                print(f"Router '{args.router}' may require PyTorch. Falling back to 'random' router.")
                controller = Controller(
                    routers=["random"],
                    strong_model=default_model_pair.strong,
                    weak_model=default_model_pair.weak,
                    middleware=[intent_selector],
                )
                controller_available = True
        except Exception as e:
            print(f"Warning: Could not initialize controller: {e}")
            print("Will only test intent detection without routing.")
    
    # Analyze a specific prompt if requested
    if args.analyze:
        result = analyze_prompt_confidence(args.analyze, args.visualize)
        
        if controller_available:
            try:
                # Get the model pair that would be selected
                model_pair = intent_selector.get_model_pair(args.analyze)
                print(f"\nSelected model pair: Strong={model_pair.strong}, Weak={model_pair.weak}")
                
                # Only attempt completion if API keys are available
                try:
                    response = controller.completion(
                        router=args.router,
                        threshold=args.threshold,
                        messages=[{"role": "user", "content": args.analyze}]
                    )
                    print(f"Routed to: {response.model}")
                    print(f"Response preview: {response.choices[0].message.content[:100]}...")
                except Exception as e:
                    print(f"Skipping actual completion due to: {e}")
            except Exception as e:
                print(f"Error during routing: {e}")
    
    # Run the test suite if requested
    if args.test:
        run_test_suite(intent_selector, controller_available, controller, args.router, args.threshold, args.visualize)
    
    # Launch web UI if requested
    if args.web:
        try:
            import gradio as gr
            from routellm.examples.intent_web_ui import create_web_ui
            print(f"Launching web UI on port {args.port}...")
            demo = create_web_ui(intent_selector, args.load)
            demo.launch(server_port=args.port)
        except ImportError:
            print("Error: gradio is not installed. Please install it with:")
            print("pip install gradio")
            print("Then run this script again.")

def run_test_suite(intent_selector, controller_available, controller=None, router="random", threshold=0.7, visualize=False):
    """Run the test suite with example prompts."""
    test_prompts = get_test_prompts()
    
    # Test the intent detection
    print("Testing intent detection...")
    for category, prompts in test_prompts.items():
        print(f"\n{category.capitalize()} prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. \"{prompt[:50]}...\" -> {intent_selector.detect_intent(prompt)}")

    # Only test the controller if it's available
    if controller_available and controller:
        # The controller will automatically select the appropriate model pair based on the detected intent
        print("\nTesting controller with intent-based routing...")
        try:
            # Test with one prompt from each category
            for category, prompts in test_prompts.items():
                test_prompt = prompts[0]
                print(f"\nRouting {category.capitalize()} prompt: \"{test_prompt[:50]}...\"")
                
                # Get the model pair that would be selected
                model_pair = intent_selector.get_model_pair(test_prompt)
                print(f"Selected model pair: Strong={model_pair.strong}, Weak={model_pair.weak}")
                
                # Only attempt completion if API keys are available
                try:
                    response = controller.completion(
                        router=router,
                        threshold=threshold,
                        messages=[{"role": "user", "content": test_prompt}]
                    )
                    print(f"Routed to: {response.model}")
                    print(f"Response preview: {response.choices[0].message.content[:100]}...")
                except Exception as e:
                    print(f"Skipping actual completion due to: {e}")
        except Exception as e:
            print(f"Error during controller testing: {e}")
    else:
        print("\nSkipping controller test as it's not available.")

    # Demonstrate confidence analysis with one example from each category
    print("\n" + "="*50)
    print("DEMONSTRATING CONFIDENCE ANALYSIS")
    print("="*50)

    try:
        # Try one prompt from each category
        for category, prompts in test_prompts.items():
            analyze_prompt_confidence(prompts[1], visualize)
        
        # Try an ambiguous prompt
        analyze_prompt_confidence("I need help with my project that involves both design and coding", visualize)
    except Exception as e:
        print(f"Error during confidence analysis demonstration: {e}")

if __name__ == "__main__":
    main()
