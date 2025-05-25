"""
Web UI for testing and visualizing intent-based routing.

This script provides a simple web interface for testing intent detection
and visualizing the routing decisions.

Usage:
  python intent_web_ui.py [--config CONFIG_PATH] [--port PORT]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Add the parent directory to sys.path to allow imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

import gradio as gr
import numpy as np
import pandas as pd
import yaml

from routellm.controller import Controller, ModelPair
from routellm.middleware import (
    IntentModelSelector, 
    IntentModelMapping,
    DomainIntentDetector,
    load_intent_config,
    create_example_config
)


def load_or_create_config(config_path: Optional[str] = None) -> IntentModelSelector:
    """
    Load intent configuration from a file or create a default one.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        An IntentModelSelector configured with the mappings
    """
    if config_path and os.path.exists(config_path):
        try:
            return load_intent_config(config_path)
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration instead.")
    
    # Create default configuration
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
    
    return IntentModelSelector(
        intent_mappings=intent_mappings,
        default_model_pair=ModelPair(strong="gpt-4", weak="gpt-3.5-turbo"),
        intent_detection_model="gpt-3.5-turbo"
    )


def create_web_ui(intent_selector: IntentModelSelector, config_path: Optional[str] = None):
    """
    Create a web UI for testing intent-based routing.
    
    Args:
        intent_selector: The IntentModelSelector to use
        config_path: Path to the configuration file
    """
    # Create domain intent detector for embedding-based intent detection
    domain_detector = DomainIntentDetector(
        intent_mappings=intent_selector.intent_mappings,
        cache_path=os.path.join(os.path.dirname(config_path) if config_path else ".", "intent_embeddings_cache.json")
    )
    
    # Load example prompts
    example_prompts = {
        "marketing": [
            "Create a Facebook ad campaign for our new accounting software targeting small businesses",
            "I need to market my digital product to accountants, what's the best approach?",
            "How can I improve the conversion rate for my SaaS landing page?",
            "Design a marketing strategy for our fintech app aimed at freelancers"
        ],
        "copywriting": [
            "Write a blog post about the latest design trends for local agencies",
            "I need ghost writing for our agency's design blog",
            "Create compelling copy for our email newsletter about interior design",
            "Write an article about sustainable design practices for our company blog"
        ],
        "technical": [
            "Explain how to implement a binary search tree in Python",
            "What's the difference between a mutex and a semaphore?",
            "Help me debug this React component that's not rendering properly",
            "How do I optimize database queries for a high-traffic website?"
        ]
    }
    
    # Add examples to domain detector
    for intent, prompts in example_prompts.items():
        domain_detector.add_examples(intent, prompts)
    
    def analyze_prompt(prompt: str):
        """Analyze a prompt and return the results."""
        # Get intent using LLM-based classifier
        llm_intent = intent_selector.detect_intent(prompt)
        
        # Get intent using embedding-based classifier
        embedding_intent = domain_detector.detect_intent(prompt)
        
        # Get confidence scores
        try:
            llm_confidence = intent_selector.analyze_intent_confidence(prompt)
            embedding_confidence = domain_detector.get_intent_confidence(prompt)
            
            # Format confidence scores for display
            llm_scores = llm_confidence.get("scores", {})
            llm_scores_str = "\n".join([f"{intent}: {score:.2f}" for intent, score in llm_scores.items()])
            
            embedding_scores_str = "\n".join([f"{intent}: {score:.2f}" for intent, score in embedding_confidence.items()])
            
            # Get model pair
            model_pair = intent_selector.get_model_pair(prompt)
            
            return (
                f"LLM-based intent: {llm_intent}",
                f"Embedding-based intent: {embedding_intent}",
                f"LLM confidence scores:\n{llm_scores_str}",
                f"Embedding confidence scores:\n{embedding_scores_str}",
                f"Selected model pair:\nStrong: {model_pair.strong}\nWeak: {model_pair.weak}"
            )
        except Exception as e:
            return (
                f"LLM-based intent: {llm_intent}",
                f"Embedding-based intent: {embedding_intent}",
                f"Error getting confidence scores: {str(e)}",
                "",
                f"Selected model pair:\nStrong: {intent_selector.get_model_pair(prompt).strong}\nWeak: {intent_selector.get_model_pair(prompt).weak}"
            )
    
    def add_example(prompt: str, intent: str):
        """Add an example prompt for a specific intent."""
        if not prompt or not intent:
            return "Please provide both a prompt and an intent."
        
        # Add example to domain detector
        domain_detector.add_examples(intent, [prompt])
        
        return f"Added example for intent '{intent}': {prompt[:50]}..."
    
    def export_examples():
        """Export examples to a file."""
        try:
            export_path = os.path.join(os.path.dirname(config_path) if config_path else ".", "intent_examples.json")
            domain_detector.export_examples(export_path)
            return f"Exported examples to {export_path}"
        except Exception as e:
            return f"Error exporting examples: {str(e)}"
    
    # Create the Gradio interface
    with gr.Blocks(title="Intent-Based Routing") as demo:
        gr.Markdown("# Intent-Based Routing Demo")
        gr.Markdown("Test intent detection and routing for different prompts.")
        
        with gr.Tab("Analyze Prompt"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Enter a prompt to analyze",
                        placeholder="e.g., Create a marketing plan for our new product",
                        lines=5
                    )
                    analyze_button = gr.Button("Analyze")
                
                with gr.Column():
                    llm_intent_output = gr.Textbox(label="LLM-Based Intent")
                    embedding_intent_output = gr.Textbox(label="Embedding-Based Intent")
                    llm_confidence_output = gr.Textbox(label="LLM Confidence Scores", lines=5)
                    embedding_confidence_output = gr.Textbox(label="Embedding Confidence Scores", lines=5)
                    model_pair_output = gr.Textbox(label="Selected Model Pair", lines=3)
            
            analyze_button.click(
                analyze_prompt,
                inputs=[prompt_input],
                outputs=[
                    llm_intent_output,
                    embedding_intent_output,
                    llm_confidence_output,
                    embedding_confidence_output,
                    model_pair_output
                ]
            )
        
        with gr.Tab("Add Examples"):
            with gr.Row():
                with gr.Column():
                    example_prompt = gr.Textbox(
                        label="Example Prompt",
                        placeholder="Enter an example prompt",
                        lines=5
                    )
                    example_intent = gr.Dropdown(
                        label="Intent",
                        choices=[mapping.intent for mapping in intent_selector.intent_mappings],
                        value=intent_selector.intent_mappings[0].intent if intent_selector.intent_mappings else None
                    )
                    add_example_button = gr.Button("Add Example")
                
                with gr.Column():
                    add_result = gr.Textbox(label="Result")
                    export_button = gr.Button("Export Examples")
                    export_result = gr.Textbox(label="Export Result")
            
            add_example_button.click(
                add_example,
                inputs=[example_prompt, example_intent],
                outputs=[add_result]
            )
            
            export_button.click(
                export_examples,
                inputs=[],
                outputs=[export_result]
            )
        
        with gr.Tab("Example Prompts"):
            gr.Markdown("### Marketing Examples")
            for i, prompt in enumerate(example_prompts["marketing"]):
                gr.Markdown(f"{i+1}. {prompt}")
            
            gr.Markdown("### Copywriting Examples")
            for i, prompt in enumerate(example_prompts["copywriting"]):
                gr.Markdown(f"{i+1}. {prompt}")
            
            gr.Markdown("### Technical Examples")
            for i, prompt in enumerate(example_prompts["technical"]):
                gr.Markdown(f"{i+1}. {prompt}")
    
    return demo


def main():
    """Main function to run the web UI."""
    parser = argparse.ArgumentParser(description="Web UI for intent-based routing")
    parser.add_argument("--config", type=str, help="Path to intent configuration file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
    parser.add_argument("--create-config", nargs='?', const="intent_config.yaml", 
                        help="Create an example configuration file (optionally specify path)")
    
    args = parser.parse_args()
    
    # Create example configuration if requested
    if args.create_config:
        config_path = args.create_config if isinstance(args.create_config, str) else "intent_config.yaml"
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
        create_example_config(config_path)
        print(f"Created example configuration at {config_path}")
        return
    
    # Load or create intent selector
    intent_selector = load_or_create_config(args.config)
    
    # Create and launch web UI
    demo = create_web_ui(intent_selector, args.config)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
