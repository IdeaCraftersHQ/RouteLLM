"""Utilities for loading and configuring causal LLM models.

Handles model and tokenizer loading with support for fine-tuned
router models and specialized token handling.
"""

import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from routellm.routers.causal_llm.configs import (
    PROMPT_FORMAT_CONFIGS,
    ModelTypeEnum,
    RouterModelConfig,
)
from routellm.routers.causal_llm.prompt_format import PromptFormat


def load_model_config(yaml_path: str):
    """Load router model configuration from YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to YAML configuration file.

    Returns
    -------
    RouterModelConfig
        Parsed model configuration.
    """
    with open(yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return RouterModelConfig(**yaml_data)


def load_prompt_format(model_id):
    """Load prompt format template for a model.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.

    Returns
    -------
    PromptFormat
        Prompt formatting template configured for generation.
    """
    prompt_format_dict = PROMPT_FORMAT_CONFIGS[model_id]
    return PromptFormat(**prompt_format_dict, is_generation=True)


def get_model(config: RouterModelConfig, model_ckpt: str, pad_token_id: int = 2):
    """Load causal LLM model from checkpoint.

    Parameters
    ----------
    config : RouterModelConfig
        Model configuration.
    model_ckpt : str
        Path to or ID of model checkpoint.
    pad_token_id : int, optional
        Padding token ID (default 2).

    Returns
    -------
    AutoModelForCausalLM
        Loaded model in bfloat16 precision.

    Raises
    ------
    NotImplementedError
        If model type is not causal.
    """
    if config.model_type == ModelTypeEnum.CAUSAL:
        return AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation=(
                "flash_attention_2" if config.flash_attention_2 else None
            ),
            attention_dropout=config.attention_dropout,
            token=os.getenv("LLAMA2_HF_TOKEN"),
        )
    else:
        raise NotImplementedError(
            f"ModelType {config.model_type} is not implemented yet!"
        )


def get_tokenizer(
    model_id, special_tokens=None, truncation_side="left", padding_side="left"
):
    """Load and configure tokenizer with special tokens.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    special_tokens : list[str], optional
        Special tokens to add (default None).
    truncation_side : str, optional
        Truncation side: "left" or "right" (default "left").
    padding_side : str, optional
        Padding side: "left" or "right" (default "left").

    Returns
    -------
    AutoTokenizer
        Configured tokenizer with special tokens added.
    """
    # Context for legacy=True: https://github.com/huggingface/transformers/issues/25176
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        legacy=True,
        truncation_side=truncation_side,
        padding_side=padding_side,
        token=os.getenv("LLAMA2_HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer


def to_openai_api_messages(system_message, classifier_message, messages):
    """Convert conversation to OpenAI chat completion format.

    Parameters
    ----------
    system_message : str
        System message to prepend.
    classifier_message : str
        Template for classifying user messages (with {question} placeholder).
    messages : list[str]
        Conversation messages alternating user/assistant.

    Returns
    -------
    list[dict]
        OpenAI-style message list with role and content keys.
    """

    ret = [{"role": "system", "content": system_message}]
    for i, turn in enumerate(messages):
        if i % 2 == 0:
            ret.append(
                {"role": "user", "content": classifier_message.format(question=turn)}
            )
        else:
            ret.append({"role": "assistant", "content": turn})
    return ret
