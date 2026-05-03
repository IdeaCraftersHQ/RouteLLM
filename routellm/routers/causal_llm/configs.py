"""Configuration classes for causal LLM-based routers.

Defines model types, prompt formatting, and router model configurations
for causal language model routing.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict

PROMPT_FORMAT_CONFIGS = {
    "meta-llama/Meta-Llama-3-8B": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "trailing_assistant": "",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>",
        "system_in_user": False,
        "bos": "<|begin_of_text|>",
        "default_system_message": "",
    },
}


class ModelTypeEnum(str, Enum):
    """Enumeration of supported causal model types."""

    CAUSAL = "causal"


class RouterModelConfig(BaseModel):
    """Configuration for causal LLM router model.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID or path (e.g., "meta-llama/Meta-Llama-3-8B").
    model_type : ModelTypeEnum
        Type of model ("causal" for causal language models).
    num_outputs : int
        Number of output classes for routing.
    special_tokens : list[str], optional
        Special tokens indicating difficulty levels (default empty).
    flash_attention_2 : bool, optional
        Enable Flash Attention 2 optimization (default False).
    attention_dropout : float, optional
        Dropout rate for attention (default 0.0).
    """

    model_id: str
    model_type: ModelTypeEnum
    num_outputs: int

    # output special tokens (e.g. [[1]], [[2]], etc.) for CAUSAL models
    special_tokens: List[str] = []
    flash_attention_2: bool = False
    attention_dropout: float = 0.0

    model_config = ConfigDict(protected_namespaces=())
