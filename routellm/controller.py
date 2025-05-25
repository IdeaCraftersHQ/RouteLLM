from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Protocol

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


class Middleware(Protocol):
    """Protocol for middleware that can modify controller behavior."""
    
    def get_model_pair(self, prompt: str) -> 'ModelPair':
        """Get a model pair based on the prompt."""
        ...


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
        middleware: Optional[List[Middleware]] = None,
    ):
        self.default_model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        self.middleware = middleware or []

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_model_pair_for_prompt(self, prompt: str) -> ModelPair:
        """
        Apply middleware to get the appropriate model pair for a prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            A ModelPair with strong and weak models appropriate for the prompt
        """
        model_pair = self.default_model_pair
        
        # Apply middleware in order
        for middleware in self.middleware:
            model_pair = middleware.get_model_pair(prompt)
            
        return model_pair

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        
        # Get the appropriate model pair based on the prompt
        model_pair = self._get_model_pair_for_prompt(prompt)
        
        # Route using the selected model pair
        routed_model = self.routers[router].route(prompt, threshold, model_pair)

        self.model_counts[router][routed_model] += 1

        return routed_model

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        
        # Define a function that applies middleware and then calculates win rate
        def calculate_with_middleware(prompt):
            model_pair = self._get_model_pair_for_prompt(prompt)
            return router_instance.calculate_strong_win_rate(prompt, model_pair=model_pair)
        
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(calculate_with_middleware)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(calculate_with_middleware)
        else:
            return prompts.parallel_apply(calculate_with_middleware)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)
        
        # Get the appropriate model pair based on the prompt
        model_pair = self._get_model_pair_for_prompt(prompt)
        
        return self.routers[router].route(prompt, threshold, model_pair)

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)
