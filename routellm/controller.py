from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Protocol

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.caching import Cache, CacheConfig
from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentChallenge
from routellm.quality import QualityManager
from routellm.resilience import Resilience, ResilienceConfig
from routellm.routers.routers import ROUTER_CLS
from routellm.traffic import TrafficManager
from routellm.types import Middleware, ModelPair

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "checkpoint_path": "routellm/sw_ranking_gpt4_augmented",
    },
    "mf": {
        "checkpoint_path": "routellm/mf_gpt4_augmented",
    },
    "bert": {
        "checkpoint_path": "routellm/bert_gpt4_augmented",
    },
    "causal_llm": {
        "checkpoint_path": "routellm/causal_llm_gpt4_augmented",
    },
}


class RoutingError(Exception):
    pass


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
        payment_gateway: Optional[PaymentGateway] = None,
        resilience_config: Optional[ResilienceConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        traffic_manager: Optional[TrafficManager] = None,
        quality_manager: Optional[QualityManager] = None,
    ):
        self.default_model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        self.middleware = middleware or []
        self.payment_gateway = payment_gateway
        self.resilience = Resilience(resilience_config)
        self.cache = Cache(cache_config)
        self.traffic_manager = traffic_manager or TrafficManager()
        self.quality_manager = quality_manager or QualityManager()

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if self.progress_bar:
            router_pbar = tqdm(total=len(routers), desc="Initializing routers")

        for router in routers:
            router_config = config.get(router, {})
            self.routers[router] = ROUTER_CLS[router](**router_config)
            if router_pbar:
                router_pbar.update(1)

    @property
    def model_pair(self) -> ModelPair:
        return self.default_model_pair

    def _get_model_pair_for_prompt(self, prompt: str) -> ModelPair:
        """Get the model pair to use for a given prompt, checking middleware."""
        for m in self.middleware:
            pair = m.get_model_pair(prompt)
            if pair:
                return pair
        return self.default_model_pair

    def _parse_model_name(self, model_name: str) -> tuple[str, float]:
        """Parse router and threshold from model name."""
        if not model_name.startswith("router-"):
            raise RoutingError(
                f"Invalid model name: {model_name}. Model name must start with 'router-'"
            )

        parts = model_name.split("-")
        if len(parts) != 3:
            raise RoutingError(
                f"Invalid model name: {model_name}. Model name must be in the format 'router-[router name]-[threshold]'"
            )

        router = parts[1]
        try:
            threshold = float(parts[2])
        except ValueError:
            raise RoutingError(
                f"Invalid threshold: {parts[2]}. Threshold must be a float."
            )

        return router, threshold

    def _validate_router_threshold(self, router: str, threshold: float):
        if router not in self.routers:
            raise RoutingError(f"Router {router} not found.")

        if not (0 <= threshold <= 1):
            raise RoutingError(f"Threshold {threshold} must be between 0 and 1.")

    async def _request_with_payment(self, call_fn):
        """Wrapper to handle 402 Payment Required challenges."""
        try:
            return await call_fn({})
        except Exception as e:
            # Check if it's a 402 challenge (LiteLLM usually surfaces this as a generic Exception with text)
            if "402" in str(e) and self.payment_gateway:
                import logging
                logging.getLogger(__name__).info("Received 402 challenge, attempting to pay...")
                
                # Extract challenge data from error (mocked for now as LiteLLM doesn't have native 402 support yet)
                # In a real scenario, we'd parse the 'WWW-Authenticate' header or body
                challenge = PaymentChallenge(
                    scheme=self.payment_gateway.name,
                    network=self.payment_gateway.networks[0],
                    amount="1", # Mock amount
                    currency="USDC",
                    payload={}
                )
                
                receipt = await self.payment_gateway.pay(challenge)
                # Retry with payment receipt in headers
                return await call_fn({"X-Payment-Receipt": receipt.tx_hash})
            raise

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
        
        prompt = kwargs["messages"][-1]["content"]

        # 1. Apply traffic rules for conditional routing
        overridden_pair = self.traffic_manager.get_model_pair(prompt, kwargs)
        if overridden_pair:
            model_pair = overridden_pair
        else:
            model_pair = self._get_model_pair_for_prompt(prompt)

        routed_model = self.routers[router].route(prompt, threshold, model_pair)

        # 2. Apply canary testing
        is_canary = self.quality_manager.should_canary()
        if is_canary:
            model_to_use = self.quality_manager.canary_config.canary_model
        else:
            model_to_use = routed_model

        self.model_counts[router][model_to_use] += 1

        # Try cache
        cache_params = {k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
        cached_res = self.cache.get(prompt, model_to_use, cache_params)
        if cached_res:
            from litellm.utils import ModelResponse
            res = ModelResponse(**cached_res)
            # Record trace for cached response too
            self.quality_manager.record_trace(prompt, model_to_use, res.model_dump(), {"cached": True, "is_canary": is_canary})
            return res

        # Fallback chain: model_to_use -> routed_model -> other model in pair
        models_to_try = [model_to_use]
        if routed_model not in models_to_try:
            models_to_try.append(routed_model)
        
        other_model = model_pair.weak if routed_model == model_pair.strong else model_pair.strong
        if other_model not in models_to_try:
            models_to_try.append(other_model)

        api_base = self.api_base
        api_key = self.api_key

        last_err = None
        for model_name in models_to_try:
            # 3. Apply load balancing
            model, balanced_key, balanced_base = self.traffic_manager.balance(model_name)
            curr_api_base = balanced_base or api_base
            curr_api_key = balanced_key or api_key

            def _call():
                kwargs_copy = dict(kwargs)
                kwargs_copy["model"] = model
                return completion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    **kwargs_copy,
                )

            try:
                # Wrap with resilience
                res = self.resilience.wrap_completion(
                    model_name,
                    _call
                )

                # Cache the response
                try:
                    self.cache.put(prompt, model_name, cache_params, res.model_dump())
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to cache response: {str(e)}")
                
                # Record trace
                self.quality_manager.record_trace(prompt, model_name, res.model_dump(), {"is_canary": is_canary and model_name == model_to_use})

                return res
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {str(e)}"
                )
                last_err = e
                continue
        
        if last_err:
            raise last_err

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
        
        prompt = kwargs["messages"][-1]["content"]
        
        # 1. Apply traffic rules for conditional routing
        overridden_pair = self.traffic_manager.get_model_pair(prompt, kwargs)
        if overridden_pair:
            model_pair = overridden_pair
        else:
            model_pair = self._get_model_pair_for_prompt(prompt)
            
        routed_model = self.routers[router].route(prompt, threshold, model_pair)
        
        # 2. Apply canary testing
        is_canary = self.quality_manager.should_canary()
        if is_canary:
            model_to_use = self.quality_manager.canary_config.canary_model
        else:
            model_to_use = routed_model
            
        self.model_counts[router][model_to_use] += 1
        
        # Try cache
        cache_params = {k: v for k, v in kwargs.items() if k not in ["messages", "model"]}
        cached_res = await self.cache.aget(prompt, model_to_use, cache_params)
        if cached_res:
            from litellm.utils import ModelResponse
            res = ModelResponse(**cached_res)
            # Record trace for cached response too
            self.quality_manager.record_trace(prompt, model_to_use, res.model_dump(), {"cached": True, "is_canary": is_canary})
            return res

        # Fallback chain: model_to_use -> routed_model -> other model in pair
        models_to_try = [model_to_use]
        if routed_model not in models_to_try:
            models_to_try.append(routed_model)
        
        other_model = model_pair.weak if routed_model == model_pair.strong else model_pair.strong
        if other_model not in models_to_try:
            models_to_try.append(other_model)

        last_err = None
        for model_name in models_to_try:
            # 3. Apply load balancing
            model, balanced_key, balanced_base = self.traffic_manager.balance(model_name)
            curr_api_base = balanced_base or self.api_base
            curr_api_key = balanced_key or self.api_key
            
            async def _call(extra_headers={}):
                kwargs_copy = dict(kwargs)
                kwargs_copy["model"] = model
                return await acompletion(
                    api_base=curr_api_base,
                    api_key=curr_api_key,
                    extra_headers=extra_headers,
                    **kwargs_copy,
                )

            try:
                res = await self.resilience.wrap_acompletion(
                    model_name,
                    lambda: self._request_with_payment(_call)
                )
                
                # 4. Validate canary if needed
                if is_canary and model_name == model_to_use:
                    passed = await self.quality_manager.validate_canary(
                        res.choices[0].message.content, 
                        prompt
                    )
                    if not passed:
                        import logging
                        logging.getLogger(__name__).warning(f"Canary model {model_name} failed validation, trying fallback.")
                        raise Exception(f"Canary validation failed for {model_name}")

                # Cache the response
                try:
                    await self.cache.aput(prompt, model_name, cache_params, res.model_dump())
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to cache response: {str(e)}")
                
                # Record trace
                self.quality_manager.record_trace(prompt, model_name, res.model_dump(), {"is_canary": is_canary and model_name == model_to_use})
                
                return res
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Model {model_name} failed, trying fallback if available. Error: {str(e)}"
                )
                last_err = e
                continue
        
        if last_err:
            raise last_err
