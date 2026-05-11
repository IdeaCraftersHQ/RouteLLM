"""Traffic management and load balancing for model endpoints.

Provides load balancing strategies and traffic rules for distributing
requests across multiple model endpoints.
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from routellm.types import ModelPair

logger = logging.getLogger(__name__)


class LoadBalancerEndpoint(BaseModel):
    """Configuration for a single load balancer endpoint.

    Attributes
    ----------
    model : str
        Model name at this endpoint.
    weight : float, optional
        Weight for weighted strategy (default 1.0).
    api_key : str, optional
        API key for this endpoint (default None).
    api_base : str, optional
        API base URL for this endpoint (default None).
    """

    model: str
    weight: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class LoadBalancerConfig(BaseModel):
    """Configuration for load balancer.

    Attributes
    ----------
    strategy : str, optional
        Load balancing strategy: "weighted" or "round-robin" (default "weighted").
    endpoints : list[LoadBalancerEndpoint]
        List of available endpoints.
    """

    strategy: str = "weighted"  # weighted, round-robin
    endpoints: List[LoadBalancerEndpoint]


class LoadBalancer:
    """Load balancer for distributing requests across endpoints.

    Supports weighted random and round-robin strategies for endpoint selection.
    """
    
    def __init__(self, config: LoadBalancerConfig):
        """Initialize load balancer.

        Parameters
        ----------
        config : LoadBalancerConfig
            Load balancer configuration.
        """
        self.config = config
        self._current_index = 0

    def select(self) -> LoadBalancerEndpoint:
        """Select endpoint based on configured strategy.

        Returns
        -------
        LoadBalancerEndpoint
            Selected endpoint.

        Raises
        ------
        ValueError
            If no endpoints are configured.
        """
        if not self.config.endpoints:
            raise ValueError("No endpoints configured for load balancer")
            
        if self.config.strategy == "weighted":
            total_weight = sum(e.weight for e in self.config.endpoints)
            r = random.uniform(0, total_weight)
            upto = 0
            for e in self.config.endpoints:
                if upto + e.weight >= r:
                    return e
                upto += e.weight
        elif self.config.strategy == "round-robin":
            print(f"[LB] select round-robin current_index={self._current_index}")
            e = self.config.endpoints[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.config.endpoints)
            print(f"[LB] select result={e.model} next_index={self._current_index}")
            return e
        
        return self.config.endpoints[0]

class TrafficRule(BaseModel):
    """Rule for conditional routing based on request payload."""
    pattern: Optional[str] = None # Regex pattern for prompt
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    strong_model: str
    weak_model: str

class TrafficManager:
    """Handles conditional routing and load balancing logic."""
    
    def __init__(
        self, 
        rules: List[TrafficRule] = None,
        load_balancers: Dict[str, LoadBalancer] = None
    ):
        self.rules = rules or []
        self.load_balancers = load_balancers or {}

    def get_model_pair(self, prompt: str, request_params: Dict[str, Any]) -> Optional[ModelPair]:
        """Check rules to see if we should override the model pair."""
        for rule in self.rules:
            # Check pattern
            if rule.pattern:
                if not re.search(rule.pattern, str(prompt), re.IGNORECASE):
                    continue
            
            # Check tokens
            max_tokens = request_params.get("max_tokens")
            if rule.min_tokens is not None and (max_tokens is None or max_tokens < rule.min_tokens):
                continue
            if rule.max_tokens is not None and (max_tokens is not None and max_tokens > rule.max_tokens):
                continue
                
            return ModelPair(strong=rule.strong_model, weak=rule.weak_model)
            
        return None

    def balance(self, model_name: str) -> tuple[str, Optional[str], Optional[str]]:
        """Return (model, api_key, api_base) if a load balancer exists for this model."""
        if model_name in self.load_balancers:
            endpoint = self.load_balancers[model_name].select()
            return endpoint.model, endpoint.api_key, endpoint.api_base
        return model_name, None, None
