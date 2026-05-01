import re
import random
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from routellm.types import ModelPair

logger = logging.getLogger(__name__)

class LoadBalancerEndpoint(BaseModel):
    model: str
    weight: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None

class LoadBalancerConfig(BaseModel):
    strategy: str = "weighted" # weighted, round-robin
    endpoints: List[LoadBalancerEndpoint]

class LoadBalancer:
    """Simple load balancer for rotating between endpoints or providers."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self._current_index = 0

    def select(self) -> LoadBalancerEndpoint:
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
