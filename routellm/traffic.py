"""Traffic management and load balancing for model endpoints.

Provides load balancing strategies and traffic rules for distributing
requests across multiple model endpoints.
"""

import logging
import random
import re
from typing import Any

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
    api_key: str | None = None
    api_base: str | None = None


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
    endpoints: list[LoadBalancerEndpoint]


class NoAllowedTarget(ValueError):
    """Every target of a load balancer was refused by the caller's check."""


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

    def select(self, allowed=None) -> LoadBalancerEndpoint:
        """Select endpoint based on configured strategy.

        Parameters
        ----------
        allowed : callable, optional
            `(model) -> bool`. A target it refuses is never selected,
            whatever the strategy; the round-robin index moves past it.

        Returns
        -------
        LoadBalancerEndpoint
            Selected endpoint.

        Raises
        ------
        ValueError
            If no endpoints are configured.
        NoAllowedTarget
            If `allowed` refuses every endpoint.
        """
        if not self.config.endpoints:
            raise ValueError("No endpoints configured for load balancer")

        pool = self.config.endpoints
        if allowed is not None:
            pool = [e for e in pool if allowed(e.model)]
            if not pool:
                raise NoAllowedTarget("no load balancer target is allowed")

        if self.config.strategy == "weighted":
            total_weight = sum(e.weight for e in pool)
            r = random.uniform(0, total_weight)
            upto = 0
            for e in pool:
                if upto + e.weight >= r:
                    return e
                upto += e.weight
        elif self.config.strategy == "round-robin":
            size = len(self.config.endpoints)
            for _ in range(size):
                e = self.config.endpoints[self._current_index]
                self._current_index = (self._current_index + 1) % size
                if e in pool:
                    return e

        return pool[0]


class TrafficRule(BaseModel):
    """Rule for conditional routing based on request payload."""

    pattern: str | None = None  # Regex pattern for prompt
    min_tokens: int | None = None
    max_tokens: int | None = None
    strong_model: str
    weak_model: str


class TrafficManager:
    """Handles conditional routing and load balancing logic."""

    def __init__(
        self,
        rules: list[TrafficRule] | None = None,
        load_balancers: dict[str, LoadBalancer] | None = None,
    ):
        self.rules = rules or []
        self.load_balancers = load_balancers or {}

    def get_model_pair(self, prompt: str, request_params: dict[str, Any]) -> ModelPair | None:
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
            if rule.max_tokens is not None and (
                max_tokens is not None and max_tokens > rule.max_tokens
            ):
                continue

            return ModelPair(strong=rule.strong_model, weak=rule.weak_model)

        return None

    def balance(self, model_name: str, allowed=None) -> tuple[str, str | None, str | None]:
        """Return (model, api_key, api_base) if a load balancer exists for this model.

        `allowed` is passed to `LoadBalancer.select`.
        """
        if model_name in self.load_balancers:
            endpoint = self.load_balancers[model_name].select(allowed)
            return endpoint.model, endpoint.api_key, endpoint.api_base
        return model_name, None, None
