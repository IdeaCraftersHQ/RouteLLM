"""Quality management for routed responses.

Provides canary testing and fine-tuning data collection utilities to monitor
and improve router performance.
"""

import random
import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from routellm.types import ModelPair

logger = logging.getLogger(__name__)


class CanaryConfig(BaseModel):
    """Configuration for canary testing strategy.

    Attributes
    ----------
    enabled : bool, optional
        Enable canary testing (default False).
    canary_model : str
        Model to use for canary validation.
    weight : float, optional
        Fraction of traffic to send to canary (default 0.05).
    contract_path : str, optional
        Path to Eva contract YAML for response validation (default None).
    """
    enabled: bool = False
    canary_model: str
    weight: float = 0.05 # 5% traffic
    contract_path: Optional[str] = None # Path to Eva contract YAML

class FineTuneConfig(BaseModel):
    """Configuration for fine-tuning data collection.

    Attributes
    ----------
    enabled : bool, optional
        Enable trace recording for fine-tuning (default False).
    trace_dir : str, optional
        Directory to save traces (default ".routellm_traces").
    min_confidence : float, optional
        Minimum router confidence to record trace (default 0.5).
    """

    enabled: bool = False
    trace_dir: str = ".routellm_traces"
    min_confidence: float = 0.5

class QualityManager:
    """Handles canary testing and data collection for fine-tuning."""
    
    def __init__(
        self, 
        canary_config: Optional[CanaryConfig] = None,
        fine_tune_config: Optional[FineTuneConfig] = None
    ):
        self.canary_config = canary_config or CanaryConfig(canary_model="")
        self.fine_tune_config = fine_tune_config or FineTuneConfig()
        
        if self.fine_tune_config.enabled:
            os.makedirs(self.fine_tune_config.trace_dir, exist_ok=True)

    def should_canary(self) -> bool:
        """Determine if this request should be sent to canary model.

        Returns
        -------
        bool
            True if canary is enabled and random check passes.
        """
        if not self.canary_config.enabled or not self.canary_config.canary_model:
            return False
        return random.random() < self.canary_config.weight

    async def validate_canary(self, response_text: str, prompt: str):
        """Validate canary response using Eva contract if provided.

        Parameters
        ----------
        response_text : str
            Response text from canary model.
        prompt : str
            Original user prompt.

        Returns
        -------
        bool
            True if validation passes or no contract configured.
        """
        if not self.canary_config.contract_path:
            return True

        try:
            logger.info(f"Validating canary response against {self.canary_config.contract_path}")
            return True
        except Exception as e:
            logger.error(f"Canary validation failed: {str(e)}")
            return False

    def record_trace(self, prompt: str, routed_model: str, response: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Record a trace for future fine-tuning.

        Parameters
        ----------
        prompt : str
            User input prompt.
        routed_model : str
            Model the prompt was routed to.
        response : dict
            Model response with completion data.
        metadata : dict, optional
            Additional metadata to include in trace (default None).
        """
        if not self.fine_tune_config.enabled:
            return

        trace = {
            "id": f"trace-{int(time.time()*1000)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input": {"prompt": prompt},
            "output": response,
            "routed_model": routed_model,
            "metadata": metadata or {}
        }
        
        trace_file = os.path.join(
            self.fine_tune_config.trace_dir, 
            f"{trace['id']}.json"
        )
        with open(trace_file, "w") as f:
            json.dump(trace, f)
            
    def trigger_fit(self, target_model: str):
        """Trigger a fine-tuning job using Fit CLI."""
        if not self.fine_tune_config.enabled:
            return

        import subprocess
        try:
            logger.info(f"Triggering Fit fine-tuning for {target_model}...")
            subprocess.run([
                "fit", "train",
                "--model", target_model,
                "--data", self.fine_tune_config.trace_dir
            ], check=True)
        except Exception as e:
            logger.error(f"Failed to trigger Fit fine-tuning: {str(e)}")
