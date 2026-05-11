"""Type definitions and protocols for routellm routing system.

This module defines the core data structures and protocols used throughout
routellm for model routing and middleware operations.
"""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from routellm.controller import Controller


@dataclass
class ModelPair:
    """Pair of strong and weak models for routing.

    Attributes
    ----------
    strong : str
        Name of the stronger/more capable model.
    weak : str
        Name of the weaker/cheaper model.
    """
    strong: str
    weak: str


class Middleware(Protocol):
    """Protocol for middleware that can modify controller behavior.

    Middleware allows customization of routing behavior by dynamically
    selecting model pairs based on prompt content.
    """

    def get_model_pair(self, prompt: str) -> 'ModelPair':
        """Get a model pair based on the prompt.

        Parameters
        ----------
        prompt : str
            User input prompt.

        Returns
        -------
        ModelPair or None
            Model pair to use for this prompt, or None to use default.
        """
        ...
