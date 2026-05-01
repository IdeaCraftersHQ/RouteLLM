from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from routellm.controller import Controller

@dataclass
class ModelPair:
    strong: str
    weak: str

class Middleware(Protocol):
    """Protocol for middleware that can modify controller behavior."""
    
    def get_model_pair(self, prompt: str) -> 'ModelPair':
        """Get a model pair based on the prompt."""
        ...
