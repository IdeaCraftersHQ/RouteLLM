from abc import ABC, abstractmethod
from .types import PaymentChallenge, PaymentReceipt

class PaymentGateway(ABC):
    """Transport-agnostic payment interface.
    Implement this to add a new payment provider to RouteLLM.
    """

    @abstractmethod
    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge. Raises PaymentError on failure."""

    @abstractmethod
    async def verify(self, receipt: PaymentReceipt) -> bool:
        """Verify a receipt is valid (for server-side use)."""

    @property
    @abstractmethod
    def networks(self) -> list[str]:
        """Supported network identifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, e.g. 'x402'."""
