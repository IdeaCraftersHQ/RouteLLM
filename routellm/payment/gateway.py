"""Abstract payment gateway interface.

Defines payment gateway protocol for handling blockchain-based payments.
"""

from abc import ABC, abstractmethod
from .types import PaymentChallenge, PaymentReceipt


class PaymentGateway(ABC):
    """Transport-agnostic payment interface.

    Implement this to add a new payment provider to RouteLLM.
    """

    @abstractmethod
    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge.

        Parameters
        ----------
        challenge : PaymentChallenge
            Payment challenge to fulfill.

        Returns
        -------
        PaymentReceipt
            Receipt confirming payment.

        Raises
        ------
        PaymentError
            If payment fails.
        """

    @abstractmethod
    async def verify(self, receipt: PaymentReceipt) -> bool:
        """Verify a receipt is valid (for server-side use).

        Parameters
        ----------
        receipt : PaymentReceipt
            Payment receipt to verify.

        Returns
        -------
        bool
            True if receipt is valid.
        """

    @property
    @abstractmethod
    def networks(self) -> list[str]:
        """Supported network identifiers.

        Returns
        -------
        list[str]
            List of supported blockchain networks.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name.

        Returns
        -------
        str
            Payment provider name (e.g., 'x402').
        """
