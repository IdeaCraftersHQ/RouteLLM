"""Abstract payment gateway interface.

Defines payment gateway protocol for handling blockchain-based payments.
"""

from abc import ABC, abstractmethod
from .types import PaymentChallenge, PaymentReceipt


class PaymentGateway(ABC):
    """Transport-agnostic payment interface.

    Implement this to add a new payment provider to RouteLLM.

    The caller does not parse the 402. It hands over the response as the
    server sent it -- `challenge.headers`, `challenge.body` and
    `challenge.resource_url` -- and the gateway reads the challenge with
    whatever library speaks its protocol. That is the only arrangement
    under which the amount, asset and payee are the server's rather than
    the caller's guess, and it is what lets a gateway support several
    protocol versions without the caller knowing they exist.

    Symmetrically, the retry header is the gateway's to name: a receipt
    reports both `header_name` and `header_value`, because the header a
    proof of payment travels on is part of the protocol version the
    server chose. x402 alone uses two -- PAYMENT-SIGNATURE for v2 and
    the legacy X-PAYMENT for v1. A gateway whose proof is a plain
    identifier can leave both at their defaults and let the receipt's
    `tx_hash` be sent on X-PAYMENT.
    """

    @abstractmethod
    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge.

        Parameters
        ----------
        challenge : PaymentChallenge
            Payment challenge to fulfill, carrying the raw 402 response.

        Returns
        -------
        PaymentReceipt
            Receipt confirming payment, naming the header the retry must
            carry the proof on.

        Raises
        ------
        PaymentError
            If payment fails.
        ValueError
            If the challenge carries nothing this gateway can read.
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
