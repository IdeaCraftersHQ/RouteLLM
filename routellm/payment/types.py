"""Data types for payment processing.

Defines payment challenges and receipts for Web3-based payment handling.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaymentChallenge:
    """Challenge requesting payment for API access.

    Attributes
    ----------
    scheme : str
        Payment scheme (e.g., "x402").
    network : str
        Blockchain network ("base", "ethereum", etc).
    amount : str
        Amount to pay as decimal string.
    currency : str
        Currency code (e.g., "USDC").
    payload : dict, optional
        Additional payment metadata (default empty).
    """

    scheme: str
    network: str
    amount: str
    currency: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentReceipt:
    """Receipt confirming payment for API access.

    Attributes
    ----------
    tx_hash : str
        Blockchain transaction hash.
    network : str
        Blockchain network where payment occurred.
    amount : str
        Amount paid as decimal string.
    currency : str
        Currency code.
    paid_at : int
        Unix timestamp of payment.
    resource : str, optional
        Resource identifier paid for (default "").
    """

    tx_hash: str
    network: str
    amount: str
    currency: str
    paid_at: int
    resource: str = ""
