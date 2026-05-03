"""Payment gateway integration for Web3-based access control.

Provides HTTP 402 Payment Required protocol and blockchain payment handling.
"""

from .gateway import PaymentGateway
from .types import PaymentChallenge, PaymentReceipt

__all__ = ["PaymentGateway", "PaymentChallenge", "PaymentReceipt"]
