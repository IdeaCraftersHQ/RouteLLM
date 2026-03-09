from dataclasses import dataclass, field
from typing import Any

@dataclass
class PaymentChallenge:
    scheme: str          # "x402"
    network: str         # "base", "ethereum"
    amount: str          # decimal string
    currency: str        # "USDC"
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass
class PaymentReceipt:
    tx_hash: str
    network: str
    amount: str
    currency: str
    paid_at: int         # unix timestamp
    resource: str = ""
