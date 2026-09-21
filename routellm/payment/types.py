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
    The wire fields (`headers`, `body`, `resource_url`) carry the 402
    response exactly as the server sent it. A gateway parses the
    challenge out of those with its own protocol library, which is the
    only way the amount, asset and payee can be the server's rather than
    this process's guess. The descriptive fields above them are a
    summary for logging and for gateways that do not speak x402.

    Attributes
    ----------
    scheme : str
        Payment scheme (e.g., "x402").
    network : str
        Blockchain network ("base", "ethereum", etc).
    amount : str
        Amount to pay as decimal string. Empty when only the server's
        own challenge states the price.
    currency : str
        Currency code (e.g., "USDC"). Empty when only the server's own
        challenge names the asset.
    payload : dict, optional
        Additional payment metadata (default empty).
    headers : dict, optional
        Response headers from the 402, lowercased (default empty).
    body : bytes, optional
        Raw response body from the 402 (default empty).
    resource_url : str, optional
        URL of the request that was refused (default empty).
    max_amount : str, optional
        The most this one payment may be, as human money ("$0.01").
        The cap belongs to the payment rather than to the gateway: one
        wallet serves every endpoint, and each endpoint may be capped
        differently, so the caller that knows which endpoint it is
        paying states the figure here. None means the caller set no
        cap, which leaves the gateway's own default standing -- never
        that the payment is unbounded (default None).
    cap_source : str, optional
        Which limit `max_amount` came from, `"global"` or
        `"endpoint"`. Carried so a refusal can name the knob that
        would change it; with two layers, "payment refused" alone
        leaves an operator guessing (default None).
    """

    scheme: str
    network: str
    amount: str
    currency: str
    payload: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    resource_url: str = ""
    max_amount: str | None = None
    cap_source: str | None = None


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
    header_name : str, optional
        Header the retry must carry the proof on. The name is part of
        the protocol version the server chose -- x402 v2 reads
        PAYMENT-SIGNATURE and v1 reads the legacy X-PAYMENT -- so the
        gateway that read the challenge reports it rather than letting
        the caller assume one (default "X-PAYMENT").
    header_value : str, optional
        Encoded proof of payment for that header. Empty means the caller
        should fall back to `tx_hash`, which suits gateways whose proof
        is just an identifier (default "").
    """

    tx_hash: str
    network: str
    amount: str
    currency: str
    paid_at: int
    resource: str = ""
    header_name: str = "X-PAYMENT"
    header_value: str = ""

    @property
    def proof(self) -> str:
        """Value to send on `header_name`.

        Returns
        -------
        str
            The encoded payload when the gateway produced one, else the
            transaction hash.
        """
        return self.header_value or self.tx_hash
