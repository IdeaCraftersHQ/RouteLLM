import os
import time

from .gateway import PaymentGateway
from .types import PaymentChallenge, PaymentReceipt


class X402Adapter(PaymentGateway):
    """Wraps the x402 PyPI package (coinbase/x402) to fulfill 402 payment challenges.

    Requires x402[evm] installed and ROUTELLM_WALLET_PRIVATE_KEY set.

    Internal flow for pay():
      1. Construct x402.schemas.PaymentRequired from challenge.payload
      2. Register EVM signer mechanism on the x402Client
      3. Call x402HTTPClient.create_payment_payload(payment_required)
      4. Map signed PaymentPayload back to RouteLLM PaymentReceipt
    """

    def __init__(
        self,
        private_key: str | None = None,
        networks: list[str] | None = None,
    ):
        self._private_key = private_key or os.environ.get("ROUTELLM_WALLET_PRIVATE_KEY", "")
        self._networks = networks or ["base", "ethereum", "polygon"]

    @property
    def name(self) -> str:
        return "x402"

    @property
    def networks(self) -> list[str]:
        return self._networks

    def _build_client(self):
        """Build and return a configured x402HTTPClient with EVM signer registered."""
        from eth_account import Account
        from x402.client import x402Client
        from x402.http.x402_http_client import x402HTTPClient
        from x402.mechanisms.evm.exact import ExactEvmScheme

        account = Account.from_key(self._private_key)

        async def signer(message: bytes) -> bytes:
            signed = account.sign_message(message)
            return signed.signature

        client = x402Client()
        network_map = {
            "base": "eip155:8453",
            "ethereum": "eip155:1",
            "polygon": "eip155:137",
        }
        for net in self._networks:
            caip2 = network_map.get(net)
            if caip2:
                client.register(caip2, ExactEvmScheme(signer=signer))

        return x402HTTPClient(client)

    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge using the x402 SDK."""
        from x402.schemas import PaymentRequired

        http_client = self._build_client()
        payment_required = PaymentRequired(**challenge.payload)
        payload = await http_client.create_payment_payload(payment_required)

        tx_hash = getattr(payload, "transaction_hash", None) or str(payload)

        return PaymentReceipt(
            tx_hash=tx_hash,
            network=challenge.network,
            amount=challenge.amount,
            currency=challenge.currency,
            paid_at=int(time.time()),
            resource=challenge.payload.get("resource", ""),
        )

    async def verify(self, receipt: PaymentReceipt) -> bool:
        """Verify a payment receipt via the x402 facilitator."""
        from x402.server import verify_payment
        return await verify_payment(receipt.tx_hash)
