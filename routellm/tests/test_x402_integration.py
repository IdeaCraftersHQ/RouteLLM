"""Integration test: full 402 retry flow through RouteLLM Controller."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentChallenge, PaymentReceipt


class FakeX402Gateway(PaymentGateway):
    """Simulates x402 payment without hitting a real chain."""

    def __init__(self):
        self._paid: list[PaymentChallenge] = []

    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        self._paid.append(challenge)
        return PaymentReceipt(
            tx_hash="0xfacilitated",
            network=challenge.network,
            amount=challenge.amount,
            currency=challenge.currency,
            paid_at=1000000,
            resource=challenge.payload.get("resource", ""),
        )

    async def verify(self, receipt: PaymentReceipt) -> bool:
        return receipt.tx_hash == "0xfacilitated"

    @property
    def networks(self) -> list[str]:
        return ["base"]

    @property
    def name(self) -> str:
        return "fake-x402"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_402_flow():
    """
    Full flow: Controller receives 402 from mock LLM, pays via gateway,
    retries with X-PAYMENT header, and returns successful response.
    """
    from unittest.mock import patch
    from routellm.controller import Controller

    gateway = FakeX402Gateway()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "The answer is 42."

    error_402 = Exception("402 Payment Required")
    error_402.status_code = 402
    error_402.response = MagicMock()
    error_402.response.json = MagicMock(return_value={
        "scheme": "x402",
        "network": "base",
        "amount": "0.001",
        "currency": "USDC",
        "resource": "https://llm.example.com/v1/chat",
    })

    call_count = 0
    received_headers = []

    async def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        received_headers.append(kwargs.get("extra_headers", {}))
        if call_count == 1:
            raise error_402
        return mock_response

    with patch("routellm.controller.acompletion", side_effect=mock_llm):
        controller = Controller(
            routers=["random"],
            strong_model="gpt-4",
            weak_model="gpt-3.5-turbo",
            payment_gateway=gateway,
        )
        result = await controller._request_with_payment(
            lambda extra_headers: mock_llm(extra_headers=extra_headers)
        )

    # LLM was called twice
    assert call_count == 2

    # Payment was made
    assert len(gateway._paid) == 1
    assert gateway._paid[0].network == "base"
    assert gateway._paid[0].currency == "USDC"

    # Second call included payment header
    assert received_headers[1].get("X-PAYMENT") == "0xfacilitated"

    # Result is the successful response
    assert result.choices[0].message.content == "The answer is 42."
