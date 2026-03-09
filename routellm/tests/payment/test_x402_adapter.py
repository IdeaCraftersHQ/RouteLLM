import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from routellm.payment.types import PaymentChallenge
from routellm.payment.x402 import X402Adapter


def test_x402_adapter_name():
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    assert adapter.name == "x402"


def test_x402_adapter_networks():
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    assert "base" in adapter.networks
    assert "ethereum" in adapter.networks


def test_x402_adapter_custom_networks():
    adapter = X402Adapter(private_key="0x" + "a" * 64, networks=["base"])
    assert adapter.networks == ["base"]


@pytest.mark.asyncio
async def test_x402_adapter_pay_calls_sdk():
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="1.00",
        currency="USDC",
        payload={"resource": "https://llm.example.com/v1/chat"},
    )

    mock_payload = MagicMock()
    mock_payload.transaction_hash = "0xdeadbeef"
    mock_http_client = MagicMock()
    mock_http_client.create_payment_payload = AsyncMock(return_value=mock_payload)

    with patch.object(adapter, "_build_client", return_value=mock_http_client):
        receipt = await adapter.pay(challenge)

    assert receipt.tx_hash == "0xdeadbeef"
    assert receipt.network == "base"
    assert receipt.currency == "USDC"
    assert receipt.resource == "https://llm.example.com/v1/chat"
