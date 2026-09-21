from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    """pay() drives the SDK's 402 handler with the server's own response.

    The adapter hands over the raw 402 -- headers, body and URL -- and
    takes back the retry header already named for the protocol version
    the SDK detected.
    """
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="1.00",
        currency="USDC",
        payload={"resource": "https://llm.example.com/v1/chat"},
        headers={"payment-required": "eyJ4NDAyVmVyc2lvbiI6IDJ9"},
        body=b"",
        resource_url="https://llm.example.com/v1/chat",
    )

    mock_payload = MagicMock()
    mock_payload.transaction_hash = "0xdeadbeef"
    mock_http_client = MagicMock()
    mock_http_client.handle_402_response = AsyncMock(
        return_value=({"PAYMENT-SIGNATURE": "c2lnbmVkLXBheWxvYWQ="}, mock_payload)
    )

    with patch.object(adapter, "_build_client", return_value=mock_http_client):
        receipt = await adapter.pay(challenge)

    mock_http_client.handle_402_response.assert_awaited_once_with(
        headers={"payment-required": "eyJ4NDAyVmVyc2lvbiI6IDJ9"},
        body=None,
        request_url="https://llm.example.com/v1/chat",
    )
    assert receipt.tx_hash == "0xdeadbeef"
    assert receipt.network == "base"
    assert receipt.currency == "USDC"
    assert receipt.resource == "https://llm.example.com/v1/chat"
    assert receipt.header_name == "PAYMENT-SIGNATURE"
    assert receipt.header_value == "c2lnbmVkLXBheWxvYWQ="


def test_x402_adapter_carries_no_verify():
    """The adapter exposes paying only.

    The SDK's verification entry point is a resource-server method
    taking the payment payload and the server's own requirements, not
    a receipt. Nothing here can supply those, so the adapter does not
    pretend to.
    """
    assert not hasattr(X402Adapter, "verify")
