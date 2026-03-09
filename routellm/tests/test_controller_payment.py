import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentChallenge, PaymentReceipt


class MockGateway(PaymentGateway):
    """Concrete gateway whose pay/verify are replaced with AsyncMocks after init."""

    async def pay(self, challenge):  # overridden in __init__ by AsyncMock
        pass  # pragma: no cover

    async def verify(self, receipt):  # overridden in __init__ by AsyncMock
        pass  # pragma: no cover

    def __init__(self):
        self.pay = AsyncMock(return_value=PaymentReceipt(
            tx_hash="0xpaid",
            network="base",
            amount="1.00",
            currency="USDC",
            paid_at=1000000,
        ))
        self.verify = AsyncMock(return_value=True)

    @property
    def networks(self):
        return ["base"]

    @property
    def name(self):
        return "mock"


@pytest.mark.asyncio
async def test_controller_retries_on_402():
    """Controller should call gateway.pay and retry when litellm raises 402."""
    from routellm.controller import Controller

    gateway = MockGateway()

    # Simulate litellm raising 402 on first call, succeeding on second
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "hello"

    error_402 = Exception("Payment Required")
    error_402.status_code = 402
    error_402.response = MagicMock()
    error_402.response.json = MagicMock(return_value={
        "scheme": "x402",
        "network": "base",
        "amount": "1.00",
        "currency": "USDC",
    })

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise error_402
        return mock_response

    with patch("routellm.controller.acompletion", side_effect=mock_acompletion):
        controller = Controller(
            routers=["random"],
            strong_model="gpt-4",
            weak_model="gpt-3.5-turbo",
            payment_gateway=gateway,
        )
        # Call the internal payment helper directly
        result = await controller._request_with_payment(
            lambda extra_headers: mock_acompletion(extra_headers=extra_headers)
        )

    assert call_count == 2
    gateway.pay.assert_awaited_once()


@pytest.mark.asyncio
async def test_controller_no_gateway_reraises_402():
    """Without a gateway, 402 errors should propagate."""
    from routellm.controller import Controller

    error_402 = Exception("Payment Required")
    error_402.status_code = 402

    async def mock_acompletion(*args, **kwargs):
        raise error_402

    with patch("routellm.controller.acompletion", side_effect=mock_acompletion):
        controller = Controller(
            routers=["random"],
            strong_model="gpt-4",
            weak_model="gpt-3.5-turbo",
            payment_gateway=None,
        )
        with pytest.raises(Exception, match="Payment Required"):
            await controller._request_with_payment(
                lambda extra_headers: mock_acompletion(extra_headers=extra_headers)
            )
