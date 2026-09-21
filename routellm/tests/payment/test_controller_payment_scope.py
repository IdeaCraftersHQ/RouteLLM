"""The controller's own 402 retry obeys the same authorisation.

There are two seams where a 402 can be paid. The transport under
litellm settles the ones it sees; the controller catches the errors
litellm raises for the rest and pays those. Scoping only the first
would close the hole and leave it open, because the controller seam
runs precisely when the transport declined -- an unauthorised endpoint
among them.

The controller knows which endpoint it was calling, so the question it
asks is the endpoint's `pay` flag rather than a URL match.
"""
import pytest

from routellm.payment.gateway import PaymentGateway
from routellm.payment.types import PaymentReceipt


class RecordingGateway(PaymentGateway):
    """A gateway that records every challenge it was asked to sign."""

    def __init__(self):
        self.signed = []

    async def pay(self, challenge):
        self.signed.append(challenge)
        return PaymentReceipt(
            tx_hash="0xpaid",
            network="base",
            amount="1.00",
            currency="USDC",
            paid_at=1000000,
        )

    @property
    def networks(self):
        return ["base"]

    @property
    def name(self):
        return "mock"


def refusal():
    """A 402 shaped like the error litellm raises for one."""
    error = Exception("Payment Required")
    error.status_code = 402
    return error


def controller_with(gateway, config, weak="gpt-3.5-turbo"):
    """Build a controller over `config`'s endpoints with `gateway`.

    The stubbed `random` router always picks the weak side, so naming
    an endpoint there is how a test drives a routed request at it.

    Caching is off. The cache is one SQLite file in the CWD, shared by
    the whole suite, so a hit from someone else's run would return an
    answer without ever reaching the payment seam -- and a test that
    asserts a payment happened would read that as success.
    """
    from routellm.caching import CacheConfig
    from routellm.controller import Controller
    from routellm.endpoints import EndpointRegistry

    return Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model=weak,
        endpoints=EndpointRegistry.from_config(config),
        payment_gateway=gateway,
        cache_config=CacheConfig(enabled=False),
    )


CONFIG = {
    "endpoints": {
        "payer": {
            "model": "gpt-4o",
            "api_base": "https://paid.example.com/v1",
            "pay": True,
        },
        "freeloader": {
            "model": "gpt-4o-mini",
            "api_base": "https://free.example.com/v1",
        },
    }
}


@pytest.mark.asyncio
async def test_an_unauthorised_endpoint_is_not_paid_on_the_error_path():
    """A 402 from an endpoint without `pay:` is re-raised, not signed.

    This is the seam that runs when the transport declined, so leaving
    it unscoped would hand back exactly the exposure the scope exists
    to remove.
    """
    gateway = RecordingGateway()
    controller = controller_with(gateway, CONFIG)

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        raise refusal()

    with pytest.raises(Exception, match="Payment Required"):
        await controller._request_with_payment(call, endpoint="freeloader")

    assert gateway.signed == []
    # One attempt, no paid replay.
    assert calls == [{}]


@pytest.mark.asyncio
async def test_an_authorised_endpoint_still_pays_on_the_error_path():
    """`pay: true` keeps the retry it always had."""
    gateway = RecordingGateway()
    controller = controller_with(gateway, CONFIG)

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        if len(calls) == 1:
            raise refusal()
        return "paid answer"

    result = await controller._request_with_payment(call, endpoint="payer")

    assert result == "paid answer"
    assert len(gateway.signed) == 1
    assert calls[1] != {}


@pytest.mark.asyncio
async def test_a_raw_model_name_is_not_payable():
    """A name no config line ever wrote cannot have asked to pay.

    The registry answers an unknown name with a passthrough endpoint.
    Treating that as authorised would make every un-configured model
    payable, which is the fail-open default all over again.
    """
    gateway = RecordingGateway()
    controller = controller_with(gateway, CONFIG)

    async def call(extra_headers):
        raise refusal()

    with pytest.raises(Exception, match="Payment Required"):
        await controller._request_with_payment(call, endpoint="some/raw-model")

    assert gateway.signed == []


@pytest.mark.asyncio
async def test_the_request_path_names_the_endpoint_it_is_calling():
    """A real routed request carries its endpoint into the payment check.

    The helper can refuse perfectly and still change nothing if the
    call site forgets to say which endpoint it is calling. Only a
    request driven through `acompletion` pins that.
    """
    from unittest.mock import patch

    gateway = RecordingGateway()
    controller = controller_with(gateway, CONFIG, weak="freeloader")

    async def always_refuses(*args, **kwargs):
        raise refusal()

    with patch("routellm.controller.acompletion", side_effect=always_refuses):
        with pytest.raises(Exception, match="Payment Required"):
            await controller.acompletion(
                router="random",
                threshold=0.5,
                messages=[{"role": "user", "content": "hi"}],
            )

    assert gateway.signed == []


@pytest.mark.asyncio
async def test_the_request_path_still_pays_an_authorised_endpoint():
    """The same routed path, an authorised endpoint: the retry happens."""
    from unittest.mock import patch

    gateway = RecordingGateway()
    controller = controller_with(gateway, CONFIG, weak="payer")

    seen = []

    async def refuse_then_serve(*args, **kwargs):
        seen.append(kwargs.get("extra_headers") or {})
        if len(seen) == 1:
            raise refusal()
        from litellm.utils import ModelResponse

        return ModelResponse(
            id="cmpl-1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "paid answer"},
                    "finish_reason": "stop",
                }
            ],
            model="gpt-4o",
        )

    with patch("routellm.controller.acompletion", side_effect=refuse_then_serve):
        response = await controller.acompletion(
            router="random",
            threshold=0.5,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.choices[0].message.content == "paid answer"
    assert len(gateway.signed) == 1
    assert seen[1] != {}
