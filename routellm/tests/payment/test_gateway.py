from routellm.payment.types import PaymentChallenge, PaymentReceipt


def test_payment_challenge_fields():
    c = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="1.00",
        currency="USDC",
        payload={},
    )
    assert c.scheme == "x402"
    assert c.amount == "1.00"


def test_payment_receipt_fields():
    r = PaymentReceipt(
        tx_hash="0xabc",
        network="base",
        amount="1.00",
        currency="USDC",
        paid_at=1234567890,
    )
    assert r.tx_hash == "0xabc"
    assert r.resource == ""


def test_gateway_requires_only_the_client_side_surface():
    """The ABC's contract is what a paying client needs, and no more.

    RouteLLM is the buyer here: it pays to consume somebody else's
    gated endpoint. Verifying a presented payment is the seller's job,
    and it needs the seller's own price terms -- which a receipt does
    not and cannot carry. A gateway that implements paying is a
    complete gateway.
    """
    from routellm.payment.gateway import PaymentGateway

    assert PaymentGateway.__abstractmethods__ == frozenset({"pay", "networks", "name"})


def test_gateway_instantiable_without_a_verify_method():
    """A gateway defining only pay/networks/name is concrete."""
    from routellm.payment.gateway import PaymentGateway

    class PayOnlyGateway(PaymentGateway):
        async def pay(self, challenge):
            raise NotImplementedError

        @property
        def networks(self):
            return ["base"]

        @property
        def name(self):
            return "pay-only"

    assert PayOnlyGateway().name == "pay-only"
