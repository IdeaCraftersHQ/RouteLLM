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
