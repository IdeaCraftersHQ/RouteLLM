"""Only endpoints the operator authorised may be paid.

`litellm.aclient_session` is process-global: one assignment puts every
async request in the process on the paying session. Without a scope,
enabling payments turns every upstream into a potential charge, and any
provider that answers 402 -- misconfigured, compromised, or simply
someone else's -- gets a signed payment.

The authorisation is the config's, per endpoint, and it is enforced on
the base URL the session sees. An endpoint that says nothing about
payment does not pay, so turning payments on authorises nothing by
itself.
"""
import base64
import json

import httpx
import litellm
import pytest

USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
PAY_TO = "0x" + "22" * 20
TEST_KEY = "0x" + "11" * 32

AUTHORISED = "https://paid.example.com/v1"
UNAUTHORISED = "https://freeloader.example.com/v1"


def challenge(resource: str) -> dict:
    """A well-formed v2 PaymentRequired for `resource`."""
    return {
        "x402Version": 2,
        "resource": {"url": resource},
        "accepts": [
            {
                "scheme": "exact",
                "network": "eip155:84532",
                "asset": USDC_BASE_SEPOLIA,
                "amount": "1000",
                "payTo": PAY_TO,
                "maxTimeoutSeconds": 60,
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }


class ChargingProvider(httpx.AsyncBaseTransport):
    """Answers every request with a valid 402 until a proof arrives.

    Records what it was actually paid, so a test can assert on the
    signing rather than on a log line.
    """

    def __init__(self):
        self.paid_urls: list[str] = []
        self.seen_urls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen_urls.append(str(request.url))
        proof = request.headers.get("PAYMENT-SIGNATURE") or request.headers.get(
            "X-PAYMENT"
        )
        if not proof:
            body = challenge(str(request.url))
            return httpx.Response(
                402,
                headers={
                    "PAYMENT-REQUIRED": base64.b64encode(
                        json.dumps(body).encode()
                    ).decode()
                },
                json=body,
                request=request,
            )
        self.paid_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "paid answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )


@pytest.fixture(autouse=True)
def restore_session():
    """litellm's session is process-global; never leak one between tests."""
    previous = litellm.aclient_session
    yield
    litellm.aclient_session = previous


@pytest.mark.asyncio
async def test_unauthorised_endpoint_is_never_signed_for():
    """A valid 402 from an unauthorised base URL buys nothing.

    This is the whole point: the challenge is well-formed and the
    wallet could sign it, so only the scope stands between a stray
    upstream and a real payment.
    """
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter, transport=provider, payable_bases=[AUTHORISED]
    )
    async with session:
        response = await session.post(
            f"{UNAUTHORISED}/chat/completions", json={"messages": []}
        )

    # The 402 comes back untouched, exactly as it would with payments
    # off: refused, not paid, and not turned into an exception either.
    assert response.status_code == 402
    assert provider.paid_urls == []
    # One attempt only. A replay would mean something was signed.
    assert len(provider.seen_urls) == 1
    assert "PAYMENT-SIGNATURE" not in response.request.headers
    assert "X-PAYMENT" not in response.request.headers


@pytest.mark.asyncio
async def test_authorised_endpoint_still_pays():
    """The same session, the same challenge, an authorised base: paid."""
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter, transport=provider, payable_bases=[AUTHORISED]
    )
    async with session:
        response = await session.post(
            f"{AUTHORISED}/chat/completions", json={"messages": []}
        )

    assert response.status_code == 200
    assert provider.paid_urls == [f"{AUTHORISED}/chat/completions"]


@pytest.mark.asyncio
async def test_one_session_separates_authorised_from_unauthorised():
    """Both travel the one global session; only the authorised one pays.

    A mixed deployment is the case the scope exists for: the session is
    shared, so the decision has to be per request, not per process.
    """
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter, transport=provider, payable_bases=[AUTHORISED]
    )
    async with session:
        paid = await session.post(
            f"{AUTHORISED}/chat/completions", json={"messages": []}
        )
        refused = await session.post(
            f"{UNAUTHORISED}/chat/completions", json={"messages": []}
        )

    assert paid.status_code == 200
    assert refused.status_code == 402
    assert provider.paid_urls == [f"{AUTHORISED}/chat/completions"]


@pytest.mark.asyncio
async def test_no_authorised_endpoint_pays_nothing():
    """`--payment-provider x402` alone authorises no endpoint at all.

    The default has to be safe: a config that names no payer leaves the
    wallet unable to spend, however many 402s arrive.
    """
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(adapter, transport=provider, payable_bases=[])
    async with session:
        response = await session.post(
            f"{AUTHORISED}/chat/completions", json={"messages": []}
        )

    assert response.status_code == 402
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_a_sibling_host_is_not_covered_by_a_prefix():
    """Authorisation is per origin, not a string prefix.

    `https://paid.example.com.evil.test` starts with the authorised
    text. Matching on the raw string would hand a lookalike host the
    wallet.
    """
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter, transport=provider, payable_bases=[AUTHORISED]
    )
    async with session:
        response = await session.post(
            "https://paid.example.com.evil.test/v1/chat/completions",
            json={"messages": []},
        )

    assert response.status_code == 402
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_a_path_outside_the_authorised_base_does_not_pay():
    """The authorised prefix is a base URL, not a whole host.

    An operator authorising `https://host/v1` did not authorise
    `https://host/admin`, and a provider serving both should only be
    able to charge on the half that was named.
    """
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter, transport=provider, payable_bases=[AUTHORISED]
    )
    async with session:
        response = await session.post(
            "https://paid.example.com/admin/chat/completions", json={"messages": []}
        )

    assert response.status_code == 402
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_the_installed_session_carries_the_scope():
    """`maybe_install_payment_session` is what the server actually calls.

    The scope has to survive that call, not only a direct
    `install_payment_session`. Dropping it there is the whole original
    defect wearing a config flag, and the session it installs is the
    process-global one every request then travels on.
    """
    from routellm.payment.transport import maybe_install_payment_session

    provider = ChargingProvider()
    gateway = maybe_install_payment_session(
        provider="x402",
        wallet_key=TEST_KEY,
        networks=["base-sepolia"],
        payable_bases=[AUTHORISED],
    )
    assert gateway is not None

    # Swap in the stand-in provider underneath the installed session,
    # so no socket opens and the scope is still the installed one.
    session = litellm.aclient_session
    session._transport._transport = provider

    refused = await session.post(
        f"{UNAUTHORISED}/chat/completions", json={"messages": []}
    )
    paid = await session.post(
        f"{AUTHORISED}/chat/completions", json={"messages": []}
    )

    assert refused.status_code == 402
    assert paid.status_code == 200
    assert provider.paid_urls == [f"{AUTHORISED}/chat/completions"]


@pytest.mark.asyncio
async def test_payments_enabled_with_no_payable_endpoint_signs_nothing():
    """The flag alone authorises nobody, through the real entry point.

    Turning payments on must not retroactively make every configured
    endpoint payable, so the default `payable_bases` has to be empty
    rather than absent.
    """
    from routellm.payment.transport import maybe_install_payment_session

    provider = ChargingProvider()
    maybe_install_payment_session(
        provider="x402", wallet_key=TEST_KEY, networks=["base-sepolia"]
    )

    session = litellm.aclient_session
    session._transport._transport = provider

    response = await session.post(
        f"{AUTHORISED}/chat/completions", json={"messages": []}
    )

    assert response.status_code == 402
    assert provider.paid_urls == []
