"""The 402 cycle runs below litellm, where headers and body survive.

litellm's exception mapper keeps a 402's status code and discards the
response it came on, so a challenge cannot be recovered from the error
it raises. These tests pin the payment cycle to an httpx transport,
underneath litellm, where the wire data is still intact.
"""
import base64
import json

import httpx
import pytest

# Real Base Sepolia USDC. The SDK's spend controls only sign for assets
# it recognises, so a made-up address is rejected before signing and the
# test would pass for the wrong reason.
USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"
PAY_TO = "0x" + "22" * 20
RESOURCE = "https://llm.example.com/v1/chat/completions"
TEST_KEY = "0x" + "11" * 32


def v2_challenge() -> dict:
    """A v2 PaymentRequired, whose price lives in the accepts entry."""
    return {
        "x402Version": 2,
        "resource": {"url": RESOURCE},
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


def v1_challenge() -> dict:
    """A v1 PaymentRequired: legacy network name, maxAmountRequired."""
    return {
        "x402Version": 1,
        "accepts": [
            {
                "scheme": "exact",
                "network": "base-sepolia",
                "maxAmountRequired": "1000",
                "resource": RESOURCE,
                "description": "llm",
                "mimeType": "application/json",
                "payTo": PAY_TO,
                "maxTimeoutSeconds": 60,
                "asset": USDC_BASE_SEPOLIA,
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }


class PayingProvider(httpx.AsyncBaseTransport):
    """A provider that refuses with 402 until a payment header arrives.

    Stands in for the network: no socket is opened and nothing settles
    on chain, but the 402 it returns is the real wire shape and the
    proof it accepts is a real signature over the challenge.
    """

    def __init__(self, challenge: dict, version: int):
        self.challenge = challenge
        self.version = version
        self.requests: list[httpx.Request] = []
        self.proofs: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        proof = request.headers.get("PAYMENT-SIGNATURE") or request.headers.get(
            "X-PAYMENT"
        )
        if not proof:
            headers = {}
            if self.version == 2:
                # v2 states the challenge on a base64 header; v1 only in
                # the body. Version detection reads exactly this.
                headers["PAYMENT-REQUIRED"] = base64.b64encode(
                    json.dumps(self.challenge).encode()
                ).decode()
            return httpx.Response(
                402, headers=headers, json=self.challenge, request=request
            )
        self.proofs.append(proof)
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


class AlwaysRefuses(httpx.AsyncBaseTransport):
    """Keeps returning 402 even once paid, to pin the retry cap."""

    def __init__(self, challenge: dict):
        self.challenge = challenge
        self.calls = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        return httpx.Response(
            402,
            headers={
                "PAYMENT-REQUIRED": base64.b64encode(
                    json.dumps(self.challenge).encode()
                ).decode()
            },
            json=self.challenge,
            request=request,
        )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.asyncio
async def test_transport_pays_and_retries_for_both_versions(version):
    """A 402 is signed and replayed, with the version the server chose.

    Nothing configures the version: the same adapter answers a v1 and a
    v2 challenge, because the SDK detects which one arrived.
    """
    from routellm.payment.x402 import X402Adapter

    challenge = v2_challenge() if version == 2 else v1_challenge()
    provider = PayingProvider(challenge, version)

    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    async with adapter.build_session(transport=provider) as session:
        response = await session.post(RESOURCE, json={"messages": []})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "paid answer"

    # Refused once, paid, retried once.
    assert len(provider.requests) == 2
    assert len(provider.proofs) == 1

    # The proof is a signed payload for the version the server spoke,
    # not a repr and not the version this client happens to prefer.
    decoded = json.loads(base64.b64decode(provider.proofs[0]))
    assert decoded["x402Version"] == version

    # The header name is the version's own: v2 signs on
    # PAYMENT-SIGNATURE, v1 on the legacy X-PAYMENT.
    retry_headers = provider.requests[1].headers
    expected = "PAYMENT-SIGNATURE" if version == 2 else "X-PAYMENT"
    assert retry_headers.get(expected)


@pytest.mark.asyncio
async def test_transport_carries_the_original_request_body():
    """The replay is the same request, not an empty one.

    A retry that drops the body would pay for a completion the provider
    never receives.
    """
    from routellm.payment.x402 import X402Adapter

    provider = PayingProvider(v2_challenge(), 2)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    async with adapter.build_session(transport=provider) as session:
        await session.post(RESOURCE, json=payload)

    assert json.loads(provider.requests[1].content) == payload


@pytest.mark.asyncio
async def test_transport_stops_at_the_sdk_retry_limit():
    """A provider that never accepts payment ends the cycle, not loops.

    Each 402 is answered by exactly one paid replay. The replay is
    marked as a retry, so a 402 coming back from it is returned rather
    than paid again -- otherwise a provider that always refuses would
    sign a fresh payment for every refusal.
    """
    from routellm.payment.x402 import X402Adapter

    provider = AlwaysRefuses(v2_challenge())
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    async with adapter.build_session(transport=provider) as session:
        response = await session.post(RESOURCE, json={"messages": []})

    # The last response is handed back as a 402 rather than retried
    # forever: one unpaid attempt plus one paid attempt.
    assert response.status_code == 402
    assert provider.calls == 2

    # The paid replay carried the retry marker. Without it the cycle has
    # nothing telling it the 402 it just got is the answer to a payment
    # it already made, and each refusal buys another one.
    assert provider.requests[-1].extensions.get("_x402_is_retry") is True


@pytest.mark.asyncio
async def test_retry_marker_stops_a_nested_payment_cascade():
    """A 402 on a paid replay is not paid a second time.

    litellm's session is shared, so a payment transport can end up
    sitting above another one. The retry marker is what stops the
    second from treating the first's paid replay as a fresh challenge
    and signing again; without it each layer multiplies the payments.
    """
    from x402.http.clients.httpx import x402AsyncTransport

    from routellm.payment.x402 import X402Adapter

    provider = AlwaysRefuses(v2_challenge())
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    nested = x402AsyncTransport(
        adapter._build_client(),
        x402AsyncTransport(adapter._build_client(), provider),
    )
    async with httpx.AsyncClient(transport=nested) as session:
        response = await session.post(RESOURCE, json={"messages": []})

    assert response.status_code == 402
    # Unpaid attempt, the inner layer's paid replay, the outer layer's
    # paid replay -- and then it stops, because that replay is marked.
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_completion_pays_below_litellm():
    """The whole cycle runs under litellm, through its own session hook.

    This is the behaviour the exception seam cannot provide: litellm
    never sees the 402 at all, because the transport settles it first.
    """
    import litellm

    from routellm.payment.x402 import X402Adapter
    from routellm.payment.transport import install_payment_session

    provider = PayingProvider(v2_challenge(), 2)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    previous = litellm.aclient_session
    try:
        install_payment_session(adapter, transport=provider)
        response = await litellm.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            api_base="https://llm.example.com/v1",
            num_retries=0,
        )
    finally:
        litellm.aclient_session = previous

    assert response.choices[0].message.content == "paid answer"
    assert len(provider.proofs) == 1
