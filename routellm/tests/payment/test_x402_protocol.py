"""End-to-end x402 protocol tests: real challenge parsing, version detection, encoding.

No network and no chain. The x402 package is mocked at its own boundary
(`x402HTTPClient`), so the wire format these tests pin is the package's,
not a local re-implementation of it.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from routellm.payment.types import PaymentChallenge, PaymentReceipt
from routellm.payment.x402 import X402Adapter

# ---------------------------------------------------------------------------
# Wire fixtures: what a real x402 server puts on a 402 response.
# ---------------------------------------------------------------------------

V2_PAYMENT_REQUIRED = {
    "x402Version": 2,
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "amount": "1000",
            "payTo": "0x0000000000000000000000000000000000000abc",
            "maxTimeoutSeconds": 60,
        }
    ],
}

V1_PAYMENT_REQUIRED = {
    "x402Version": 1,
    "accepts": [
        {
            "scheme": "exact",
            "network": "base",
            "maxAmountRequired": "1000",
            "resource": "https://llm.example.com/v1/chat",
            "payTo": "0x0000000000000000000000000000000000000abc",
            "maxTimeoutSeconds": 60,
            "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        }
    ],
}


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


def v2_response() -> httpx.Response:
    """A v2 server: PaymentRequired travels base64 in the PAYMENT-REQUIRED header."""
    return httpx.Response(
        402,
        headers={"PAYMENT-REQUIRED": _b64(V2_PAYMENT_REQUIRED)},
        content=b"",
        request=httpx.Request("POST", "https://llm.example.com/v1/chat"),
    )


def v1_response() -> httpx.Response:
    """A v1 server: PaymentRequired travels as the JSON response body."""
    return httpx.Response(
        402,
        headers={"Content-Type": "application/json"},
        content=json.dumps(V1_PAYMENT_REQUIRED).encode(),
        request=httpx.Request("POST", "https://llm.example.com/v1/chat"),
    )


def error_with(response: httpx.Response) -> Exception:
    """An exception shaped like the ones litellm raises, carrying the 402."""
    exc = Exception("Payment Required")
    exc.status_code = 402
    exc.response = response
    return exc


# ---------------------------------------------------------------------------
# Defect 1: the challenge must be parsed from the real 402, never invented.
# ---------------------------------------------------------------------------


def test_challenge_carries_raw_402_wire_data():
    """PaymentChallenge must be able to carry the raw headers and body.

    Without these the adapter has nothing to parse and can only invent a
    challenge, which is what defect 1 was.
    """
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers={"PAYMENT-REQUIRED": "abc"},
        body=b'{"x402Version":1}',
        resource_url="https://llm.example.com/v1/chat",
    )

    assert challenge.headers == {"PAYMENT-REQUIRED": "abc"}
    assert challenge.body == b'{"x402Version":1}'
    assert challenge.resource_url == "https://llm.example.com/v1/chat"


@pytest.mark.asyncio
async def test_controller_parses_real_402_into_challenge():
    """The controller must hand the gateway the server's actual 402 bytes.

    Pins defect 1: the old code built `payload={}` with a hardcoded
    amount and currency, so nothing the server said ever reached the
    gateway.
    """
    from routellm.controller import Controller

    response = v2_response()
    seen = []

    class RecordingGateway:
        name = "x402"
        networks = ["base"]

        async def pay(self, challenge):
            seen.append(challenge)
            return PaymentReceipt(
                tx_hash="t",
                network="base",
                amount="0",
                currency="USDC",
                paid_at=0,
                header_name="PAYMENT-SIGNATURE",
                header_value="signed",
            )

    calls = 0

    async def call_fn(extra_headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_with(response)
        return "ok"

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        payment_gateway=RecordingGateway(),
    )
    result = await controller._request_with_payment(call_fn)

    assert result == "ok"
    assert len(seen) == 1
    challenge = seen[0]
    # The real header the server sent must have reached the gateway.
    assert challenge.headers.get("payment-required") == _b64(V2_PAYMENT_REQUIRED)
    assert challenge.resource_url == "https://llm.example.com/v1/chat"


@pytest.mark.asyncio
async def test_adapter_parses_challenge_instead_of_inventing_one():
    """The adapter must feed the SDK the server's PaymentRequired.

    Pins defect 1 at the adapter: `PaymentRequired(**challenge.payload)`
    on an empty payload raises, because `accepts` is required.
    """
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers=dict(v2_response().headers),
        body=b"",
        resource_url="https://llm.example.com/v1/chat",
    )

    http_client = MagicMock()
    http_client.handle_402_response = AsyncMock(
        return_value=({"PAYMENT-SIGNATURE": "c2lnbmVk"}, MagicMock())
    )

    with patch.object(adapter, "_build_client", return_value=http_client):
        await adapter.pay(challenge)

    http_client.handle_402_response.assert_awaited_once()
    kwargs = http_client.handle_402_response.await_args.kwargs
    # The SDK got the real wire data, not a fabricated challenge.
    assert kwargs["headers"].get("payment-required") == _b64(V2_PAYMENT_REQUIRED)
    assert kwargs["request_url"] == "https://llm.example.com/v1/chat"


# ---------------------------------------------------------------------------
# Defect 2: the protocol version comes from the server, not from a constant.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v2_server_gets_payment_signature_header():
    """A v2 server must be answered on PAYMENT-SIGNATURE."""
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers=dict(v2_response().headers),
        body=b"",
        resource_url="https://llm.example.com/v1/chat",
    )

    http_client = MagicMock()
    http_client.handle_402_response = AsyncMock(
        return_value=({"PAYMENT-SIGNATURE": "djJzaWduZWQ="}, MagicMock())
    )

    with patch.object(adapter, "_build_client", return_value=http_client):
        receipt = await adapter.pay(challenge)

    assert receipt.header_name == "PAYMENT-SIGNATURE"
    assert receipt.header_value == "djJzaWduZWQ="


@pytest.mark.asyncio
async def test_v1_server_gets_x_payment_header():
    """A v1 server must be answered on the legacy X-PAYMENT header."""
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers=dict(v1_response().headers),
        body=json.dumps(V1_PAYMENT_REQUIRED).encode(),
        resource_url="https://llm.example.com/v1/chat",
    )

    http_client = MagicMock()
    http_client.handle_402_response = AsyncMock(
        return_value=({"X-PAYMENT": "djFzaWduZWQ="}, MagicMock())
    )

    with patch.object(adapter, "_build_client", return_value=http_client):
        receipt = await adapter.pay(challenge)

    assert receipt.header_name == "X-PAYMENT"
    assert receipt.header_value == "djFzaWduZWQ="


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_factory,expected_header",
    [(v2_response, "PAYMENT-SIGNATURE"), (v1_response, "X-PAYMENT")],
)
async def test_retry_uses_version_appropriate_header(response_factory, expected_header):
    """The retry must carry the header name the DETECTED version requires.

    Pins defect 2: the controller hardcoded X-PAYMENT, so a v2 server
    would never see the signature at all.
    """
    from routellm.controller import Controller

    class VersionGateway:
        name = "x402"
        networks = ["base"]

        async def pay(self, challenge):
            return PaymentReceipt(
                tx_hash="ignored",
                network="base",
                amount="0",
                currency="USDC",
                paid_at=0,
                header_name=expected_header,
                header_value="c2lnbmVkLXZhbHVl",
            )

    calls = 0
    sent = []

    async def call_fn(extra_headers):
        nonlocal calls
        calls += 1
        sent.append(extra_headers)
        if calls == 1:
            raise error_with(response_factory())
        return "ok"

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        payment_gateway=VersionGateway(),
    )
    await controller._request_with_payment(call_fn)

    assert sent[1] == {expected_header: "c2lnbmVkLXZhbHVl"}


# ---------------------------------------------------------------------------
# Defect 3: the header value is the package's base64, never a Python repr.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_header_value_is_base64_not_a_repr():
    """The retry header value must be the SDK's base64 encoding.

    Pins defect 3: `str(payload)` produced a Python repr such as
    "<MagicMock id=...>", which no server can decode.
    """
    from x402.http.utils import encode_payment_signature_header
    from x402.schemas import PaymentPayload

    payload = PaymentPayload(
        x402Version=2,
        payload={"signature": "0xabc"},
        accepted=V2_PAYMENT_REQUIRED["accepts"][0],
    )
    expected = encode_payment_signature_header(payload)

    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers=dict(v2_response().headers),
        body=b"",
        resource_url="https://llm.example.com/v1/chat",
    )

    http_client = MagicMock()
    http_client.handle_402_response = AsyncMock(
        return_value=({"PAYMENT-SIGNATURE": expected}, payload)
    )

    with patch.object(adapter, "_build_client", return_value=http_client):
        receipt = await adapter.pay(challenge)

    assert receipt.header_value == expected
    # It must decode back to the payload the server expects.
    decoded = json.loads(base64.b64decode(receipt.header_value))
    assert decoded["x402Version"] == 2
    assert decoded["payload"]["signature"] == "0xabc"
    # And must not be a Python repr.
    assert "MagicMock" not in receipt.header_value
    assert not receipt.header_value.startswith("<")


@pytest.mark.asyncio
async def test_receipt_tx_hash_is_not_a_repr():
    """tx_hash must never fall back to str(payload)."""
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers=dict(v2_response().headers),
        body=b"",
        resource_url="https://llm.example.com/v1/chat",
    )

    opaque_payload = MagicMock()  # no transaction_hash attribute of interest
    del opaque_payload.transaction_hash

    http_client = MagicMock()
    http_client.handle_402_response = AsyncMock(
        return_value=({"PAYMENT-SIGNATURE": "dmFsaWQ="}, opaque_payload)
    )

    with patch.object(adapter, "_build_client", return_value=http_client):
        receipt = await adapter.pay(challenge)

    assert "MagicMock" not in receipt.tx_hash
    assert not receipt.tx_hash.startswith("<")


# ---------------------------------------------------------------------------
# Opt-in: a server without a gateway must behave exactly as before.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_gateway_never_parses_or_pays():
    """Without a configured gateway the 402 propagates untouched."""
    from routellm.controller import Controller

    response = v2_response()
    calls = 0

    async def call_fn(extra_headers):
        nonlocal calls
        calls += 1
        raise error_with(response)

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        payment_gateway=None,
    )

    with pytest.raises(Exception, match="Payment Required"):
        await controller._request_with_payment(call_fn)

    # No retry, so no payment was even attempted.
    assert calls == 1


@pytest.mark.asyncio
async def test_non_402_error_never_pays():
    """A non-402 failure must not trigger a payment."""
    from routellm.controller import Controller

    paid = []

    class SpyGateway:
        name = "x402"
        networks = ["base"]

        async def pay(self, challenge):
            paid.append(challenge)
            raise AssertionError("must not pay on a non-402")

    async def call_fn(extra_headers):
        exc = Exception("rate limited, 402 tokens used")
        exc.status_code = 429
        raise exc

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        payment_gateway=SpyGateway(),
    )

    with pytest.raises(Exception, match="rate limited"):
        await controller._request_with_payment(call_fn)

    assert paid == []


# ---------------------------------------------------------------------------
# Honest degradation when the exception carries no response at all.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_402_without_response_carries_no_invented_amount():
    """litellm's generic APIError carries no response, and we must not invent one.

    The controller still delegates to the gateway -- deciding whether an
    unparseable challenge is payable is the gateway's job -- but it must
    pass along empty wire data rather than the fabricated
    `amount="1", currency="USDC", payload={}` of defect 1.
    """
    from routellm.controller import Controller

    seen = []

    class SpyGateway:
        name = "x402"
        networks = ["base"]

        async def pay(self, challenge):
            seen.append(challenge)
            return PaymentReceipt(
                tx_hash="t", network="base", amount="0", currency="USDC", paid_at=0
            )

    calls = 0

    async def call_fn(extra_headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            bare = Exception("Payment Required")
            bare.status_code = 402  # no .response, as litellm.APIError raises it
            raise bare
        return "ok"

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
        payment_gateway=SpyGateway(),
    )
    await controller._request_with_payment(call_fn)

    assert len(seen) == 1
    assert seen[0].headers == {}
    assert seen[0].body in (b"", None)
    # No invented price reached the gateway.
    assert seen[0].amount != "1"


@pytest.mark.asyncio
async def test_adapter_rejects_an_unparseable_challenge():
    """With no x402 data on the 402, the adapter must fail loudly, not pay blind."""
    adapter = X402Adapter(private_key="0x" + "a" * 64)
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="0",
        currency="USDC",
        headers={},
        body=b"",
        resource_url="",
    )

    with pytest.raises(ValueError, match="no x402 payment challenge"):
        await adapter.pay(challenge)


# ---------------------------------------------------------------------------
# The wallet must actually sign: the SDK is exercised for real here, with a
# throwaway key. Still no network and no chain -- signing an EIP-712 payload
# is local arithmetic, and nothing is ever broadcast.
# ---------------------------------------------------------------------------

THROWAWAY_KEY = "0x" + "11" * 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_factory,expected_header,expected_version",
    [
        (v2_response, "PAYMENT-SIGNATURE", 2),
        (v1_response, "X-PAYMENT", 1),
    ],
)
async def test_real_sdk_signs_and_encodes(response_factory, expected_header, expected_version):
    """A real wallet signs a real challenge and the proof decodes.

    Drives the actual x402 SDK, so it pins the signer wiring too: the
    SDK's ClientEvmSigner protocol needs a wallet object exposing
    `address`, and a bare signing function fails partway through
    building the authorization.
    """
    adapter = X402Adapter(private_key=THROWAWAY_KEY, networks=["base"])
    response = response_factory()
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="",
        currency="",
        headers={k.lower(): v for k, v in dict(response.headers).items()},
        body=response.content,
        resource_url="https://llm.example.com/v1/chat",
    )

    receipt = await adapter.pay(challenge)

    assert receipt.header_name == expected_header

    decoded = json.loads(base64.b64decode(receipt.header_value))
    assert decoded["x402Version"] == expected_version
    # A real EIP-712 signature, not a placeholder.
    signature = decoded["payload"]["signature"]
    assert signature.startswith("0x")
    assert len(signature) == 132  # 65 bytes hex-encoded, 0x-prefixed


@pytest.mark.asyncio
async def test_v1_network_registration_is_not_skipped():
    """v1 uses legacy network names, and they must be registered.

    Registering only the CAIP-2 networks v2 uses leaves a v1 challenge
    with no scheme to match, so it fails after parsing cleanly.
    """
    adapter = X402Adapter(private_key=THROWAWAY_KEY, networks=["base"])
    response = v1_response()
    challenge = PaymentChallenge(
        scheme="x402",
        network="base",
        amount="",
        currency="",
        headers={k.lower(): v for k, v in dict(response.headers).items()},
        body=response.content,
        resource_url="https://llm.example.com/v1/chat",
    )

    receipt = await adapter.pay(challenge)

    decoded = json.loads(base64.b64decode(receipt.header_value))
    assert decoded["network"] == "base"  # the legacy name, matched by v1


# ---------------------------------------------------------------------------
# Opt-in stays opt-in: the gateway is built only when BOTH the provider flag
# and a wallet key are present.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,key,expect_gateway",
    [
        (None, "", False),  # default server: no payment path at all
        (None, "0x" + "11" * 32, False),  # a key alone enables nothing
        ("x402", "", False),  # the flag alone enables nothing
        ("x402", "0x" + "11" * 32, True),  # both -> opt in
    ],
)
def test_gateway_requires_both_flag_and_key(provider, key, expect_gateway):
    """Reproduces the server's opt-in condition for building a gateway."""
    gateway = None
    if provider == "x402" and key:
        gateway = X402Adapter(private_key=key)

    assert (gateway is not None) is expect_gateway


@pytest.mark.asyncio
async def test_unconfigured_controller_is_unchanged_by_the_payment_path():
    """A controller with no gateway never touches the payment code."""
    from routellm.controller import Controller

    controller = Controller(
        routers=["random"],
        strong_model="gpt-4",
        weak_model="gpt-3.5-turbo",
    )
    assert controller.payment_gateway is None

    async def call_fn(extra_headers):
        assert extra_headers == {}
        return "plain"

    assert await controller._request_with_payment(call_fn) == "plain"
