"""How much a single payment may be, and which limit said no.

An authorised endpoint still names its own price: the 402 states the
amount, and signing it is unconditional once the endpoint is allowed to
charge. Authorisation answers *who* may charge; nothing yet answered
*how much*. A compromised or merely mispriced upstream inside the scope
can therefore name any figure and have it signed, bounded only by the
x402 SDK's own default.

Two layers bound it here. `--max-payment` is a process-wide ceiling no
endpoint may exceed, and an endpoint's own `max_payment:` may lower it
further but never raise it; the effective cap is the smaller of the
two. With two layers, a bare "payment refused" is not actionable, so a
refusal names the limit that refused it.

Units are the SDK's to convert. A cap is written in human money
("$0.01") while a challenge states an atomic on-chain amount against an
asset with its own decimals. `spend_controls.max_amount_per_payment`
takes the Money string and the SDK resolves it against the scheme's
default asset, so nothing here multiplies by a power of ten.
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
OTHER = "https://other.example.com/v1"

# USDC carries 6 decimals, so these atomic figures are $0.001 and $0.01.
CHEAP = "1000"
DEAR = "10000"


def challenge(resource: str, amount: str) -> dict:
    """A well-formed v2 PaymentRequired asking `amount` for `resource`."""
    return {
        "x402Version": 2,
        "resource": {"url": resource},
        "accepts": [
            {
                "scheme": "exact",
                "network": "eip155:84532",
                "asset": USDC_BASE_SEPOLIA,
                "amount": amount,
                "payTo": PAY_TO,
                "maxTimeoutSeconds": 60,
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }


class ChargingProvider(httpx.AsyncBaseTransport):
    """Charges `amount` for every request until a proof arrives."""

    def __init__(self, amount: str = CHEAP):
        self.amount = amount
        self.paid_urls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        proof = request.headers.get("PAYMENT-SIGNATURE") or request.headers.get("X-PAYMENT")
        if not proof:
            body = challenge(str(request.url), self.amount)
            return httpx.Response(
                402,
                headers={"PAYMENT-REQUIRED": base64.b64encode(json.dumps(body).encode()).decode()},
                json=body,
                request=request,
            )
        self.paid_urls.append(str(request.url))
        return httpx.Response(200, json={"ok": True}, request=request)


@pytest.fixture(autouse=True)
def restore_session():
    """litellm's session is process-global; never leak one between tests."""
    previous = litellm.aclient_session
    yield
    litellm.aclient_session = previous


# ---------------------------------------------------------------------
# The limits themselves: min(global, endpoint), and who refused.
# ---------------------------------------------------------------------


def test_the_global_ceiling_applies_to_an_endpoint_naming_none():
    """An endpoint that sets no cap of its own still gets the ceiling."""
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$0.01")

    assert limits.effective(AUTHORISED) == ("$0.01", "global")


def test_an_endpoint_may_lower_the_ceiling():
    """The smaller of the two wins, and it is named as the endpoint's."""
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$0.01", per_base={AUTHORISED: "$0.002"})

    assert limits.effective(AUTHORISED) == ("$0.002", "endpoint")


def test_an_endpoint_may_not_raise_the_ceiling():
    """A larger per-endpoint figure is clamped to the global one.

    This is the direction that matters: a config line must never widen
    what the operator set process-wide, or the ceiling is advisory.
    """
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$0.01", per_base={AUTHORISED: "$5"})

    assert limits.effective(AUTHORISED) == ("$0.01", "global")


def test_an_endpoint_cap_applies_with_no_global_ceiling():
    """Per-endpoint alone still binds; the ceiling is not a precondition."""
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(per_base={AUTHORISED: "$0.002"})

    assert limits.effective(AUTHORISED) == ("$0.002", "endpoint")


def test_neither_layer_set_keeps_the_sdk_default():
    """No cap anywhere means no cap of ours, never an unbounded one.

    Answering `None` here has to leave the SDK's own default standing.
    Substituting `spend_controls=False` would read as "no cap
    configured" and silently remove the only ceiling there was.
    """
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits()

    assert limits.effective(AUTHORISED) == (None, None)


def test_equal_layers_name_the_global_ceiling():
    """A tie is not the endpoint lowering anything, so the ceiling owns it."""
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$0.01", per_base={AUTHORISED: "$0.01"})

    assert limits.effective(AUTHORISED) == ("$0.01", "global")


def test_a_per_base_cap_is_matched_by_origin_and_prefix():
    """A cap keyed on a base URL covers the paths under it.

    The request URL is a full path, never the bare base, so a cap that
    only matched an exact string would never fire.
    """
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$1", per_base={AUTHORISED: "$0.002"})

    assert limits.effective(f"{AUTHORISED}/chat/completions") == (
        "$0.002",
        "endpoint",
    )


def test_a_cap_does_not_leak_to_another_base():
    """One endpoint's lower cap must not bind a different endpoint."""
    from routellm.payment.limits import PaymentLimits

    limits = PaymentLimits(global_cap="$1", per_base={AUTHORISED: "$0.002"})

    assert limits.effective(f"{OTHER}/chat/completions") == ("$1", "global")


# ---------------------------------------------------------------------
# Seam one: the scoped transport under litellm.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_challenge_under_the_cap_is_paid():
    """The cap binds without blocking a payment the operator allowed."""
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.01"),
    )
    async with session:
        response = await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert response.status_code == 200
    assert provider.paid_urls == [f"{AUTHORISED}/chat/completions"]


@pytest.mark.asyncio
async def test_a_challenge_over_the_global_ceiling_is_refused_by_name():
    """Over the ceiling nothing is signed, and the message says which one."""
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=DEAR)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.005"),
    )
    async with session:
        with pytest.raises(Exception) as caught:
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    message = str(caught.value)
    assert "--max-payment" in message
    assert "$0.005" in message
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_a_challenge_over_the_endpoint_cap_names_the_endpoint():
    """The lower of the two refused, so the message must not blame the other."""
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=DEAR)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$1", per_base={AUTHORISED: "$0.005"}),
    )
    async with session:
        with pytest.raises(Exception) as caught:
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    message = str(caught.value)
    assert "max_payment" in message
    assert "$0.005" in message
    # The ceiling did not refuse this; saying so would send the operator
    # to raise the wrong knob.
    assert "--max-payment" not in message
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_an_endpoint_cap_lower_than_the_global_one_wins():
    """Two endpoints, one ceiling: only the one that lowered it refuses.

    Both requests travel the one process-global session, so the cap has
    to be chosen per request from the URL, not fixed on the client.
    """
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED, OTHER],
        limits=PaymentLimits(global_cap="$0.01", per_base={AUTHORISED: "$0.0005"}),
    )
    async with session:
        with pytest.raises(Exception):
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})
        allowed = await session.post(f"{OTHER}/chat/completions", json={"messages": []})

    assert allowed.status_code == 200
    assert provider.paid_urls == [f"{OTHER}/chat/completions"]


@pytest.mark.asyncio
async def test_no_configured_cap_keeps_the_sdk_default_ceiling():
    """`pay: true` with no cap anywhere is still bounded.

    The SDK defaults to $1 per payment. Passing our "nothing
    configured" state through as `spend_controls=False` would turn the
    unconfigured case into the unbounded one.
    """
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    # $2, above the SDK's own $1 default and above nothing of ours.
    provider = ChargingProvider(amount="2000000")
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(),
    )
    async with session:
        with pytest.raises(Exception):
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_a_session_built_from_the_gateway_carries_its_limits():
    """`build_session` reads the cap off the gateway it belongs to.

    Since `install_payment_session` sets the limits onto the gateway
    rather than passing them alongside, that read is the whole path.
    A `build_session` ignoring them would leave every session
    uncapped while the gateway still reported a limit.
    """
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.scope import PaymentScope
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=DEAR)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    adapter.limits = PaymentLimits(global_cap="$0.005")

    session = adapter.build_session(transport=provider, scope=PaymentScope([AUTHORISED]))
    async with session:
        with pytest.raises(Exception) as caught:
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert "--max-payment" in str(caught.value)
    assert provider.paid_urls == []


@pytest.mark.asyncio
async def test_the_installed_session_carries_the_limits():
    """`maybe_install_payment_session` is what the server actually calls.

    A cap the helper honours but the real entry point drops is no cap
    at all, and this is the call that builds the process-global
    session every request then travels on.
    """
    from routellm.payment.limits import PaymentLimits
    from routellm.payment.transport import maybe_install_payment_session

    provider = ChargingProvider(amount=DEAR)
    gateway = maybe_install_payment_session(
        provider="x402",
        wallet_key=TEST_KEY,
        networks=["base-sepolia"],
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.005"),
    )
    assert gateway is not None

    session = litellm.aclient_session
    session._transport._transport = provider

    with pytest.raises(Exception) as caught:
        await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert "--max-payment" in str(caught.value)
    assert provider.paid_urls == []


# ---------------------------------------------------------------------
# Seam two: the controller's own 402 retry, below litellm's exceptions.
# ---------------------------------------------------------------------


def refusal():
    """A 402 shaped like the error litellm raises for one."""
    error = Exception("Payment Required")
    error.status_code = 402
    return error


def controller_with(gateway, config, weak, limits=None):
    """Build a controller over `config`'s endpoints with `gateway`.

    Caching is off: the SQLite cache is one file in the CWD shared by
    the whole suite, and a hit from another run would answer without
    ever reaching the payment seam.
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
        payment_limits=limits,
        cache_config=CacheConfig(enabled=False),
    )


CONFIG = {
    "endpoints": {
        "cheap": {
            "model": "gpt-4o",
            "api_base": AUTHORISED,
            "pay": True,
            "max_payment": "$0.002",
        },
        "plain": {
            "model": "gpt-4o-mini",
            "api_base": OTHER,
            "pay": True,
        },
    }
}


class CapturingGateway:
    """A gateway recording the cap each challenge arrived with."""

    def __init__(self):
        self.caps: list[tuple] = []

    async def pay(self, challenge):
        from routellm.payment.types import PaymentReceipt

        self.caps.append((challenge.max_amount, challenge.cap_source))
        return PaymentReceipt(
            tx_hash="0xpaid",
            network="base",
            amount="1.00",
            currency="USDC",
            paid_at=1,
        )

    @property
    def networks(self):
        return ["base"]

    @property
    def name(self):
        return "mock"


@pytest.mark.asyncio
async def test_the_error_path_carries_the_endpoint_cap():
    """The controller seam knows the endpoint, so it states its cap.

    This seam fires exactly when the transport declined, so a cap
    enforced only in the transport is a cap with a hole in it.
    """
    from routellm.payment.limits import PaymentLimits

    gateway = CapturingGateway()
    controller = controller_with(
        gateway, CONFIG, weak="cheap", limits=PaymentLimits(global_cap="$0.01")
    )

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        if len(calls) == 1:
            raise refusal()
        return "paid answer"

    result = await controller._request_with_payment(call, endpoint="cheap")

    assert result == "paid answer"
    assert gateway.caps == [("$0.002", "endpoint")]


@pytest.mark.asyncio
async def test_the_error_path_falls_back_to_the_global_ceiling():
    """An endpoint naming no cap still pays under the process ceiling."""
    from routellm.payment.limits import PaymentLimits

    gateway = CapturingGateway()
    controller = controller_with(
        gateway, CONFIG, weak="plain", limits=PaymentLimits(global_cap="$0.01")
    )

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        if len(calls) == 1:
            raise refusal()
        return "paid answer"

    await controller._request_with_payment(call, endpoint="plain")

    assert gateway.caps == [("$0.01", "global")]


@pytest.mark.asyncio
async def test_the_routed_request_path_carries_the_cap():
    """A real routed request has to reach the gateway with its cap.

    The helper can resolve the cap perfectly and change nothing if the
    call site never passes an endpoint whose cap can be looked up.
    """
    from unittest.mock import patch

    from routellm.payment.limits import PaymentLimits

    gateway = CapturingGateway()
    controller = controller_with(
        gateway, CONFIG, weak="cheap", limits=PaymentLimits(global_cap="$0.01")
    )

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
    assert gateway.caps == [("$0.002", "endpoint")]


@pytest.mark.asyncio
async def test_the_adapter_enforces_the_cap_the_challenge_states():
    """The real adapter, not a stand-in, refuses an over-cap challenge.

    The controller seam builds its own client per payment, so the cap
    has to reach `X402Adapter.pay` and be installed there.
    """
    from routellm.payment.types import PaymentChallenge
    from routellm.payment.x402 import X402Adapter

    body = challenge(f"{AUTHORISED}/chat/completions", DEAR)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    with pytest.raises(Exception) as caught:
        await adapter.pay(
            PaymentChallenge(
                scheme="x402",
                network="base-sepolia",
                amount="",
                currency="",
                headers={"payment-required": base64.b64encode(json.dumps(body).encode()).decode()},
                body=json.dumps(body).encode(),
                resource_url=f"{AUTHORISED}/chat/completions",
                max_amount="$0.005",
                cap_source="endpoint",
            )
        )

    message = str(caught.value)
    assert "max_payment" in message
    assert "$0.005" in message


@pytest.mark.asyncio
async def test_the_adapter_pays_a_challenge_under_the_cap():
    """The same adapter and cap, a cheaper challenge: signed."""
    from routellm.payment.types import PaymentChallenge
    from routellm.payment.x402 import X402Adapter

    body = challenge(f"{AUTHORISED}/chat/completions", CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    receipt = await adapter.pay(
        PaymentChallenge(
            scheme="x402",
            network="base-sepolia",
            amount="",
            currency="",
            headers={"payment-required": base64.b64encode(json.dumps(body).encode()).decode()},
            body=json.dumps(body).encode(),
            resource_url=f"{AUTHORISED}/chat/completions",
            max_amount="$0.005",
            cap_source="endpoint",
        )
    )

    assert receipt.header_value


# ---------------------------------------------------------------------
# Config and CLI: where the two layers are actually written.
# ---------------------------------------------------------------------


def test_an_endpoint_declares_its_own_cap():
    """`max_payment:` is config, read off the endpoint that wrote it."""
    from routellm.endpoints import EndpointRegistry

    registry = EndpointRegistry.from_config(CONFIG)

    assert registry.get("cheap").max_payment == "$0.002"
    assert registry.get("plain").max_payment is None


def test_a_malformed_cap_is_refused_at_load():
    """A cap nobody can parse must fail loudly, not fall back to none.

    Accepting "lots" and quietly dropping it leaves the operator
    believing an endpoint is capped when it is not.
    """
    import pytest as _pytest

    from routellm.endpoints import EndpointRegistry

    with _pytest.raises(Exception):
        EndpointRegistry.from_config(
            {
                "endpoints": {
                    "bad": {
                        "model": "gpt-4o",
                        "api_base": AUTHORISED,
                        "pay": True,
                        "max_payment": "lots",
                    }
                }
            }
        )


def test_a_cap_on_an_unpayable_endpoint_is_refused_at_load():
    """A cap without `pay:` is a config that cannot mean what it says.

    It reads as a bound on spending at an endpoint that can never
    spend, which is at best dead config and at worst a belief that
    `max_payment:` is what enables paying.
    """
    import pytest as _pytest

    from routellm.endpoints import EndpointRegistry

    with _pytest.raises(Exception):
        EndpointRegistry.from_config(
            {
                "endpoints": {
                    "nopay": {
                        "model": "gpt-4o",
                        "api_base": AUTHORISED,
                        "max_payment": "$0.01",
                    }
                }
            }
        )


def test_the_registry_reports_per_base_caps():
    """The scope is keyed on base URLs, so the caps must be too."""
    from routellm.endpoints import EndpointRegistry

    registry = EndpointRegistry.from_config(CONFIG)

    assert registry.payment_caps() == {AUTHORISED: "$0.002"}


@pytest.mark.parametrize("tight_first", [True, False])
def test_two_endpoints_on_one_base_enforce_the_lowest_cap(tight_first):
    """One base URL, one cap: the lower one, whichever was written first.

    The scope collapses endpoints sharing a base into one
    authorisation, so there is a single cap to enforce for it. Taking
    the higher would let one endpoint's config quietly raise another's
    limit -- and taking whichever came last would make the config's
    line order decide the limit, which is the same bug wearing a
    passing test on one ordering.
    """
    from routellm.endpoints import EndpointRegistry

    loose = {
        "model": "gpt-4o",
        "api_base": AUTHORISED,
        "pay": True,
        "max_payment": "$0.05",
    }
    tight = {
        "model": "gpt-4o-mini",
        "api_base": AUTHORISED,
        "pay": True,
        "max_payment": "$0.002",
    }
    ordered = {"tight": tight, "loose": loose} if tight_first else {"loose": loose, "tight": tight}

    registry = EndpointRegistry.from_config({"endpoints": ordered})

    assert registry.payment_caps() == {AUTHORISED: "$0.002"}


def test_a_cap_is_not_reported_for_an_endpoint_that_cannot_pay():
    """`payment_caps` reads `pay:` itself rather than trusting the loader.

    Config validation already refuses a cap without `pay:`, so this
    state only arises when an endpoint is built directly. Reporting a
    cap for it would key the limits on a base URL the scope never
    authorised -- a cap that looks enforced and binds nothing.
    """
    from routellm.endpoints import Endpoint, EndpointRegistry

    registry = EndpointRegistry.from_config(
        {"endpoints": {"plain": {"model": "gpt-4o", "api_base": OTHER}}}
    )
    unpayable = Endpoint.model_construct(
        name="sneaky",
        model="gpt-4o",
        api_base=AUTHORISED,
        pay=False,
        max_payment="$0.002",
    )
    registry._endpoints["sneaky"] = unpayable

    assert registry.payment_caps() == {}


def test_the_server_builds_limits_from_the_flag_and_the_config():
    """`--max-payment` and `max_payment:` meet in one object.

    `routellm.openai_server` parses `sys.argv` at import, so this runs
    in a subprocess with the flag actually on the command line. That
    is the only way to pin that the flag exists, is named what the
    refusal message tells operators to change, and reaches the limits
    -- rather than testing a helper the server never calls with it.
    """
    import json
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.argv = ["openai_server", "--max-payment", "$0.01"]
        import routellm.openai_server as server
        from routellm.endpoints import EndpointRegistry

        registry = EndpointRegistry.from_config({CONFIG!r})
        limits = server.payment_limits_for(
            registry,
            max_payment=server.args.max_payment,
            default_base=server.args.base_url,
        )
        print(json.dumps({{
            "capped": limits.effective({AUTHORISED!r} + "/chat/completions"),
            "uncapped": limits.effective({OTHER!r} + "/chat/completions"),
        }}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr

    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen["capped"] == ["$0.002", "endpoint"]
    assert seen["uncapped"] == ["$0.01", "global"]


def test_the_server_hands_the_limits_to_both_seams():
    """The lifespan is what wires them; a limits object it drops caps nothing.

    Two seams pay a 402 and both have to be capped, so this pins the
    installed session and the controller rather than the helper that
    computes the numbers. Wiring left untested is exactly how a cap
    ends up correct and unreachable.
    """
    import json
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import asyncio, json, sys
        sys.argv = [
            "openai_server", "--max-payment", "$0.01",
            "--payment-provider", "x402", "--wallet-key-env", "TEST_WALLET",
            "--routers", "random",
        ]
        import os
        os.environ["TEST_WALLET"] = {TEST_KEY!r}

        import routellm.openai_server as server
        from routellm.endpoints import EndpointRegistry

        server.build_registry = (
            lambda file_config, origin: EndpointRegistry.from_config({CONFIG!r})
        )
        server.load_config = lambda explicit=None: type(
            "Loaded", (), {{"layers": [], "data": {{}}}}
        )()

        async def main():
            async with server.lifespan(None):
                import litellm
                transport = litellm.aclient_session._transport
                controller = server.CONTROLLER
                print(json.dumps({{
                    "session_capped": transport._limits_probe(
                        {AUTHORISED!r} + "/chat/completions"
                    ),
                    "controller_capped": list(
                        controller._payment_cap("cheap")
                    ),
                    "controller_default": list(
                        controller._payment_cap("plain")
                    ),
                }}))

        asyncio.run(main())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr

    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen["session_capped"] == ["$0.002", "endpoint"]
    assert seen["controller_capped"] == ["$0.002", "endpoint"]
    assert seen["controller_default"] == ["$0.01", "global"]
