"""How much the wallet may spend in total, across every payment.

A per-payment cap bounds one payment. It says nothing about how many
payments there are, so N requests each comfortably under the cap spend
N times the cap. An upstream inside the scope that is compromised,
misconfigured or merely mispriced cannot name a figure above the cap,
but it can name one just under it again and again, and nothing so far
counts.

`--payment-budget` is that count. One figure for the process, debited
at both seams a payment can be authorised at, refusing once the
remainder no longer covers what the next payment would authorise.

What is debited is the *authorised* amount, never a settled one. At
signing nothing has settled on chain -- a signed PaymentPayload has no
transaction hash -- so the only figure knowable at either seam is the
amount the wallet was authorised to spend, which is the effective cap
for that payment. That is deliberately conservative: a provider that
challenges but never settles still consumes budget, so the ledger
under-spends rather than over-spends.

The budget is process-wide and in memory, exactly as `--max-payment`
is. A restart clears it.
"""

import asyncio
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
# The ledger itself: what it admits, what it refuses, what it reports.
# ---------------------------------------------------------------------


def test_an_unset_budget_admits_everything():
    """No `--payment-budget` means no cumulative limit at all.

    Per-payment caps still apply; this layer simply abstains. An
    unset budget that silently defaulted to some figure would refuse
    payments an operator never asked to bound.
    """
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget()

    for _ in range(100):
        assert budget.debit("$1000") is None
    assert budget.remaining is None


def test_a_zero_budget_refuses_the_first_payment():
    """Zero is a budget of nothing, which is not the same as unset."""
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0")

    assert budget.remaining == "$0"
    refusal = budget.debit("$0.001")
    assert refusal is not None
    assert "budget" in refusal.lower()


def test_payments_draw_the_budget_down_until_it_no_longer_covers_one():
    """Each debit reduces the remainder; the one that overruns is refused."""
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    assert budget.debit("$0.004") is None
    assert budget.remaining == "$0.006"
    assert budget.debit("$0.004") is None
    assert budget.remaining == "$0.002"

    refusal = budget.debit("$0.004")
    assert refusal is not None
    # The refused payment must not be debited, or a refusal would
    # consume budget it never spent.
    assert budget.remaining == "$0.002"


def test_a_payment_that_exactly_exhausts_the_budget_is_allowed():
    """Spending the last of it is spending within it, not beyond it."""
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    assert budget.debit("$0.01") is None
    assert budget.remaining == "$0"
    assert budget.debit("$0.000001") is not None


def test_the_refusal_names_the_budget_and_both_amounts():
    """A budget refusal is not a cap refusal and must not read like one.

    A per-payment cap names a limit an operator could raise for this
    one call. An exhausted budget is the process's whole allowance
    gone, so the message has to say which, and state how much is left
    against how much was asked.
    """
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")
    budget.debit("$0.009")

    refusal = budget.debit("$0.005")

    assert "budget" in refusal.lower()
    assert "--payment-budget" in refusal
    assert "$0.001" in refusal  # remaining
    assert "$0.005" in refusal  # requested
    assert "$0.01" in refusal  # the budget as written

    # The figure and the knob have to sit together. Stating the total
    # without naming what set it leaves an operator knowing the number
    # and not where to change it, so the budget is quoted as the value
    # of `--payment-budget` rather than as a bare amount.
    assert "$0.01 set by --payment-budget" in refusal


def test_a_malformed_budget_is_refused_when_it_is_built():
    """A budget nobody can parse must not be silently dropped.

    Dropping it would leave the operator believing a total limit is in
    force when none is.
    """
    from routellm.payment.limits import PaymentBudget

    with pytest.raises(ValueError):
        PaymentBudget("half a dollar")


def test_a_budget_naming_an_asset_is_refused():
    """A budget is stated in USD, like every other money figure here."""
    from routellm.payment.limits import PaymentBudget

    with pytest.raises(ValueError):
        PaymentBudget("0.01 USDT")


# ---------------------------------------------------------------------
# Concurrency: two payments racing the last of the budget.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_debits_cannot_both_take_the_last_of_the_budget():
    """Check and debit are one step, so the remainder never goes negative.

    Merely running two coroutines proves nothing: `debit` never
    awaits, so under a single event loop it cannot be interrupted no
    matter how many callers race it, and a ledger that read the
    remainder outside its lock would still pass.

    The interleaving therefore has to be forced from inside the
    critical section. Patching `parse_cap` -- which `debit` calls
    between reading the remainder and writing it back -- lets the
    second caller in at exactly the moment a split check-and-debit
    would be wrong. Both then see the same remainder and both spend
    it, unless the two steps are genuinely one.
    """
    import threading

    from routellm.payment import limits as limits_module
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    real_parse = limits_module.parse_cap
    entered = threading.Event()
    release = threading.Event()
    first = True

    def slow_parse(value):
        nonlocal first
        if first:
            first = False
            # Hold the first debit open between its read and its
            # write, and let the second one run into it.
            entered.set()
            release.wait(timeout=5)
        return real_parse(value)

    limits_module.parse_cap = slow_parse
    try:
        results = []

        def spend():
            results.append(budget.debit("$0.006"))

        a = threading.Thread(target=spend)
        a.start()
        entered.wait(timeout=5)

        b = threading.Thread(target=spend)
        b.start()
        # The second caller is now either blocked on the lock (correct)
        # or already past a stale read (racy). Let the first finish.
        release.set()
        a.join(timeout=5)
        b.join(timeout=5)
    finally:
        limits_module.parse_cap = real_parse
        release.set()

    # Together they ask $0.012 against $0.01, so exactly one wins.
    assert results.count(None) == 1
    assert budget.remaining == "$0.004"


def test_a_debit_is_visible_the_moment_its_caller_is_told_it_succeeded():
    """The write lands before the lock is released, not after.

    A ledger that decided under the lock and then wrote outside it
    would admit two payments that only one remainder covers: each
    decides against a remainder the other has not yet reduced.

    `parse_cap` is the last call a debit makes while still deciding,
    so holding the first caller there and running the second to
    completion reproduces exactly that ordering. Under a correct
    ledger the second blocks on the lock; under a split one it decides
    against a stale remainder and both succeed.
    """
    import threading

    from routellm.payment import limits as limits_module
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    real_parse = limits_module.parse_cap
    deciding = threading.Event()
    go = threading.Event()
    armed = [True]

    def slow_parse(value):
        if armed[0]:
            armed[0] = False
            parsed = real_parse(value)
            # Decided but not yet written: let the other caller run
            # the whole way through from here.
            deciding.set()
            go.wait(timeout=5)
            return parsed
        return real_parse(value)

    results = []

    def spend():
        results.append(budget.debit("$0.006"))

    limits_module.parse_cap = slow_parse
    try:
        first = threading.Thread(target=spend)
        first.start()
        deciding.wait(timeout=5)

        second = threading.Thread(target=spend)
        second.start()
        # Give the second caller time to finish if nothing excludes it.
        second.join(timeout=2)
        go.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        limits_module.parse_cap = real_parse
        go.set()

    # $0.012 asked against $0.01: exactly one may be admitted, and the
    # remainder must never go negative.
    assert results.count(None) == 1
    assert budget.remaining == "$0.004"


@pytest.mark.asyncio
async def test_many_concurrent_debits_never_overspend_the_budget():
    """Under real contention the total admitted never exceeds the budget."""
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.10")

    async def spend():
        await asyncio.sleep(0)
        return budget.debit("$0.01")

    results = await asyncio.gather(*(spend() for _ in range(50)))

    # $0.10 buys exactly ten payments of $0.01 and no more.
    assert results.count(None) == 10
    assert budget.remaining == "$0"


def test_concurrent_debits_across_threads_never_overspend():
    """The sync `completion()` path is not on any event loop.

    An `asyncio.Lock` would bind to one loop and exclude nothing
    between threads, so the ledger has to hold under plain threading
    too. Every thread is released from one barrier, so the debits
    genuinely overlap rather than queueing behind each other.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.10")
    start = threading.Barrier(32)

    def spend(_):
        start.wait(timeout=5)
        return budget.debit("$0.01")

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(spend, range(32)))

    # $0.10 buys exactly ten payments of $0.01 and no more.
    assert results.count(None) == 10
    assert budget.remaining == "$0"


def test_the_ledger_refuses_to_be_built_on_a_loop_bound_primitive():
    """An `asyncio.Lock` would silently stop excluding off its own loop.

    The ledger is reached from the sync path, which is on no loop at
    all, and from whichever loop a caller happens to drive
    `acompletion` with. A primitive created on one loop does not
    exclude callers on another, so the mutex has to be a plain
    threading one.
    """
    import asyncio as _asyncio

    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    assert not isinstance(budget._lock, _asyncio.Lock)


def test_the_ledger_holds_across_separate_event_loops():
    """A library caller may drive each request from its own loop.

    `asyncio.run` per call is an ordinary way to use an async API from
    sync code, and the ledger outlives any one loop. A primitive bound
    to a loop would raise or silently stop excluding here.
    """
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    async def spend():
        return budget.debit("$0.004")

    assert asyncio.run(spend()) is None
    assert asyncio.run(spend()) is None
    assert asyncio.run(spend()) is not None
    assert budget.remaining == "$0.002"


# ---------------------------------------------------------------------
# Seam one: the installed session's transport.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sequence_of_payments_fits_inside_the_budget():
    """Two payments the budget covers both go through."""
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=PaymentBudget("$0.002"),
    )
    async with session:
        for _ in range(2):
            response = await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})
            assert response.status_code == 200

    assert len(provider.paid_urls) == 2


@pytest.mark.asyncio
async def test_the_budget_exhausts_part_way_through_a_sequence():
    """The payment that would overrun is refused, and nothing is signed.

    This is the whole point of the layer: every one of these is under
    the per-payment cap, so the cap alone would sign all of them.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=PaymentBudget("$0.002"),
    )

    async with session:
        await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})
        await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

        with pytest.raises(Exception) as caught:
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    message = str(caught.value)
    assert "budget" in message.lower()
    assert "--payment-budget" in message
    # The third was refused before signing, so the provider never saw
    # a proof for it.
    assert len(provider.paid_urls) == 2


@pytest.mark.asyncio
async def test_a_refused_payment_does_not_consume_the_budget():
    """An over-cap payment is never authorised, so it spends nothing.

    Debiting a payment the cap refused would let a mispriced upstream
    drain the budget without ever being paid.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=DEAR)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    budget = PaymentBudget("$0.01")

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.005"),
        budget=budget,
    )
    async with session:
        with pytest.raises(Exception):
            await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert provider.paid_urls == []
    assert budget.remaining == "$0.01"


@pytest.mark.asyncio
async def test_a_request_outside_the_scope_never_touches_the_budget():
    """An unpayable 402 comes back unpaid, so it spends nothing."""
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    budget = PaymentBudget("$0.01")

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=budget,
    )
    async with session:
        response = await session.post(f"{OTHER}/chat/completions", json={"messages": []})

    assert response.status_code == 402
    assert budget.remaining == "$0.01"


@pytest.mark.asyncio
async def test_concurrent_requests_race_the_last_of_the_budget():
    """Two real requests, overlapping, with budget for only one.

    The session is the process-global one every request travels on, so
    this is the race as a user meets it rather than as the ledger sees
    it.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=PaymentBudget("$0.001"),
    )

    async def call():
        return await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    async with session:
        outcomes = await asyncio.gather(call(), call(), return_exceptions=True)

    paid = [o for o in outcomes if not isinstance(o, BaseException)]
    refused = [o for o in outcomes if isinstance(o, BaseException)]

    assert len(paid) == 1
    assert len(refused) == 1
    assert "budget" in str(refused[0]).lower()
    assert len(provider.paid_urls) == 1


@pytest.mark.asyncio
async def test_the_installed_session_carries_the_budget():
    """`maybe_install_payment_session` is what the server actually calls.

    A budget the helper honours but the real entry point drops is no
    budget at all.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import maybe_install_payment_session

    provider = ChargingProvider(amount=CHEAP)
    gateway = maybe_install_payment_session(
        provider="x402",
        wallet_key=TEST_KEY,
        networks=["base-sepolia"],
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=PaymentBudget("$0.001"),
    )
    assert gateway is not None

    session = litellm.aclient_session
    session._transport._transport = provider

    await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    with pytest.raises(Exception) as caught:
        await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert "--payment-budget" in str(caught.value)
    assert len(provider.paid_urls) == 1


# ---------------------------------------------------------------------
# Seam two: the controller's own 402 retry, below litellm's exceptions.
# ---------------------------------------------------------------------


def refusal():
    """A 402 shaped like the error litellm raises for one."""
    error = Exception("Payment Required")
    error.status_code = 402
    return error


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


def controller_with(gateway, config, weak, limits=None, budget=None):
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
        payment_budget=budget,
        cache_config=CacheConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_the_controller_seam_debits_the_budget():
    """This seam fires exactly when the transport declined.

    A budget enforced in only one of the two is not a budget, so a
    payment signed here has to draw the same remainder down.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits

    budget = PaymentBudget("$0.01")
    gateway = CapturingGateway()
    controller = controller_with(
        gateway,
        CONFIG,
        weak="cheap",
        limits=PaymentLimits(global_cap="$0.01"),
        budget=budget,
    )

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        if len(calls) == 1:
            raise refusal()
        return "paid answer"

    result = await controller._request_with_payment(call, endpoint="cheap")

    assert result == "paid answer"
    # The endpoint's own $0.002 is what was authorised, so that is the
    # figure debited -- not the $0.01 ceiling.
    assert budget.remaining == "$0.008"


@pytest.mark.asyncio
async def test_the_controller_seam_refuses_once_the_budget_is_gone():
    """Exhausted here refuses like an over-cap payment does, and says so."""
    from routellm.payment.limits import PaymentBudget, PaymentLimits

    budget = PaymentBudget("$0.003")
    gateway = CapturingGateway()
    controller = controller_with(
        gateway,
        CONFIG,
        weak="cheap",
        limits=PaymentLimits(global_cap="$0.01"),
        budget=budget,
    )

    async def always_402(extra_headers):
        raise refusal()

    async def call_once(extra_headers):
        if not gateway.caps:
            raise refusal()
        return "paid answer"

    # First payment fits: $0.002 of $0.003.
    assert await controller._request_with_payment(call_once, endpoint="cheap") == ("paid answer")

    with pytest.raises(Exception) as caught:
        await controller._request_with_payment(always_402, endpoint="cheap")

    message = str(caught.value)
    assert "budget" in message.lower()
    assert "--payment-budget" in message
    # Nothing was signed for the refused one.
    assert len(gateway.caps) == 1


@pytest.mark.asyncio
async def test_both_seams_draw_down_one_shared_remainder():
    """One budget, two seams: spending at one leaves less for the other.

    Two ledgers would be two budgets, and the process would spend
    twice what the operator allowed.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    budget = PaymentBudget("$0.003")
    limits = PaymentLimits(global_cap="$0.001")

    provider = ChargingProvider(amount=CHEAP)
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=limits,
        budget=budget,
    )

    # Seam one spends $0.001 of the $0.003.
    async with session:
        await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})
    assert budget.remaining == "$0.002"

    # Seam two now sees the reduced remainder, and its own $0.002
    # endpoint cap exactly finishes it.
    gateway = CapturingGateway()
    controller = controller_with(
        gateway,
        CONFIG,
        weak="cheap",
        limits=PaymentLimits(global_cap="$0.01"),
        budget=budget,
    )

    calls = []

    async def call(extra_headers):
        calls.append(extra_headers)
        if len(calls) == 1:
            raise refusal()
        return "paid answer"

    await controller._request_with_payment(call, endpoint="cheap")
    assert budget.remaining == "$0"


@pytest.mark.asyncio
async def test_no_budget_leaves_the_controller_seam_paying_as_before():
    """Unset means this layer abstains; the cap still binds."""
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

    assert await controller._request_with_payment(call, endpoint="cheap") == ("paid answer")
    assert gateway.caps == [("$0.002", "endpoint")]


@pytest.mark.asyncio
async def test_a_402_that_was_never_paid_does_not_consume_the_budget():
    """A 402 coming back means nothing was signed, so nothing was spent.

    The payment cycle can decline for reasons of its own -- the SDK
    gives up, or the retry is refused again -- and the response is the
    402 the caller would have seen without a wallet. Keeping the
    reservation would let such an upstream drain the budget without
    ever being paid.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits
    from routellm.payment.transport import install_payment_session
    from routellm.payment.x402 import X402Adapter

    class NeverAccepts(httpx.AsyncBaseTransport):
        """Answers 402 even once a proof arrives."""

        def __init__(self):
            self.seen = 0

        async def handle_async_request(self, request):
            self.seen += 1
            body = challenge(str(request.url), CHEAP)
            return httpx.Response(
                402,
                headers={"PAYMENT-REQUIRED": base64.b64encode(json.dumps(body).encode()).decode()},
                json=body,
                request=request,
            )

    provider = NeverAccepts()
    adapter = X402Adapter(private_key=TEST_KEY, networks=["base-sepolia"])
    budget = PaymentBudget("$0.01")

    session = install_payment_session(
        adapter,
        transport=provider,
        payable_bases=[AUTHORISED],
        limits=PaymentLimits(global_cap="$0.001"),
        budget=budget,
    )
    async with session:
        response = await session.post(f"{AUTHORISED}/chat/completions", json={"messages": []})

    assert response.status_code == 402
    # The signature was never accepted, so the budget is untouched and
    # the next request still has the whole of it.
    assert budget.remaining == "$0.01"


@pytest.mark.asyncio
async def test_a_failed_payment_at_the_controller_seam_is_refunded():
    """`pay` raising means nothing was signed, so nothing was spent.

    Without the refund a gateway failing repeatedly -- a wallet with
    no funds, an unreachable facilitator -- would exhaust the budget
    having paid nobody, and later legitimate payments would be refused
    for spend that never happened.
    """
    from routellm.payment.limits import PaymentBudget, PaymentLimits

    class FailingGateway:
        """A gateway whose payments always raise."""

        def __init__(self):
            self.attempts = 0

        async def pay(self, challenge):
            self.attempts += 1
            raise RuntimeError("facilitator unreachable")

        @property
        def networks(self):
            return ["base"]

        @property
        def name(self):
            return "mock"

    budget = PaymentBudget("$0.01")
    gateway = FailingGateway()
    controller = controller_with(
        gateway,
        CONFIG,
        weak="cheap",
        limits=PaymentLimits(global_cap="$0.01"),
        budget=budget,
    )

    async def call(extra_headers):
        raise refusal()

    for _ in range(10):
        with pytest.raises(Exception):
            await controller._request_with_payment(call, endpoint="cheap")

    assert gateway.attempts == 10
    # Ten failed attempts at $0.002 would have spent $0.02 -- twice the
    # budget -- had the reservations been kept.
    assert budget.remaining == "$0.01"


def test_a_budget_refusal_reads_differently_from_a_cap_refusal():
    """The two refusals must not be confusable.

    A cap refusal names a limit that could be raised for this one
    call; an exhausted budget is the whole allowance for the process
    gone. An operator reading the wrong one turns the wrong knob, so
    each has to name its own.
    """
    from routellm.payment.limits import PaymentBudget, refusal_message

    cap_refusal = refusal_message("$0.01", "global", Exception("too dear"))

    budget = PaymentBudget("$0.01")
    budget.debit("$0.009")
    budget_refusal = budget.debit("$0.005")

    # The budget refusal names its own knob and not the cap's.
    assert "--payment-budget" in budget_refusal
    assert "--max-payment" not in budget_refusal

    # And the cap refusal does not claim a budget is exhausted.
    assert "--max-payment" in cap_refusal
    assert "budget" not in cap_refusal.lower()


def test_an_uncapped_payment_cannot_be_counted_against_a_budget():
    """With a budget set, a payment carrying no cap has no figure to debit.

    Admitting it would let an unbounded payment through a layer whose
    whole job is to bound the total, and debiting zero would record it
    as free. Refusing says which knob makes it countable.
    """
    from routellm.payment.limits import PaymentBudget

    budget = PaymentBudget("$0.01")

    refusal = budget.debit(None)

    assert refusal is not None
    assert "--max-payment" in refusal
    assert budget.remaining == "$0.01"


def test_an_unset_budget_still_admits_an_uncapped_payment():
    """With no budget there is nothing to count against, so nothing changes."""
    from routellm.payment.limits import PaymentBudget

    assert PaymentBudget().debit(None) is None


# ---------------------------------------------------------------------
# The flag, and the wiring that has to reach both seams.
# ---------------------------------------------------------------------


def in_server(argv, body):
    """Run `body` against a freshly imported server with `argv` on the line.

    `routellm.openai_server` parses `sys.argv` at import, so a flag can
    only be pinned in a subprocess with it actually on the command
    line. Testing a helper the server never calls with the real flag
    would pin nothing.
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        f"""
        import json, sys
        sys.argv = {argv!r}
        import routellm.openai_server as server
        """
    ) + textwrap.dedent(body)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_server_builds_a_budget_from_the_flag():
    """`--payment-budget` is where the figure is actually written."""
    seen = in_server(
        ["openai_server", "--payment-budget", "$0.25"],
        """
        budget = server.payment_budget_for(server.args.payment_budget)
        print(json.dumps({"remaining": budget.remaining}))
        """,
    )

    assert seen["remaining"] == "$0.25"


def test_the_server_builds_no_budget_when_the_flag_is_absent():
    """Unset is not zero, and must not become a budget of nothing."""
    seen = in_server(
        ["openai_server"],
        """
        budget = server.payment_budget_for(server.args.payment_budget)
        print(json.dumps({"budget": budget}))
        """,
    )

    assert seen["budget"] is None


def test_the_server_accepts_a_zero_budget_as_a_real_one():
    """Zero is a deliberate 'spend nothing', distinct from unset.

    The two must not collapse: unset leaves the total unbounded, while
    "$0" refuses the first payment.
    """
    seen = in_server(
        ["openai_server", "--payment-budget", "$0"],
        """
        budget = server.payment_budget_for(server.args.payment_budget)
        print(json.dumps({
            "remaining": budget.remaining,
            "refuses": budget.debit("$0.000001") is not None,
        }))
        """,
    )

    assert seen["remaining"] == "$0"
    assert seen["refuses"] is True


def test_the_flag_exists_and_defaults_to_unset():
    """A budget the CLI cannot express is a budget nobody can set."""
    seen = in_server(
        ["openai_server"],
        """
        actions = {
            option: action
            for action in server.parser._actions
            for option in action.option_strings
        }
        print(json.dumps({
            "present": "--payment-budget" in actions,
            "default": actions["--payment-budget"].default,
        }))
        """,
    )

    assert seen["present"] is True
    assert seen["default"] is None


def test_the_server_hands_one_budget_to_both_seams():
    """The wiring is the whole feature: one ledger, both seams.

    A budget built at startup and handed to only one of them leaves
    the other spending freely, which is the hole this layer exists to
    close. Run in a subprocess because it installs a process-global
    session and builds the real controller.
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]

    script = textwrap.dedent(
        f"""
        import asyncio, json, sys
        sys.path.insert(0, {str(repo_root)!r})
        sys.argv = [
            "routellm.openai_server",
            "--payment-provider", "x402",
            "--payment-budget", "$0.25",
            "--max-payment", "$0.01",
            "--base-url", {AUTHORISED!r},
        ]
        import os
        os.environ["ROUTELLM_WALLET_KEY"] = {TEST_KEY!r}

        from routellm import openai_server as server
        from routellm.endpoints import EndpointRegistry

        server.build_registry = staticmethod(
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
                session_budget = transport._budget_probe()
                print(json.dumps({{
                    "session_budget": session_budget,
                    "controller_budget": (
                        None if controller.payment_budget is None
                        else controller.payment_budget.remaining
                    ),
                    "shared": (
                        controller.payment_budget
                        is transport._budget_probe(obj=True)
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
    assert seen["session_budget"] == "$0.25"
    assert seen["controller_budget"] == "$0.25"
    # Not merely equal figures: the same ledger, or the process spends
    # the budget twice.
    assert seen["shared"] is True
