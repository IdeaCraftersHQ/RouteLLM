"""Install the payment cycle underneath litellm, scoped to who may charge.

A 402 challenge lives in the response's headers and body, and paying it
means replaying the request. litellm's exception mapper keeps only the
status code -- the generic `APIError` it raises for a 402 carries no
`response` at all -- so nothing downstream of it can read a challenge,
let alone sign one. The payment cycle therefore belongs below litellm,
on the HTTP client it sends through.

`litellm.aclient_session` is litellm's own hook for that: when it is
set, litellm hands that exact `httpx.AsyncClient` to the provider SDK
and never closes it. Putting the x402 transport on that session means
litellm never sees the 402, because the transport has already paid it
and replayed the request by the time a response comes back.

That session is process-global, which is the whole reason a scope is
needed. One assignment puts every async request in the process on the
paying client, so without a scope a mixed deployment pays whichever
upstream answers 402 -- not only the one the operator meant to pay. The
scope is built from the endpoints that asked to pay, and a request
outside it never reaches the payment cycle at all.

The session is installed only for a server that asked to pay and has a
key to pay with. Anything else leaves litellm's own client in place,
unchanged.
"""

import logging

logger = logging.getLogger(__name__)


def install_payment_session(gateway, transport=None, payable_bases=None, limits=None, budget=None):
    """Route litellm's async requests through `gateway`'s paying client.

    Parameters
    ----------
    gateway : X402Adapter
        Gateway able to build a paying httpx client.
    transport : httpx.AsyncBaseTransport, optional
        Transport underneath the payment one. Defaults to httpx's own.
    payable_bases : iterable[str], optional
        Base URLs a payment may be signed for. An empty iterable
        authorises nothing, so every 402 comes back unpaid. None leaves
        the session unscoped and is only for a caller that has already
        narrowed the client to a single upstream.
    limits : PaymentLimits, optional
        How much a single payment may be, process-wide and per base
        URL. Set onto `gateway` rather than passed alongside it: the
        gateway pays at two seams -- this session, and its own `pay`
        for the 402s litellm turns into exceptions -- and one cap has
        to bind both. Two places to hold it is two places to forget
        it. None caps nothing of ours, which leaves the SDK's own
        default per-payment ceiling standing.
    budget : PaymentBudget, optional
        The cumulative total every payment is debited against. Set
        onto `gateway` for the same reason `limits` is: the gateway
        pays at two seams and one ledger has to serve both, or the
        process spends the budget twice. None leaves the total
        unbounded.

    Returns
    -------
    httpx.AsyncClient
        The session that was installed.
    """
    import litellm

    scope = None
    if payable_bases is not None:
        from routellm.payment.scope import PaymentScope

        scope = PaymentScope(payable_bases)

    if limits is not None:
        gateway.limits = limits

    if budget is not None:
        gateway.budget = budget

    session = gateway.build_session(transport=transport, scope=scope)
    litellm.aclient_session = session
    return session


def maybe_install_payment_session(
    provider,
    wallet_key,
    networks=None,
    payable_bases=(),
    limits=None,
    budget=None,
):
    """Install the paying session only when payment was actually asked for.

    Both halves are required. The provider flag alone has nothing to
    sign with, and a wallet key alone was never opted in; in either
    case litellm keeps its own session, so an unconfigured server takes
    the same code path and the same connections it always did.

    The flag does not authorise anyone to charge. That is stated per
    endpoint, in the config, and arrives here as `payable_bases`; the
    default is empty, so switching payments on without marking an
    endpoint payable installs a session that never signs anything.

    Parameters
    ----------
    provider : str or None
        Value of `--payment-provider`. Only "x402" is recognised.
    wallet_key : str
        Private key read from the configured environment variable.
    networks : list[str], optional
        Networks the wallet may pay on. Adapter default when None.
    payable_bases : iterable[str], optional
        Base URLs of the endpoints the operator authorised to charge.
        Empty by default, which authorises nothing.
    limits : PaymentLimits, optional
        Per-payment caps, from `--max-payment` and the endpoints'
        `max_payment:`. None leaves the SDK's own default standing.
    budget : PaymentBudget, optional
        The cumulative total from `--payment-budget`. None leaves the
        total unbounded, which is what an operator who set no budget
        asked for.

    Returns
    -------
    PaymentGateway or None
        The gateway when a session was installed, None otherwise.
    """
    if provider != "x402" or not wallet_key:
        return None

    from routellm.payment.x402 import X402Adapter

    gateway = X402Adapter(private_key=wallet_key, networks=networks)
    install_payment_session(gateway, payable_bases=payable_bases, limits=limits, budget=budget)

    bases = list(payable_bases or ())
    if bases:
        logger.info(
            "x402 payments enabled for %d base URL(s): %s. A 402 from "
            "anywhere else is returned unpaid.",
            len(bases),
            ", ".join(sorted(bases)),
        )
    else:
        logger.warning(
            "x402 payments were enabled but no endpoint is marked "
            "payable, so nothing will ever be paid. Set `pay: true` on "
            "the endpoints allowed to charge this wallet."
        )
    return gateway
