"""Install the payment cycle underneath litellm.

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

The session is process-global, so it is installed only for a server
that asked to pay and has a key to pay with. Anything else leaves
litellm's own client in place, unchanged.
"""

import logging

logger = logging.getLogger(__name__)


def install_payment_session(gateway, transport=None):
    """Route litellm's async requests through `gateway`'s paying client.

    Parameters
    ----------
    gateway : X402Adapter
        Gateway able to build a paying httpx client.
    transport : httpx.AsyncBaseTransport, optional
        Transport underneath the payment one. Defaults to httpx's own.

    Returns
    -------
    httpx.AsyncClient
        The session that was installed.
    """
    import litellm

    session = gateway.build_session(transport=transport)
    litellm.aclient_session = session
    return session


def maybe_install_payment_session(provider, wallet_key, networks=None):
    """Install the paying session only when payment was actually asked for.

    Both halves are required. The provider flag alone has nothing to
    sign with, and a wallet key alone was never opted in; in either
    case litellm keeps its own session, so an unconfigured server takes
    the same code path and the same connections it always did.

    Parameters
    ----------
    provider : str or None
        Value of `--payment-provider`. Only "x402" is recognised.
    wallet_key : str
        Private key read from the configured environment variable.
    networks : list[str], optional
        Networks the wallet may pay on. Adapter default when None.

    Returns
    -------
    PaymentGateway or None
        The gateway when a session was installed, None otherwise.
    """
    if provider != "x402" or not wallet_key:
        return None

    from routellm.payment.x402 import X402Adapter

    gateway = X402Adapter(private_key=wallet_key, networks=networks)
    install_payment_session(gateway)
    logger.info(
        "x402 payments enabled: 402 challenges are signed and retried "
        "below litellm, on its shared async session"
    )
    return gateway
