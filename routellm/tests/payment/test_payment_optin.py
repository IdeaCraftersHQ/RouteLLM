"""Payment stays opt-in: no flag, no wallet, no new client.

The transport seam replaces litellm's shared session, which every
request then travels on. That is only acceptable while it is reached
exclusively by a server that asked for payments: an unconfigured server
has to keep litellm's own session, its own code path and its own
latency.
"""

import litellm
import pytest


@pytest.fixture(autouse=True)
def restore_session():
    """Never leak a session between tests; it is process-global state."""
    previous = litellm.aclient_session
    yield
    litellm.aclient_session = previous


def test_no_provider_installs_nothing():
    """No --payment-provider leaves litellm's session untouched."""
    from routellm.payment.transport import maybe_install_payment_session

    litellm.aclient_session = None
    gateway = maybe_install_payment_session(provider=None, wallet_key="0x" + "11" * 32)

    assert gateway is None
    assert litellm.aclient_session is None


def test_provider_without_wallet_key_installs_nothing():
    """The flag alone cannot pay, so it must not change the client.

    Installing a payment session that has no key to sign with would
    swap every request onto a new transport and then fail the first
    402 anyway.
    """
    from routellm.payment.transport import maybe_install_payment_session

    litellm.aclient_session = None
    gateway = maybe_install_payment_session(provider="x402", wallet_key="")

    assert gateway is None
    assert litellm.aclient_session is None


def test_provider_with_wallet_key_installs_a_paying_session():
    """Both present is the only combination that changes anything."""
    from x402.http.clients.httpx import x402AsyncTransport

    from routellm.payment.transport import maybe_install_payment_session
    from routellm.payment.x402 import X402Adapter

    litellm.aclient_session = None
    gateway = maybe_install_payment_session(provider="x402", wallet_key="0x" + "11" * 32)

    assert isinstance(gateway, X402Adapter)
    assert litellm.aclient_session is not None
    assert isinstance(litellm.aclient_session._transport, x402AsyncTransport)
