"""HTTP 402 Payment Required adapter using x402 library.

Implements W3C Payment Handler for fulfilling 402 challenges via blockchain.
"""

import os
import time

from .gateway import PaymentGateway
from .types import PaymentChallenge, PaymentReceipt


class X402Adapter(PaymentGateway):
    """x402-based payment gateway for EVM blockchains.

    Wraps the x402 PyPI package (coinbase/x402) to fulfill 402 payment
    challenges. Requires x402[evm] installed and ROUTELLM_WALLET_PRIVATE_KEY set.

    Internal flow for pay():
    1. Register EVM signer mechanisms on the x402Client
    2. Hand the raw 402 headers and body to
       x402HTTPClient.handle_402_response, which detects the protocol
       version, decodes the server's PaymentRequired, signs it, and
       returns the retry header under its version-correct name
    3. Map the encoded payload back to a RouteLLM PaymentReceipt
    """

    def __init__(
        self,
        private_key: str | None = None,
        networks: list[str] | None = None,
    ):
        """Initialize x402 payment adapter.

        Parameters
        ----------
        private_key : str, optional
            EVM private key for signing transactions. If not provided,
            uses ROUTELLM_WALLET_PRIVATE_KEY environment variable.
        networks : list[str], optional
            Supported EVM networks (default ["base", "ethereum", "polygon"]).
        """
        self._private_key = private_key or os.environ.get("ROUTELLM_WALLET_PRIVATE_KEY", "")
        self._networks = networks or ["base", "ethereum", "polygon"]

    @property
    def name(self) -> str:
        """Get provider name.

        Returns
        -------
        str
            "x402"
        """
        return "x402"

    @property
    def networks(self) -> list[str]:
        return self._networks

    # Legacy v1 network names mapped to the CAIP-2 identifiers v2 uses.
    _CAIP2 = {
        "base": "eip155:8453",
        "base-sepolia": "eip155:84532",
        "ethereum": "eip155:1",
        "polygon": "eip155:137",
        "avalanche": "eip155:43114",
    }

    def _build_client(self):
        """Build an x402HTTPClient with the wallet registered for both versions.

        `register_exact_evm_client` is the package's own registration
        helper, and it registers the scheme twice: once under the CAIP-2
        networks v2 addresses, and once under the legacy names v1 uses.
        Registering only the v2 side -- as a hand-rolled loop over
        `client.register` does -- leaves a v1 challenge with no scheme to
        match, so the payment fails after the challenge parses cleanly.

        Returns
        -------
        x402HTTPClient
            Client able to sign for either protocol version.
        """
        from eth_account import Account
        from x402.client import x402Client
        from x402.http.x402_http_client import x402HTTPClient
        from x402.mechanisms.evm.exact import register_exact_evm_client
        from x402.mechanisms.evm.signers import EthAccountSigner

        # EthAccountSigner satisfies the SDK's ClientEvmSigner protocol:
        # it exposes the wallet `address` and `sign_typed_data`. A bare
        # signing function does not, and the SDK fails on `.address`
        # partway through building the authorization.
        signer = EthAccountSigner(Account.from_key(self._private_key))

        networks = [self._CAIP2[n] for n in self._networks if n in self._CAIP2]

        client = x402Client()
        register_exact_evm_client(client, signer, networks=networks or None)

        return x402HTTPClient(client)

    def build_session(self, transport=None, scope=None):
        """Build an httpx client that settles 402s before returning.

        The payment cycle needs the response headers, the response body
        and a way to replay the request. All three exist only below the
        HTTP client: litellm's exception mapper keeps a 402's status
        code and drops the response it arrived on, so a challenge
        cannot be recovered from the error it raises.

        The SDK's own `x402AsyncTransport` wraps another transport and
        does the whole cycle -- version detection, challenge decoding,
        signing, the replay and the retry cap -- so this only supplies
        the wallet and the transport underneath it.

        Parameters
        ----------
        transport : httpx.AsyncBaseTransport, optional
            Transport that actually reaches the provider. Defaults to
            httpx's own, which is what a live run uses; a test passes a
            stand-in so no socket is opened.
        scope : PaymentScope, optional
            Base URLs a payment may be signed for. When given, a
            request outside it never reaches the payment cycle and a
            402 from it is returned unpaid. None leaves the session
            unscoped, which is only ever right for a caller that has
            already narrowed the client to one upstream.

        Returns
        -------
        httpx.AsyncClient
            Client whose requests pay and retry on a 402, within the
            scope when one was given.
        """
        import httpx
        from x402.http.clients.httpx import x402AsyncTransport

        client = self._build_client()

        if scope is None:
            return httpx.AsyncClient(
                transport=x402AsyncTransport(client, transport)
            )

        from routellm.payment.scope import scoped_payment_transport

        return httpx.AsyncClient(
            transport=scoped_payment_transport(scope, client, transport)
        )

    async def pay(self, challenge: PaymentChallenge) -> PaymentReceipt:
        """Fulfill a 402 payment challenge using the x402 SDK.

        The challenge is decoded from the server's own 402 response.
        `handle_402_response` detects the protocol version from what the
        server sent -- a base64 PAYMENT-REQUIRED header for v2, a JSON
        body for v1 -- decodes the PaymentRequired, signs it with the
        configured wallet, and returns the retry header already named
        for that version. Choosing the version here instead would make
        the client's configuration override the server's protocol.

        Parameters
        ----------
        challenge : PaymentChallenge
            Challenge carrying the raw 402 headers, body and URL.

        Returns
        -------
        PaymentReceipt
            Receipt whose `header_name` and `header_value` are the
            header the retry must carry.

        Raises
        ------
        ValueError
            If the challenge carries no x402 data to parse.
        """
        if not challenge.headers and not challenge.body:
            raise ValueError(
                "the 402 response carried no x402 payment challenge to sign: "
                "no PAYMENT-REQUIRED header and no body"
            )

        http_client = self._build_client()
        payment_headers, payload = await http_client.handle_402_response(
            headers=challenge.headers,
            body=challenge.body or None,
            request_url=challenge.resource_url,
        )

        if not payment_headers:
            raise ValueError("the x402 client produced no payment header")

        header_name, header_value = next(iter(payment_headers.items()))

        # The proof of payment is the encoded payload, never a repr of
        # the object: a signed PaymentPayload has no transaction hash,
        # because nothing has settled on chain yet at this point.
        tx_hash = getattr(payload, "transaction_hash", None) or header_value

        return PaymentReceipt(
            tx_hash=tx_hash,
            network=challenge.network,
            amount=challenge.amount,
            currency=challenge.currency,
            paid_at=int(time.time()),
            resource=challenge.resource_url or challenge.payload.get("resource", ""),
            header_name=header_name,
            header_value=header_value,
        )
