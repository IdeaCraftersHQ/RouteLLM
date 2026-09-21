"""Which base URLs the wallet is allowed to spend on.

`litellm.aclient_session` is process-global: assigning it once puts
every async request in the process on the same client. A payment
transport installed there therefore sits in front of every upstream,
not only the one the operator meant to pay, and any provider that
answers 402 -- misconfigured, compromised, or simply someone else's --
can have a payment signed for it.

The authorisation is per endpoint and comes from the config, so this
module holds only the matching: a set of authorised base URLs, and a
transport that consults it before letting the payment cycle see a
request at all.

Matching is per origin plus path prefix, never raw string containment.
`https://api.example.com/v1` authorises `https://api.example.com/v1/
chat/completions` and neither `https://api.example.com.evil.test/v1`
nor `https://api.example.com/admin`.
"""

import logging
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


def _normalise(base: str) -> tuple[str, str, str] | None:
    """Split a base URL into the `(scheme, host:port, path)` it authorises.

    Parameters
    ----------
    base : str
        Base URL from an endpoint's `api_base`, or the controller
        default.

    Returns
    -------
    tuple[str, str, str] or None
        Lowercased scheme, lowercased netloc and the path with any
        trailing slash removed, or None when `base` names no host and
        so authorises nothing.
    """
    if not base:
        return None

    parts = urlsplit(base.strip())
    if not parts.scheme or not parts.netloc:
        return None

    return parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/")


class PaymentScope:
    """The set of base URLs a wallet may be spent on.

    An empty scope authorises nothing, which is what makes the default
    safe: enabling a payment provider without marking any endpoint
    payable leaves the wallet unable to spend.

    Attributes
    ----------
    bases : tuple
        The normalised `(scheme, netloc, path)` triples this scope
        covers.
    """

    def __init__(self, bases=()):
        """Build a scope over the given base URLs.

        Parameters
        ----------
        bases : iterable[str], optional
            Base URLs the operator authorised. Entries naming no host
            are dropped, with a warning: silently keeping one would
            leave the operator believing an endpoint may pay when it
            may not.
        """
        normalised = []
        for base in bases or ():
            parsed = _normalise(base)
            if parsed is None:
                logger.warning(
                    "ignoring unusable payable base URL %r: it names no "
                    "scheme and host, so no request can match it",
                    base,
                )
                continue
            normalised.append(parsed)

        self.bases = tuple(dict.fromkeys(normalised))

    def __bool__(self) -> bool:
        """Whether any base URL is authorised.

        Returns
        -------
        bool
            False when the scope is empty, in which case nothing pays.
        """
        return bool(self.bases)

    def allows(self, url) -> bool:
        """Whether a payment may be signed for `url`.

        Parameters
        ----------
        url : str or httpx.URL
            The URL of the request that was refused with a 402.

        Returns
        -------
        bool
            True only when `url` sits under an authorised base.
        """
        target = _normalise(str(url))
        if target is None:
            return False

        scheme, netloc, path = target
        for base_scheme, base_netloc, base_path in self.bases:
            if scheme != base_scheme or netloc != base_netloc:
                continue
            # A path prefix only counts on a segment boundary, so
            # `/v1` does not authorise `/v1beta`.
            if base_path and not (
                path == base_path or path.startswith(base_path + "/")
            ):
                continue
            return True
        return False


def scoped_payment_transport(scope: PaymentScope, client, transport=None):
    """Build an x402 transport that only pays inside `scope`.

    The result is an `x402AsyncTransport`, so everything that already
    reads litellm's session -- and the SDK's own retry markers, version
    detection and replay -- keeps working unchanged. The one difference
    is that a request whose URL is outside the scope is handed straight
    to the transport underneath, so the payment cycle never sees it.
    There is nothing to refuse late and nothing to sign: a 402 from an
    unauthorised upstream comes back exactly as it would with payments
    switched off.

    Refusing inside the SDK instead would not do. Its
    `handle_payment_required` hook can only ask for an unpaid retry,
    never decline -- returning None is what means "pay" -- and a
    client-side `AbortResult` raises out of the transport as a
    `PaymentError` rather than handing back the 402 a caller without a
    wallet would have seen.

    Parameters
    ----------
    scope : PaymentScope
        The base URLs a payment may be signed for.
    client : x402Client or x402HTTPClient
        Payment client the SDK transport signs with.
    transport : httpx.AsyncBaseTransport, optional
        Transport that actually reaches the provider.

    Returns
    -------
    x402AsyncTransport
        A payment transport gated on `scope`.
    """
    from x402.http.clients.httpx import x402AsyncTransport

    class _ScopedX402Transport(x402AsyncTransport):
        """An x402 transport that declines to look at unpayable URLs."""

        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            """Send `request`, entering the payment cycle only if allowed.

            Parameters
            ----------
            request : httpx.Request
                The outgoing request.

            Returns
            -------
            httpx.Response
                The response, settled and replayed when the URL is
                payable and the server asked for payment.
            """
            if scope.allows(request.url):
                return await super().handle_async_request(request)

            logger.debug(
                "not payable: %s is outside the configured payment scope, "
                "so a 402 from it is returned unpaid",
                request.url,
            )
            return await self._transport.handle_async_request(request)

    return _ScopedX402Transport(client, transport)
