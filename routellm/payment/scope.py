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


def scoped_payment_transport(
    scope: PaymentScope, client, transport=None, limits=None, budget=None
):
    """Build an x402 transport that only pays inside `scope`, and only so much.

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
    The amount is bounded the same way, and for the same reason the
    scope exists: one session serves every upstream, so the cap has to
    be chosen per request from its URL rather than fixed on the client
    once. The SDK enforces it -- `set_spend_controls` takes the cap as
    a money string and resolves it against the scheme's own default
    asset, which is the only place the asset's decimals are known --
    and its refusal is restated in terms of the limit that refused.

    Passing no cap leaves the SDK's default spend controls untouched.
    `spend_controls=False` would read as "nothing configured" and
    remove the only ceiling an unconfigured deployment has.

    Parameters
    ----------
    scope : PaymentScope
        The base URLs a payment may be signed for.
    client : x402Client or x402HTTPClient
        Payment client the SDK transport signs with.
    transport : httpx.AsyncBaseTransport, optional
        Transport that actually reaches the provider.
    limits : PaymentLimits, optional
        Per-payment caps. None caps nothing of ours.
    budget : PaymentBudget, optional
        The cumulative ledger every authorised payment is debited
        against. The cap bounds one payment; this bounds their sum,
        which the cap alone cannot, since any number of payments may
        each sit just under it. None leaves the total unbounded.

    Returns
    -------
    x402AsyncTransport
        A payment transport gated on `scope`, on `limits` and on
        `budget`.
    """
    from x402.http.clients.httpx import PaymentError, x402AsyncTransport

    def _signing_client():
        """Return the object `set_spend_controls` lives on.

        `x402AsyncTransport` accepts either an `x402Client` or the
        `x402HTTPClient` wrapping one, and spend controls belong to
        the inner client in both cases.
        """
        return getattr(client, "_client", client)

    class _ScopedX402Transport(x402AsyncTransport):
        """An x402 transport that declines unpayable URLs and over-cap prices."""

        def _limits_probe(self, url):
            """Return the cap this transport would enforce for `url`.

            The cap is chosen per request inside `handle_async_request`,
            so there is otherwise nothing on the installed session to
            read it off. A caller checking that the process-global
            session really carries the operator's limits needs to ask
            without sending a request and paying one.

            Parameters
            ----------
            url : str
                A request URL.

            Returns
            -------
            list
                `[cap, source]`, or `[None, None]` when nothing of
                ours caps it.
            """
            if limits is None:
                return [None, None]
            return list(limits.effective(url))

        def _budget_probe(self, obj=False):
            """Return the ledger this transport debits, or its remainder.

            The budget is consulted per request inside
            `handle_async_request`, so there is otherwise nothing on
            the installed session to read it off. A caller checking
            that the process-global session really carries the
            operator's budget -- and that it is the same ledger the
            controller holds, not a second one -- needs to ask without
            sending a request and paying one.

            Parameters
            ----------
            obj : bool, optional
                True returns the ledger itself, so a caller can check
                identity. False returns its remainder as money.

            Returns
            -------
            PaymentBudget or str or None
                The ledger, its remainder, or None when none is set.
            """
            if obj:
                return budget
            return None if budget is None else budget.remaining

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
                payable, the price is within the limit, and the server
                asked for payment.

            Raises
            ------
            PaymentError
                When the challenge is above the effective cap, or when
                the cumulative budget no longer covers it. The message
                names which limit refused it.
            """
            if not scope.allows(request.url):
                logger.debug(
                    "not payable: %s is outside the configured payment "
                    "scope, so a 402 from it is returned unpaid",
                    request.url,
                )
                return await self._transport.handle_async_request(request)

            cap, source = (None, None)
            if limits is not None:
                cap, source = limits.effective(request.url)

            if cap is not None:
                # Chosen per request: the session is shared, so a cap
                # installed once would bind whichever endpoint happened
                # to be configured last.
                _signing_client().set_spend_controls(
                    {"max_amount_per_payment": cap}
                )

            debited = False
            if budget:
                # Reserved before anything is signed: the budget has to
                # refuse before a wallet is authorised, never after.
                # The figure is the cap, because that is the amount the
                # wallet is about to be authorised to spend -- nothing
                # has settled yet, so no smaller figure is knowable.
                refused = budget.debit(cap)
                if refused is not None:
                    raise PaymentError(refused)
                debited = True

            try:
                response = await super().handle_async_request(request)
            except PaymentError as exc:
                # Nothing was signed, so the reservation was never
                # spent and goes back. Keeping it would let an
                # unpayable upstream drain the budget by being refused
                # over and over.
                if debited:
                    budget.refund(cap)
                if cap is None or "max_amount_per_payment" not in str(exc):
                    raise
                from routellm.payment.limits import refusal_message

                raise PaymentError(refusal_message(cap, source, exc)) from exc
            except BaseException:
                if debited:
                    budget.refund(cap)
                raise

            if debited and response.status_code == 402:
                # A 402 came back: either the request never entered the
                # payment cycle or the cycle gave up, and in both cases
                # nothing was signed.
                budget.refund(cap)

            return response

    return _ScopedX402Transport(client, transport)
