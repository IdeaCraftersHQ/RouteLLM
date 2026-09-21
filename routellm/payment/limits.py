"""How much a single payment may be, and which limit said no.

Authorisation answers who may charge the wallet. It does not answer
how much: once an endpoint carries `pay: true`, a 402 from it names
its own price and that price is signed. An upstream inside the scope
that is compromised, misconfigured or merely mispriced can therefore
name any figure, bounded only by the x402 SDK's own default.

Two layers bound it. `--max-payment` is a process-wide ceiling, and an
endpoint's `max_payment:` may lower it further but never raise it; the
effective cap is the smaller of the two. Clamping rather than trusting
the endpoint is the point of the ceiling -- a config line that could
widen it would make it advisory.

With two layers a bare "payment refused" is not actionable, so
`effective` reports which layer owns the number it returned and the
refusal names that layer rather than the pair.

Units are the SDK's to convert. A cap is written in human money
("$0.01"); a challenge states an atomic on-chain amount against an
asset with its own decimals. The SDK's
`spend_controls.max_amount_per_payment` takes the money string and
resolves it against the scheme's default asset, so nothing here
multiplies by a power of ten. Comparing two caps is a different
question -- both sides are money, in the same unit -- and that is the
only arithmetic this module does.
"""

from decimal import Decimal, InvalidOperation

from routellm.payment.scope import _normalise

# Which layer a cap came from, and the knob an operator would turn to
# change it. A refusal quotes these so the message points at one.
GLOBAL = "global"
ENDPOINT = "endpoint"

CAP_KNOB = {
    GLOBAL: "--max-payment",
    ENDPOINT: "the endpoint's `max_payment:`",
}


def parse_cap(value):
    """Return `value` as a comparable decimal, or raise if it is not money.

    The SDK's `Money` is a human string like "$0.01". Its own parser is
    the authority on what it will accept, so this defers to it and only
    turns the result into something two caps can be compared with.

    Parameters
    ----------
    value : str or int or float
        A cap as written in the config or on the command line.

    Returns
    -------
    decimal.Decimal
        The amount, in the same unit the SDK reads it in.

    Raises
    ------
    ValueError
        If `value` is not money the SDK could act on. Refusing here is
        what keeps a cap nobody can parse from being silently dropped,
        which would leave the operator believing a limit is in force.
    """
    from x402.schemas.helpers import parse_money

    parsed = parse_money(value)
    if parsed.get("symbol"):
        raise ValueError(
            f"payment cap {value!r} names the asset {parsed['symbol']}, but a "
            "cap is stated in USD: write it as \"$0.01\". A per-asset cap is "
            "an atomic amount against one token, which is not what this sets."
        )
    try:
        return Decimal(parsed["amount"])
    except InvalidOperation as exc:  # pragma: no cover - parse_money guards it
        raise ValueError(f"unusable payment cap {value!r}") from exc


class PaymentLimits:
    """The per-payment ceiling, process-wide and per base URL.

    Attributes
    ----------
    global_cap : str or None
        The `--max-payment` ceiling, as written. None when the
        operator set none, which leaves the SDK's own default standing
        rather than removing every limit.
    per_base : dict
        Base URL to the cap that endpoint wrote, as written.
    """

    def __init__(self, global_cap=None, per_base=None):
        """Build the limits, refusing any cap that is not money.

        Parameters
        ----------
        global_cap : str, optional
            Process-wide ceiling. None means none was set.
        per_base : dict, optional
            Base URL to per-endpoint cap. Each is validated here, so a
            malformed one fails at startup rather than at the first
            402, when a refusal is indistinguishable from a broken
            upstream.

        Raises
        ------
        ValueError
            If any cap is not a money string.
        """
        self.global_cap = global_cap
        self._global_amount = None if global_cap is None else parse_cap(global_cap)

        self.per_base = dict(per_base or {})
        self._bases = []
        for base, cap in self.per_base.items():
            amount = parse_cap(cap)
            parsed = _normalise(base)
            if parsed is None:
                raise ValueError(
                    f"payment cap {cap!r} is set for {base!r}, which names no "
                    "scheme and host, so no request could ever match it"
                )
            self._bases.append((parsed, cap, amount))

    def __bool__(self) -> bool:
        """Whether any cap is configured at either layer."""
        return self._global_amount is not None or bool(self._bases)

    def _endpoint_cap(self, url):
        """Return the per-endpoint cap covering `url`, if any.

        The scope is keyed on base URLs while a request carries a full
        path, so matching is the scope's own origin-plus-prefix rule.
        An exact string match would never fire.

        Parameters
        ----------
        url : str or httpx.URL
            The URL of the request that was refused.

        Returns
        -------
        tuple or None
            `(cap_as_written, amount)` for the most specific matching
            base, or None when no endpoint capped this URL.
        """
        target = _normalise(str(url))
        if target is None:
            return None

        scheme, netloc, path = target
        best = None
        for (base_scheme, base_netloc, base_path), cap, amount in self._bases:
            if scheme != base_scheme or netloc != base_netloc:
                continue
            # A path prefix only counts on a segment boundary, so `/v1`
            # does not cap `/v1beta`.
            if base_path and not (
                path == base_path or path.startswith(base_path + "/")
            ):
                continue
            # The longest matching prefix wins, so a cap on a
            # sub-path is not overridden by a broader one.
            if best is None or len(base_path) > best[0]:
                best = (len(base_path), cap, amount)

        return None if best is None else (best[1], best[2])

    def effective(self, url):
        """Return the cap that binds a payment for `url`, and whose it is.

        Parameters
        ----------
        url : str or httpx.URL
            The URL of the request that was refused with a 402. A bare
            base URL works too, which is what the config-level tests
            and the controller seam pass.

        Returns
        -------
        tuple
            `(cap, source)` where `cap` is the money string to enforce
            and `source` is `"global"` or `"endpoint"`. `(None, None)`
            when neither layer set one, which means the SDK's own
            default stands -- never that there is no limit.
        """
        endpoint = self._endpoint_cap(url)

        if endpoint is None:
            if self._global_amount is None:
                return None, None
            return self.global_cap, GLOBAL

        cap, amount = endpoint
        if self._global_amount is None:
            return cap, ENDPOINT

        # Strictly lower, so a tie is the ceiling's: the endpoint
        # lowered nothing, and naming it would send an operator to the
        # wrong knob.
        if amount < self._global_amount:
            return cap, ENDPOINT
        return self.global_cap, GLOBAL


def refusal_message(cap, source, cause):
    """Explain a refused payment in terms of the limit that refused it.

    The SDK raises one message for every rejection by
    `max_amount_per_payment`, and it advises raising that control --
    which is not a knob any operator of this process has. With two
    layers, it also cannot say which of them was in force. Both are
    restated here in terms the config actually offers.

    Parameters
    ----------
    cap : str
        The money string that was enforced.
    source : str
        `"global"` or `"endpoint"`.
    cause : Exception
        The SDK's own refusal, quoted so the asset and amount it names
        are not lost.

    Returns
    -------
    str
        A message naming the limit and the knob that changes it.
    """
    return (
        f"payment refused: the challenge exceeds the {source} per-payment "
        f"limit of {cap}, set by {CAP_KNOB[source]}. Raise that limit to "
        f"allow it. Underlying refusal: {cause}"
    )
