"""Who may charge is stated per endpoint, in the config.

`--payment-provider x402` supplies a wallet; it does not name anyone
allowed to spend it. That is `pay: true` on the endpoints the operator
chose, alongside `model:` and `api_base:` where the rest of an
endpoint's shape already lives.

An endpoint that says nothing about payment does not pay, so adding the
flag to a running server authorises exactly the endpoints it is written
on and no others.
"""

import sys

import pytest

from routellm.endpoints import EndpointRegistry


def _server_module():
    """Import the server with a clean argv.

    `routellm.openai_server` parses `sys.argv` at import, so importing
    it under pytest's own argv aborts the interpreter. Every other
    suite works around this by spawning a subprocess; the scope
    resolver is a plain function, so setting argv for the import is
    enough here.
    """
    if "routellm.openai_server" in sys.modules:
        return sys.modules["routellm.openai_server"]

    previous = sys.argv
    sys.argv = ["routellm.openai_server"]
    try:
        import routellm.openai_server as server
    finally:
        sys.argv = previous
    return server


def registry(config: dict) -> EndpointRegistry:
    """Build a registry from an inline config mapping."""
    return EndpointRegistry.from_config(config)


def test_an_endpoint_does_not_pay_unless_it_says_so():
    """The default is not paying. Silence is never consent to charge."""
    reg = registry(
        {"endpoints": {"cloud": {"model": "gpt-4o", "api_base": "https://a.example.com/v1"}}}
    )

    assert reg.resolve("cloud").pay is False
    assert reg.payable_bases() == []


def test_only_the_endpoints_marked_payable_are_collected():
    """`pay: true` is what puts a base URL in the scope, nothing else."""
    reg = registry(
        {
            "endpoints": {
                "payer": {
                    "model": "gpt-4o",
                    "api_base": "https://paid.example.com/v1",
                    "pay": True,
                },
                "freeloader": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://free.example.com/v1",
                },
            }
        }
    )

    assert reg.payable_bases() == ["https://paid.example.com/v1"]


def test_a_payable_endpoint_without_a_base_falls_back_to_the_default():
    """An endpoint setting no `api_base` is reached at the default one.

    Authorising it has to authorise the URL it is actually called on,
    or `pay: true` would silently do nothing.
    """
    reg = registry({"endpoints": {"payer": {"model": "gpt-4o", "pay": True}}})

    assert reg.payable_bases(default_base="https://default.example.com/v1") == [
        "https://default.example.com/v1"
    ]


def test_a_payable_endpoint_with_no_base_at_all_authorises_nothing():
    """No `api_base` and no default leaves nothing to authorise.

    Better an endpoint that cannot pay than a scope with an empty entry
    in it, which would be an authorisation nobody can read.
    """
    reg = registry({"endpoints": {"payer": {"model": "gpt-4o", "pay": True}}})

    assert reg.payable_bases() == []


def test_several_payable_endpoints_on_one_base_authorise_it_once():
    """Two endpoints behind one gateway are one authorisation."""
    reg = registry(
        {
            "endpoints": {
                "big": {
                    "model": "gpt-4o",
                    "api_base": "https://paid.example.com/v1",
                    "pay": True,
                },
                "small": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://paid.example.com/v1",
                    "pay": True,
                },
            }
        }
    )

    assert reg.payable_bases() == ["https://paid.example.com/v1"]


def test_pay_is_rejected_when_it_is_not_a_boolean():
    """`pay: "yes"` must not quietly become an authorisation.

    A string that happens to be truthy is exactly the mistake that
    would hand a wallet to an endpoint nobody meant to authorise.
    """
    with pytest.raises(ValueError):
        registry(
            {
                "endpoints": {
                    "payer": {
                        "model": "gpt-4o",
                        "api_base": "https://paid.example.com/v1",
                        "pay": "yes",
                    }
                }
            }
        )


def test_the_server_authorises_only_the_endpoints_that_asked():
    """The server resolves the scope from the very registry it routes on.

    This is the wiring an operator relies on: `pay: true` in the YAML
    has to reach the session litellm sends every request on, or the
    flag is documentation rather than enforcement. Reading it off a
    second, separately parsed copy of the config would let the two
    drift.
    """
    payable_bases_for = _server_module().payable_bases_for

    reg = registry(
        {
            "endpoints": {
                "payer": {
                    "model": "gpt-4o",
                    "api_base": "https://paid.example.com/v1",
                    "pay": True,
                },
                "freeloader": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://free.example.com/v1",
                },
            }
        }
    )

    assert payable_bases_for(reg, default_base=None) == ["https://paid.example.com/v1"]


def test_the_server_authorises_nothing_when_no_endpoint_asked():
    """A config with endpoints but no `pay:` leaves the wallet unspendable."""
    payable_bases_for = _server_module().payable_bases_for

    reg = registry(
        {
            "endpoints": {
                "plain": {
                    "model": "gpt-4o",
                    "api_base": "https://free.example.com/v1",
                }
            }
        }
    )

    assert payable_bases_for(reg, default_base="https://default.example.com/v1") == []
