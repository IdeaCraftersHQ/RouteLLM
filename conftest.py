"""Root conftest: stub heavy ML/torch deps so unit tests run without GPU stack.

Strategy: stub routellm.routers.routers entirely (it imports torch/transformers
at module level), then stub its transitive heavy deps so nothing blows up at
import time. The "random" router used in payment tests is implemented in Python
with no heavy deps and is registered via ROUTER_CLS.

Also stubs x402 so X402Adapter tests can exercise the adapter logic without
the real PyPI package installed.
"""
import sys
from unittest.mock import MagicMock


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__path__ = []
    m.__file__ = ""
    m.__spec__ = None
    m.__package__ = name.split(".")[0]
    return m


# ---------------------------------------------------------------------------
# Stub the entire routers module before controller.py imports it.
# We register a fake ROUTER_CLS with the "random" router so Controller.__init__
# can instantiate it.
# ---------------------------------------------------------------------------

class _FakeRandomRouter:
    NO_PARALLEL = True

    def route(self, prompt, threshold, model_pair):
        return model_pair.weak

    def calculate_strong_win_rate(self, prompt, model_pair=None):
        return 0.5


_fake_routers_mod = _stub("routellm.routers.routers")
_fake_routers_mod.ROUTER_CLS = {"random": lambda **kw: _FakeRandomRouter()}

# Only the module is stubbed, never the `routellm.routers` package:
# that directory has no __init__.py, so a stub there would shadow the
# namespace package and hide the real, torch-free `base` and `registry`
# siblings that the registry tests import.
sys.modules["routellm.routers.routers"] = _fake_routers_mod

# ---------------------------------------------------------------------------
# x402: prefer the real SDK, stub it only when it is genuinely absent.
#
# The package is a declared dependency (x402[evm]), and it owns the wire
# format -- header names, base64 encoding, version detection. A stub that
# shadows it unconditionally means the protocol tests assert against this
# file rather than against the protocol, which hides exactly the defects
# those tests exist to catch. So the stub is now a fallback for an
# environment without the extra installed, not the default.
# ---------------------------------------------------------------------------

def _x402_is_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("x402.http") is not None
    except (ImportError, ValueError):
        return False


if not _x402_is_installed():

    class _FakePaymentRequired:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _x402_schemas = _stub("x402.schemas")
    _x402_schemas.PaymentRequired = _FakePaymentRequired

    _x402_server = _stub("x402.server")
    _x402_server.verify_payment = MagicMock(return_value=True)

    _x402_pkg = _stub("x402")
    _x402_pkg.schemas = _x402_schemas
    _x402_pkg.server = _x402_server

    sys.modules.setdefault("x402", _x402_pkg)
    sys.modules["x402.schemas"] = _x402_schemas
    sys.modules["x402.server"] = _x402_server
