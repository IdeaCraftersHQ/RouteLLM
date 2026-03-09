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

sys.modules.setdefault("routellm.routers", _stub("routellm.routers"))
sys.modules["routellm.routers.routers"] = _fake_routers_mod

# ---------------------------------------------------------------------------
# Stub x402 package so X402Adapter tests run without the real SDK installed.
# PaymentRequired just needs to be constructable with **kwargs.
# ---------------------------------------------------------------------------

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
