"""Root conftest: stub heavy ML/torch deps so unit tests run without GPU stack.

Strategy: stub routellm.routers.routers entirely (it imports torch/transformers
at module level), then stub its transitive heavy deps so nothing blows up at
import time. The "random" router used in payment tests is implemented in Python
with no heavy deps and is registered via ROUTER_CLS.

Also stubs x402 so X402Adapter tests can exercise the adapter logic without
the real PyPI package installed.
"""
import os

# MUST come before anything imports litellm, directly or transitively.
#
# `import litellm` fetches its model cost map over HTTPS at import time.
# Offline -- or under a socket blocker -- that turns every module that
# imports the controller into a collection error, so the suite cannot even
# be collected without a network. litellm ships the same map inside the
# package, and nothing in this repo reads pricing, so the local copy is
# both sufficient and authoritative here.
#
# `setdefault`, not assignment: an operator who deliberately exports the
# variable keeps their choice.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

# ---------------------------------------------------------------------------
# Network blocker: the suite must make zero non-loopback connections.
#
# Installed here, at import of the root conftest, so it is live before
# collection imports anything. Subprocesses are covered by prepending the
# `_nonet` directory (a `sitecustomize`) to their PYTHONPATH from a
# `subprocess.Popen` hook, whatever `env=` the test passes. See
# routellm/tests/_nonet/sitecustomize.py and TEST_INDEX.md.
#
# Opt out of the whole thing with ROUTELLM_TEST_NO_NETWORK=0; opt single
# tests out with `@pytest.mark.network` plus ROUTELLM_TEST_NETWORK=1.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

_NONET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routellm", "tests", "_nonet")
_spec = importlib.util.spec_from_file_location(
    "_routellm_nonet", os.path.join(_NONET_DIR, "sitecustomize.py")
)
_nonet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nonet)

_BLOCK = os.environ.get(_nonet.ENV_FLAG) != "0"
_blocking = _BLOCK
_log_path = ""

if _BLOCK:
    os.environ[_nonet.ENV_FLAG] = "1"
    if not os.environ.get(_nonet.ENV_LOG):
        _fd, _log_path = tempfile.mkstemp(prefix="routellm-nonet-", suffix=".log")
        os.close(_fd)
        os.environ[_nonet.ENV_LOG] = _log_path
    _nonet.install()

    _real_popen_init = subprocess.Popen.__init__

    def _popen_init(self, args, *a, **kw):
        if _blocking:
            env = dict(kw.get("env") if kw.get("env") is not None else os.environ)
            old = env.get("PYTHONPATH")
            env["PYTHONPATH"] = _NONET_DIR + (os.pathsep + old if old else "")
            env[_nonet.ENV_FLAG] = "1"
            env[_nonet.ENV_LOG] = os.environ[_nonet.ENV_LOG]
            if os.environ.get("PYTEST_CURRENT_TEST"):
                env[_nonet.ENV_TEST] = os.environ["PYTEST_CURRENT_TEST"]
            kw["env"] = env
        return _real_popen_init(self, args, *a, **kw)

    subprocess.Popen.__init__ = _popen_init


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


# ---------------------------------------------------------------------------
# Network blocker: markers, opt-out, and the per-test zero-attempt check.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


def _live_network_enabled() -> bool:
    return os.environ.get("ROUTELLM_TEST_NETWORK") == "1"


def pytest_collection_modifyitems(config, items):
    """`network` tests need the live network: skip unless explicitly enabled."""
    if _live_network_enabled():
        return
    skip = pytest.mark.skip(reason="live network test: set ROUTELLM_TEST_NETWORK=1 to run")
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(skip)


def _log_size() -> int:
    try:
        return os.path.getsize(os.environ[_nonet.ENV_LOG])
    except (KeyError, OSError):
        return 0


@pytest.fixture(autouse=True)
def _no_network(request):
    """Fail any test that attempted a non-loopback connection.

    The blocker raises at the call site, but product code may catch and
    swallow that (a cache lookup that "just misses"). The attempt is
    logged regardless, and this fixture fails the test that made it.
    `blocker_probe` tests attempt on purpose and are exempt.
    """
    global _blocking
    live = bool(request.node.get_closest_marker("network")) and _live_network_enabled()
    if live and _BLOCK:
        _nonet.uninstall()
        _blocking = False
    start = _log_size()
    yield
    if live and _BLOCK:
        _nonet.install()
        _blocking = True
    if not _BLOCK or live or request.node.get_closest_marker("blocker_probe"):
        return
    path = os.environ.get(_nonet.ENV_LOG)
    if not path or _log_size() <= start:
        return
    with open(path, encoding="utf-8") as fh:
        fh.seek(start)
        attempts = [ln.split("\t")[0] for ln in fh.read().splitlines() if ln]
    pytest.fail(
        "test attempted non-loopback network access (blocked): "
        + ", ".join(sorted(set(attempts)))
        + ". Stub or replay it; mark @pytest.mark.network only for a deliberately live test.",
        pytrace=False,
    )


def pytest_sessionfinish(session, exitstatus):
    if _log_path:
        try:
            os.unlink(_log_path)
        except OSError:
            pass
