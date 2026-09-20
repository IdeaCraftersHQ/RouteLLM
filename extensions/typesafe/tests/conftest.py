"""Stub the torch-bound router module so these tests import cleanly.

`routellm.routers.routers` imports torch and transformers at module
level, and `routellm.middleware.__init__` reaches it transitively
(intent_config -> controller -> routers.routers). Nothing under test
here needs a real router class, so the module is replaced before the
first `routellm` import. `routellm.routers.registry`, which the
discovery test drives, stays real: it is torch-free and is imported as
its own module, not through the stub.

This mirrors what the repo-root conftest does for the core suite; the
extension carries its own copy because pytest's rootdir is this
package, so the root conftest never loads.
"""
import sys
from unittest.mock import MagicMock


def _stub(name):
    """Build a MagicMock that passes as an imported module.

    Parameters
    ----------
    name : str
        Fully qualified module name to impersonate.

    Returns
    -------
    unittest.mock.MagicMock
        Mock carrying the dunder attributes the import system reads.
    """
    module = MagicMock()
    module.__name__ = name
    module.__path__ = []
    module.__file__ = ""
    module.__spec__ = None
    module.__package__ = name.split(".")[0]
    return module


_routers_stub = _stub("routellm.routers.routers")
_routers_stub.ROUTER_CLS = {}

# Only the module is stubbed, never the `routellm.routers` package:
# that directory has no __init__.py, so a stub there would shadow the
# namespace package and hide the real `base` and `registry` siblings.
sys.modules["routellm.routers.routers"] = _routers_stub
