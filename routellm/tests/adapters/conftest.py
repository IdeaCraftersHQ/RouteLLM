"""Fixtures of the adapter tests: one harness server per pytest process.

`harness` is reset before each test. Every adapter in `routellm/adapters/`
is mounted by discovery; the reference adapter of the harness's own tests
is added through `ROUTELLM_ADAPTER_MODULES`.
"""

import pytest

from routellm.tests.adapters.conformance import start_harness

REFERENCE = "routellm.tests.adapters.reference_adapter"


@pytest.fixture(scope="session")
def _harness_server(tmp_path_factory):
    server = start_harness(tmp_path_factory.mktemp("adapters"), modules=REFERENCE)
    yield server
    server.close()


@pytest.fixture
def harness(_harness_server):
    _harness_server.reset()
    return _harness_server
