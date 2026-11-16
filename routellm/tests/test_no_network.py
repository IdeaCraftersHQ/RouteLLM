"""The suite's socket blocker: loopback in, everything else refused.

Each test that attempts a blocked connection is marked `blocker_probe`,
which exempts it from the per-test zero-attempt check that every other
test is held to.
"""

import os
import socket
import subprocess
import sys
import tempfile
import threading

import pytest

# TEST-NET-1 (RFC 5737): never routable, so a connect that slipped past
# the blocker would hang or fail rather than reach a real host.
OFFSITE = "192.0.2.1"


def _accept_once(srv):
    """Accept one connection, then close it (runs in a thread)."""
    srv.accept()[0].close()


def _log() -> str:
    path = os.environ["ROUTELLM_TEST_NO_NETWORK_LOG"]
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.blocker_probe
def test_non_loopback_connect_is_refused_naming_the_host():
    s = socket.socket()
    try:
        with pytest.raises(ConnectionRefusedError, match=r"192.0.2.1"):
            s.connect((OFFSITE, 443))
    finally:
        s.close()
    assert "192.0.2.1:443" in _log()


@pytest.mark.blocker_probe
def test_create_connection_to_a_hostname_is_refused():
    with pytest.raises(ConnectionRefusedError, match=r"api.openai.com"):
        socket.create_connection(("api.openai.com", 443), timeout=1)


@pytest.mark.blocker_probe
def test_dns_lookup_of_a_remote_name_is_refused():
    with pytest.raises(ConnectionRefusedError, match=r"example.com"):
        socket.getaddrinfo("example.com", 80)


@pytest.mark.blocker_probe
def test_udp_sendto_off_host_is_refused():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(ConnectionRefusedError):
            s.sendto(b"x", (OFFSITE, 53))
    finally:
        s.close()


@pytest.mark.blocker_probe
def test_ipv6_non_loopback_is_refused():
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("2001:db8::1", 443), timeout=1)


def test_loopback_tcp_still_works():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=_accept_once, args=(srv,))
    t.start()
    try:
        socket.create_connection(("localhost", port), timeout=2).close()
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    finally:
        t.join(2)
        srv.close()


def test_unix_socket_still_works():
    # macOS caps a socket path near 100 bytes; pytest's tmp_path is longer.
    d = tempfile.mkdtemp(prefix="rl")
    path = os.path.join(d, "s")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    t = threading.Thread(target=_accept_once, args=(srv,))
    t.start()
    try:
        c = socket.socket(socket.AF_UNIX)
        c.connect(path)
        c.close()
    finally:
        t.join(2)
        srv.close()


_CHILD = (
    "import socket,sys\n"
    "try:\n"
    f"    socket.create_connection(('{OFFSITE}', 443), timeout=1)\n"
    "except ConnectionRefusedError as e:\n"
    "    print('BLOCKED', e); sys.exit(0)\n"
    "except OSError as e:\n"
    "    print('LEAKED', e); sys.exit(3)\n"
)


@pytest.mark.blocker_probe
@pytest.mark.parametrize("env", [None, {"PATH": "/usr/bin:/bin"}, {"PYTHONPATH": "."}])
def test_subprocess_is_blocked_whatever_env_the_test_passes(env):
    """A child that gets its own explicit env is blocked all the same."""
    r = subprocess.run(
        [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "BLOCKED" in r.stdout
    assert "192.0.2.1:443" in _log()


@pytest.mark.network
def test_network_marker_lifts_the_blocker_for_live_tests():
    """Runs only with ROUTELLM_TEST_NETWORK=1; then connect is not refused."""
    s = socket.socket()
    s.settimeout(0.5)
    try:
        try:
            s.connect((OFFSITE, 443))
        except ConnectionRefusedError as e:
            assert "network blocked" not in str(e)
        except OSError:
            pass
    finally:
        s.close()
