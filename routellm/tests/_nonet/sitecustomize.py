"""Outbound-network blocker for the test suite. Stdlib only.

Loaded two ways, so one implementation covers every Python process a
test starts:

* in-process: `conftest.py` loads this file by path and calls `install()`;
* subprocess: the directory holding this file is prepended to the child's
  `PYTHONPATH`, and the interpreter imports it as `sitecustomize` at
  startup, which calls `install()` when `ROUTELLM_TEST_NO_NETWORK=1`.

Allowed: Unix sockets, loopback (127.0.0.0/8, ::1), `localhost`, the
unspecified address. Everything else raises `NetworkBlockedError` naming
the host, and the attempt is appended to the file named by
`ROUTELLM_TEST_NO_NETWORK_LOG`, so an attempt that product code swallows
still fails the test that made it.
"""

import ipaddress
import os
import socket

ENV_FLAG = "ROUTELLM_TEST_NO_NETWORK"
ENV_LOG = "ROUTELLM_TEST_NO_NETWORK_LOG"
ENV_TEST = "ROUTELLM_TEST_NO_NETWORK_TEST"


class NetworkBlockedError(ConnectionRefusedError):
    """Raised for any non-loopback network access under the blocker."""


_installed = False
_orig = {}


def _is_local(host) -> bool:
    if host is None or host == "" or host == b"":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    h = host.strip("[]").split("%", 1)[0].lower()
    if h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


def _record(host, port, how):
    path = os.environ.get(ENV_LOG)
    if not path:
        return
    test = os.environ.get(ENV_TEST) or os.environ.get("PYTEST_CURRENT_TEST", "")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{host}:{port}\t{how}\t{test}\n")
    except OSError:
        pass


def _refuse(host, port, how):
    _record(host, port, how)
    raise NetworkBlockedError(
        f"network blocked in tests: {how} to {host!r}:{port}. Only loopback and "
        f"Unix sockets are allowed; stub or replay it, or mark the test "
        f"@pytest.mark.network and run with ROUTELLM_TEST_NETWORK=1."
    )


def _check_addr(address, how):
    # AF_UNIX: str/bytes path. AF_INET/INET6: tuple with host first.
    if isinstance(address, tuple) and address:
        host = address[0]
        port = address[1] if len(address) > 1 else ""
        if not _is_local(host):
            _refuse(host, port, how)


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    S = socket.socket
    for name in ("connect", "connect_ex", "sendto"):
        _orig[name] = getattr(S, name)

    def connect(self, address):
        _check_addr(address, "connect")
        return _orig["connect"](self, address)

    def connect_ex(self, address):
        _check_addr(address, "connect_ex")
        return _orig["connect_ex"](self, address)

    def sendto(self, data, *args):
        if args:
            _check_addr(args[-1], "sendto")
        return _orig["sendto"](self, data, *args)

    S.connect, S.connect_ex, S.sendto = connect, connect_ex, sendto

    _orig["getaddrinfo"] = socket.getaddrinfo

    def getaddrinfo(host, port, *a, **kw):
        if not _is_local(host):
            _refuse(host, port, "getaddrinfo")
        return _orig["getaddrinfo"](host, port, *a, **kw)

    socket.getaddrinfo = getaddrinfo

    for fname in ("gethostbyname", "gethostbyname_ex"):
        _orig[fname] = getattr(socket, fname)

        def make(fn, label):
            def wrapped(host):
                if not _is_local(host):
                    _refuse(host, "", label)
                return _orig[label](host)

            return wrapped

        setattr(socket, fname, make(fname, fname))


def uninstall() -> None:
    """Restore the real socket functions (used by the opt-out marker)."""
    global _installed
    if not _installed:
        return
    S = socket.socket
    S.connect, S.connect_ex, S.sendto = _orig["connect"], _orig["connect_ex"], _orig["sendto"]
    socket.getaddrinfo = _orig["getaddrinfo"]
    socket.gethostbyname = _orig["gethostbyname"]
    socket.gethostbyname_ex = _orig["gethostbyname_ex"]
    _installed = False


if os.environ.get(ENV_FLAG) == "1":
    install()
