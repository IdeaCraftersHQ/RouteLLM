"""Hosts a `local`-tagged service may and may not name; shared by every local-base test."""

#: Accepted as written (no DNS).
LOCAL_BASES = [
    "http://127.0.0.1:11434/v1",
    "http://127.8.9.1/v1",
    "http://localhost:8000",
    "http://LOCALHOST:8000",
    "http://[::1]:8000/v1",
    "http://10.1.2.3/v1",
    "http://172.16.0.5:80",
    "http://172.31.255.1",
    "http://192.168.1.20:8080/v1",
    "http://169.254.10.10",
    "http://[fd12:3456::1]:80",
    "http://[fe80::1]",
    "http://box.local:80",
    "http://box.local./v1",
    "http://gpu.internal/v1",
    "http://app.localhost:3000",
    "localhost:11434",  # scheme-less
    "http://user:pw@127.0.0.1:1/v1",
]

#: Public, near-miss or hostless: refused without `allow_public_base`.
NON_LOCAL_BASES = [
    "https://api.openai.com/v1",
    "https://api.openai.com",
    "http://8.8.8.8/v1",
    "http://172.32.0.1",  # just outside RFC1918 172.16/12
    "http://100.100.1.1:80",  # shared/CGNAT space: not this network
    "http://gpu.lan/v1",  # .lan: unreserved, resolver-defined
    "http://ollama:11434",  # single-label: search domains may resolve it anywhere
    "http://0.0.0.0:11434",  # unspecified
    "http://192.0.2.1",  # documentation range, not private
    "http://[::ffff:8.8.8.8]",  # mapped public v4
    "http://192.169.0.1",
    "http://[2001:4860:4860::8888]",
    "https://evil.example.com/v1",
    "https://localhost.evil.com/v1",
    "https://box.local.evil.com/v1",
    "https://notinternal.com",
    "http://127.0.0.1.evil.com",
    "https://evil.com/x@127.0.0.1",
    "http://",
    "http://:8080",
]
