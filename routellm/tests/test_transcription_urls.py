"""`audio_url` fetching: operator allowlist, public addresses only, pinned peer.

Fetching is off by default. Enabled, it needs an exact-hostname allowlist,
and every address the host resolves to must be public unless the operator
opts in to private ones. The connection goes to the validated IP, so a
second DNS answer cannot redirect it. No real DNS or network: the resolver
is mocked, and the "public" fetch runs over `httpx.MockTransport`.
"""

import functools
import json

import httpx
import pytest

from routellm import transcription
from routellm.tests.fake_transcriber import FakeTranscriber
from routellm.transcription import TranscriberConfig, TranscriptionError

HOST = "audio.example.test"
PUBLIC = "93.184.216.34"


def cfg(base="http://127.0.0.1:1/v1", **kw):
    kw.setdefault("allow_urls", True)
    kw.setdefault("allowed_url_hosts", [HOST])
    return TranscriberConfig(api_base=base, **kw)


def msg(url):
    return [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": url}}]}]


def resolver(monkeypatch, *addresses):
    calls = []

    def fake(host, port):
        calls.append((host, port))
        return list(addresses)

    monkeypatch.setattr(transcription, "_resolve", fake)
    return calls


@pytest.fixture
def stt():
    fake = FakeTranscriber()
    yield fake
    fake.close()


def test_a_host_off_the_allowlist_is_refused_before_resolving(monkeypatch):
    calls = resolver(monkeypatch, PUBLIC)

    with pytest.raises(TranscriptionError) as err:
        transcription.transcribe_sync(msg("http://other.example.test/a.wav"), cfg())

    assert err.value.status_code == 400
    assert calls == []


def test_urls_enabled_with_no_allowlist_refuse_everything(monkeypatch):
    calls = resolver(monkeypatch, PUBLIC)

    with pytest.raises(TranscriptionError) as err:
        transcription.transcribe_sync(msg(f"http://{HOST}/a.wav"), cfg(allowed_url_hosts=[]))

    assert err.value.status_code == 400
    assert calls == []


def test_userinfo_in_the_url_is_refused(monkeypatch):
    resolver(monkeypatch, PUBLIC)

    with pytest.raises(TranscriptionError) as err:
        transcription.transcribe_sync(msg(f"http://user:pw@{HOST}/a.wav"), cfg())

    assert err.value.status_code == 400


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.9",
        "169.254.169.254",
        "0.0.0.0",
        "224.0.0.1",
        "100.64.0.1",
        "::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "fe80::1",
        "fd00::1",
    ],
)
def test_an_allowlisted_host_resolving_to_a_non_public_address_is_refused(monkeypatch, address):
    resolver(monkeypatch, address)

    with pytest.raises(TranscriptionError) as err:
        transcription.transcribe_sync(msg(f"http://{HOST}/a.wav"), cfg())

    assert err.value.status_code == 400


def test_one_bad_address_among_good_ones_refuses(monkeypatch):
    resolver(monkeypatch, PUBLIC, "10.0.0.5")

    with pytest.raises(TranscriptionError):
        transcription.transcribe_sync(msg(f"http://{HOST}/a.wav"), cfg())


def test_a_resolution_failure_is_refused(monkeypatch):
    resolver(monkeypatch)

    with pytest.raises(TranscriptionError):
        transcription.transcribe_sync(msg(f"http://{HOST}/a.wav"), cfg())


def _mock_client(monkeypatch, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=b"pcm", headers={"content-type": "audio/wav"})
        return httpx.Response(200, content=json.dumps({"text": "ok"}).encode())

    transport = httpx.MockTransport(handler)
    real = httpx.Client
    monkeypatch.setattr(transcription.httpx, "Client", functools.partial(real, transport=transport))
    real_async = httpx.AsyncClient
    monkeypatch.setattr(
        transcription.httpx, "AsyncClient", functools.partial(real_async, transport=transport)
    )


def test_an_allowlisted_public_host_is_fetched_over_the_pinned_ip(monkeypatch):
    resolver(monkeypatch, PUBLIC)
    seen = []
    _mock_client(monkeypatch, seen)

    # Case and a trailing dot do not defeat the exact-host match.
    out = transcription.transcribe_sync(msg("http://Audio.Example.TEST./a.wav"), cfg())

    get = seen[0]
    assert get.method == "GET"
    assert get.url.host == PUBLIC  # pinned: the name is never looked up again
    assert get.headers["host"] == HOST
    assert out[0]["content"][0]["type"] == "text"


def test_the_async_path_pins_too(monkeypatch):
    import asyncio

    resolver(monkeypatch, PUBLIC)
    seen = []
    _mock_client(monkeypatch, seen)

    asyncio.run(transcription.transcribe(msg(f"https://{HOST}/a.wav"), cfg()))

    get = seen[0]
    assert get.url.host == PUBLIC
    assert get.headers["host"] == HOST
    assert get.extensions["sni_hostname"] == HOST


def test_a_redirect_is_not_followed(monkeypatch, stt):
    target = FakeTranscriber()
    stt.redirect_to = f"{target.base}/media/secret.wav"
    resolver(monkeypatch, "127.0.0.1")
    try:
        with pytest.raises(TranscriptionError) as err:
            transcription.transcribe_sync(
                msg(f"{stt.base}/media/a.wav".replace("127.0.0.1", HOST)),
                cfg(base=stt.base, allow_private_url_hosts=True),
            )
        assert err.value.public == "audio_url could not be fetched"
        assert stt.gets == ["/v1/media/a.wav"]
        assert target.gets == []
    finally:
        target.close()


def test_private_hosts_need_their_own_explicit_opt_in(monkeypatch, stt):
    resolver(monkeypatch, "127.0.0.1")
    url = f"{stt.base}/media/a.wav".replace("127.0.0.1", HOST)

    with pytest.raises(TranscriptionError):
        transcription.transcribe_sync(msg(url), cfg(base=stt.base))
    assert stt.gets == []

    transcription.transcribe_sync(msg(url), cfg(base=stt.base, allow_private_url_hosts=True))
    assert stt.gets == ["/v1/media/a.wav"]
