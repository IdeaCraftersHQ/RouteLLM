"""The response cache is isolated per principal and per data class.

The upstream is a thread on 127.0.0.1 that counts what reaches it, so every
assertion is about calls that crossed the wire. A hit is an unchanged count.
"""

import asyncio
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from routellm.auth import Principal, current_principal
from routellm.caching import Cache, CacheConfig, cache_namespace
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot
from routellm.policy import parse_policy
from routellm.tests.test_server_cache_resilience import Upstream

MSGS = [{"role": "user", "content": "the same prompt"}]
ALICE = Principal("alice", allowed_classes=("a", "b", "none", "sem", "blocked"))
BOB = Principal("bob", allowed_classes=("a", "b", "none", "sem", "blocked"))

_LOCAL = {"treatment": "raw", "reach": ["local"], "tiers": ["default"]}
CLASSES = {
    "a": {**_LOCAL},
    "b": {**_LOCAL},
    "none": {**_LOCAL, "cache": "none"},
    "sem": {**_LOCAL, "cache": "both"},
    "blocked": {"treatment": "block", "reach": ["local"], "tiers": ["default"], "cache": "exact"},
}


@pytest.fixture
def upstream():
    fake = Upstream()
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))
    yield
    current_principal.set(None)


def _controller(fake, tmp_path, policy=True, **cache):
    leaf = {
        "model": "openai/m",
        "api_base": fake.base,
        "api_key_env": "FAKE_KEY",
        "tags": ["local"],
    }
    registry = EndpointRegistry.from_config(
        {
            "endpoints": {"loc": leaf},
            "tiers": {
                "default": {"router": "random", "threshold": 0.0, "strong": "loc", "weak": "loc"}
            },
            "strict_registry": True,
        }
    )
    pol = parse_policy({"classes": CLASSES}, "p.yaml") if policy else None
    return Controller(
        routers=["random"],
        endpoints=registry,
        policy=pol,
        strict_registry=True,
        cache_config=CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"), **cache),
    )


def _ask(controller, cls=None, principal=None, messages=MSGS, **kwargs):
    return asyncio.run(
        controller.acompletion(
            messages=messages,
            model="default",
            declared_class=cls,
            principal=principal,
            **kwargs,
        )
    )


# ---------------------------------------------------------------- keying


def test_principal_a_entry_is_never_served_to_b(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    _ask(c, "a", BOB)
    assert len(upstream.hits) == 2
    _ask(c, "a", ALICE)
    _ask(c, "a", BOB)
    assert len(upstream.hits) == 2


def test_anonymous_never_shares_with_a_principal(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    _ask(c, "a", None)
    assert len(upstream.hits) == 2


def test_class_change_on_the_same_prompt_misses(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    _ask(c, "b", ALICE)
    assert len(upstream.hits) == 2
    _ask(c, "a", ALICE)
    _ask(c, "b", ALICE)
    assert len(upstream.hits) == 2


def test_no_policy_no_auth_still_caches(upstream, tmp_path):
    c = _controller(upstream, tmp_path, policy=False)
    _ask(c)
    _ask(c)
    assert len(upstream.hits) == 1


def test_no_policy_declared_class_still_splits_the_key(upstream, tmp_path):
    c = _controller(upstream, tmp_path, policy=False)
    _ask(c, "x")
    _ask(c, "y")
    assert len(upstream.hits) == 2


def test_namespace_digest_hides_the_principal_and_class(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    raw = open(tmp_path / "c.db", "rb").read()
    assert b"alice" not in raw
    ns = cache_namespace("alice", "a")
    assert ns and "alice" not in ns
    assert cache_namespace("alice", "a") != cache_namespace("alice", "b")
    assert cache_namespace("alice", "a") != cache_namespace("bob", "a")
    assert cache_namespace(None, "a") != cache_namespace("", "a")
    assert cache_namespace("a", "b") != cache_namespace("ab", None)


def test_no_principal_no_class_is_the_empty_namespace():
    assert cache_namespace(None, None) == ""
    assert cache_namespace(None, "a") != ""
    assert cache_namespace("alice", None) != ""


def test_key_depends_on_namespace(tmp_path):
    cache = Cache(CacheConfig(enabled=True, db_path=str(tmp_path / "c.db")))
    k = lambda ns: cache._get_key("p", "m", {}, ns)  # noqa: E731
    assert k(cache_namespace("alice", "a")) != k(cache_namespace("bob", "a"))
    assert k(cache_namespace("alice", "a")) != k(cache_namespace("alice", "b"))


# ---------------------------------------------------------------- class mode


def test_cache_none_class_neither_reads_nor_stores(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "none", ALICE)
    _ask(c, "none", ALICE)
    assert len(upstream.hits) == 2
    rows = sqlite3.connect(tmp_path / "c.db").execute("SELECT count(*) FROM completion_cache")
    assert rows.fetchone()[0] == 0


def test_block_class_is_never_stored(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "blocked", ALICE)
    _ask(c, "blocked", ALICE)
    assert len(upstream.hits) == 2
    rows = sqlite3.connect(tmp_path / "c.db").execute("SELECT count(*) FROM completion_cache")
    assert rows.fetchone()[0] == 0


def test_a_none_class_hit_does_not_leak_from_a_cached_class(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    _ask(c, "none", ALICE)
    assert len(upstream.hits) == 2


def test_stream_is_neither_read_nor_stored(upstream, tmp_path):
    c = _controller(upstream, tmp_path)
    _ask(c, "a", ALICE)
    assert len(upstream.hits) == 1
    touched = []
    c.cache.get = lambda *a, **k: touched.append("get")
    c.cache.put = lambda *a, **k: touched.append("put")
    asyncio.run(_drain(c))
    assert touched == []
    assert len(upstream.hits) == 2
    rows = sqlite3.connect(tmp_path / "c.db").execute("SELECT count(*) FROM completion_cache")
    assert rows.fetchone()[0] == 1


async def _drain(c):
    stream = await c.acompletion(
        messages=MSGS,
        model="default",
        declared_class="a",
        principal=ALICE,
        stream=True,
    )
    try:
        async for _ in stream:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------- semantic


def _embed_fake(cache):
    vocab: dict[str, int] = {}

    def fake(text):
        vec = np.zeros(64, dtype=np.float32)
        vec[vocab.setdefault("x", len(vocab))] = 1.0  # every text alike
        return vec

    cache._get_embedding = fake


def test_semantic_hit_never_crosses_classes_or_principals(tmp_path):
    cache = Cache(CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"), semantic_enabled=True))
    _embed_fake(cache)
    ns = cache_namespace("alice", "a")
    cache.put("hello there", "m", {"_context": "c"}, {"r": 1}, namespace=ns)
    assert cache.get("hello there!", "m", {"_context": "c"}, namespace=ns) == {"r": 1}
    assert (
        cache.get("hello there!", "m", {"_context": "c"}, namespace=cache_namespace("alice", "b"))
        is None
    )
    assert (
        cache.get("hello there!", "m", {"_context": "c"}, namespace=cache_namespace("bob", "a"))
        is None
    )
    assert cache.get("hello there!", "m", {"_context": "c"}) is None


def test_exact_mode_class_skips_the_semantic_lookup(tmp_path):
    cache = Cache(CacheConfig(enabled=True, db_path=str(tmp_path / "c.db"), semantic_enabled=True))
    _embed_fake(cache)
    ns = cache_namespace("alice", "a")
    cache.put("hello there", "m", {"_context": "c"}, {"r": 1}, namespace=ns)
    assert cache.get("hello there!", "m", {"_context": "c"}, namespace=ns, semantic=False) is None


def test_exact_class_never_embeds(upstream, tmp_path):
    c = _controller(upstream, tmp_path, semantic_enabled=True)
    calls = []
    c.cache._get_embedding = lambda t: calls.append(t) or np.zeros(4, dtype=np.float32)
    _ask(c, "a", ALICE)
    _ask(c, "a", ALICE)
    assert calls == []  # exact class: no embedding on put or get


def test_both_class_embeds(upstream, tmp_path):
    c = _controller(upstream, tmp_path, semantic_enabled=True)
    calls = []
    c.cache._get_embedding = lambda t: calls.append(t) or np.ones(4, dtype=np.float32)
    _ask(c, "sem", ALICE)
    assert calls


# ---------------------------------------------------------------- policy


def test_old_rows_are_never_served(tmp_path):
    path = tmp_path / "c.db"
    cache = Cache(CacheConfig(enabled=True, db_path=str(path)))
    legacy = hashlib.sha256(
        json.dumps({"prompt": "p", "model": "m", "params": {}}, sort_keys=True).encode()
    ).hexdigest()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO completion_cache (key, prompt, model, response, created_at) "
        "VALUES (?, 'p', 'm', '{\"r\": 1}', 9999999999)",
        (legacy,),
    )
    conn.commit()
    conn.close()
    assert cache.get("p", "m", {}) is None
    assert cache.get("p", "m", {}, namespace=cache_namespace("alice", None)) is None
