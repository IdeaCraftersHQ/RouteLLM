"""The semantic cache fuzzes the final user turn and nothing else.

Two conversations that end in the same words must never share an answer
unless everything before those words is identical too: history, system
prompt, images, tools and sampling params all shape the reply.
"""

import json
import sqlite3
import time

import numpy as np

from routellm.caching import Cache, CacheConfig
from routellm.controller import Controller


def _cache(tmp_path, embed=None):
    cache = Cache(
        CacheConfig(
            enabled=True,
            db_path=str(tmp_path / "cache.db"),
            semantic_enabled=True,
            semantic_threshold=0.95,
        )
    )
    # Same text, same vector; distinct texts orthogonal unless `embed`
    # says otherwise. No provider is called.
    vocab: dict[str, int] = {}

    def fake(text):
        text = embed(text) if embed else text
        vec = np.zeros(64, dtype=np.float32)
        vec[vocab.setdefault(text, len(vocab))] = 1.0
        return vec

    cache._get_embedding = fake
    return cache


def _store(cache, messages, answer, **kwargs):
    params = Controller._cache_params({"messages": messages, **kwargs})
    cache.put(messages[-1]["content"], "m", params, {"answer": answer})


def _lookup(cache, messages, **kwargs):
    params = Controller._cache_params({"messages": messages, **kwargs})
    return cache.get(messages[-1]["content"], "m", params)


_A = [{"role": "user", "content": "alpha"}, {"role": "user", "content": "same ending"}]
_B = [{"role": "user", "content": "beta"}, {"role": "user", "content": "same ending"}]


def test_different_history_with_the_same_last_message_misses(tmp_path):
    cache = _cache(tmp_path)
    _store(cache, _A, "answer-a")

    assert _lookup(cache, _A) == {"answer": "answer-a"}
    assert _lookup(cache, _B) is None


def test_different_system_prompt_misses(tmp_path):
    cache = _cache(tmp_path)
    one = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "q"}]
    two = [{"role": "system", "content": "be verbose"}, {"role": "user", "content": "q"}]
    _store(cache, one, "terse")

    assert _lookup(cache, two) is None


def test_different_tools_miss(tmp_path):
    cache = _cache(tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    _store(cache, msgs, "no-tools")

    assert _lookup(cache, msgs, tools=[{"type": "function"}]) is None


def test_different_image_in_the_last_turn_misses(tmp_path):
    cache = _cache(tmp_path)

    def turn(url):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]

    one, two = turn("data:image/png;base64,AAAA"), turn("data:image/png;base64,BBBB")
    params_one = Controller._cache_params({"messages": one})
    params_two = Controller._cache_params({"messages": two})
    cache.put("what is this", "m", params_one, {"answer": "cat"})

    assert cache.get("what is this", "m", params_two) is None


def test_different_sampling_params_miss(tmp_path):
    cache = _cache(tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    _store(cache, msgs, "cold", temperature=0)

    assert _lookup(cache, msgs, temperature=1) is None


def test_similar_final_turn_in_identical_context_still_hits(tmp_path):
    cache = _cache(tmp_path, embed=lambda t: t.lower().rstrip("?!. "))
    ctx = [{"role": "user", "content": "context"}]
    _store(cache, [*ctx, {"role": "user", "content": "What time is it?"}], "noon")

    hit = _lookup(cache, [*ctx, {"role": "user", "content": "what time is it"}])

    assert hit == {"answer": "noon"}


def test_caller_without_a_context_digest_gets_exact_semantics(tmp_path):
    cache = _cache(tmp_path)
    cache.put("same ending", "m", {"_conversation": "conv-a"}, {"answer": "a"})

    assert cache.get("same ending", "m", {"_conversation": "conv-b"}) is None
    assert cache.get("same ending", "m", {"_conversation": "conv-a"}) == {"answer": "a"}


def test_database_from_before_the_scope_column_opens_and_never_serves_semantically(tmp_path):
    path = tmp_path / "cache.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE completion_cache (key TEXT PRIMARY KEY, prompt TEXT, model TEXT, "
        "response TEXT, embedding BLOB, created_at INTEGER)"
    )
    legacy = np.zeros(64, dtype=np.float32)
    legacy[0] = 1.0
    conn.execute(
        "INSERT INTO completion_cache VALUES (?,?,?,?,?,?)",
        (
            "k",
            "same ending",
            "m",
            json.dumps({"answer": "old"}),
            legacy.tobytes(),
            int(time.time()),
        ),
    )
    conn.commit()
    conn.close()

    cache = _cache(tmp_path)
    cache._get_embedding = lambda text: legacy

    assert _lookup(cache, _A) is None
    _store(cache, _A, "new")
    assert _lookup(cache, _A) == {"answer": "new"}


def test_context_digest_ignores_only_the_last_turns_text():
    base = Controller._cache_params({"messages": [{"role": "user", "content": "x"}]})
    other = Controller._cache_params({"messages": [{"role": "user", "content": "y"}]})
    assert base["_context"] == other["_context"]
    assert base["_conversation"] != other["_conversation"]
