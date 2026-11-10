"""The redact treatment through the controller: egress, restore, refusal, scrubbing.

Runs the real `RedactTransform` and the real token store behind a
recording provider (see `test_request_transform`), under a policy whose
`pii` class has treatment `redact` and may reach the cloud.
"""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path

import pytest

from routellm.class_tiers import check_decision_endpoint
from routellm.config import ConfigError
from routellm.policy import PolicyError, parse_policy
from routellm.redact.placeholders import Redactor
from routellm.redact.tokens import TokenStore
from routellm.redact.transform import RedactTransform
from routellm.requirements import Requirements
from routellm.tests.test_redact_transform import analyzer, leaked
from routellm.tests.test_request_transform import (  # noqa: F401  (fixtures)
    args_of,
    ask,
    build,
    messages,
    provider,
    text_of,
)
from routellm.transform import TransformError, TransformRunner

POLICY = {
    "classes": {
        "pii": {
            "treatment": "redact",
            "reach": ["local", "cloud"],
            "tiers": ["default"],
            "cache": "exact",
        },
        "open": {"treatment": "raw", "reach": ["local", "cloud"], "tiers": ["default"]},
    },
    "default_class": "pii",
}
NAME = "Jean Tremblay"
LOCAL_FIRST = {
    "default": {"router": "pin_strong", "threshold": 0.0, "strong": "loc", "weak": "cld"}
}


@pytest.fixture
def store(tmp_path):
    return TokenStore(str(tmp_path / "tokens.db"))


def make(tmp_path, store, *, analyze=analyzer, **kw):
    transform = RedactTransform(Redactor(store, analyze=analyze))
    kw.setdefault("tags", {"loc": ["local"]})
    return build(
        [transform], tmp_path, policy=parse_policy(POLICY, "p"), strict_registry=True, **kw
    )


def say(controller, text, **kw):
    return ask(controller, messages(text), declared_class=kw.pop("declared_class", "pii"), **kw)


def sync_say(controller, text, **kw):
    return controller.completion(
        model="default", messages=messages(text), declared_class="pii", **kw
    )


def provider_saw(provider):
    return json.dumps([c["messages"] for c in provider.calls])


# --- the main call ---------------------------------------------------------------------


def test_cloud_gets_placeholders_client_gets_the_restored_text(provider, store, tmp_path):
    res = say(make(tmp_path, store), f"Write to {NAME}")
    assert leaked(provider_saw(provider)) == [] and "[[PERSON_" in provider_saw(provider)
    assert res.choices[0].message.content == f"got: Write to {NAME}"


def test_the_sync_entry_point_is_held_to_the_same(provider, store, tmp_path):
    res = sync_say(make(tmp_path, store), f"Write to {NAME}")
    assert leaked(provider_saw(provider)) == []
    assert res.choices[0].message.content == f"got: Write to {NAME}"


def test_a_stream_split_over_three_chunks_is_restored(provider, store, tmp_path):
    provider.stream_cuts = [8, 12]  # "got: [[P" "ERSO" "N_1]]"
    chunks = drain_class(make(tmp_path, store))
    assert text_of(chunks) == f"got: {NAME}"
    assert not any("[[" in (c.choices[0].delta.content or "") for c in chunks if c.choices)


def test_the_sync_stream_is_restored_too(provider, store, tmp_path):
    provider.stream_cuts = [8, 12]
    stream = sync_say(make(tmp_path, store), NAME, stream=True)
    assert text_of(list(stream)) == f"got: {NAME}"


def test_tool_call_arguments_go_out_redacted_and_come_back_restored(provider, store, tmp_path):
    history = [
        {"role": "user", "content": "look it up"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": json.dumps({"who": NAME})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": f"{NAME} found"},
    ]
    provider.tool_args = '{"who": "[[PERSON_1]]"}'
    res = ask(make(tmp_path, store), history, declared_class="pii")
    assert leaked(provider_saw(provider)) == []
    assert json.loads(res.choices[0].message.tool_calls[0].function.arguments) == {"who": NAME}


def test_streamed_tool_arguments_split_mid_placeholder_are_restored(provider, store, tmp_path):
    provider.tool_args = '{"who": "[[PERSON_1]]"}'
    provider.stream_cuts = [14, 18]
    chunks = drain_class(make(tmp_path, store))
    assert json.loads(args_of(chunks)) == {"who": NAME}


def test_the_sibling_fallback_is_redacted_too(provider, store, tmp_path):
    provider.fail = {"openai/cld"}
    tiers = {"default": {"router": "pin_strong", "threshold": 0.0, "strong": "cld", "weak": "cld2"}}
    res = say(make(tmp_path, store, tiers=tiers), f"Write to {NAME}")
    assert [c["model"] for c in provider.calls] == ["openai/cld", "openai/cld2"]
    assert leaked(provider_saw(provider)) == []
    assert res.choices[0].message.content == f"got: Write to {NAME}"


def test_a_local_leaf_stays_raw(provider, store, tmp_path):
    res = say(make(tmp_path, store, tiers=LOCAL_FIRST), f"Write to {NAME}")
    assert provider.calls[0]["model"] == "openai/loc"
    assert NAME in provider_saw(provider)
    assert res.choices[0].message.content == f"got: Write to {NAME}"


def test_a_raw_class_is_not_redacted(provider, store, tmp_path):
    res = say(make(tmp_path, store), f"Write to {NAME}", declared_class="open")
    assert NAME in provider_saw(provider)
    assert res.choices[0].message.content == f"got: Write to {NAME}"


# --- refusal ---------------------------------------------------------------------------


def exploding(text, language):
    raise RuntimeError(f"analyzer exploded on {NAME}")


def test_a_redactor_exception_is_refused_with_no_provider_call(provider, store, tmp_path):
    controller = make(tmp_path, store, analyze=exploding)
    with pytest.raises(TransformError):
        say(controller, f"Write to {NAME}")
    with pytest.raises(TransformError):
        sync_say(controller, f"Write to {NAME}")
    with pytest.raises(TransformError):
        drain_class(controller)
    assert provider.calls == []


def drain_class(controller, text=NAME):
    async def go():
        res = await controller.acompletion(
            model="default", messages=messages(text), stream=True, declared_class="pii"
        )
        return [c async for c in res]

    return asyncio.run(go())


def test_a_restore_failure_is_refused_not_retried_on_another_leaf(provider, store, tmp_path):
    class BadRestore(RedactTransform):
        def inverse(self, text, state):
            raise RuntimeError("cannot restore")

    controller = build(
        [BadRestore(Redactor(store, analyze=analyzer))],
        tmp_path,
        policy=parse_policy(POLICY, "p"),
        strict_registry=True,
        tags={"loc": ["local"]},
    )
    with pytest.raises(TransformError):
        say(controller, f"Write to {NAME}")
    assert len(provider.calls) == 1


def test_a_redact_class_with_no_redactor_is_refused_at_the_call(provider, store, tmp_path):
    controller = make(tmp_path, store)
    controller.transforms = TransformRunner([])
    with pytest.raises(TransformError):
        say(controller, f"Write to {NAME}")
    assert provider.calls == []


def test_a_policy_that_lets_a_redact_class_off_host_needs_the_redactor(tmp_path):
    with pytest.raises(ConfigError, match="redact"):
        build([], tmp_path, policy=parse_policy(POLICY, "p"), strict_registry=True)


def test_a_redact_class_that_stays_local_needs_no_redactor(tmp_path):
    local_only = {
        "classes": {
            "pii": {"treatment": "redact", "reach": ["local"], "tiers": ["default"]},
        },
        "default_class": "pii",
    }
    build(
        [],
        tmp_path,
        policy=parse_policy(local_only, "p"),
        strict_registry=True,
        tags={"cld": ["local"], "cld2": ["local"], "loc": ["local"]},
    )


# --- nothing raw at rest ---------------------------------------------------------------


def files_text(root: Path) -> str:
    out = ""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "tokens.db":
            out += path.read_bytes().decode("utf-8", "replace")
    return out


def mapping_ids(store):
    with sqlite3.connect(store.path) as c:
        return [r[0] for r in c.execute("SELECT id FROM mappings")]


def test_traces_cache_and_logs_hold_no_value_and_no_mapping_id(provider, store, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    controller = make(tmp_path, store)
    for _ in range(2):  # the second is not served from a cache
        say(controller, f"Write to {NAME} <jean.t@example.com>")
        drain_class(controller)
    assert len(provider.calls) == 4
    at_rest = files_text(tmp_path) + caplog.text
    assert leaked(at_rest) == []
    ids = mapping_ids(store)
    assert len(ids) == 4 and not any(i in at_rest for i in ids)


def test_a_redact_class_is_never_cached_even_when_its_policy_says_exact(provider, store, tmp_path):
    controller = build(
        [RedactTransform(Redactor(store, analyze=analyzer))],
        tmp_path,
        cache=True,
        policy=parse_policy(POLICY, "p"),
        strict_registry=True,
        tags={"loc": ["local"]},
    )
    say(controller, f"Write to {NAME}")
    say(controller, f"Write to {NAME}")
    assert len(provider.calls) == 2
    db = tmp_path / "cache.db"
    if db.exists():
        with sqlite3.connect(db) as c:
            tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            assert all(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0 for t in tables)


# --- auxiliary calls -------------------------------------------------------------------


def test_aux_calls_for_a_redact_class_are_local_only(store, tmp_path):
    c = make(tmp_path, store)
    assert c._aux_denial("cld", "pii") == "redact:pii:aux_not_local"
    assert c._aux_denial("loc", "pii") is None
    assert c._aux_denial("cld", "open") is None  # raw class: unchanged
    assert c._aux_denial("cld", None) is None


def test_the_main_call_may_still_reach_the_cloud_through_the_transform(store, tmp_path):
    c = make(tmp_path, store)
    assert c._can_serve("cld", Requirements(data_class="pii")) is None
    assert c._policy_reason("cld", Requirements(data_class="pii")) is None


def test_a_tier_is_local_only_when_every_leaf_is(store, tmp_path):
    c = make(tmp_path, store)
    assert c._aux_denial("default", "pii") is not None  # reaches cld


def test_a_router_that_embeds_through_the_cloud_is_skipped(store, tmp_path):
    c = make(tmp_path, store)
    c.embedding_endpoint = "cld"
    c.routers["fake"] = type("R", (), {"requires_embedding": True})()
    assert c._router_denial("fake", "pii") == "redact:pii:aux_not_local"
    c.embedding_endpoint = "loc"
    assert c._router_denial("fake", "pii") is None


def test_the_semantic_cache_never_embeds_a_redact_class(store, tmp_path):
    c = make(tmp_path, store)
    c.cache.config.semantic_enabled = True
    assert c._semantic_cache_allowed("pii") is False


def test_a_decision_endpoint_must_be_local_when_a_class_redacts(store, tmp_path):
    c = make(tmp_path, store)
    with pytest.raises(PolicyError, match="pii"):
        check_decision_endpoint(c.policy, c.endpoints, "cld")
    check_decision_endpoint(c.policy, c.endpoints, "loc")
