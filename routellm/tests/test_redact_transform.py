"""The redact transform on its own: what it rewrites, leaves raw, refuses and restores."""

import json
import logging
import re

import pytest

from routellm.redact import analyze_patterns
from routellm.redact.placeholders import Redactor
from routellm.redact.tokens import MappingError, TokenStore
from routellm.redact.transform import RedactTransform, union_languages
from routellm.transform import TransformContext, TransformError, TransformRunner

NAMES = ("Jean Tremblay", "Marie Gagnon", "Alice Smith")
CANARIES = (*NAMES, "jean.t@example.com", "046 454 286", "Québec")


def analyzer(text, language):
    spans = [s.to_dict() for s in analyze_patterns(text, language)]
    for word in (*NAMES, "Québec"):
        for m in re.finditer(re.escape(word), text):
            spans.append({"start": m.start(), "end": m.end(), "category": "person", "score": 0.9})
    return spans


def ctx(reach=("cloud",), treatment="redact", **kw):
    return TransformContext(
        endpoint="e",
        model="m",
        reach=frozenset(reach),
        data_class="pii",
        treatment=treatment,
        request_id="req-1",
        **kw,
    )


@pytest.fixture
def store(tmp_path):
    return TokenStore(str(tmp_path / "t.db"))


@pytest.fixture
def transform(store):
    return RedactTransform(Redactor(store, analyze=analyzer), non_text="pass")


def leaked(obj) -> list[str]:
    blob = json.dumps(obj, ensure_ascii=False)
    return [c for c in CANARIES if c in blob]


MESSAGES = [
    {"role": "system", "content": "Client: Jean Tremblay, jean.t@example.com"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "NAS 046 454 286 pour Marie Gagnon"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"who": "Alice Smith", "city": "Québec"}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "found Alice Smith in Québec"},
    {"role": "assistant", "content": "Done for Jean Tremblay"},
]


# --- forward --------------------------------------------------------------------------


def test_every_message_part_is_redacted(transform):
    out, state = transform.forward(MESSAGES, ctx())
    assert state is not None
    assert leaked(out) == []
    assert out[1]["content"][1] == MESSAGES[1]["content"][1]  # non-text part untouched
    assert "[[PERSON_" in out[0]["content"]


def test_a_part_that_is_not_text_is_refused_by_default(store):
    strict = RedactTransform(Redactor(store, analyze=analyzer))
    runner = TransformRunner([strict])
    with pytest.raises(TransformError):
        runner.forward(MESSAGES, ctx())
    out, _ = runner.forward(MESSAGES, ctx(("local",)))  # local leaf: raw, nothing to refuse
    assert out is MESSAGES


def test_tool_call_arguments_are_redacted_even_when_json_escapes_them(transform):
    escaped = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {
                        "name": "f",
                        "arguments": json.dumps({"city": "Québec", "q": "Jean Tremblay"}),
                    },
                }
            ],
        }
    ]
    assert "\\u00e9" in escaped[0]["tool_calls"][0]["function"]["arguments"]
    out, _ = transform.forward(escaped, ctx())
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert leaked(out) == [] and "\\u00e9" not in args and "Qu" not in args
    parsed = json.loads(args)
    assert set(parsed) == {"city", "q"}


def test_legacy_function_call_arguments_are_redacted(transform):
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "f", "arguments": '{"a": "Jean Tremblay"}'},
        }
    ]
    out, _ = transform.forward(msgs, ctx())
    assert leaked(out) == []


def test_arguments_that_are_not_json_are_still_redacted(transform):
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "f", "arguments": "who=Jean Tremblay"},
                }
            ],
        }
    ]
    out, _ = transform.forward(msgs, ctx())
    assert leaked(out) == []


def test_the_callers_messages_are_not_mutated(transform):
    before = json.dumps(MESSAGES)
    transform.forward(MESSAGES, ctx())
    assert json.dumps(MESSAGES) == before


@pytest.mark.parametrize("reach", [("cloud",), ("local", "cloud"), ("vpn",), ()])
def test_every_leaf_that_is_not_only_local_is_redacted(transform, reach):
    out, state = transform.forward(MESSAGES, ctx(reach))
    assert state is not None and leaked(out) == []


def test_a_local_leaf_is_left_raw(transform):
    out, state = transform.forward(MESSAGES, ctx(("local",)))
    assert out is MESSAGES and state is None


@pytest.mark.parametrize("treatment", ["raw", "block", None])
def test_other_treatments_are_not_this_transforms_business(transform, treatment):
    out, state = transform.forward(MESSAGES, ctx(treatment=treatment))
    assert out is MESSAGES and state is None


def test_selects_only_a_redact_class(transform):
    assert transform.selects(TransformContext(treatment="redact", data_class="pii"))
    assert not transform.selects(TransformContext(treatment="raw", data_class="open"))
    assert not transform.selects(TransformContext())


def test_the_output_holds_no_mapping_id_and_the_log_no_value(transform, store, caplog):
    caplog.set_level(logging.DEBUG)
    out, state = transform.forward(MESSAGES, ctx())
    ids = [r[0] for r in _mapping_ids(store)]
    assert ids and not any(i in json.dumps(out) for i in ids)
    assert leaked(caplog.text) == []
    assert not any(i in caplog.text for i in ids)


def _mapping_ids(store):
    import sqlite3

    with sqlite3.connect(store.path) as c:
        return c.execute("SELECT id FROM mappings").fetchall()


def test_one_request_uses_one_mapping(transform, store):
    transform.forward(MESSAGES, ctx())
    assert store.row_counts()["mappings"] == 1


def test_text_without_a_span_is_unchanged(transform):
    plain = [{"role": "user", "content": "hello there"}]
    out, _ = transform.forward(plain, ctx())
    assert out == plain


# --- refusal --------------------------------------------------------------------------


def test_a_redactor_exception_refuses_the_request(store):
    def boom(text, language):
        raise RuntimeError("analyzer exploded on Jean Tremblay")

    runner = TransformRunner([RedactTransform(Redactor(store, analyze=boom))])
    with pytest.raises(TransformError) as exc:
        runner.forward(MESSAGES, ctx())
    assert "Jean Tremblay" not in exc.value.reason


def test_a_span_outside_the_text_refuses_the_request(store):
    bad = lambda text, language: [{"start": 0, "end": 9999, "category": "person", "score": 1}]  # noqa: E731
    runner = TransformRunner([RedactTransform(Redactor(store, analyze=bad))])
    with pytest.raises(TransformError):
        runner.forward(MESSAGES, ctx())


def test_an_unreadable_mapping_refuses_the_request(store, monkeypatch):
    def gone(*a, **k):
        raise MappingError("expired")

    monkeypatch.setattr(store, "resolve", gone)
    runner = TransformRunner([RedactTransform(Redactor(store, analyze=analyzer))])
    with pytest.raises(TransformError):
        runner.forward(MESSAGES, ctx())


# --- restore --------------------------------------------------------------------------


def roundtrip(transform, text):
    out, state = transform.forward([{"role": "user", "content": text}], ctx())
    return out[0]["content"], state


def test_text_is_restored(transform):
    sent, state = roundtrip(transform, "Écrire à Jean Tremblay (jean.t@example.com)")
    assert transform.inverse(f"Reply to: {sent}", state) == (
        "Reply to: Écrire à Jean Tremblay (jean.t@example.com)"
    )


def test_a_value_restored_into_arguments_keeps_the_json_valid(transform):
    value = 'Jean "JT" Tremblay\\o'
    t = RedactTransform(
        Redactor(
            transform.redactor.store,
            analyze=lambda s, lang: (
                [{"start": 0, "end": len(value), "category": "person", "score": 1}]
                if s == value
                else []
            ),
        )
    )
    out, state = t.forward([{"role": "user", "content": value}], ctx())
    placeholder = out[0]["content"]
    restored = t.inverse_arguments(json.dumps({"who": placeholder}), state)
    assert json.loads(restored) == {"who": value}
    assert t.inverse(placeholder, state) == value  # text is not escaped


def test_an_unknown_placeholder_is_left_as_written(transform):
    _, state = roundtrip(transform, "Jean Tremblay")
    assert transform.inverse("see [[PERSON_99]]", state) == "see [[PERSON_99]]"


def test_a_placeholder_split_over_three_pieces_is_restored(transform):
    sent, state = roundtrip(transform, "Jean Tremblay")
    assert re.fullmatch(r"\[\[PERSON_\d+\]\]", sent)
    cut = [sent[:3], sent[3:7], sent[7:]]
    restorer = transform.restorer(state)
    out = "".join(restorer.feed(p) for p in cut) + restorer.finish()
    assert out == "Jean Tremblay"


def test_a_held_tail_that_never_completes_is_released_as_written(transform):
    _, state = roundtrip(transform, "Jean Tremblay")
    restorer = transform.restorer(state)
    out = restorer.feed("price [[PERS") + restorer.finish()
    assert out == "price [[PERS"


def test_argument_pieces_split_mid_placeholder_are_restored_as_json(transform):
    sent, state = roundtrip(transform, "Jean Tremblay")
    args = json.dumps({"who": sent})
    cut = [args[:10], args[10:14], args[14:]]
    restorer = transform.arguments_restorer(state)
    out = "".join(restorer.feed(p) for p in cut) + restorer.finish()
    assert json.loads(out) == {"who": "Jean Tremblay"}


def test_languages_are_unioned_so_a_fr_name_is_found_under_en(store):
    def per_language(text, language):
        return (
            [{"start": 0, "end": 4, "category": "person", "score": 1}] if language == "fr" else []
        )

    wrapped = union_languages(per_language, ("en", "fr"))
    assert wrapped("Jean x", "en") == [{"start": 0, "end": 4, "category": "person", "score": 1}]
