"""Reversible placeholders and the local token store.

Pure Python: spans come from the pattern tier or an injected analyzer, so
none of this needs the `redact` extra.
"""

import logging
import os
import re
import stat

import pytest

from routellm.redact import analyze_patterns
from routellm.redact.placeholders import (
    MappingError,
    MappingExpired,
    RedactContext,
    RedactError,
    Redactor,
    split_incomplete,
)
from routellm.redact.tokens import TokenStore

NAMES = ("Jean Tremblay", "Marie Gagnon", "Alice Smith")
FR = "Bonjour, Jean Tremblay (jean.t@example.com) est né le 12 mars 1985. NAS 046 454 286. Écrire à Marie Gagnon."
EN = "Hi, Alice Smith <alice@example.org> was born on March 3, 1990. Cc Jean Tremblay; ask Alice Smith again."


def analyzer(text, language):
    """Pattern tier plus a toy name finder standing in for NER."""
    spans = [s.to_dict() for s in analyze_patterns(text, language)]
    for name in NAMES:
        for m in re.finditer(re.escape(name), text):
            spans.append({"start": m.start(), "end": m.end(), "category": "person", "score": 0.9})
    return sorted(spans, key=lambda s: s["start"])


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return TokenStore(str(tmp_path / "state" / "tokens.db"), clock=clock)


@pytest.fixture
def redactor(store):
    return Redactor(store, analyze=analyzer)


def ctx(request_id="r1", conv="c1", principal="team-a", language="en"):
    return RedactContext(
        principal=principal, request_id=request_id, conversation_id=conv, language=language
    )


PLACEHOLDER = re.compile(r"\[\[[A-Z_]+_\d+\]\]")


# --- round trip ---------------------------------------------------------


@pytest.mark.parametrize("text,lang", [(FR, "fr"), (EN, "en")])
def test_round_trip_restores_exactly(redactor, text, lang):
    c = ctx(language=lang)
    out, mid = redactor.redact(text, c)
    assert out != text
    for secret in ("Tremblay", "Gagnon", "Smith", "example", "046 454 286", "1985", "1990"):
        assert secret not in out
    assert redactor.restore(out, mid, c) == text


def test_placeholders_are_category_typed(redactor):
    out, _ = redactor.redact(FR, ctx(language="fr"))
    found = PLACEHOLDER.findall(out)
    assert {p.split("_")[0] for p in found} >= {"[[PERSON", "[[EMAIL", "[[SIN"}


def test_text_without_spans_is_untouched(redactor):
    out, mid = redactor.redact("nothing sensitive here", ctx())
    assert out == "nothing sensitive here"
    assert redactor.restore(out, mid, ctx()) == out


# --- determinism within a request / conversation ------------------------


def test_same_value_same_placeholder_within_request(redactor):
    out, _ = redactor.redact(EN, ctx())
    smith = [m for m in PLACEHOLDER.findall(out)]
    # Alice Smith appears twice: one placeholder, used twice
    assert len(smith) != len(set(smith))
    assert len({p for p in smith if p.startswith("[[PERSON")}) == 2


def test_same_value_same_placeholder_across_calls_of_one_request(redactor):
    a, _ = redactor.redact("Alice Smith", ctx())
    b, _ = redactor.redact("so said Alice Smith", ctx())
    assert a in b


def test_distinct_values_get_distinct_placeholders(redactor):
    out, _ = redactor.redact("Alice Smith and Jean Tremblay", ctx())
    a, b = PLACEHOLDER.findall(out)
    assert a != b


# --- unlinkable across requests unless opted in -------------------------


def test_placeholders_not_linkable_across_requests(redactor):
    outs = [redactor.redact("Alice Smith", ctx(request_id=f"r{i}"))[0] for i in range(5)]
    assert len(set(outs)) == 5


def test_numbering_not_reused_across_requests(redactor):
    first = set(PLACEHOLDER.findall(redactor.redact(EN, ctx(request_id="a"))[0]))
    second = set(PLACEHOLDER.findall(redactor.redact(EN, ctx(request_id="b"))[0]))
    assert not first & second


def test_stable_across_requests_links_values_in_a_conversation(store):
    r = Redactor(store, analyze=analyzer, stable_across_requests=True)
    a, _ = r.redact("Alice Smith", ctx(request_id="r1"))
    b, _ = r.redact("Alice Smith", ctx(request_id="r2"))
    assert a == b


def test_stable_mode_needs_a_conversation_id(store):
    r = Redactor(store, analyze=analyzer, stable_across_requests=True)
    with pytest.raises(RedactError):
        r.redact("Alice Smith", RedactContext(principal="p", request_id="r"))


# --- collision with text already in the prompt --------------------------


def test_placeholder_literal_in_prompt_is_not_reused(redactor):
    c = ctx()
    text = "Keep [[PERSON_1]] and [[PERSON_2]] as is. Alice Smith asked."
    out, mid = redactor.redact(text, c)
    assert "Alice Smith" not in out
    assigned = [p for p in PLACEHOLDER.findall(out) if p not in ("[[PERSON_1]]", "[[PERSON_2]]")]
    assert len(assigned) == 1
    # the literals survive the round trip untouched
    assert redactor.restore(out, mid, c) == text
    # and a model echoing the literal does not get it "restored"
    assert redactor.restore("see [[PERSON_1]]", mid, c) == "see [[PERSON_1]]"


def test_lenient_spelling_in_prompt_is_reserved_too(redactor):
    c = ctx()
    text = "[[ person_1 ]] vs Alice Smith"
    out, mid = redactor.redact(text, c)
    assert redactor.restore(out, mid, c) == text


def test_existing_mapping_colliding_with_prompt_literal_fails_closed(store):
    r = Redactor(store, analyze=analyzer, stable_across_requests=True)
    out, _ = r.redact("Alice Smith", ctx(request_id="r1"))
    with pytest.raises(RedactError):
        r.redact(f"{out} Alice Smith", ctx(request_id="r2"))


# --- restore tolerance and refusal --------------------------------------


def test_restore_tolerates_case_and_inner_spacing(redactor):
    c = ctx()
    out, mid = redactor.redact("Alice Smith", c)
    ph = PLACEHOLDER.findall(out)[0]
    mangled = ph.replace("[[", "[[ ").replace("]]", " ]]").lower()
    assert redactor.restore(f"said {mangled}.", mid, c) == "said Alice Smith."


@pytest.mark.parametrize(
    "mangled",
    [
        "[[PERSON_1",
        "PERSON_1]]",
        "[PERSON_1]",
        "[[PERSON_1]",
        "PERSON_1",
        "[[PERSON_",
        "[[PERSON_11]]",
        "[[PERSON]]",
        "[[PERSON_1 ]x",
    ],
)
def test_mangled_placeholder_left_alone_never_guessed(redactor, mangled):
    c = ctx()
    out, mid = redactor.redact("Alice Smith", c)
    assert out == "[[PERSON_1]]"
    assert redactor.restore(mangled, mid, c) == mangled


def test_multi_digit_numbers_resolve_exactly(redactor):
    c = ctx()

    def many(text, language):
        return [
            {"start": m.start(), "end": m.end(), "category": "person", "score": 1.0}
            for m in re.finditer(r"Alice Smith\d+", text)
        ]

    r = Redactor(redactor.store, analyze=many)
    text = " ".join(f"Alice Smith{i}" for i in range(12))
    out, mid = r.redact(text, c)
    assert len(set(PLACEHOLDER.findall(out))) == 12
    assert r.restore(out, mid, c) == text


def test_restore_is_single_pass(redactor):
    c = ctx()
    out, mid = redactor.redact("Alice Smith", c)
    ph = PLACEHOLDER.findall(out)[0]
    assert redactor.restore(f"{ph}{ph}", mid, c) == "Alice SmithAlice Smith"


def test_unknown_placeholder_in_response_untouched(redactor):
    c = ctx()
    _, mid = redactor.redact("Alice Smith", c)
    assert redactor.restore("[[IBAN_9999]]", mid, c) == "[[IBAN_9999]]"


# --- split for streaming -------------------------------------------------


@pytest.mark.parametrize(
    "text,pending",
    [
        ("hello [[PERS", "[[PERS"),
        ("hello [[PERSON_1", "[[PERSON_1"),
        ("hello [[PERSON_1]", "[[PERSON_1]"),
        ("hello [", "["),
        ("hello [[", "[["),
        ("hello [[PERSON_1]]", ""),
        ("hello", ""),
        ("a [x] b", ""),
    ],
)
def test_split_incomplete(text, pending):
    safe, held = split_incomplete(text)
    assert held == pending
    assert safe + held == text


# --- expiry, scope, mode --------------------------------------------------


def test_expired_mapping_fails_closed(redactor, clock):
    c = ctx()
    out, mid = redactor.redact("Alice Smith", c)
    clock.now += 24 * 3600 + 1
    with pytest.raises(MappingExpired):
        redactor.restore(out, mid, c)


def test_not_expired_just_before_ttl(redactor, clock):
    c = ctx()
    out, mid = redactor.redact("Alice Smith", c)
    clock.now += 24 * 3600 - 1
    assert redactor.restore(out, mid, c) == "Alice Smith"


def test_ttl_is_configurable(tmp_path, clock):
    s = TokenStore(str(tmp_path / "t.db"), ttl_s=60, clock=clock)
    r = Redactor(s, analyze=analyzer)
    out, mid = r.redact("Alice Smith", ctx())
    clock.now += 61
    with pytest.raises(MappingExpired):
        r.restore(out, mid, ctx())


def test_expired_rows_are_purged(store, redactor, clock):
    redactor.redact("Alice Smith", ctx())
    clock.now += 24 * 3600 + 1
    assert store.purge_expired() > 0
    assert store.row_counts() == {"mappings": 0, "tokens": 0}


def test_unknown_mapping_id_fails_closed(redactor):
    with pytest.raises(MappingError):
        redactor.restore("[[PERSON_1]]", "nope", ctx())


def test_other_principal_cannot_restore(redactor):
    out, mid = redactor.redact("Alice Smith", ctx(principal="team-a"))
    with pytest.raises(MappingError):
        redactor.restore(out, mid, ctx(principal="team-b"))


def test_other_conversation_cannot_restore(store):
    r = Redactor(store, analyze=analyzer, stable_across_requests=True)
    out, mid = r.redact("Alice Smith", ctx(conv="c1"))
    with pytest.raises(MappingError):
        r.restore(out, mid, ctx(conv="c2"))


def test_other_request_cannot_restore(redactor):
    out, mid = redactor.redact("Alice Smith", ctx(request_id="r1"))
    with pytest.raises(MappingError):
        redactor.restore(out, mid, ctx(request_id="r2"))


# --- the store itself ------------------------------------------------------


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_store_files_and_dir_are_private(store, redactor):
    redactor.redact("Alice Smith", ctx())
    assert mode(store.path) == 0o600
    assert mode(store.key_path) == 0o600
    assert mode(os.path.dirname(store.path)) == 0o700


def test_loose_existing_file_is_tightened(tmp_path):
    p = tmp_path / "tokens.db"
    p.write_bytes(b"")
    os.chmod(p, 0o644)
    TokenStore(str(p))
    assert mode(p) == 0o600


def test_umask_cannot_loosen(tmp_path):
    old = os.umask(0)
    try:
        s = TokenStore(str(tmp_path / "d" / "t.db"))
        assert mode(s.path) == 0o600 and mode(s.key_path) == 0o600
    finally:
        os.umask(old)


def test_values_not_stored_as_lookup_keys(store, redactor):
    redactor.redact("Alice Smith", ctx())
    rows = store.dump_lookup_keys()
    assert rows and all("Alice" not in k and len(k) == 64 for k in rows)


def test_lookup_key_depends_on_gateway_key_and_scope(tmp_path):
    a = TokenStore(str(tmp_path / "a" / "t.db"))
    b = TokenStore(str(tmp_path / "b" / "t.db"))
    assert a.lookup_key("s", "Alice Smith") != b.lookup_key("s", "Alice Smith")
    assert a.lookup_key("s1", "Alice Smith") != a.lookup_key("s2", "Alice Smith")


def test_key_is_persisted_across_opens(tmp_path):
    p = str(tmp_path / "t.db")
    assert TokenStore(p).lookup_key("s", "v") == TokenStore(p).lookup_key("s", "v")


def test_mapping_id_unguessable_and_distinct(redactor):
    ids = {redactor.redact("Alice Smith", ctx())[1] for _ in range(5)}
    assert len(ids) == 5 and all(len(i) >= 22 for i in ids)


def test_nothing_logged_and_errors_carry_no_values(redactor, caplog, clock):
    caplog.set_level(logging.DEBUG)
    c = ctx()
    out, mid = redactor.redact("Alice Smith alice@example.org", c)
    redactor.restore(out, mid, c)
    clock.now += 10**6
    with pytest.raises(MappingExpired) as ei:
        redactor.restore(out, mid, c)
    assert "Alice" not in str(ei.value) and "example.org" not in str(ei.value)
    assert "Alice" not in caplog.text and "example.org" not in caplog.text


def test_context_repr_has_no_text():
    assert "Alice" not in repr(ctx())
