"""Pattern recognizers: pure Python, run on a base install."""

import pytest

from routellm.redact.recognizers import (
    Span,
    analyze_patterns,
    compile_client_id_patterns,
    find_amounts,
    find_client_ids,
    find_date_of_birth,
    find_emails,
    find_postal_codes,
    find_ramq,
    find_sin,
    luhn_valid,
    resolve_overlaps,
)


def texts(text, spans):
    return [text[s.start : s.end] for s in spans]


# -- Luhn / SIN ---------------------------------------------------------


def test_luhn_known_values():
    assert luhn_valid("046454286")
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("046454287")
    assert not luhn_valid("")
    assert not luhn_valid("12a4")


@pytest.mark.parametrize("sin", ["046 454 286", "046-454-286", "046454286", "130 692 544"])
def test_sin_valid_forms_found(sin):
    text = f"mon NAS: {sin}."
    spans = find_sin(text)
    assert texts(text, spans) == [sin]
    assert spans[0].category == "sin"


@pytest.mark.parametrize("sin", ["046 454 287", "123 456 789", "046-454-287", "000 000 001"])
def test_sin_failing_luhn_rejected(sin):
    assert find_sin(f"ref {sin} end") == []


def test_sin_mixed_separators_rejected():
    assert find_sin("046 454-286") == []


def test_sin_not_cut_out_of_longer_digit_runs():
    assert find_sin("0464542861") == []
    assert find_sin("4111 1111 1111 1111") == []


def test_sin_cue_raises_score():
    plain = find_sin("code 046 454 286")[0].score
    cued = find_sin("NAS 046 454 286")[0].score
    assert cued > plain


# -- RAMQ ---------------------------------------------------------------


@pytest.mark.parametrize("ramq", ["DUPJ85031234", "DUPJ 8503 1234", "TREM 9155 0112"])
def test_ramq_valid(ramq):
    text = f"carte {ramq} ok"
    assert texts(text, find_ramq(text)) == [ramq]


@pytest.mark.parametrize(
    "bad",
    [
        "DUPJ85133412",  # month 13
        "DUPJ85030012",  # day 00
        "DUPJ85037012",  # day 70
        "DUP85031234",  # three letters
        "dupj85031234",  # lowercase
        "XDUPJ85031234",  # glued to a longer token
        "DUPJ850312345",  # nine digits
    ],
)
def test_ramq_invalid(bad):
    assert find_ramq(f"carte {bad} ok") == []


# -- postal code --------------------------------------------------------


@pytest.mark.parametrize("pc", ["H2X 1Y4", "h2x1y4", "K1A-0B1", "M5V 2T6"])
def test_postal_valid(pc):
    text = f"adresse {pc}, Montréal"
    assert texts(text, find_postal_codes(text)) == [pc]


@pytest.mark.parametrize(
    "bad", ["D2X 1Y4", "H2X 1U4", "W1A 1A1", "H2X 1Y", "H22 1Y4", "AH2X 1Y4", "H2X 1Y44"]
)
def test_postal_invalid(bad):
    assert find_postal_codes(f"adresse {bad}") == []


# -- amounts ------------------------------------------------------------


@pytest.mark.parametrize(
    "amount",
    ["1 234,56 $", "12,50 €", "1\u00a0234,56\u00a0$", "1 234 567,89 €", "300 $", "75 CAD", "9,5 €"],
)
def test_amount_valid(amount):
    text = f"total {amount} dû"
    assert texts(text, find_amounts(text)) == [amount]


@pytest.mark.parametrize("bad", ["total 1234,56", "il a 12 chats", "version 2,5", "$"])
def test_amount_needs_a_currency(bad):
    assert find_amounts(bad) == []


def test_amount_not_cut_from_a_longer_number():
    assert find_amounts("x 11.234,56 $") == []
    assert find_amounts("ref 1.5 €") == []


# -- date of birth ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "date"),
    [
        ("Né le 12 mars 1985 à Laval", "12 mars 1985"),
        ("née le 12/03/1985", "12/03/1985"),
        ("date de naissance : 12-03-1985", "12-03-1985"),
        ("DDN 1985-03-12", "1985-03-12"),
        ("naissance: 1er février 1990", "1er février 1990"),
        ("Date de naissance: 3 août 2001", "3 août 2001"),
    ],
)
def test_dob_fr(text, date):
    assert texts(text, find_date_of_birth(text, "fr")) == [date]


@pytest.mark.parametrize(
    ("text", "date"),
    [
        ("born on March 12, 1985", "March 12, 1985"),
        ("Date of birth: 03/12/1985", "03/12/1985"),
        ("DOB: 25/12/1985", "25/12/1985"),
        ("birthday 1985-03-12", "1985-03-12"),
        ("born 12 March 1985", "12 March 1985"),
        ("DOB Sept. 9, 1970", "Sept. 9, 1970"),
    ],
)
def test_dob_en(text, date):
    assert texts(text, find_date_of_birth(text, "en")) == [date]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("réunion le 12 mars 1985", "fr"),  # a date, no birth cue
        ("meeting on March 12, 1985", "en"),
        ("né le 31/02/1985", "fr"),  # not a calendar date
        ("né le 12/13/1985", "fr"),
        ("born 13/13/1985", "en"),
        ("né le 12 marzo 1985", "fr"),  # unknown month
        ("born on March 12, 1850", "en"),  # before 1900
        ("born 12 mars 1985", "en"),  # French month, English text
    ],
)
def test_dob_rejected(text, language):
    assert find_date_of_birth(text, language) == []


def test_dob_numeric_order_follows_language():
    # 03/12: 3 December in French, March 12 in English; both are dates.
    assert find_date_of_birth("né le 03/12/1985", "fr")
    assert find_date_of_birth("born 03/12/1985", "en")
    # 25/12 only reads day-first; 12/25 only month-first.
    assert find_date_of_birth("né le 12/25/1985", "fr") == []
    assert find_date_of_birth("born 12/25/1985", "en")


def test_dob_unknown_language():
    with pytest.raises(ValueError, match="unsupported language"):
        find_date_of_birth("born 1985-03-12", "de")


# -- email --------------------------------------------------------------


def test_email():
    text = "écrire à jean.dupont+x@ex-ample.co.uk ou a@b"
    assert texts(text, find_emails(text)) == ["jean.dupont+x@ex-ample.co.uk"]


# -- client ids ---------------------------------------------------------


def test_client_ids_match_configured_patterns_only():
    pats = compile_client_id_patterns([r"CLI-\d{6}", r"\bAC[A-Z]{2}\d{4}\b"])
    text = "dossier CLI-123456 et ACXY0042; CLI-12 non"
    assert texts(text, find_client_ids(text, pats)) == ["CLI-123456", "ACXY0042"]
    assert find_client_ids(text, []) == []


def test_client_id_bad_regex_fails_loud():
    with pytest.raises(ValueError, match="bad regex"):
        compile_client_id_patterns(["(unclosed"])


def test_client_id_empty_matching_regex_rejected():
    with pytest.raises(ValueError, match="empty string"):
        compile_client_id_patterns(["x*"])


def test_client_id_non_string_rejected():
    with pytest.raises(ValueError, match="expected a string"):
        compile_client_id_patterns([42])


# -- composition --------------------------------------------------------


def test_overlap_keeps_strongest():
    kept = resolve_overlaps([Span(0, 10, "a", 0.5), Span(5, 12, "b", 0.9), Span(20, 25, "c", 0.1)])
    assert [s.category for s in kept] == ["b", "c"]


def test_overlap_tie_prefers_longer():
    kept = resolve_overlaps([Span(0, 4, "short", 0.5), Span(0, 9, "long", 0.5)])
    assert [s.category for s in kept] == ["long"]


def test_analyze_patterns_all_categories_with_offsets():
    text = (
        "Né le 12 mars 1985, NAS 046 454 286, RAMQ DUPJ 8503 1234, H2X 1Y4, "
        "1 234,56 $, jean@ex.fr, id CLI-123456"
    )
    pats = compile_client_id_patterns([r"CLI-\d{6}"])
    spans = analyze_patterns(text, "fr", pats)
    got = {s.category: text[s.start : s.end] for s in spans}
    assert got == {
        "date_of_birth": "12 mars 1985",
        "sin": "046 454 286",
        "ramq": "DUPJ 8503 1234",
        "postal_code": "H2X 1Y4",
        "amount": "1 234,56 $",
        "email": "jean@ex.fr",
        "client_id": "CLI-123456",
    }
    assert spans == sorted(spans, key=lambda s: s.start)
    assert all(0 < s.score <= 1 for s in spans)


def test_span_dict_shape():
    assert Span(1, 2, "sin", 0.9).to_dict() == {
        "start": 1,
        "end": 2,
        "category": "sin",
        "score": 0.9,
    }


def test_empty_text():
    assert analyze_patterns("", "en") == []
