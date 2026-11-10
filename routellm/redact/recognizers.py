"""Pattern recognizers for personal data, pure Python.

Nothing here imports Presidio, spaCy or any other optional dependency;
the functions take text and return spans, so they are testable and
usable on a base install. `routellm.redact.presidio_adapter` wraps them
for the Presidio engine, which adds the named-entity categories (person,
location) on top.

A span is `{start, end, category, score}`, offsets in characters into
the text given. Categories:

    sin, ramq, postal_code, amount, date_of_birth, client_id, email

Guards named in the docs are real checks, not shape matches: a SIN must
pass the Luhn checksum, a RAMQ number must carry a calendar-plausible
birth month and day, a postal code must use the letters Canada Post
issues, a date of birth must be a real calendar date after a cue word.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

__all__ = [
    "LANGUAGES",
    "Span",
    "analyze_patterns",
    "compile_client_id_patterns",
    "find_amounts",
    "find_client_ids",
    "find_date_of_birth",
    "find_emails",
    "find_postal_codes",
    "find_ramq",
    "find_sin",
    "luhn_valid",
    "resolve_overlaps",
]

#: Languages the recognizers know the formats of.
LANGUAGES: frozenset[str] = frozenset({"fr", "en"})


@dataclass(frozen=True)
class Span:
    """A detected span; `start`/`end` are character offsets, end exclusive."""

    start: int
    end: int
    category: str
    score: float

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "category": self.category,
            "score": self.score,
        }


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a string of digits."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _check_language(language: str) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language {language!r}; expected one of {sorted(LANGUAGES)}")


# -- context cues ------------------------------------------------------

_CUE_WINDOW = 40


def _has_cue(text: str, start: int, cue: re.Pattern[str]) -> bool:
    return cue.search(text[max(0, start - _CUE_WINDOW) : start]) is not None


# -- SIN ---------------------------------------------------------------

_SIN = re.compile(r"(?<![\d-])\d{3}([ -]?)\d{3}\1\d{3}(?![\d-])")
_SIN_CUE = re.compile(r"\b(?:sin|nas|social insurance|assurance sociale)\b", re.I)


def find_sin(text: str) -> list[Span]:
    """Canadian SIN: nine digits, spaced, dashed or bare, Luhn-valid."""
    out = []
    for m in _SIN.finditer(text):
        if not luhn_valid(re.sub(r"\D", "", m.group())):
            continue
        score = 0.95 if _has_cue(text, m.start(), _SIN_CUE) else 0.8
        out.append(Span(m.start(), m.end(), "sin", score))
    return out


# -- RAMQ --------------------------------------------------------------

_RAMQ = re.compile(r"(?<![A-Za-z0-9])[A-Z]{4}( ?)(\d{4})\1(\d{4})(?![A-Za-z0-9])")


def find_ramq(text: str) -> list[Span]:
    """Quebec health number: four letters then eight digits.

    The digits are YYMMDDNN: birth year, month (+50 for women) and day,
    then two administrative digits. A month or day outside the calendar
    range is not a RAMQ number.
    """
    out = []
    for m in _RAMQ.finditer(text):
        digits = m.group(2) + m.group(3)
        month, day = int(digits[2:4]), int(digits[4:6])
        if month > 50:
            month -= 50
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        out.append(Span(m.start(), m.end(), "ramq", 0.9))
    return out


# -- postal code -------------------------------------------------------

# Canada Post never issues D, F, I, O, Q, U; W and Z do not start a code.
_POSTAL = re.compile(
    r"(?<![A-Za-z0-9])[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]"
    r"[ -]?\d[ABCEGHJ-NPRSTV-Z]\d(?![A-Za-z0-9])",
    re.I,
)


def find_postal_codes(text: str) -> list[Span]:
    """Canadian postal code, `A1A 1A1` (space, dash or none; any case)."""
    return [Span(m.start(), m.end(), "postal_code", 0.85) for m in _POSTAL.finditer(text)]


# -- amounts -----------------------------------------------------------

_SP = "[ \u00a0\u202f]"
_AMOUNT = re.compile(
    rf"(?<![\d,.])(?:\d{{1,3}}(?:{_SP}\d{{3}})+|\d+)(?:,\d{{1,2}})?{_SP}?(?:\$|€|(?:CAD|EUR|USD)\b)"
)


def find_amounts(text: str) -> list[Span]:
    """French-format money: `1 234,56 $`, `12,50 €`, `300 CAD`.

    Space (also no-break) groups thousands, comma marks decimals, the
    currency follows the number. A bare number is not an amount.
    """
    return [Span(m.start(), m.end(), "amount", 0.8) for m in _AMOUNT.finditer(text)]


# -- date of birth -----------------------------------------------------

_MONTHS = {
    "fr": {
        "janvier": 1,
        "février": 2,
        "fevrier": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
        "decembre": 12,
    },
    "en": {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    },
}
_DOB_CUE = {
    "fr": re.compile(
        r"(?:\bn[ée]e?\s+(?:le\s+)?|date\s+de\s+naissance|\bnaissance|\bddn\b|\bdob\b)\W{0,12}$",
        re.I,
    ),
    "en": re.compile(
        r"(?:\bborn(?:\s+on)?|date\s+of\s+birth|\bbirth\s*date|\bbirthday|\bdob\b|\bd\.o\.b\.?)\W{0,12}$",
        re.I,
    ),
}
_NUM_DATE = re.compile(r"(?<!\d)(\d{1,2})([/.-])(\d{1,2})\2(\d{4})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DMY_TEXT_DATE = re.compile(r"(?<!\d)(\d{1,2})(?:er)?\s+([A-Za-zéèûôîÀ-ÿ]+)\s+(\d{4})(?!\d)")
_EN_TEXT_DATE = re.compile(
    r"(?<![A-Za-z])([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)"
)


def _real_date(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
    except ValueError:
        return False
    return y >= 1900


def _date_candidates(text: str, language: str) -> Iterable[tuple[int, int, bool]]:
    """Yield (start, end, valid) for each date-shaped run in `text`."""
    months = _MONTHS[language]
    for m in _ISO_DATE.finditer(text):
        yield m.start(), m.end(), _real_date(int(m[1]), int(m[2]), int(m[3]))
    for m in _NUM_DATE.finditer(text):
        a, b, y = int(m[1]), int(m[3]), int(m[4])
        if language == "fr":
            ok = _real_date(y, b, a)
        else:
            # Month first in English, day first when that is the only
            # reading that is a date (`25/12/1985`).
            ok = _real_date(y, a, b) or _real_date(y, b, a)
        yield m.start(), m.end(), ok
    if language == "fr":
        for m in _DMY_TEXT_DATE.finditer(text):
            mon = months.get(m[2].lower())
            yield m.start(), m.end(), mon is not None and _real_date(int(m[3]), mon, int(m[1]))
    else:
        for m in _EN_TEXT_DATE.finditer(text):
            mon = months.get(m[1].lower())
            yield m.start(), m.end(), mon is not None and _real_date(int(m[3]), mon, int(m[2]))
        for m in _DMY_TEXT_DATE.finditer(text):
            mon = months.get(m[2].lower())
            yield m.start(), m.end(), mon is not None and _real_date(int(m[3]), mon, int(m[1]))


def find_date_of_birth(text: str, language: str) -> list[Span]:
    """A real calendar date that follows a birth cue, in `language`.

    A date alone is not personal data; a date after "né le", "date de
    naissance", "born", "DOB" is. The span covers the date only.
    Formats: ISO, `dd/mm/yyyy` (`mm/dd/yyyy` in English, day first when
    the month cannot be), `12 mars 1985`, `March 12, 1985`.
    """
    _check_language(language)
    cue = _DOB_CUE[language]
    out = []
    for start, end, valid in _date_candidates(text, language):
        if valid and _has_cue(text, start, cue):
            out.append(Span(start, end, "date_of_birth", 0.85))
    return out


# -- email -------------------------------------------------------------

_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"
)


def find_emails(text: str) -> list[Span]:
    """Email address.

    Presidio ships its own, but its validator asks `tldextract` for the
    public-suffix list, which fetches it over the network. This one
    does not, and the adapter uses it instead.
    """
    return [Span(m.start(), m.end(), "email", 0.9) for m in _EMAIL.finditer(text)]


# -- client ids --------------------------------------------------------


def compile_client_id_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    """Compile configured client-id regexes, failing loud on a bad one."""
    out = []
    for raw in patterns:
        if not isinstance(raw, str):
            raise ValueError(f"redact.client_id_patterns: expected a string, got {raw!r}")
        try:
            rx = re.compile(raw)
        except re.error as exc:
            raise ValueError(f"redact.client_id_patterns: bad regex {raw!r}: {exc}") from exc
        if rx.fullmatch(""):
            raise ValueError(f"redact.client_id_patterns: {raw!r} matches the empty string")
        out.append(rx)
    return out


def find_client_ids(text: str, patterns: Iterable[re.Pattern[str]]) -> list[Span]:
    """Spans matching any configured client-id pattern."""
    out = []
    for rx in patterns:
        out.extend(
            Span(m.start(), m.end(), "client_id", 0.9)
            for m in rx.finditer(text)
            if m.end() > m.start()
        )
    return out


# -- composition -------------------------------------------------------


def resolve_overlaps(spans: Iterable[Span]) -> list[Span]:
    """Keep the strongest of any overlapping spans; return in text order.

    Higher score wins, then the longer span, then the earlier start.
    """
    ranked = sorted(spans, key=lambda s: (-s.score, -(s.end - s.start), s.start))
    kept: list[Span] = []
    for s in ranked:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: (s.start, s.end))


_FINDERS: list[Callable[[str, str], list[Span]]] = [
    lambda t, _lang: find_sin(t),
    lambda t, _lang: find_ramq(t),
    lambda t, _lang: find_postal_codes(t),
    lambda t, _lang: find_amounts(t),
    find_date_of_birth,
    lambda t, _lang: find_emails(t),
]


def analyze_patterns(
    text: str, language: str, client_id_patterns: Iterable[re.Pattern[str]] = ()
) -> list[Span]:
    """Every pattern recognizer over `text`, overlaps resolved.

    The no-extra entry point: no named-entity recognition, so no person
    or location, but also nothing to install.
    """
    _check_language(language)
    spans = [s for finder in _FINDERS for s in finder(text, language)]
    spans.extend(find_client_ids(text, client_id_patterns))
    return resolve_overlaps(spans)
