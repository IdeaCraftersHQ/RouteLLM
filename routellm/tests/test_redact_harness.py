"""The harness without a server: corpus, scoring, thresholds, digest and the gate."""

import json
from pathlib import Path

import pytest

from routellm.config import ConfigError
from routellm.policy import parse_policy
from routellm.redact import harness
from routellm.redact import recognizers as rec
from routellm.redact.config import _analyzer, build_redact_transform
from routellm.redact.gate import Gate, config_digest, digest_material, verify_report
from routellm.redact.harness import THRESHOLDS, Doc, judge, load_canaries, load_corpus, score
from routellm.redact.harness_checks import leaked, linkable, reidentifying
from routellm.redact.placeholders import Redactor
from routellm.redact.tokens import TokenStore
from routellm.redact.transform import RedactTransform
from routellm.tests.redact_report import write_report
from routellm.tests.test_redact_transform import analyzer as stub_analyzer
from routellm.tests.test_request_transform import build, provider  # noqa: F401  (fixture)

RAW = {"engine": "patterns", "client_id_patterns": [r"ACME-\d{5}"]}
FIXTURES = Path(harness.CORPUS_DIR)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def patterns_scores(corpus, raw=RAW):
    return judge(score(corpus, _analyzer(raw, ("fr", "en"))), "patterns")


# --- corpus ------------------------------------------------------------------------------


def test_every_span_sits_where_it_says(corpus):
    for doc in corpus + load_canaries():
        for s in doc.spans:
            assert doc.text[s["start"] : s["end"]] == s["text"], doc.id


def test_each_category_has_fr_and_en_fixtures(corpus):
    langs: dict[str, set[str]] = {}
    for doc in corpus:
        for s in doc.spans:
            langs.setdefault(s["category"], set()).add(doc.lang)
    assert langs and all(v == {"fr", "en"} for v in langs.values()), langs


def test_the_corpus_covers_the_task_s_categories(corpus):
    cats = {s["category"] for d in corpus for s in d.spans}
    assert {
        "person", "address", "postal_code", "sin", "ramq", "date_of_birth", "phone",
        "email", "amount", "client_id",
    } <= cats  # fmt: skip


def test_the_corpus_has_negatives_and_luhn_invalid_sins(corpus):
    negatives = [d for d in corpus if d.id.startswith("negatives")]
    assert len(negatives) >= 10 and all(not d.spans for d in negatives)
    assert any("123 456 789" in d.text for d in negatives)
    assert any("dated 03/15/2024" in d.text or "12 mars 2024" in d.text for d in negatives)


def test_a_misplaced_span_is_refused(tmp_path):
    bad = {
        "id": "x",
        "lang": "en",
        "text": "abc",
        "spans": [{"start": 0, "end": 2, "category": "c", "text": "zz"}],
    }
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not match"):
        load_corpus(tmp_path)


# --- scoring -----------------------------------------------------------------------------


def span(start, end, cat):
    return {"start": start, "end": end, "category": cat}


def test_score_counts_hits_misses_and_false_positives():
    docs = [
        Doc("a", "en", "x" * 20, ({"start": 0, "end": 4, "category": "sin", "text": "xxxx"},)),
        Doc("b", "en", "y" * 20, ()),
    ]

    def analyze(text, lang):
        return [span(0, 4, "sin")] if text[0] == "x" else [span(2, 6, "sin"), span(0, 3, "email")]

    got = score(docs, analyze)
    assert got["sin"] == {
        "gold": 1, "predicted": 2, "tp": 1, "fn": 0, "fp": 1, "recall": 1.0, "precision": 0.5,
    }  # fmt: skip
    assert got["email"]["fp"] == 1 and got["email"]["recall"] is None


def test_overlap_below_half_or_in_another_category_is_a_miss():
    doc = Doc("a", "en", "x" * 20, ({"start": 0, "end": 10, "category": "sin", "text": "x" * 10},))
    assert score([doc], lambda t, lang: [span(0, 4, "sin")])["sin"]["tp"] == 0
    assert score([doc], lambda t, lang: [span(0, 10, "ramq")])["sin"]["tp"] == 0


# --- thresholds --------------------------------------------------------------------------


def test_the_pattern_engine_meets_its_table(corpus):
    rows = patterns_scores(corpus)
    assert set(THRESHOLDS["patterns"]) <= set(rows)
    assert {c for c, r in rows.items() if not r["passed"]} == set()


@pytest.mark.parametrize("category", sorted(THRESHOLDS["patterns"]))
def test_dropping_a_recognizer_puts_its_category_under_threshold(corpus, category):
    inner = _analyzer(RAW, ("fr", "en"))
    rows = judge(
        score(corpus, lambda t, lang: [s for s in inner(t, lang) if s["category"] != category]),
        "patterns",
    )
    assert rows[category]["passed"] is False
    assert {c for c, r in rows.items() if not r["passed"]} == {category}


def test_dropping_the_sin_finder_itself_fails_sin(corpus, monkeypatch):
    monkeypatch.setattr(rec, "find_sin", lambda text: [])
    assert patterns_scores(corpus)["sin"]["passed"] is False


def test_missing_client_id_patterns_fail_client_id(corpus):
    assert patterns_scores(corpus, {"engine": "patterns"})["client_id"]["passed"] is False


def test_a_category_without_a_row_is_reported_not_gated(corpus):
    rows = patterns_scores(corpus)
    assert rows["address"]["threshold"] is None and rows["address"]["passed"] is True


def test_a_new_category_in_the_output_does_not_fail_the_table(corpus):
    inner = _analyzer(RAW, ("fr", "en"))
    rows = judge(
        score(corpus, lambda t, lang: [*inner(t, lang), span(0, 1, "brand_new")]), "patterns"
    )
    assert rows["brand_new"]["passed"] is True
    assert {c for c, r in rows.items() if not r["passed"]} == set()


def test_a_gated_category_the_corpus_never_mentions_fails():
    assert judge({}, "patterns")["sin"]["passed"] is False


# --- pure checks -------------------------------------------------------------------------


def test_leaked_sees_separator_variants_and_case():
    assert leaked("sin 046454286", ["046 454 286"]) == ["046 454 286"]
    assert leaked("A@EXAMPLE.COM", ["a@example.com"]) == ["a@example.com"]
    assert leaked("[[SIN_1]]", ["046 454 286"]) == []


def test_reidentifying_flags_a_dob_and_postal_pair_with_no_name():
    pair = "née le 12 mars 1985, code H2X 3K8"
    assert reidentifying([pair]) == [pair]
    assert reidentifying(["née le [[DATE_OF_BIRTH_1]], code [[POSTAL_CODE_1]]"]) == []
    assert reidentifying(["code H2X 3K8 seul"]) == []
    assert reidentifying(["a", "b"], ["12 mars 1985", "H2X 3K8"]) == []
    assert reidentifying(["12 mars 1985 H2X 3K8"], ["12 mars 1985", "H2X 3K8"])


def test_linkable_reports_a_placeholder_shared_across_requests():
    assert linkable([{"[[SIN_1]]"}, {"[[SIN_1]]", "[[SIN_2]]"}]) == ["[[SIN_1]]"]
    assert linkable([{"[[SIN_1]]"}, {"[[SIN_2]]"}]) == []


# --- digest ------------------------------------------------------------------------------


def test_the_digest_follows_what_changes_recall():
    base = config_digest(RAW)
    assert config_digest({**RAW, "client_id_patterns": [r"X-\d+"]}) != base
    assert config_digest({**RAW, "languages": ["fr"]}) != base
    assert config_digest({**RAW, "engine": "presidio"}) != base
    assert config_digest({**RAW, "client_id_patterns": [r"ACME-\d{5}", "b"]}) != base


def test_the_digest_ignores_order_and_what_is_not_recall():
    base = config_digest(
        {"engine": "patterns", "languages": ["fr", "en"], "client_id_patterns": ["a", "b"]}
    )
    other = {
        "engine": "patterns", "languages": ["en", "fr"], "client_id_patterns": ["b", "a"],
        "store": {"path": "x"}, "non_text": "pass", "harness_report": "r.json",
    }  # fmt: skip
    assert config_digest(other) == base


def test_the_digest_follows_models_and_versions(monkeypatch):
    from routellm.redact import gate

    raw = {"engine": "presidio"}
    base = config_digest(raw)
    assert config_digest({**raw, "models": {"fr": "fr_core_news_md"}}) != base
    real = gate._version
    monkeypatch.setattr(gate, "_version", lambda d: "9.9.9" if d == "fr-core-news-lg" else real(d))
    assert config_digest(raw) != base
    monkeypatch.setattr(gate, "_version", lambda d: "9.9.9" if d == "routellm" else real(d))
    assert config_digest(raw) != base


def test_the_digest_follows_the_recognizer_sources(monkeypatch):
    from routellm.redact import gate

    base = config_digest(RAW)
    monkeypatch.setattr(gate, "_sources_hash", lambda: "changed")
    assert config_digest(RAW) != base
    assert set(digest_material(RAW)) >= {
        "engine",
        "languages",
        "client_id_patterns",
        "models",
        "routellm",
    }


# --- the gate ----------------------------------------------------------------------------


def test_a_passing_current_report_is_accepted(tmp_path):
    verify_report(write_report(RAW, tmp_path / "r.json"), RAW)


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda r: r.update(digest="sha256:0"), "stamped"),
        (lambda r: r.update(passed=False), "failed"),
        (lambda r: r.update(schema=0), "schema"),
        (lambda r: r["categories"].update(sin={"passed": False}), "sin"),
        (lambda r: r["checks"].pop("leak"), "leak"),
        (lambda r: r["checks"].update(linkability={"passed": False}), "linkability"),
        (lambda r: r.update(categories={}), "categories"),
    ],
)
def test_a_stale_or_failing_report_is_refused(tmp_path, mutate, why):
    path = Path(write_report(RAW, tmp_path / "r.json"))
    report = json.loads(path.read_text())
    mutate(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ConfigError, match=why):
        verify_report(str(path), RAW)


def test_a_config_changed_after_the_run_is_refused(tmp_path):
    path = write_report(RAW, tmp_path / "r.json")
    with pytest.raises(ConfigError, match="stamped"):
        verify_report(path, {**RAW, "client_id_patterns": []})


def test_no_path_a_missing_file_and_bad_json_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="harness_report"):
        verify_report(None, RAW)
    with pytest.raises(ConfigError, match="cannot read"):
        verify_report(str(tmp_path / "none.json"), RAW)
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ConfigError, match="not JSON"):
        verify_report(str(tmp_path / "bad.json"), RAW)


def test_the_report_path_is_resolved_against_the_config_file(tmp_path):
    (tmp_path / "conf").mkdir()
    write_report(RAW, tmp_path / "conf" / "r.json")
    Gate({**RAW, "harness_report": "r.json"}, str(tmp_path / "conf" / "c.yaml")).check()


def test_the_section_accepts_harness_report_and_arms_the_gate(tmp_path):
    raw = {**RAW, "store": {"path": str(tmp_path / "t.db")}, "harness_report": "r.json"}
    transform = build_redact_transform(raw, str(tmp_path / "c.yaml"))
    assert transform.harness_gate.report == str(tmp_path / "r.json")
    with pytest.raises(ConfigError, match="harness_report must be a path"):
        build_redact_transform({**raw, "harness_report": 3}, None)


# --- the policy load ---------------------------------------------------------------------

POLICY = {
    "classes": {"pii": {"treatment": "redact", "reach": ["local", "cloud"], "tiers": ["default"]}},
    "default_class": "pii",
}
LOCAL_ONLY = {
    "classes": {"pii": {"treatment": "redact", "reach": ["local"], "tiers": ["default"]}},
    "default_class": "pii",
}


def gated(tmp_path, report, policy=POLICY):
    tags = {"loc": ["local"]}
    if policy is LOCAL_ONLY:
        tags = {"loc": ["local"], "cld": ["local"], "cld2": ["local"]}
    transform = RedactTransform(Redactor(TokenStore(str(tmp_path / "t.db")), analyze=stub_analyzer))
    transform.harness_gate = Gate({**RAW, "harness_report": report})
    return build(
        [transform], tmp_path, policy=parse_policy(policy, "p"), strict_registry=True, tags=tags
    )


def test_redact_plus_cloud_loads_with_a_passing_stamped_report(provider, tmp_path):
    gated(tmp_path, write_report(RAW, tmp_path / "r.json"))


def test_redact_plus_cloud_refuses_to_load_on_a_digest_mismatch(provider, tmp_path):
    report = write_report({**RAW, "languages": ["fr"]}, tmp_path / "r.json")
    with pytest.raises(ConfigError, match="stamped"):
        gated(tmp_path, report)


def test_redact_plus_cloud_refuses_to_load_without_or_with_a_failing_report(provider, tmp_path):
    with pytest.raises(ConfigError, match="harness_report"):
        gated(tmp_path, None)
    with pytest.raises(ConfigError, match="failed"):
        gated(tmp_path, write_report(RAW, tmp_path / "r.json", passed=False))


def test_a_local_only_redact_class_needs_no_report(provider, tmp_path):
    gated(tmp_path, None, LOCAL_ONLY)
