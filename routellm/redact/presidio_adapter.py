"""Presidio engine around the pattern recognizers.

Imported lazily, only by `routellm.redact.analyze`; importing
`routellm.redact` on a base install never reaches this file. Requires
the `redact` extra (`presidio-analyzer`, `spacy`) and one installed
spaCy model per language.

Runs offline by construction:

- Presidio's own `SpacyNlpEngine` calls `spacy download` when a model
  is missing. That is a runtime fetch, so the models are checked first
  and a missing one is an error naming the install step.
- Presidio's email recognizer asks `tldextract` for the public-suffix
  list, which is fetched over the network. The registry here is built
  explicitly and takes the pure-Python email pattern instead.
"""

from dataclasses import dataclass, field

from routellm.redact.recognizers import (
    LANGUAGES,
    Span,
    compile_client_id_patterns,
    find_addresses,
    find_amounts,
    find_client_ids,
    find_date_of_birth,
    find_emails,
    find_phones,
    find_postal_codes,
    find_ramq,
    find_sin,
    resolve_overlaps,
)

__all__ = ["DEFAULT_MODELS", "PresidioAnalyzer", "RedactUnavailable"]

#: spaCy model per language. Overridable through `redact.models`.
DEFAULT_MODELS: dict[str, str] = {"fr": "fr_core_news_lg", "en": "en_core_web_lg"}

# Presidio entity type -> span category.
_CATEGORY = {
    "PERSON": "person",
    "LOCATION": "location",
    "PHONE_NUMBER": "phone",
    "IBAN_CODE": "iban",
    "CREDIT_CARD": "card",
}

_PHONE_REGIONS = ("CA", "FR", "US")


class RedactUnavailable(RuntimeError):
    """The `redact` extra or a spaCy model is not installed."""


@dataclass
class PresidioAnalyzer:
    """Presidio `AnalyzerEngine` plus the pattern recognizers.

    Build once, call `analyze` per message; construction loads the
    spaCy models, which is the slow part.
    """

    client_id_patterns: tuple[str, ...] = ()
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    def __post_init__(self) -> None:
        self._client_ids = compile_client_id_patterns(self.client_id_patterns)
        self._engine = self._build_engine()

    def _build_engine(self):
        try:
            import spacy.util
            from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer.predefined_recognizers import (
                CreditCardRecognizer,
                IbanRecognizer,
                PhoneRecognizer,
                SpacyRecognizer,
            )
            from presidio_analyzer.recognizer_registry import RecognizerRegistry
        except ImportError as exc:
            raise RedactUnavailable(
                "PII analysis needs the `redact` extra: pip install 'routellm[redact]' "
                "(then install the spaCy models, see docs/redact.md)"
            ) from exc

        unknown = set(self.models) - LANGUAGES
        if unknown:
            raise ValueError(f"redact.models: unsupported language(s) {sorted(unknown)}")
        missing = [m for m in self.models.values() if not spacy.util.is_package(m)]
        if missing:
            raise RedactUnavailable(
                f"spaCy model(s) not installed: {', '.join(missing)}. "
                "Install them once, ahead of time (docs/redact.md); "
                "routellm never downloads a model at runtime."
            )

        client_ids = self._client_ids

        class _Wrapped(EntityRecognizer):
            """A pure-Python finder, presented to Presidio as a recognizer."""

            def __init__(self, entity, language, finder):
                self._finder = finder
                super().__init__(
                    supported_entities=[entity],
                    supported_language=language,
                    name=f"routellm_{entity}",
                )

            def load(self) -> None:  # nothing to load
                pass

            def analyze(self, text, entities, nlp_artifacts=None):
                return [
                    RecognizerResult(self.supported_entities[0], s.start, s.end, s.score)
                    for s in self._finder(text, self.supported_language)
                ]

        wrapped = {
            "SIN": lambda t, _l: find_sin(t),
            "RAMQ": lambda t, _l: find_ramq(t),
            "POSTAL_CODE": lambda t, _l: find_postal_codes(t),
            "AMOUNT": lambda t, _l: find_amounts(t),
            "DATE_OF_BIRTH": find_date_of_birth,
            "EMAIL": lambda t, _l: find_emails(t),
            # Presidio's own PHONE_NUMBER scores at the bare threshold; this
            # one is scored to stand clear of it and wins any overlap.
            "PHONE": lambda t, _l: find_phones(t),
            "ADDRESS": lambda t, _l: find_addresses(t),
            "CLIENT_ID": lambda t, _l: find_client_ids(t, client_ids),
        }
        self._wrapped_category = {k: k.lower() for k in wrapped}

        languages = sorted(self.models)
        registry = RecognizerRegistry(supported_languages=languages)
        for lang in languages:
            registry.add_recognizer(CreditCardRecognizer(supported_language=lang))
            registry.add_recognizer(IbanRecognizer(supported_language=lang))
            registry.add_recognizer(
                PhoneRecognizer(supported_language=lang, supported_regions=_PHONE_REGIONS)
            )
            registry.add_recognizer(SpacyRecognizer(supported_language=lang))
            for entity, finder in wrapped.items():
                registry.add_recognizer(_Wrapped(entity, lang, finder))

        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": k, "model_name": v} for k, v in self.models.items()],
            }
        ).create_engine()
        return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=languages)

    def analyze(self, text: str, language: str, score_threshold: float = 0.4) -> list[Span]:
        """Spans for `text`, in text order, overlaps resolved."""
        if language not in self.models:
            raise ValueError(
                f"language {language!r} not configured; configured: {sorted(self.models)}"
            )
        results = self._engine.analyze(
            text=text, language=language, score_threshold=score_threshold
        )
        spans = []
        for r in results:
            category = _CATEGORY.get(r.entity_type) or self._wrapped_category.get(r.entity_type)
            if category is not None:
                spans.append(Span(r.start, r.end, category, round(float(r.score), 4)))
        return resolve_overlaps(spans)
