# PII analysis

`routellm.redact` finds personal data in a message and reports where.
It runs in-process and offline.

```python
from routellm.redact import analyze

analyze("Né le 12 mars 1985, NAS 046 454 286", "fr")
# [{"start": 6, "end": 18, "category": "date_of_birth", "score": 0.85},
#  {"start": 24, "end": 35, "category": "sin", "score": 0.95}]
```

`analyze(text, language)` takes `fr` or `en` and returns spans in text
order, non-overlapping: `{start, end, category, score}`, offsets in
characters, end exclusive. Where two recognizers claim the same text the
higher score wins, then the longer span.

## Categories

| Category | Source | Notes |
|----------|--------|-------|
| `person`, `location` | spaCy NER via Presidio | needs `redact` |
| `phone`, `iban`, `card` | Presidio | card and IBAN are checksum-validated; needs `redact` |
| `email` | pattern | |
| `sin` | pattern | nine digits, spaced, dashed or bare; Luhn-validated |
| `ramq` | pattern | four letters, eight digits (`DUPJ 8503 1234`); the digits must hold a plausible birth month and day |
| `postal_code` | pattern | `A1A 1A1`; only letters Canada Post issues |
| `amount` | pattern | French format with a currency: `1 234,56 $`, `12,50 €` |
| `date_of_birth` | pattern | a real date after a birth cue ("né le", "date de naissance", "born", "DOB") |
| `client_id` | config | your regexes |

A date with no birth cue is not reported, and neither is a bare number
without a currency: the patterns would otherwise flag most of a message.

## Two tiers

- Base install: `routellm.redact.analyze_patterns(text, language)` runs
  the pattern recognizers only. Nothing to install, no person or
  location.
- `routellm[redact]`: `analyze` adds the Presidio categories.

Without the extra, `analyze` raises `RedactUnavailable` naming it. Core
installs never import Presidio or spaCy.

## Install

```sh
pip install "routellm[redact]"
pip install \
  "https://github.com/explosion/spacy-models/releases/download/fr_core_news_lg-3.8.0/fr_core_news_lg-3.8.0-py3-none-any.whl" \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
```

The models are an install step, done once, ahead of time (about 550 MB
and 380 MB). Nothing is downloaded at runtime: a missing model raises
`RedactUnavailable`, and the engine is built with the network paths of
Presidio and spaCy closed (no model download, no public-suffix-list
fetch). The test suite proves it with sockets blocked.

Smaller models work if you name them: `redact.models.fr:
fr_core_news_md`. Recall on names and places drops with size.

## Configuration

```yaml
redact:
  client_id_patterns: ['CLI-\d{6}']
  models: {fr: fr_core_news_lg, en: en_core_web_lg}
```

```python
from routellm.redact import analyze, build_analyzer

analyze(text, "fr", client_id_patterns=[r"CLI-\d{6}"])
build_analyzer({"client_id_patterns": [...]}).analyze(text, "fr")  # Span objects
```

`analyze` builds the engine on first use and reuses it per configuration;
building loads the models, which takes seconds.

## Licences

| Component | Version | Licence |
|-----------|---------|---------|
| presidio-analyzer | 2.2.365 | MIT |
| spacy | 3.8.16 | MIT |
| en_core_web_lg | 3.8.0 | MIT |
| fr_core_news_lg | 3.8.0 | LGPL-LR (data licence of its Sequoia treebank) |

The French model is not part of this package, is not bundled in its
wheel or sdist, and is installed by you; its licence governs that model
file, not this code. If its terms do not suit your distribution, point
`redact.models.fr` at another French model.

## Limits

- Language is declared by the caller; there is no detection.
- Street addresses are covered only as far as the NER model tags them
  as locations.
- Amounts use French separators; `$1,234.56` is not matched.
