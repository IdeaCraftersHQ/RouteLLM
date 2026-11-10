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

## Reversible placeholders

`Redactor` swaps spans for `[[CATEGORY_n]]` and puts the values back on
the way return. Pure Python: it takes spans from `analyze` (needs the
extra) or from any `analyze(text, language)` you pass in.

```python
from routellm.redact import RedactContext, Redactor, TokenStore

redactor = Redactor(TokenStore("/var/lib/routellm/tokens.db"))
ctx = RedactContext(principal="team-a", request_id="r1", language="fr")
text, mapping_id = redactor.redact("Écrire à Marie Gagnon", ctx)
# "Écrire à [[PERSON_1]]"
redactor.restore(model_reply, mapping_id, ctx)
```

- Placeholder: `[[` + category upper-cased + `_` + number + `]]`. Same
  value, same placeholder within the scope; matching is exact, so
  `Jean` and `jean` are two values and each restores as written.
- Scope: one request by default. Numbers are never reused for a
  principal, so two requests cannot be linked through their placeholders.
  `Redactor(..., stable_across_requests=True)` scopes to the conversation
  (`RedactContext.conversation_id` required): the same value keeps its
  placeholder for as long as the mapping lives.
- Text already shaped like a placeholder (`[[PERSON_1]]`, `[[ person_1 ]]`)
  is reserved: never assigned, never restored. If a mapping that already
  exists would collide with it, `redact` raises `RedactError`.
- `restore` is one pass. It forgives case and spaces inside the brackets.
  Anything else a model might do (`[[PERSON_1`, `PERSON_1]]`, a changed
  number) is left as written, never guessed. A mapping that is unknown,
  expired or belongs to another principal, request or conversation raises
  `MappingError` / `MappingExpired`; no text is returned.
- `split_incomplete(chunk)` -> `(safe, pending)`: hold `pending` (a
  trailing `[[PERS`) until the next stream chunk.

### Token store

`TokenStore(path, key=None, ttl_s=86400)`: SQLite on the gateway host,
read by nothing but `restore`. Never logged, traced, cached, audited or
sent upstream.

- Directory 0700, database and key file 0600 (a looser existing file is
  tightened).
- Values are found by `HMAC-SHA256(gateway key, scope || value)`; the
  key is `<path>.key`, generated on first use, or passed in.
- The values themselves are stored in the clear by default, as the
  response cache stores prompts: the file mode is the protection. For
  more, turn on AES-GCM (see [Protecting stored values](#protecting-stored-values))
  or put the directory on an encrypted volume.
- A mapping and each token expire `ttl_s` after creation / last use;
  expired rows are purged on the next write.

### Protecting stored values

The store's value column goes through a `ValueCipher`
(`routellm.redact.cipher`): `protect(value, context)` /
`reveal(stored, context)`.

| `redact.store.encryption` | Value column | Needs |
|---------------------------|--------------|-------|
| `none` (default) | plaintext; database and key files 0600, directory 0700 | nothing |
| `aes-gcm` | AES-256-GCM, fresh 96-bit nonce per value, bound to its scope and placeholder | `pip install "routellm[redact-encrypt]"` and a 32-byte key |

```yaml
redact:
  store:
    encryption: aes-gcm
    key_file: /etc/routellm/redact.key   # mode 0600; or key_env: ROUTELLM_REDACT_KEY
```

The key is base64 or hex of 32 random bytes (`openssl rand -base64 32`).
Startup stops if `aes-gcm` is chosen and the extra, the key, its 0600
mode or its length is wrong. A tampered row, a row moved to another
placeholder, a wrong key and a row written under the other setting all
fail as an unreadable mapping (the request is refused); ciphertext is
never returned as a value. Rows live for `ttl_s` (a day by default), so
changing the setting only costs the mappings in flight.

The lookup column (an HMAC under the gateway key) is not the value and
does not change. `cryptography` 50.0.2 is the pinned dependency
(Apache-2.0 OR BSD-3-Clause, released 2026-09-30).

## The redact treatment

A policy class with `treatment: redact` may reach cloud leaves
(`reach: [local, cloud]`) because this transform runs first. It is built
from a `redact:` section of the server config and runs ahead of any
`transforms:` entry.

```yaml
policy: policy.yaml            # classes.pii: {treatment: redact, reach: [local, cloud], ...}
redact:
  engine: presidio             # presidio (default) | patterns (no NER: no person, no location)
  languages: [fr, en]          # every text is analysed under each, spans merged
  client_id_patterns: ['CLI-\d{6}']
  non_text: refuse             # refuse (default) | pass
  store:
    path: state/tokens.db      # default $XDG_STATE_HOME/routellm/redact/tokens.db
    ttl_s: 86400
    encryption: none
```

**Which calls.** For a request whose effective class has treatment
`redact`, every call to a leaf that is not exactly `local` (cloud, vpn,
a leaf with no reach label) goes out redacted: system, user, assistant
and tool messages, the text parts of multimodal content, and tool-call
and legacy function-call arguments. Arguments are JSON, so the strings
inside are redacted (a dump escapes `é`, which would hide it) and the
document is re-serialised. This holds for the primary, the sibling
fallback, pool members, shadow and compare targets. A `local` leaf gets
the request raw: redaction is for egress.

**Restore.** The reply is restored locally, before it reaches the client:
message content, reasoning, tool-call arguments (values are JSON-escaped
into the arguments, so they stay valid), streamed or not. A placeholder
split across chunks is held until it completes; a held tail that never
completes is released as written. A placeholder the model invented or
mangled is left as written. The mapping is read once, before the call; a
reply is restored from memory.

**Refusal, never raw.** A redactor, analyzer or store error, a span
outside the text, an unreadable mapping, or a part that is not text
(an image, audio, a file; `non_text: pass` sends those as they are)
ends the request with a 500 `transform_failed` before any provider is
called. No fallback leaf is tried. A restore error does the same. A
policy that lets a redact class leave the host with no `redact:` section
stops the server at startup; a request that reaches a non-local leaf
without a redactor is refused at the call.

**Auxiliary calls.** The transform rewrites the main call only. Every
other call that carries request content (hosted and embedding routers,
the semantic cache's embedding, the intent detector, the comparison
judge, the decision classifier) is local-only for a redact class: its
endpoint must have only the `local` reach label, else it is skipped
(`redact:<class>:aux_not_local`) and, for the decision classifier,
startup fails. A redact class takes no part in the semantic cache.

**At rest.**

- Cache: a request in a redact class is never looked up or stored.
- Traces: a non-`raw` class keeps digests only, whatever `capture_content`.
- Logs: this package logs no text; litellm's DEBUG records, which print
  messages, are dropped while a redact transform is configured.
- Mapping ids appear in no cache key, trace or log line.
- The values live in the token store and nowhere else, protected as above.

**Limits.** The transform sees `messages`. Other request fields
(`user`, `metadata`, tool and schema descriptions, a message's `name`)
go upstream as sent. A transcriber runs before the class is known; keep
it local with a redact class. Recall is the analyzers': measure it
(the leak and recall harness gates the treatment for production).

## Licences

| Component | Version | Licence |
|-----------|---------|---------|
| cryptography (`redact-encrypt`) | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
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
