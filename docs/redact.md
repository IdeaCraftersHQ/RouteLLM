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
| `iban`, `card` | Presidio | card and IBAN are checksum-validated; needs `redact` |
| `phone` | pattern + Presidio | NANP and French formats (`514-555-0123`, `+33 6 12 34 56 78`) scored 0.8, ahead of Presidio's borderline 0.4; ten bare digits only after a cue (`tel`, `cell`) |
| `address` | pattern | civic number + street type + capitalised name, fr/en: `1234 rue Saint-Denis`, `123 Main St. Apt 4`, `45 avenue du Parc, app. 5` |
| `email` | pattern | |
| `sin` | pattern | nine digits, spaced, dashed or bare; Luhn-validated |
| `ramq` | pattern | four letters, eight digits (`DUPJ 8503 1234`); the digits must hold a plausible birth month and day |
| `postal_code` | pattern | `A1A 1A1`; only letters Canada Post issues |
| `amount` | pattern | French (`1 234,56 $`, `12,50 €`) or English (`$1,234.56`, `USD 1,234.56`, `1,234.56 $`) format; a currency is always required |
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
  harness_report: redact-report.json   # required for redact + cloud; relative to this file
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

### Request fields beyond `messages`

Every field of a call is in one list. A field in none is refused, so a
field added upstream never leaves the host unseen; an unset one (`null`,
`[]`, `{}`) carries nothing and passes.

| Field | Treatment |
|-------|-----------|
| `messages[].content`, text parts, tool-call and `function_call` arguments | redacted, restored in the reply |
| `messages[].name`, `refusal`, `reasoning_content` | redacted |
| `messages[].tool_calls[].id`, `tool_call_id`, function names | kept; refused if the redactor would change one (an identifier cannot be swapped) |
| `user`, `metadata` (values) | redacted (`metadata` is not forwarded over HTTP: the server reads the class from it, then drops it) |
| `tools`, `functions`: descriptions, parameter `description`/`title`, `enum`, `default`, `examples`, `const` | redacted |
| `tools`, `functions`: names, property names, `required`, `type`, `format`, `$ref`, `pattern` | kept; refused if the redactor would change one |
| `response_format` `json_schema`: `description` and the schema, as for tools | redacted |
| `prediction` (prefill) | redacted |
| `tool_choice`, `function_call` | kept (a keyword, or a function name checked as above) |
| `stream_options` | kept (`include_usage` only) |
| `model`, `stream`, `temperature`, `top_p`, `n`, `max_tokens`, `max_completion_tokens`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `parallel_tool_calls`, `reasoning_effort`, `extra_headers`, `timeout` | allowed as sent |
| `audio`, `web_search_options`, a tool that is not a function, a `response_format` that is not text, `json_object` or `json_schema` | refused |
| any other field | refused |

Redacted definitions and prefills are inputs: nothing in them comes back
in a reply, so only message text and tool-call arguments are restored. A
refusal is `500 transform_failed` with no provider call. A local leaf
gets every field as sent. Over HTTP, a body field the server does not
declare is dropped before routing (see `ChatCompletionRequest`), so it
never goes upstream for any class; a library call that passes one is
refused for a redact class. A message `name` is redacted like text, and a
provider that restricts its charset may then reject it.

### Transcriber

`transcriber:` runs before a request's class is known, so its audio
cannot be redacted. When any class has treatment `redact` or `block`,
every transcriber that can run (the global one and each tier's) must be
tagged `local`, else startup fails naming the transcriber and the
classes (`PolicyError`). A transcriber in `mode: passthrough` is
unaffected. Per request, a redact or block class declared with a
transcriber that is not local is refused (403, `transcriber:<class>:not_local`)
before any audio is sent.

### Video preprocessor

`video:` is the same case: the video goes to the preprocessing service
before the class is known. With any `redact` or `block` class, every
active preprocessor (global and per tier) must be tagged `local`, else
startup fails (`PolicyError`); `mode: refuse` or `passthrough` is
unaffected. Per request, a redact or block class with a non-local
preprocessor is refused (403, `video:<class>:not_local`) before upload.
The returned transcript and on-screen text are redacted like any user
text. Frames are images and cannot be: under the default
`non_text: refuse` a video that yields frames is refused for a `redact`
class; `non_text: pass` forwards them.

**Gate.** A policy that lets a `redact` class reach the cloud loads only
when `redact.harness_report` names a passing report stamped for the
config in effect (see [The harness](#the-harness)). A policy whose redact
classes are local-only needs none.

**Limits.** Recall is the analyzers': measure it (the harness below gates
the treatment).

## The harness

```sh
python -m routellm.redact.harness --config server.yaml --out redact-report.json
```

Reads the `redact:` section of `--config`, measures it and writes a JSON
report. Exit 0 when everything passes, 1 on a failed threshold or check,
2 when it could not run. `--corpus DIR` replaces the shipped fixtures.

**Recall.** The analyzers the section builds run over the labelled corpus
(`routellm/tests/fixtures/redact/*.jsonl`: synthetic fr and en text; each
line `{id, lang, text, spans: [{start, end, category, text}]}`, one file
per category plus `negatives.jsonl`: dates that are not birth dates,
invoice numbers, versions, Luhn-invalid SINs). A predicted span is right
when it has the labelled category and overlaps it by at least half. The
`THRESHOLDS` table in `routellm/redact/harness.py` gives each category a
minimum recall and precision per engine; under either, the report fails.
A category with fixtures and no row is reported, not gated, so a new
recognizer is measured as soon as it has fixtures and gated once it has a
row. `client_id` is scored against the corpus's `ACME-NNNNN` ids: use
`client_id_patterns` that match them, or pass a `--corpus` with your own
synthetic ids.

**Leak.** One run of the real server (policy, auth, audit, cache,
traces with `capture_content`, the section's redact transform) against a
recording provider that echoes the last message. Conversations carrying
the canary values (`canaries.jsonl`) travel as system, multimodal part,
tool-call arguments, tool result and user text, streamed and not. The
outbound requests, logs (in-process records, stdout, stderr), traces,
cache, audit events and, with `store.encryption: aes-gcm`, the token store
must hold no canary, as written or with separators removed. (A plaintext
token store holds values by design and is not scanned.) A control request
through a `raw` class must show its text in the provider request, the
traces and the cache: a scan that cannot see it is blind and fails. The
client must read its original text back.

**Re-identification.** No outbound request holds a date and a postal code
together (matched by shape, independent of the recognizers), name or no
name. Two identical requests from one principal carry disjoint
placeholders (the default per-request scope), so they cannot be linked.

**Digest and gate.** The report carries `digest`, `sha256` over the
engine, languages, client-id patterns, spaCy models and their installed
versions plus presidio and spaCy versions (presidio engine), the routellm
version and a hash of the recognizer sources. Store, `non_text` and
`harness_report` are not in it. At startup, if a policy lets a redact
class reach the cloud, `redact.harness_report` must name a report that
exists, has this digest, and in which every category row and the three
checks (`leak`, `reidentification`, `linkability`) passed. Otherwise the
server refuses to start, naming the reason; any change to the recognizer
config calls for a new run.

CI runs the pattern-engine harness (`redact-harness` job, config in
`routellm/tests/fixtures/redact/harness.yaml`). The presidio engine needs
the extra and models; its test is marked `redact` and skipped without
them.

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

- `routellm.redact.detect_language(text)` returns `(lang, confidence)`
  from stop-words, elisions, contractions and diacritics (no model).
  `languages_for(text, languages)` picks the detected language only when
  confidence is at least 0.7, else all configured ones; short text,
  names, numbers and mixed text stay on all languages. The transform
  still analyses under every configured language until it is wired to it.
- Addresses need a civic number, a street type and a capitalised name;
  a name in all lower-case (`12 rue de la paix`) is missed, and a
  number-led capitalised phrase ending in a street word (`2024 Capital
  Drive`) is flagged. Addresses beyond that rely on the NER location tag.
- Ten bare digits are a phone number only after a cue word.
