# routellm-typesafe

TypeSafe Jev routing for [routellm](https://github.com/lm-sys/RouteLLM).

Two pieces, both backed by TypeSafe's hosted System One API rather than
local checkpoints:

- **`JevRouter`** — the `jev` router. Asks Jev one Noul (yes/no)
  question about the prompt and uses the returned probability as the
  strong-model win rate.
- **`JevIntentDetector`** — classifies a prompt with a single Choice
  question instead of embeddings, so it needs no examples. Drops into
  the `intent_detector` slot of `IntentModelSelector`.

## Install

```bash
pip install -e extensions/routellm_typesafe
```

Installing is the whole wiring. The package declares a
`routellm.routers` entry point, so routellm discovers the `jev` name at
import; no code change and no import of this package are needed.

Set `TYPESAFE_API_KEY` (required). Optionally set `TYPESAFE_BASE_URL`
(default `https://api.typesafe.ai`) and `TYPESAFE_DEFAULT_MODEL`
(default `jev-latest`, currently resolving to `jev-1.13.0`). Pin
`TYPESAFE_DEFAULT_MODEL=jev-1.13.0` once a threshold is calibrated
against a specific model version: `jev-latest` moves on release.

## Config

```yaml
jev:
    # model and timeout default to the SDK env (TYPESAFE_DEFAULT_MODEL, 10s)
    # calibrated 2026-09-20 against jev-1.13.0; pin model: jev-1.13.0 to keep thresholds stable
    max_prompt_chars: 100000
    # prompt_file: prompts/jev.example.yaml  # copy + edit wording without code changes
```

## Router

```python
from routellm.controller import Controller

controller = Controller(routers=["jev"], strong_model="gpt-4", weak_model="gpt-3.5-turbo")
```

## Intent detector

```python
from routellm.middleware import IntentModelMapping, IntentModelSelector
from routellm_typesafe import JevIntentDetector

mappings = [
    IntentModelMapping(intent="code", description="programming and debugging", ...),
    IntentModelMapping(intent="chat", description="casual conversation", ...),
]
selector = IntentModelSelector(..., intent_detector=JevIntentDetector(mappings))
```

## Prompt file

Both classes read their wording from one optional YAML file, so the
question and criteria can be edited without touching code. Each reads
only its own section and ignores the rest, so a future adapter can add
a section to the same file. Precedence: explicit kwarg > prompt file >
built-in default.

```python
from routellm_typesafe import JevIntentDetector, JevRouter

router = JevRouter(prompt_file="prompts/jev.example.yaml")
detector = JevIntentDetector(mappings, prompt_file="prompts/jev.example.yaml")
```

See `prompts/jev.example.yaml` in this package for the format: a
`router` section (`instructions`, `criteria.true`, `criteria.false`)
and an `intent_detector` section (`instructions`,
`general_description`).

## Tests

From the repository root:

```bash
PYTHONPATH=extensions/routellm_typesafe:. python -m pytest extensions/routellm_typesafe/tests -q
```
