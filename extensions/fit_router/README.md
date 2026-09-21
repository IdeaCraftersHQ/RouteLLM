# routellm-fit

A routellm router trained on your own scored traffic.

The other routers score a prompt against something fixed: an embedding
model, a classifier, a yes/no question. This one learns from what your
endpoints actually produced. Score your traces, train an advisor on the
result, serve it, and point a tier at `router: fit`.

Optional. Nothing in routellm needs it, and routellm core gains no
dependency by its existing.

## Install

```
pip install -e ./extensions/fit_router
```

Training additionally needs `trl`, `transformers` and torch:

```
pip install -e './extensions/fit_router[training]'
```

## 1. Get scored traces

This package starts where `routellm.quality_scores` leaves off, so do
that first — see [the runbook](../../docs/runbooks/traces-to-quality.md):

```
python -m routellm.quality_scores score \
    --traces .routellm_traces \
    --scorer rubric:./rubric.yaml \
    --out scores.jsonl
```

## 2. Train

```
python -m routellm_fit.train --scores scores.jsonl --out advisor/
```

The label is the **argmax endpoint per prompt**: of every endpoint that
answered a given prompt, the one whose mean score was highest. Two
things follow from that.

The mean, not the best single answer, so one lucky response does not
outrank an endpoint that is consistently better. And a prompt only one
endpoint ever answered is dropped, because there was no choice to
learn. If your traffic never sends the same prompt to two endpoints,
this router has nothing to train on and the command says so.

`--dry-run` builds the dataset and prints it, without touching any
training dependency:

```
$ python -m routellm_fit.train --scores scores.jsonl --out advisor/ --dry-run
3 training examples
  16164be0fe63ae66 -> cloud_strong (reward 0.700; cloud_strong=0.700, local_fast=0.650)
  53eae7058490affd -> local_fast (reward 0.850; cloud_strong=0.300, local_fast=0.850)
  eea17685f2048f6b -> cloud_strong (reward 0.900; cloud_strong=0.900, local_fast=0.200)
```

Read that output before training. The `candidates` column is how close
each decision was: a dataset where every margin is 0.01 will train a
router that is guessing.

Without the training dependencies the command refuses, naming what is
missing:

```
$ python -m routellm_fit.train --scores scores.jsonl --out advisor/
error: training needs trl, which fit's training extra installs: pip
install -e '.[training]'. Use --dry-run to build and inspect the
dataset without them.
```

The check is deliberate. fit's own `GRPOTrainer.train` catches its
ImportError and falls back to a simplified loop that trains no model,
which would leave you with an empty advisor and an exit code of 0.

## 3. Serve

```
fit serve --model advisor/
```

## 4. Route on it

```yaml
tiers:
  coding:
    router: fit
    threshold: 0.6
    strong: cloud_strong
    weak: local_fast
```

The router asks the advisor about the prompt and uses the returned
confidence as the strong-model win rate, clamped to `[0, 1]`. The
advice's `domain` and `steering_text` are ignored: routellm needs one
number.

An unreachable advisor raises a `RuntimeError` naming the endpoint, so
the controller's fallback chain sees an ordinary failure rather than a
silent default.

By default it talks to `http://localhost:8080`. The router takes an
`endpoint` and a `timeout_ms`, and an `advisor` object for a local
export or a test.

## Tests

```
pip install -e './extensions/fit_router[dev]'
python -m pytest extensions/fit_router/tests
```

They live with the extension, as `extensions/typesafe`'s do, and are
not in the main repo's `TEST_INDEX.md`.

## How it registers

Through the `routellm.routers` entry point group declared in this
package's `pyproject.toml`. Installing the package is the whole
wiring: nothing imports `routellm_fit` to make `fit` resolvable.
