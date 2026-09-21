# Traces to quality

`quality:` on an endpoint is a number someone typed into YAML once.
Nobody retypes it when the model behind the endpoint changes, and every
`quality_desc` selector orders on it. This runbook replaces that guess
with a measurement, in five steps: record traces, score them, aggregate
the scores, point the config at the result, and split the numbers by
area.

Nothing here runs on the serving path. Scoring and aggregation are
offline, against traffic that already happened.

Every command below was run against a scratch directory; the output is
pasted as it came back.

---

## 1. Turn tracing on

Tracing is off by default. Turn it on in code:

```python
from routellm.controller import Controller
from routellm.quality import FineTuneConfig, QualityManager

controller = Controller(
    routers=["mf"],
    strong_model="gpt-4o",
    weak_model="ollama/qwen",
)
controller.quality_manager = QualityManager(
    fine_tune_config=FineTuneConfig(
        enabled=True,
        trace_dir=".routellm_traces",
        max_files=10_000,
        max_bytes=512 * 1024 * 1024,
    )
)
```

One JSON file per request, named `trace-<epoch_ms>-<hex>.json`:

```
$ ls .routellm_traces | head -3
trace-1789949030797-b7f4f5.json
trace-1789949030798-5723cb.json
trace-1789949030798-91b18b.json
```

The body is fit's `trace-format-v1` plus a `routellm` block naming the
decision that produced it: the endpoint that answered, the request
model, the deepest tier, its area, the full decision path, the cache
and canary flags, latency, the router and its win rate.

**What the caps mean for disk.** Before each write the trace directory
is checked against `max_files` and `max_bytes`, and the oldest traces
are deleted until both hold. Set either to `0` for unbounded. The
directory listing is cached for 60 seconds and after every delete, so a
busy server stats the directory about once a minute, not once a
request. The first time a cap trips it logs one WARNING naming the cap;
after that it is silent. At the defaults, roughly 1 KB per trace, the
file cap binds first at about 10 MB.

Writes are `.json.tmp` then `os.replace`, so a scorer running
concurrently never reads half a file. Recording never raises: any
failure is logged at WARNING and swallowed, because a trace must not
fail the request it describes.

---

## 2. Score the traces

Scoring needs fit:

```
pip install 'routellm[fit]'
```

Start with a rubric. It is regex over the answer text, weighted, fully
deterministic, and makes no network call:

```yaml
# rubric.yaml
patterns:
  - ["\\bbecause\\b", 0.5]
  - ["\\bstep\\b", 0.3]
  - ["\\bfirst\\b", 0.2]
```

The score is the matched weight over the total weight, so an answer
matching every pattern scores 1.0 and one matching none scores 0.0.
Matching is case-insensitive.

```
$ python -m routellm.quality_scores score \
    --traces .routellm_traces \
    --scorer rubric:./rubric.yaml \
    --out scores.jsonl
INFO wrote 106 scored traces to scores.jsonl
```

One JSONL row per trace:

```json
{
    "trace_id": "trace-1789949030797-b7f4f5",
    "endpoint": "local_fast",
    "request_model": "coding",
    "tier": "coding_fast",
    "area": "coding",
    "cached": false,
    "is_canary": false,
    "score": 1.0,
    "breakdown": {
        "\\bbecause\\b": 0.5,
        "\\bstep\\b": 0.3,
        "\\bfirst\\b": 0.2
    },
    "scorer": "rubric:./rubric.yaml",
    "scored_at": "2026-09-21T00:04:28Z",
    "prompt_hash": "6995801374f0e93f"
}
```

Appending is the default, and a trace id already in the file is skipped,
so re-running after more traffic scores only what is new. `--rescore`
scores everything again, which is what you want after changing the
rubric. `--limit N` stops after N traces.

### The four scorer specs

| Spec | What it does | Calls an LLM |
|------|--------------|--------------|
| `rubric:<path.yaml>` | Weighted regex over the answer | no |
| `module:<pkg.mod>:<factory>` | Your own scorer | your choice |
| `composite:<dim>,<dim>` | fit's built-in dimension scorers | no |
| `judge:<model>` | An LLM judge, once per trace | yes |

**`composite:` measures nothing.** fit's `DimensionScorer` returns a
hardcoded 0.5 for every input, so `composite:accuracy,relevance,safety`
gives every endpoint the same number. It is a wiring smoke test, not a
measurement, and the command warns as much when you use it:

```
WARNING composite: measures nothing. fit's DimensionScorer returns a
constant 0.5 for every input, so 'composite:accuracy,relevance,safety'
is a wiring smoke test, not a measurement. Use rubric: or module: for
real numbers.
```

Lead with `rubric:`. Reach for `module:` when your project has a real
scorer of its own; the factory just has to return something with
`score(output, context) -> Reward`.

`judge:` calls an LLM once per trace and is refused unless you also pass
`--allow-llm`, with the call count in the message so the spend is
visible before it is paid.

---

## 3. Aggregate into the sidecar

```
$ python -m routellm.quality_scores aggregate \
    --scores scores.jsonl \
    --config config.yaml \
    --out quality.yaml \
    --min-samples 30
INFO wrote 2 rated endpoints to quality.yaml
```

```yaml
version: 1
generated_at: '2026-09-21T00:04:28Z'
min_samples: 30
transform: linear
source: traces
area_source: config
endpoints:
  cloud_strong:
    quality: 100
    n: 40
    by_area:
      coding:
        quality: 100
        n: 40
  local_fast:
    quality: 39
    n: 66
    by_area:
      coding:
        quality: 81
        n: 32
      copywriting:
        quality: 0
        n: 34
```

**The transform.** `--transform linear`, the default, is
`round(100 * mean_score)` clamped to `[0, 100]`. fit scores live in
`[0, 1]` and `Endpoint.quality` is `ge=0, le=100`, so the number is the
mean reward as a percentage and nothing more. It is recorded in the
sidecar's own `transform` key so a reader knows what they are looking
at.

`--transform percentile` ranks the rated endpoints against each other
and spreads them over `[0, 100]`. Use it when every scorer output
clusters — which `composite:` guarantees. It compares endpoints to each
other, so it needs at least two rated; with fewer it falls back to
linear and warns.

**What counts.** A trace counts when it was scored and was not served
from cache. Cached answers measure the cache, not the endpoint that
once produced them. Canary traces DO count: they are real answers from
a real endpoint, and dropping them would make a measurement depend on
how much canary traffic an endpoint happened to get.

**Below the threshold.** An endpoint with fewer than `--min-samples`
scored traces is left OUT of `endpoints:` entirely, not written with a
null quality. Absent means unrated, and step 4 then leaves its YAML
`quality` alone. The same threshold applies per area: an area under it
drops out of `by_area` while the endpoint's overall number may still be
there.

`--config` is optional and is read only to resolve areas. Without it,
`by_area` is keyed by the raw tier name and the sidecar records
`area_source: tier` instead of `config`.

---

## 4. Point the config at the sidecar

```yaml
# config.yaml
quality_from: ./quality.yaml
quality_from_override: true   # the default
```

A relative path resolves against the config file's own directory, so
`--config /etc/routellm.yaml` works from any working directory.

**Precedence: the sidecar wins.** It is measured; the YAML number is a
guess someone typed once and forgot. Each override logs one INFO line
at startup:

```
INFO quality: local_fast 60 -> 39 (n=66, from the sidecar)
INFO quality: cloud_strong 88 -> 100 (n=40, from the sidecar)
```

`quality_from_override: false` flips it: an explicit YAML `quality`
wins and the sidecar fills only the endpoints that set none. That mode
logs one INFO per endpoint it declined to override, so the decision is
still visible.

Under either setting, an endpoint the sidecar does not name is never
touched and keeps exactly the ordering it has today.

A sidecar whose `generated_at` is more than 30 days old logs one
WARNING naming the age. Never an error: an old measurement still beats
no measurement.

The merge happens after the registry is built and before the Controller
resolves selectors, so `quality_desc` orders on the merged numbers.

---

## 5. Areas, and what `by_area` buys

Pairing resolves once, at startup, not per request. A selector
therefore knows only the tier it sits in, and that tier's name is the
only handle on what kind of work it does. So areas are named explicitly:

```yaml
areas:
  coding:      [coding, coding_quality, coding_fast]
  copywriting: [copywriting, copywriting_quality, copywriting_fast]
```

Each value is the list of tiers belonging to that area. A tier in no
list has no area and orders on the overall number, exactly as every
tier did before areas existed. A tier in two areas is a load error
naming both.

The sidecar above is the case that makes this worth doing.
`local_fast` measures 39 overall, well below `cloud_strong`'s 100 — but
81 within `coding`, because its bad answers are all copywriting. A
`quality_desc` selector inside `coding_fast` reads 81, not 39, and the
candidate table says which number it used:

```
local_fast(model=ollama/qwen, quality=81 [coding], cost=-, context=-)
```

An endpoint with no measurement for an area falls back to its overall
number rather than dropping to the unrated bucket: measured everywhere
except here is still better known than measured nowhere.

Filling the area also closes the loop. The controller stamps each
trace with the area of the tier that answered it, so the next
aggregation has areas without being handed `--config` at all.

Per-request per-area quality is out of scope. Pairing is a startup
step, and making it per-request means re-running every selector on
every call.

---

## How often

The sidecar is a file. Regenerate it by hand or from cron:

```cron
0 4 * * * cd /srv/routellm && \
  python -m routellm.quality_scores score --traces .routellm_traces \
    --scorer rubric:./rubric.yaml --out scores.jsonl && \
  python -m routellm.quality_scores aggregate --scores scores.jsonl \
    --config config.yaml --out quality.yaml
```

**The server reads it once, at startup. There is no hot reload.** A
freshly written sidecar changes nothing until the server restarts, so
schedule the regeneration and the restart together, or accept that the
numbers are as old as the last restart.

Scoring appends, so the nightly run above costs only the traces added
since yesterday. Aggregation re-reads the whole scores file every time,
which is what makes the numbers stable rather than drifting with the
last batch.

---

## Related

- `routellm/quality.py` — trace recording and the caps
- `routellm/quality_scores.py` — scoring, aggregation, sidecar loading
- `config.example.yaml` — `quality_from` and `areas:`, commented
- `examples/multitier.yaml` — a shipped `areas:` block
- [US-0106](../stories/US-0106-traces-to-quality.md)
