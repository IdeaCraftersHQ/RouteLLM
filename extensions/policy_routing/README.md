# routellm-policy-routing

Class-aware cache + routing policy mapping for routellm.

Translates **c12n** classification labels into a routellm
**CachePolicy** (skip / lookup / store / TTL / namespace / match
strategy) per the cache policy table from showcase scenario 3
(**T-0104**).

## Status

Paper-stable. Wire-up to routellm core's cache hook ships when
**US-0102** (cache-policy-hooks) lands.

## Install

```sh
pip install -e extensions/policy_routing
```

## Use

```python
from policy_routing import classify_to_policy, CacheAction
from policy_routing.integrations import build_cache_hook

# Direct mapping:
policy = classify_to_policy(request, classification_result)
if policy.action == CacheAction.SKIP:
    ...

# Or as a routellm cache hook (US-0102):
hook = build_cache_hook(classifier=my_c12n_classifier)
controller = Controller(..., cache_policy=hook)  # planned API
```

## Cache policy table

| c12n signal             | action            | ttl    | match    | namespace      |
| ----------------------- | ----------------- | ------ | -------- | -------------- |
| `PII: EMAIL/PHONE/SSN`  | skip              | 0      | exact    | tenant         |
| `Jailbreak: high`       | skip              | 0      | exact    | (n/a)          |
| `Toxicity: high`        | skip              | 0      | exact    | (n/a)          |
| `CodeContent: <lang>`   | lookup_and_store  | 7d     | exact    | workspace      |
| `Domain: legal`         | lookup_and_store  | 15m    | both     | workspace      |
| `Domain: math|cs`       | lookup_and_store  | 7d     | both     | workspace      |
| easy + low-cost         | skip              | 0      | exact    | (n/a)          |
| hard + high-cost        | lookup_and_store  | 7d     | semantic | workspace      |
| `OutputFormat: code/json/yaml` | lookup_and_store | 7d  | exact    | workspace      |
| (else)                  | lookup_and_store  | 24h    | both     | workspace      |

Precedence top-to-bottom (safety wins).

## Spec links

- US-0102 — `docs/stories/US-0102-cache-policy-hooks.md`
- Scenario 3 — `~/.ops/.tlc/tracks/tools-showcase-scenarios/scenarios/3-class-aware-llm-gateway.md`
- T-0193 — workspace_id namespace default
- T-0179 — wsm intra-tenant cache leak audit

## Test

```sh
cd extensions/policy_routing
pytest
```
