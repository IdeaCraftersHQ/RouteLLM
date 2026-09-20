# US-0103: TypeSafe Jev routing

**Status:** paper
**Persona:** API consumer
**Slug:** RLM-NEW-JEV-ROUTING

## User goal

As a developer, I want a router backed by TypeSafe's Jev model so
routing decisions come from a hosted judgment call instead of a
locally-trained classifier, and I want the same Jev-backed
detection available to intent-based model selection (US-0009) so
both routing and intent middleware can share one provider.

## Context

Existing routers (`mf`, `causal_llm`, `bert`, `sw_ranking`,
`random`) all run local models or heuristics. TypeSafe's Jev is a
hosted System One model that turns a prompt into a typed judgment;
a `jev` router calls it to score prompt difficulty and pick
strong/weak accordingly, at a documented confidence threshold.

Intent-based routing (US-0009) currently detects intent via a
configured LLM call inline in `IntentModelSelector`. This story
extracts that into a pluggable `intent_detector` slot so a
`JevIntentDetector` (calling the same TypeSafe SDK) and the
existing `DomainIntentDetector`-style detection both fit the same
interface. Below a confidence floor, `JevIntentDetector` returns
`"general"` rather than guessing.

The TypeSafe SDK is an optional dependency: importing `routellm`
must not require it. The `ImportError` naming the missing extra
surfaces only when a `Controller` is actually constructed with
`jev` selected, not at module import time.

## Acceptance criteria

- [ ] `routers=["jev"]` routes a hard prompt to strong and a
      trivial prompt to weak at a documented threshold
- [ ] Router config accepts `model`, `timeout`, `max_prompt_chars`;
      API key comes only from `TYPESAFE_API_KEY`
- [ ] `IntentModelSelector` accepts an `intent_detector` and
      delegates `detect_intent` to it; `JevIntentDetector` and
      `DomainIntentDetector` both fit that slot
- [ ] Below the confidence floor, `JevIntentDetector` returns
      `"general"`
- [ ] SDK missing → `ImportError` naming the extra at `Controller`
      construction, not at `import routellm`
- [ ] Response model id is logged so thresholds can be pinned to a
      versioned model

## Implementation notes

Planned locations (not yet present):

- `routellm/routers/typesafe/router.py` — `jev` router, registered
  in `ROUTER_CLS` (`routellm/routers/routers.py`)
- `routellm/middleware/jev_intent_detector.py` — `JevIntentDetector`
- `IntentModelSelector` (`routellm/middleware/intent_model_selector.py`)
  gains an `intent_detector` kwarg

## E2E tests

pending

## Related

- US-0001 — strong/weak routing via cost threshold (sibling router
  family; `jev` joins `ROUTER_CLS`)
- US-0009 — intent-based routing middleware (`IntentModelSelector`
  gains the `intent_detector` slot this story defines)
