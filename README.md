# RouteLLM

The probabilistic quality router. A framework for serving and evaluating LLM routers.

[ [Blog](http://lmsys.org/blog/2024-07-01-routellm/) ] [ [Paper](https://arxiv.org/abs/2406.18665) ]

<p align="center">
  <img src="assets/router.png" width="50%" />
</p>

Our core features include:

- Drop-in replacement for OpenAI's client (or launch an OpenAI-compatible server) to route simpler queries to cheaper models.
- Trained routers are provided out of the box, which we have shown to **reduce costs by up to 85%** while maintaining **95% GPT-4 performance** on widely-used benchmarks like MT Bench.
- Benchmarks also demonstrate that these routers achieve the same performance as commercial offerings while being **>40% cheaper**. 
- Easily extend the framework to include new routers and compare the performance of routers across multiple benchmarks.
- **NEW**: Intent-based routing middleware that allows you to route queries to specialized models based on detected intents.
- **NEW**: Advanced reliability and performance features including auto-retry, circuit breaker, semantic caching, load balancing, and canary testing.

## Advanced Features

RouteLLM now supports a suite of enterprise-grade features for reliability, performance, and quality management:

### Reliability & Resilience
Ensure your system remains operational even when providers fail:
- **Auto-Retry**: Exponential backoff for transient failures (e.g., 429, 5xx). Optional [rate-limit cooldown](#rate-limit-cooldown) skips a rate-limited endpoint until its reset instead.
- **Circuit Breaker**: Prevents cascading failures by opening when success rates drop, automatically entering `HALF_OPEN` state to test recovery.
- **Fallbacks**: Automatically falls back to the alternative model in your pair if the routed model fails after retries.
- **Timeouts**: Enforce strict latency requirements on every call.

### Performance Optimization
Reduce costs and latency with intelligent data management:
- **Simple Caching**: Persistent exact-match caching using SQLite.
- **Semantic Caching**: Intelligent caching based on prompt similarity using vector embeddings. Avoid redundant calls for semantically identical queries.
- **Load Balancing**: Rotate between multiple endpoints, providers, or API keys with **Weighted** or **Round-Robin** strategies.

### Quality & Continuous Improvement
Maintain high standards and automate model improvement:
- **Canary Testing**: Safely split a percentage of traffic to a new model candidate for live validation.
- **Trace Collection**: Standardized request/response recording, in **Fit**'s trace format with the routing decision attached.
- **Measured Quality**: Score recorded traces offline and let selectors order on the measurement instead of a hand-written number.
- **Contract Enforcement**: conceptual integration points for **Eva** to enforce output quality contracts.

## Usage: Advanced Configuration

Configure these features by passing specialized config objects to the `Controller`:

```python
from routellm.controller import Controller
from routellm.resilience import ResilienceConfig
from routellm.caching import CacheConfig
from routellm.traffic import TrafficManager, TrafficRule, LoadBalancer, LoadBalancerConfig, LoadBalancerEndpoint
from routellm.quality import QualityManager, FineTuneConfig, CanaryConfig

# 1. Configure Load Balancing for a specific model name
lb_config = LoadBalancerConfig(
    strategy="round-robin",
    endpoints=[
        LoadBalancerEndpoint(model="openai/gpt-4-turbo", api_key="KEY_A"),
        LoadBalancerEndpoint(model="azure/gpt-4", api_base="https://endpoint-b.com")
    ]
)

# 2. Setup Traffic Management with Rules
traffic_manager = TrafficManager(
    rules=[
        # Route requests containing "code" to a specific pair
        TrafficRule(pattern=".*code.*", strong_model="codellama-34b", weak_model="gpt-3.5-turbo")
    ],
    load_balancers={"gpt-4": LoadBalancer(lb_config)}
)

client = Controller(
    routers=["mf"],
    strong_model="gpt-4",
    weak_model="gpt-3.5-turbo",
    # Resilience: 3 retries, 30s timeout, circuit breaker enabled
    resilience_config=ResilienceConfig(max_retries=3, timeout_ms=30000),
    # Caching: off by default; enable it, with semantic matching at 95% similarity
    cache_config=CacheConfig(enabled=True, semantic_enabled=True, semantic_threshold=0.95),
    traffic_manager=traffic_manager,
    # Quality: 5% Canary traffic and record traces for fine-tuning
    quality_manager=QualityManager(
        canary_config=CanaryConfig(enabled=True, canary_model="gpt-4o", weight=0.05),
        fine_tune_config=FineTuneConfig(enabled=True)  # digests only; capture_content=True stores text
    )
)
```

## Installation

**From PyPI**
```
pip install "routellm[serve]"
```

**From source**

```
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM
pip install -e ".[serve]"
```

**Optional extras**

| Extra | Adds | Needed for |
|-------|------|------------|
| `serve` | fastapi, uvicorn, `mcp` | the OpenAI-compatible server (`mcp` is imported by litellm on any request carrying `tools`) |
| `ml` | torch, transformers, datasets, huggingface_hub | the `mf`, `bert`, `causal_llm` and `sw_ranking` routers; calibration |
| `eval` | matplotlib, sglang, tiktoken, psutil | the evaluation scripts |
| `pairing` | `hop-top-aim` (models.dev client) | policy-based tier selectors |

The base install and `serve` run the `random` router and threshold routing
over config tiers with none of the ML stack (the install is hundreds of MB
rather than several GB). A router that runs a model fails at start-up with an
error naming the missing extra; install it with:

```
pip install "routellm[serve,ml]"
```

If you installed the previous default (which bundled torch), add `ml` to keep
the model-backed routers.

Supported litellm versions: `>=1.106.4,<1.107`. CI runs the suite on the oldest
and the newest version in that range.

**PII analysis (optional)**

`routellm.redact.analyze(text, "fr")` returns character spans for personal
data (person, location, email, phone, IBAN, card, Canadian SIN, RAMQ,
postal code, amounts, date of birth, your own client ids), fully offline.
Pattern recognizers work on a base install; the named-entity tier needs
the `redact` extra and two spaCy models. A policy class with `treatment: redact`
reaches cloud leaves as reversible placeholders restored locally; the optional
`redact-encrypt` extra encrypts the local token store. See [docs/redact.md](docs/redact.md).

**Policy-based pairing (optional)**

A tier side may select over the configured endpoints instead of naming
one, using endpoint tags plus the [models.dev](https://models.dev)
catalog. That needs the `pairing` extra, whose client is not on PyPI
and requires Python 3.11 or newer:

```
pip install "routellm[pairing]"
```

To work against a checkout of the client instead:

```
pip install -e /path/to/poly-aim/py
```

Inspect how a config's selectors resolve without starting a server:

```
python -m routellm.pairing --config config.yaml
```

## Quickstart

Let's walkthrough replacing an existing OpenAI client to route queries between LLMs instead of using only a single model.

1. First, let's replace our OpenAI client by initializing the RouteLLM controller with the `mf` router. By default, RouteLLM will use the best-performing config:
```python
import os
from routellm.controller import Controller

os.environ["OPENAI_API_KEY"] = "sk-XXXXXX"
# Replace with your model provider, we use Anyscale's Mixtral here.
os.environ["ANYSCALE_API_KEY"] = "esecret_XXXXXX"

client = Controller(
  routers=["mf"],
  strong_model="gpt-4-1106-preview",
  weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
)
```
Above, we pick `gpt-4-1106-preview` as the strong model and `anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1` as the weak model, setting the API keys accordingly. You can route between different model pairs or providers by updating the model names as described in [Model Support](#model-support).

Want to route to local models? Check out [Routing to Local Models](examples/routing_to_local_models.md).

You can also use our intent-based routing middleware to route queries to specialized models based on detected intents (see [Intent-Based Routing](#intent-based-routing) below).

2. Each routing request has a *cost threshold* that controls the tradeoff between cost and quality. We should calibrate this based on the types of queries we receive to maximize routing performance. As an example, let's calibrate our threshold for 50% GPT-4 calls using data from Chatbot Arena.
```
> python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5 --config config.example.yaml
For 50.0% strong model calls for mf, threshold = 0.11593
```
This means that we want to use `0.11593` as our threshold so that approximately 50% of all queries (those that require GPT-4 the most) will be routed to it (see [Threshold Calibration](#threshold-calibration) for details).

3. Now, let's update the `model` field when we generate completions to specify the router and threshold to use:
```python
response = client.chat.completions.create(
  # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)
```
That's it! Now, requests with be routed between the strong and weak model depending on what is required, **saving costs while maintaining a high quality of responses**.

Depending on your use case, you might want to consider using a different model pair, modifying the configuration, or calibrating the thresholds based on the types of queries you receive to improve performance.

### Server & Demo

Instead of using the Python SDK, you can also launch an OpenAI-compatible server that will work with any existing OpenAI client, using similar steps:
```
> export OPENAI_API_KEY=sk-XXXXXX
> export ANYSCALE_API_KEY=esecret_XXXXXX
> python -m routellm.openai_server --routers mf --strong-model gpt-4-1106-preview --weak-model anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:6060 (Press CTRL+C to quit)
```

Once the server is launched, you can start a local router chatbot to see how different messages are routed.
```
python -m examples.router_chat --router mf --threshold 0.11593
```

<p align="center">
  <img src="assets/chat-interface.png" width="50%" />
</p>

### Model Support

In the above examples, GPT-4 and Mixtral 8x7B are used as the model pair, but you can modify this using the `strong-model` and `weak-model` arguments.

We leverage [LiteLLM](https://github.com/BerriAI/litellm) to support chat completions from a wide-range of open-source and closed models. In general, you need a setup an API key and point to the provider with the appropriate model name. Alternatively, you can also use **any OpenAI-compatible endpoint** by prefixing the model name with `openai/` and setting the `--base-url` and `--api-key` flags.

The `mf` and `sw_ranking` routers and the semantic cache embed prompts through a registry endpoint (see [How a router embeds a prompt](#how-a-router-embeds-a-prompt)); there is no default embedding provider.

Instructions for setting up your API keys for popular providers:
- Local models with Ollama: see [this guide](examples/routing_to_local_models.md)
- [Anthropic](https://litellm.vercel.app/docs/providers/anthropic#api-keys)
- [Gemini - Google AI Studio](https://litellm.vercel.app/docs/providers/gemini#sample-usage)
- [Amazon Bedrock](https://litellm.vercel.app/docs/providers/bedrock#required-environment-variables)
- [Together AI](https://litellm.vercel.app/docs/providers/togetherai#api-keys)
- [Anyscale Endpoints](https://litellm.vercel.app/docs/providers/anyscale#api-key)

For other model providers, find instructions [here](https://litellm.vercel.app/docs/providers) or raise an issue.

## Motivation

Different LLMs vary widely in their costs and capabilities, which leads to a dilemma when deploying them: routing all queries to the most capable model leads to the highest-quality responses but can be very expensive, while routing queries to smaller models can save costs but may result in lower-quality responses. 

*LLM routing* offers a solution to this. We introduce a router that looks at queries and routes simpler queries to smaller, cheaper models, saving costs while maintaining quality. We focus on routing between 2 models: a stronger, more expensive model and a cheaper but weaker model. Each request is also associated with a _cost threshold_ that determines the cost-quality tradeoff of that request - a higher cost threshold leads to lower cost but may lead to lower-quality responses.

The research in this repository was conducted in [collaboration with Anyscale](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses), and we are grateful for their help and support.

## Server

RouteLLM offers a lightweight OpenAI-compatible server for routing requests based on different routing strategies:

```
python -m routellm.openai_server --routers mf jev
```

The config is discovered rather than named; pass `--config <path>` for a one-off file.

- `--routers` specifies the list of routers available to the server. For instance, here, the server is started with two available routers, `mf` and `jev` (see below for the list of routers). The first one is the default for a request that names none.
- `--config` names an explicit config file, merged last over the discovered chain (system, user, project, `ROUTELLM_CONFIG`). Omit it and the server uses whatever it discovers; with nothing on the chain it falls back to our best-performing router configuration and routes the flat `--strong-model`/`--weak-model` pair. Run `python -m routellm.config paths` to see the chain (see [Configuration](#configuration) for details).
- `--strong-model` and `--weak-model` name that flat pair. Both together or neither; each may name a configured endpoint or a raw model name.
- `--default-threshold` is the threshold used by a level that names none and whose request carries none. Default `0.5`.
- `--host` is the interface to bind. Default `127.0.0.1`. Unless `auth:` is configured the server is unauthenticated, so a wider bind exposes it to that network (and logs a warning): configure [caller identity](#caller-identity) or widen it only behind a proxy that authenticates. Changed: the server previously bound `0.0.0.0`. Pass `--host 0.0.0.0` to restore that, only behind an authenticating proxy.
- `--port` is the port to listen on. Default `6060`.

### Caller identity

The server is unauthenticated unless the config has an `auth:` section (see `config.example.yaml`). With `auth.keys_file` set, every route except `GET /health` needs `Authorization: Bearer <key>`; a missing or unknown key is a `401`. The keys file maps the SHA-256 of each key to a caller, so no key is ever stored in clear:

```yaml
keys:
  - key_sha256: "<printf %s "$KEY" | shasum -a 256>"
    principal: team-a
    profile: default          # optional
    allowed_classes: [public] # optional
```

An entry holding a clear-text key stops startup. Keys are compared in constant time and never logged. The matched caller is a `routellm.auth.Principal` on `request.state.principal` and is passed to `Controller.acompletion(principal=...)`; it is never sent upstream. Swap the static map for another source by implementing `routellm.auth.PrincipalResolver`. With auth on, widening `--host` no longer warns.

#### Several clients behind one key

A local proxy that holds one key can name its own clients per request. Opt in on that entry only; every other entry behaves as above and ignores the header, so a caller cannot change its identity by sending it:

```yaml
keys:
  - key_sha256: "<sha256 of the proxy's key>"
    principal: gateway
    profile: default                 # fallback for clients below
    allowed_classes: [public]        # fallback for clients below
    client_header: X-Route-Client    # a valid header name
    clients:                         # non-empty; names are 1-64 of [A-Za-z0-9._-]
      web: {profile: web, allowed_classes: [public, internal]}
      batch: {}
```

| Request | Resolves to |
|---|---|
| key valid, header absent | `gateway` |
| key valid, header `web` | `gateway/web` (its `profile`/`allowed_classes`, else the entry's) |
| key valid, header names an unlisted client | `403`, naming the header, never the value |
| entry without `client_header`, any header | the entry's own principal |
| key unknown | `401` |

Policy `principals.gateway/web`, audit, cache isolation and redaction scope all key on the resolved principal, and a policy principal naming an unlisted `<entry>/<client>` stops startup like any other principal without a key. The header is never sent upstream. Use it only behind a proxy that sets the header itself and strips any the caller sent: whoever holds the key can name any listed client. The keys file's contents are not config, so no new key appears under `auth:`, which stays a system-layer pinned section; protect the file itself.

For most use-cases, **we recommend the `mf` router** as we have evaluated it to be very strong and lightweight.

### Endpoints

An endpoint gives a reachable model a stable name, so routing, caching, and traces refer to `cloud_strong` rather than to whatever litellm model string happens to sit behind it. Endpoints are declared under the `endpoints:` key of the `--config` file:

```yaml
endpoints:
  cloud_strong:
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    tags: [cloud, tools]
    quality: 90
  ollama_qwen:
    model: ollama_chat/qwen3:8b
    api_base: http://127.0.0.1:11500
    tags: [local]
```

Each endpoint carries its own `api_base` and its own credential, so one server can span a cloud provider and two local runtimes at once. `api_key_env` names an environment variable read at call time — never the key itself — so a config loads on a machine that holds none of the keys. A missing variable is an error naming the endpoint and the variable, raised when that endpoint is called.

The global `--base-url` and `--api-key` flags remain as the defaults an endpoint falls back to when it sets none of its own, so an existing flat deployment keeps working unchanged. Precedence for one call is: a load balancer registered on the routed name, then the endpoint's own values, then the two flags.

Names match `[A-Za-z0-9_]+`; hyphens are reserved for the model-name grammar below. A name the registry does not know is used as a raw litellm model name with the global defaults, and logged once as a warning.

Endpoint names are also the cache, resilience, and trace keys. Adopting them therefore changes existing SQLite cache keys, and entries written under the old raw model names go unread.

#### Governance tags

Tags are free-form, with one rule: a tag written `<ns>:<value>` is namespaced, and its namespace must be known. Two conventions are parsed, not just matched:

- **reach**: the subset of `local`, `vpn`, `cloud` an endpoint carries (an endpoint on your own box that is also reachable over a private network is `local` and `vpn`);
- **`jur:<cc>`**: at most one, a lowercase ISO 3166-1 alpha-2 code (`jur:ca`, never `jur:CA`).

`jur` is the built-in namespace; add your own with `tag_namespaces: [team]`. An unknown namespace (`jurr:ca`), a second `jur:` tag or a malformed code fails the config load, naming the endpoint and the tag. `Endpoint.reach` and `Endpoint.jur` expose the parsed view.

Setting `governance: true` (top-level, off by default) additionally requires every endpoint to carry at least one reach tag and exactly one `jur:` tag.

#### Policy file

`policy: <path>` (top-level) loads a policy file mapping data classes to what may happen to them. It denies by default and fails closed: any load error stops startup, naming the file, class and field. A relative path resolves against the config file that sets it. Loading a policy implies `governance: true` and `strict_registry: true`.

```yaml
default_class: null              # a class name, or null to refuse a request with none
classes:
  public:
    treatment: raw               # raw | redact | block | vault
    reach: [local, vpn, cloud]   # reach labels an endpoint may carry
    jur: []                      # allowed jurisdictions; empty = any
    tiers: [default]             # entry tiers a request of this class may use
    cache: exact                 # none | exact | both
  internal:
    treatment: block
    reach: [local, vpn]
    jur: [ca]
    tiers: [private]
  sealed:
    treatment: vault
principals:
  analyst: {classes: [public, internal], projects: [research]}  # projects: policy_routing policies a header may select
```

- `cache` defaults to `exact` for `raw` and `none` for the other treatments.
- `cache` is enforced per request: `none` makes no lookup and no store; `exact` serves only an identical conversation and never asks for an embedding; `both` also allows the semantic match, when `cache.semantic_enabled` is on. Whatever a `block` or `redact` class sets, it is never cached: a redact class's replies cannot be restored without that request's own mapping, and a blocked class is never stored.
- `reach` and `tiers` need at least one entry on every class except `vault`; an empty list would allow nothing.
- `vault`: every request is refused with no model call; it takes no `reach`, `jur`, `tiers`, and `cache` must be `none`.
- `block`: `cloud` in `reach` is a load error.
- `redact`: `cloud` in `reach` is allowed: the main call to a non-local leaf goes out as placeholders and the reply is restored locally (see [docs/redact.md](docs/redact.md#the-redact-treatment)). Without a `redact:` config section the server refuses to start; every auxiliary call that carries the class's content stays local-only.
- An unknown class on a request is refused; so is a missing class when `default_class` is null. A principal reaches only the classes listed for it; an unlisted principal reaches none.

#### Class to tiers

`classes.<name>.tiers` lists the entry tiers a request of that class may use. Two checks hold it.

- **Request**: the tier the model name addresses must be in the list of the request's effective class, else `403` in the API error shape, naming class and tier (`data class 'internal' may not use tier 'general' (allowed: private)`). The effective class is the declared one, or the decision classifier's when that is stricter; the check runs after the decision step and before any routing, so a refused request calls nothing. A model name that addresses no tier (a flat pair) and a request with no class are refused too.
- **Startup**: for each non-vault class, every leaf reachable from each allowed tier must carry only reach labels within the class's `reach`, and a `jur:` within its `jur` (when set). Reachable means both sides of every level, nested tiers, the sibling a failed pick falls back to, and every member of a candidate pool; plus, for every class, traffic-rule pairs, the canary model, middleware pairs, and the targets of a load balancer keyed on a reachable leaf. Any violation refuses to start, listing each as class, tier, leaf and offending tag. A tier named in `tiers` that the registry lacks is an error too.

`routellm.class_tiers` holds both (`check_entry_tier`, `check_class_tiers`); `PolicyDenied` (status 403) is what the request path raises.

**Every candidate, every path.** Under a policy the same reach and `jur` rule is applied again to each endpoint a request could be sent to, at request time (`Controller._can_serve`, through `routellm.class_tiers.class_denial`): tier sides (a tier-valued side passes only if every leaf under it does), a side forced by capabilities or by the class, the fallback sibling, candidate-pool members, load-balancer targets, the canary model, traffic-rule and middleware pairs, and every comparison target. A denied candidate is never called and is never replaced by another class or tier: when none is left the request fails `403` with the reason `policy:<class>:<reach|jur>`. A sibling dropped for its class after the primary failed upstream ends the same way, with `403` rather than the upstream error. Nothing changes without a policy, or for a class that may reach the endpoint.

- **Comparison**: a shadow target the request's class may not use is dropped before anything is called and listed in the primary's path entry as `comparison_denied` (`endpoint`, `reason`). `POST /v1/compare` refuses with `403` when the class may not enter the named `tier` or may not use any listed model, and calls nothing. The class is the strictest of the declared one, the decision classifier's and the selected [routing policy](#policy-routing)'s floor; under a policy a client request without one is refused. The operator's replay run, which declares no class, is not class-gated by the policy; an [egress gate](#egress-gates) still judges it, with the class undecided.
- **Intent tier switch**: a tier an intent maps to that is not in the class's `tiers` is ignored, with no error: the request stays in the tier it entered, and the path entry records `intent_ignored` with the tier named. Intent tiers a class may enter are walked at startup like any other allowed tier.

#### Request data class and operation

A caller declares what it sends, in either of:

- headers `X-Data-Class` and `X-Operation`;
- body `metadata: {data_class, operation}` (OpenAI-compatible; the rest of `metadata` is dropped).

If the header and the body name the same field with different values the request is a `400`. `data_class` matches `[A-Za-z0-9][A-Za-z0-9_.-]{0,62}`; `operation` is free-form `<tool>.<step>`, 1 to 128 visible characters, recorded only (audit) and never used for routing. Both ride on `request.state.request_class` and `routellm.request_class.current_request_class`, next to the principal, and are never forwarded: not to an upstream, a shadow or comparison call, or the decision classifier.

With a policy loaded: an unknown class, or none declared when `default_class` is null, is a `400` naming the class; a vaulted class, or a class the caller may not send, is a `403`. Only the policy's `principals` decides what a caller may send; the keys file's `allowed_classes` is unused (one startup warning if set), and every principal the policy names must hold a key (startup error). A policy with `principals` needs `auth:`: with auth off no request has a principal, so each is refused (`403`). The validated class is the one passed to the decision classifier.

Without a policy: a declared class is checked against the keys file's `allowed_classes` (`403` if absent from it; empty means none), and handed to the decision classifier only if it is one of its `data_classes`, else ignored. `POST /v1/compare` obeys the same rules.
- Unknown keys, reach labels, treatments or malformed `jur` codes are load errors.
- The classifier's `decision.data_classes` must all be policy classes (the policy may list more, for classes callers declare); otherwise startup fails naming the class.

`routellm.policy.Policy` is the loaded object (`routellm.openai_server.POLICY`; None without a `policy:` key): `resolve(declared)` returns `(name, ClassPolicy)` or raises `PolicyRefusal` with `code` `unknown_class`, `missing_class` or `vault`; `principal_allows(principal, name)`; `classes[name].reach/jur/tiers/treatment/cache`.

#### Strict endpoint keys

Unknown keys under an endpoint are a load error naming the endpoint and the key, with a "did you mean" for near misses (`api_base_env` for `api_base`). A silently ignored key would leave the endpoint on its provider's default base.

An endpoint tagged `local` must set `api_base`. Without one the call would go to the provider's default (public) base, so a leaf meant to stay on your machine would send its prompts to the cloud; the config is refused instead. Drop the tag if the endpoint is meant to be remote.

The host of that `api_base` must also be non-public, judged as written (after `${VAR}` expansion, no DNS lookup, so a load never touches the network): loopback (`127.0.0.0/8`, `::1`, `localhost`), private and link-local ranges (RFC 1918, `fc00::/7`, `169.254.0.0/16`, `fe80::/10`), or a name ending `.local`, `.internal` or `.localhost`. Any other host is refused at load, including shared/CGNAT space (`100.64.0.0/10`), `.lan` and single-label names such as a compose service (`http://ollama:11434`), since a resolver search domain can send those anywhere; use an IP literal or a `.internal` alias for those, because a `local` tag on a public host would send prompts off the network. A tunnel that reaches a local server through a public name needs an explicit opt-out:

```yaml
endpoints:
  tunnelled:
    model: openai/some-model
    api_base: https://tunnel.example.com/v1
    tags: [local]
    allow_public_base: true   # dangerous: `local` no longer means the prompt stays on your network
```

`allow_public_base` defaults to `false`. Prefer a private address or a `.internal` name to the opt-out.

#### Strict registry mode

By default a model name that is not a registry endpoint is passed to litellm as a raw model string, with the controller's `api_base`/`api_key`. `strict_registry: true` (top-level) or `--strict-registry` closes that on every path; it is off by default. Either one turns it on; the flag cannot turn a config's `true` off.

- `resolve` of an unknown name raises `ValueError`, so a fallback, pool member or any other late lookup fails closed.
- Checked at startup, each refused with the name and where it came from: `--strong-model` / `--weak-model` (and the controller's flat pair), traffic-rule models, `canary_model`, load-balancer keys and targets, and the pairs or raw detection model a middleware carries. The legacy default pair (used when no flag and no `default` tier is given) is never invented: the server refuses to start without a `default` tier or two endpoint flags.
- A middleware pair returned per request is checked per request; a raw one is a `RoutingError` (HTTP 400).
- Comparison and shadow targets, and `intents.model` / `intents.upstream`, already must be registry endpoints.
- Every name above must be an endpoint, not a tier. A load-balancer target is an endpoint name: the call carries that endpoint's model, base, key and params (the balancer's own `api_base`/`api_key` still win).

#### Environment variables in config values

One config file can serve several hosts whose endpoint URLs differ. Any string value (in mappings and lists; never keys) may read the environment:

| Syntax | Meaning |
|--------|---------|
| `${VAR}` | the value; unset is a load error |
| `${VAR:-default}` | `default` when unset or empty |
| `${VAR:?message}` | load error with `message` when unset or empty |
| `$$` | a literal `$` |

A bare `$NAME` is left alone, so header templates and model ids containing `$` need no escaping.

```yaml
endpoints:
  box:
    model: openai/my-model
    api_base: ${MY_BOX_BASE:-http://127.0.0.1:11434/v1}
    tags: [local]
```

Expansion runs once, on the merged config. Layers still merge first (system, user, project, `ROUTELLM_CONFIG`, `--config`), so a higher layer's literal replaces a lower layer's `${VAR}`; the replaced variable is never read and need not be set. An unset variable fails the load with the file that set the value, the dotted key (`endpoints.box.api_base`) and the variable.

Do not interpolate secrets. Keys stay in `api_key_env`, read at call time. A `${...}` in a field named like a secret (`api_key`, `secret`, `token`, `password`, `authorization`) is refused.

`python -m routellm.config show` prints the text as written, plus the variables each key reads, and needs none of them set. Add `--resolve` to print the expanded values.

### Tiers

A tier is a named strong/weak pair whose sides may themselves be tiers, so a request addressed to a tier walks a tree, one router call per level on the original prompt:

```yaml
tiers:
  premium:
    router: mf
    threshold: 0.33
    strong: cloud_strong
    weak: local_glm
  default:
    router: mf
    threshold: 0.12
    strong: premium
    weak: ollama_qwen
```

A tier may name any registered router. `jev` is registered by the typesafe extension rather than the base package, so a tier routing over it needs `pip install -e ./extensions/typesafe`; without it the server reports the unknown router at startup.

A tier naming no `router` or `threshold` inherits it, first hit winning: its own value, the parent level's resolved value, the request-level value from the model name, then `--routers[0]` and `--default-threshold`. The response records where each level's values came from, so a routing decision is explainable without re-running it:

```python
out = controller.completion(
    model="default", messages=[{"role": "user", "content": "hello"}]
)
for entry in out._hidden_params["routellm_path"]:
    print({k: entry[k] for k in
           ("tier", "router", "router_from", "threshold", "threshold_from", "picked")})
```

```
{'tier': 'default', 'router': 'random', 'router_from': 'tier', 'threshold': 0.12, 'threshold_from': 'tier', 'picked': 'premium'}
{'tier': 'premium', 'router': 'random', 'router_from': 'tier', 'threshold': 0.33, 'threshold_from': 'tier', 'picked': 'cloud_strong'}
```

Over the HTTP server the same path arrives on a non-streamed response under a top-level `routellm` key, which OpenAI clients ignore. A streamed response carries it in the server logs only, since its chunks must keep the SSE shape the client parses.

The chat endpoint accepts what agent clients send: nested `tools`, `tool_choice` as a string or object, `tool_calls` and `role: "tool"` turns, null or absent `content`, content parts including `image_url`, and `response_format` of type `json_schema`. These are forwarded unchanged. Besides the sampling fields, an explicit allowlist passes through: `stream_options`, `max_completion_tokens`, `parallel_tool_calls` and `reasoning_effort`. Any other field in a body is dropped.

An allowlisted flag is advisory, and a routed provider may not take it: litellm refuses `parallel_tool_calls` on an `ollama_chat/` leaf, for one. By default each leaf is sent without the advisory flags (`parallel_tool_calls`, `reasoning_effort`, `stream_options`) its provider does not list as supported, and a leaf that does list one still gets it. Nothing else is dropped by default: an unsupported `tools`, `response_format` or sampling value still fails the call rather than being ignored. The endpoint's `drop_params:` changes that for one leaf: `true` drops whatever its provider rejects (litellm's `drop_params`), `false` drops nothing, and a list of names drops those for that leaf, supported or not, on top of the default. At DEBUG the server logs the names of the params dropped for a call and the endpoint, never their values.

**Cache isolation.** An entry is keyed on the prompt, model and params plus a namespace: a SHA256 digest of the authenticated principal and the request's effective data class. A principal's entries are never served to another principal, nor to an unauthenticated caller; an entry stored under one class is never served under another, exactly or semantically (the semantic lookup filters on the namespace). The cache database holds the digest, not the names. With no auth and no policy the namespace is constant and the cache behaves as before. The key format changed: entries written by earlier versions are never read.

A streamed response is never cached. The first chunk is read before the routed leaf is committed to, so a leaf that fails before it hands over to its sibling; a failure after it ends the stream with an `error` event (no `[DONE]`) and the sibling is not called. The trace is recorded when the stream ends, and canary validation does not apply to streams. All of this holds on the async path only, which is what the server uses: the synchronous `Controller.completion(stream=True)` hands back the provider's stream as it is, with no first-chunk read, no sibling fallback and no trace. Every outbound upstream call carries an `X-RouteLLM-Hop` header, merged with the endpoint's own `extra_headers`. A request carrying an `X-RouteLLM-Hop` header is refused with 508 before any routing, so a leaf that fronts this server cannot loop back into it.

Tier and endpoint names share one namespace and are validated together at load: every side must name a known endpoint or tier, the graph must be acyclic, and it may not nest deeper than four tier levels. A name that is both a tier and a router is rejected too.

If a tier fails, the request falls back to the other side of the level it was decided at. A tier-valued sibling is descended by its `weak` side without running any router, so the failure path stays deterministic.

### Model names

Clients address the router through the `model` field, in any of four forms:

| `model` | Means |
|---|---|
| `premium` | the `premium` tier, inheriting router and threshold |
| `router-premium` | the same tier, for clients that expect a `router-` prefix |
| `router-mf-0.5` | the legacy flat form: the `mf` router at threshold `0.5`, entering the `default` tier when one is configured and the flat pair otherwise |
| `premium:router-mf-0.5` | the `premium` tier, with `mf` and `0.5` supplied at request level |

A request-level router and threshold sit below a tier's own values in the inheritance order, so they fill in the levels that name none rather than overriding the levels that do.

The `default` tier is what makes the legacy form keep working: an existing OpenAI client sending `router-mf-0.5` reaches the tree without changing its model field. Where that tier comes from depends on what you pass:

| `--config` has a `default` tier | `--strong-model` + `--weak-model` | Result |
|---|---|---|
| yes | either | the config tier; the flags are ignored for it and stay the flat pair |
| no | both given | an implicit `default` tier from the two, running `--routers[0]` at `--default-threshold` |
| no | neither given | no `default` tier at all — none is invented and none is listed on `/v1/models`. `router-<r>-<thr>` still routes, against the historic `gpt-4-1106-preview` / `anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1` pair, and the server warns at startup that it assumed it |

The two flags must be given together: one alone cannot form a pair and is an argument error. Which `default` tier is in effect is logged at startup.

### `GET /v1/models`

Many OpenAI clients probe the model list before their first request. The server answers with one id per configured tier, plus `router-<name>-<default_threshold>` for each loaded router — the legacy form needs a concrete threshold, and only the server's default is a usable id, though any other threshold still routes:

```
> curl -s localhost:6060/v1/models
{"object":"list","data":[
  {"id":"default","object":"model","created":1758326400,"owned_by":"routellm",
   "routellm":{"kind":"tier","capabilities":{"vision":true,"tools":true,"context":262144},
               "unknown":["reasoning","max_output"]}},
  {"id":"router-mf-0.5","object":"model","created":1758326400,"owned_by":"routellm",
   "routellm":{"kind":"router"}}]}
```

`created` is the moment the process came up: the listing is derived from the config, so it does not change while the server runs.

Each entry carries an additive `routellm` object. For a tier id it holds the union of what every reachable leaf can do plus `unknown`, the keys no reachable leaf knows; for a `router-<name>-<thr>` id it holds `kind: router`, plus the flat pair's capabilities when the controller carries one. The standard OpenAI keys keep their exact values and order, and unknown keys are ignored by every OpenAI client.

### Selectors

A tier side may say what it wants instead of naming one endpoint, and the winner is chosen once at startup from the configured endpoints' own tags plus the [models.dev](https://models.dev) catalog:

```yaml
tiers:
  default:
    router: mf
    threshold: 0.12
    strong: {select: "tool_call:true reasoning:true", order: quality_desc}
    weak: {select: "tag:local", order: cost_asc}
```

`select` is a space-separated list of terms, all ANDed. A `tag:<label>` term is answered locally from the endpoint's own `tags`; a capability term is answered from the merged [capabilities](#capabilities); every other `key:value` term is a models.dev fact. Bare free text is rejected — a policy says what a model *is*, never what its name looks like. `order` picks the winner among the matches, one of `cost_asc`, `cost_desc`, `quality_asc`, `quality_desc` (default), `context_desc`, `max_output_desc`.

A `tag:` term also takes `!` and `|`: `!tag:cloud` keeps endpoints that do not carry `cloud`; `tag:jur:ca|jur:fr` keeps endpoints carrying at least one alternative; `!tag:jur:ca|jur:fr` keeps those carrying none. Alternatives must share one namespace (`jur:ca|jur:fr`) or all be reach labels (`local|vpn|cloud`). `!` applies to `tag:` terms only. Any `tag:` term naming an unknown namespace, an empty label or alternative, or mixed namespaces is a startup error, never a match-all. Terms stay ANDed: `tag:vpn !tag:jur:fr`.

The capability terms are `vision:true`, `tools:true`, `structured_output:true`, `reasoning:true`, `open_weights:true`, `context:>=N`, `context:<=N`, `max_output:>=N`, `max_output:<=N`, and `input:image`. `tools` is an alias of the catalog's `tool_call`: both spellings work and mean the same thing, because both are answered from `Capabilities.tools`, which already merged the catalog's value. A range term needs its comparison — a bare `context:200000` is an error naming the term and showing `context:>=200000`, because "exactly this context window" is never what an operator means.

Selectors need the optional `pairing` extra. Endpoints models.dev does not list, such as anything served by Ollama, are tag-only candidates: they satisfy `tag:` terms and fail every catalog term. Explain a pick without starting a server:

```
python -m routellm.pairing --config config.example.yaml
```

```
-> local_glm(model=openai/glm-5.2, quality=85, cost=-, context=-)
   ollama_qwen(model=ollama_chat/qwen3:8b, quality=None, cost=-, context=-)
```

Every selector is resolved to an endpoint name before the tier graph is validated a second time, so nothing downstream ever sees one.

`free:true` keeps only endpoints whose models.dev record prices input and output at exactly 0; `free:false` keeps the rest. An endpoint the catalog does not list, or lists without a price, is not free: a missing price is not a price of zero, so an uncatalogued `:free` id fails `free:true` unless the endpoint states its own price.

**Price override.** An endpoint may carry `price: {input: 0, output: 0}` (USD per 1M tokens; both keys required, non-negative numbers, no other keys) or `free: true` (shorthand for 0 and 0; `free: false` and `price` together with `free` are rejected at load). It replaces the catalog price for that endpoint everywhere a price is read: the `free:` selector term, `cost_asc`/`cost_desc` ordering, and comparison `cost_usd` and budgets. An explicit override wins over the catalog, whether the catalog says paid or free. Use it for local and OpenAI-compatible models the catalog lacks, or new ids it has not listed yet.

```yaml
endpoints:
  local_a: {model: ollama_chat/a, api_base: "http://127.0.0.1:11434", tags: [local], free: true}
  gw_new:  {model: openrouter/vendor/new:free, tags: [cloud], price: {input: 0, output: 0}}
```

#### Candidate pools

`pool: N` keeps the best `N` matches instead of one. The first is the side, as before; the rest are tried in order, after it and before the other side, when it fails or is cooling after a rate limit ([below](#rate-limit-cooldown)). Every pool member passes the same capability check a sibling does, and a member that cannot serve the request is dropped.

```yaml
tiers:
  default:
    strong: {select: "tag:paid", order: quality_desc}
    weak: {select: "tag:cheap free:true", order: quality_desc, pool: 3}
```

`pool` defaults to 1. A pool applies to tier sides only; a side that names an endpoint, and a flat `strong_model`/`weak_model` pair, has none. A side's pool is built from its own `select`, so a tier whose sides select `tag:local` can never list a cloud endpoint, whatever its pool size.

The decision path records the walk on its last entry, `walk`: each endpoint skipped as `cooling` (with its `until`) or `failed`, ending with the one that `served`. It is absent when the pick answered at once.

### Rate-limit cooldown

Off by default. With `resilience.rate_limit_cooldown: true`, a 429 is not retried against the same endpoint. The endpoint cools until the reset the upstream stated, and the walk moves on at once; a cooling endpoint is skipped without a call until then. A 429 under cooldown is not a fault, so it does not count toward the circuit breaker, and the provider SDK's own retries are switched off for the call (a 429 would otherwise reach the endpoint again after the upstream's `Retry-After`).

The reset is read, in the order of `rate_limit_reset_headers`, from the response headers of the error (and the error body's `metadata.headers`, which some gateways repeat there): `Retry-After` as seconds or an HTTP date, any other reset header as epoch milliseconds, epoch seconds, or seconds by magnitude. With none stated, `rate_limit_default_cooldown_s` applies, or `rate_limit_daily_cooldown_s` when the message contains one of `rate_limit_daily_markers` (a daily quota). No cooldown exceeds `rate_limit_max_cooldown_s`. When every candidate is cooling the request fails with `EndpointsCooling` naming each endpoint and when it is next tried. The server answers it with a 503, a generic JSON error carrying the request id, and `Retry-After` set to the soonest cooldown end in whole seconds (rounded up, at least 1); the endpoint names and times stay in the server log. This holds for streamed requests too, since the failure comes before the first chunk; a stream that fails later has already sent its 200 and ends with an `error` event instead. The cooldown is per process and in memory.

A rate-limit error event inside a stream is handled the same: before the first chunk the sibling answers; after it the endpoint cools for the next request. A stream that fails after its first chunk counts as one breaker failure for that endpoint, and a clean end as one success; the breaker hears nothing at the first chunk.

`retry_on_timeout: false` stops a timed-out call being retried. The abandoned attempt may still be running upstream, so on a paid endpoint a retry can be billed twice; the default (true) keeps the previous behaviour.

### Free-model roster

Free model lists churn. Diff a config's free endpoints against a provider's public models listing (report only, nothing is edited):

```
python -m routellm.roster --url <models-listing-url> --prefix <litellm-prefix>/ [--suffix :free] [--config path]
```

It lists ids the roster marks free that the config lacks, configured ids no longer listed, and configured ids carrying an `expiration_date`.

### Capabilities

Routing picks on difficulty and on policy; capabilities add *fitness*. A request carrying an image should never be sent to a text-only model, and the first sign of trouble should not be a provider error after the tokens were spent.

Any endpoint may state what it can do:

```yaml
endpoints:
  local_server:
    model: openai/qwen3-coder
    api_base: http://127.0.0.1:11800/v1
    capabilities:
      vision: false
      tools: true
      structured_output: true
      reasoning: false
      context: 262144
      max_output: 32768
      modalities_in: [text]
    strict: false
```

Three sources fill the record, highest precedence first: the explicit `capabilities:` block, the deprecated capability tags, then the [models.dev](https://models.dev) record the pairing path already fetches. Tags sit *between* the block and the catalog because a hand-written capability tag on a catalogued model is nearly always a correction to a catalog the operator found wrong, and an operator who wants to override a tag has the block.

An omitted field is `null`, which means **not known** — a different answer from `false`, which means known not to. Nothing is inferred: a field is filled only by a source that states it.

The tags `tools`, `vision` and `long_context` still populate the block for one more release, logging one deprecation warning per alias per process (`long_context` sets `context: 200000`, a floor an operator asserts rather than a measurement). Tags themselves are not deprecated: they keep their governance, performance and expertise meaning and stay selectable with `tag:`.

#### The two opposite defaults for unknown

At **selection** time — resolving a selector at startup — an unknown capability **fails** its term. An endpoint that has not declared `vision` does not satisfy `vision:true` and is dropped. Startup is where an operator can see the gap and fix it, `--capabilities` prints exactly which endpoint is unknown on which key, and silently selecting a model that might not do the job is worse than a clear "no candidate matched".

At **request** time — checking a side before its router runs — an unknown capability **serves**, and the assumption is logged once per endpoint and requirement at INFO. `strict: false` is the default, so a config that declares nothing about any of its endpoints keeps routing exactly as it always did. `strict: true` on an endpoint flips it: unknown means no, for an endpoint you would rather refuse than surprise.

Both defaults are deliberate. Read together: *at startup, prove it; at request time, assume it unless told otherwise.*

#### What a request needs

Five facts are read off the request and nothing else: `vision` from an `image_url` or `input_image` part in some message's content, `audio` from an `input_audio` or `audio_url` part, `tools` from a non-empty `tools` or `functions`, `structured_output` from a `response_format` of `json_schema` or `json_object`, and `context_needed` from the prompt's token count plus `max_tokens`. `stream`, `n`, `temperature`, `seed` and the rest never affect routing.

`context` is judged only when both the count and the endpoint's window are known: the count is an estimate, and it never refuses a request whose endpoint declares no context window.

#### Audio input

A request carrying an audio part (`{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`, or `audio_url`) needs an endpoint that serves audio. An endpoint declares it with `capabilities.audio: true`, by listing `audio` in `modalities_in`, or through its models.dev record; `modalities_in: [text]` or `audio: false` declares the opposite.

```yaml
endpoints:
  local_audio:
    model: openai/<your-audio-model>
    api_base: http://127.0.0.1:11434/v1
    capabilities:
      modalities_in: [text, audio]
```

Audio is **fail-closed**, unlike the other requirements: an endpoint that says nothing about audio does not serve it, `strict` or not, because a text-only model would answer without ever hearing the clip. A tier takes a side only if it (or a leaf reachable from it) declares audio; with none, the request is refused with a 400 naming `audio` before any upstream call, and the audio is never dropped or sent to a model outside the tier. The path entry shows `capability_requirement: "audio"` when audio forced the side. A request with an image and audio needs a side declaring both. Selectors accept `audio:true`.

Transport matters. litellm's native `ollama_chat/` and `ollama/` providers drop audio parts (and `ollama_chat/` fails while counting tokens over them), so routellm treats those endpoints as not audio capable and logs a warning if you declared otherwise. Reach an audio-capable Ollama model through its OpenAI-compatible route: `openai/<model>` with `api_base: http://host:11434/v1`.

**Audio in the context estimate.** `litellm.token_counter` cannot count audio, so each `input_audio` part adds an advisory flat cost per second of audio on top of the text count, and a long clip can therefore exceed a leaf's `context` and be refused (400) like any oversized prompt. The duration of a WAV is read from its header; any other format is guessed from the byte length at a deliberately low bitrate for that format (mp3 and aac 32 kbit/s, ogg, opus and webm 24 kbit/s, flac 64 kbit/s, anything else 32 kbit/s), so the guess errs long. An `audio_url` part carries no payload here and adds nothing. The rate is a top-level key, default 32 tokens per second, the high end of what providers publish:

```yaml
audio_tokens_per_second: 32      # positive number; lower it for a provider that bills audio more cheaply
```

It matters when audio reaches a model (`mode: passthrough`, or no transcriber) and for judging clip size before a transcriber runs. With `mode: transcribe` the audio is replaced by its transcript before routing, so only the transcript is counted and the audio is never counted twice. A leaf that declares no `context` is never refused on size.

#### Transcribe-then-route

Instead of finding an audio-capable model, a tier can turn each audio part into text first and route the result as an ordinary text request. Any side may then serve it, and the tier's usual rules apply to the transcript. Configure a speech-to-text service that speaks OpenAI's `/v1/audio/transcriptions`:

```yaml
transcriber:                      # global default; omit to keep audio routing as above
  mode: transcribe                # transcribe | passthrough (default transcribe)
  api_base: http://127.0.0.1:9000/v1
  api_key_env: TRANSCRIBER_API_KEY   # optional; the variable's NAME, read per call
  model: whisper-1                # default whisper-1
  timeout_ms: 60000               # one call; default 60000
  max_audio_bytes: 26214400       # per audio part, decoded; default 25 MiB
  language: null                  # default language hint; null lets the service detect it
  format: "[{start}-{end}] {speaker}: {text}"     # a line with a speaker (the default)
  format_no_speaker: "[{start}-{end}] {text}"     # a line without one (the default)
  allow_urls: false               # fetch audio_url parts? default false: refused
  tags: [local]                   # `local` declares the service runs on your own machine/network
  allow_public_base: false        # DANGEROUS: lets a `local` transcriber name a public host; default false

tiers:
  default:
    strong: cloud_strong
    weak: local_glm
    transcriber:                  # per tier: overrides the global keys it names
      language: fr
  keep_audio:
    strong: local_audio
    weak: local_audio2
    transcriber: {mode: passthrough}   # this tier routes audio to an audio-capable side
```

A transcriber tagged `local` must name a loopback, private (RFC 1918 / ULA), link-local, `localhost`, `.local`, `.internal` or `.localhost` host in `api_base`, judged as written after `${VAR}` expansion with no DNS lookup; a public, `.lan`, single-label or shared-space (100.64/10) host fails at load, globally and per tier, with the same rule as a `local` endpoint. `allow_public_base: true` (a real boolean) is the explicit, dangerous opt-out. An untagged transcriber may use any host. Unknown keys under `transcriber:` are a load error, as are unknown placeholders in a template. A tier's keys are laid over the global section; `mode: passthrough` (or no `transcriber:` at all) leaves audio routing exactly as described above. The transcriber used is the one on the tier the request **addresses** (the global section when it addresses none), decided before any intent re-routing. Because of that, a tier that accepts intent routing (`default`, or `intent_routing: true`) and that an `intents:` mapping can send to a local-only tier (see Private tiers) must itself use a `local` transcriber, or the config is rejected at load. Middleware supplied in code rather than through `intents:` is not visible to this check.

For every `input_audio` part the server POSTs the decoded bytes as multipart to `<api_base>/audio/transcriptions` with `response_format=verbose_json` and `timestamp_granularities[]=segment`, plus `language` when one applies, and replaces the part with a text part, one line per segment:

```
[00:01.2-00:04.8] Speaker 1: Shall we start?
[00:05.0-00:06.5] Speaker 2: Yes.
```

`{start}` and `{end}` are `MM:SS.t` (`H:MM:SS.t` from an hour up). `{speaker}` is read from `segments[].speaker`, a diarization extension some services add and the OpenAI API does not; speakers are renumbered `Speaker 1`, `Speaker 2`... by first appearance within each audio part. A segment with no speaker, or a service that sends none, is rendered with `format_no_speaker`. A service that returns no segments yields its bare `text`; silence yields `[no speech detected]`. Several audio parts are transcribed in order, each becoming its own text part in place, so surrounding text and image parts are untouched.

**Language.** The request's language is the `X-RouteLLM-Transcribe-Language` header (a code such as `fr` or `pt-BR`), or `transcribe_language=` on `Controller.acompletion`; it wins over the config's `language`. With neither, the service detects it.

**`audio_url`** parts are refused with a 400 unless `allow_urls: true` **and** `allowed_url_hosts` lists the exact hostnames they may name (compared lower-cased, without a trailing dot; an empty list refuses every URL; credentials in the URL are refused). The host is resolved once and every address must be public: loopback, private, link-local, multicast, reserved, unspecified and IPv4-mapped forms of those are refused, and the connection is pinned to the checked address, so a second DNS answer cannot move it. Redirects are not followed, and the download is capped at `max_audio_bytes` while streaming and bounded by `timeout_ms`. `allow_private_url_hosts: true` lifts the public-address check and is **dangerous**: any allowlisted name that points inward then lets a client make the server reach internal services. Leave it off unless the allowlisted hosts are yours.

**Fails closed.** A transcriber that is unreachable, errors, answers unreadably, times out (`timeout_ms`) or is sent audio over `max_audio_bytes` ends the request, before any model is called, with a JSON error carrying the request id and no upstream detail:

| Cause | Status |
|---|---|
| unreachable, error, unreadable answer | 502 |
| timeout | 504 |
| audio over `max_audio_bytes` | 413 |
| malformed part, bad language, `audio_url` refused | 400 |

The raw audio is never forwarded to a model instead, and no other transcriber is tried: only the one configured for the addressed tier is ever called, and there is no retry. What the transcriber said goes to the server log under the request id.

**Private tiers.** A tier is *local-only* when every endpoint it can reach carries the `local` tag (the label `tag:local` selectors already use). Such a tier whose transcriber is not tagged `local` fails at load, so its audio cannot leave for a cloud transcriber; tag the transcriber `local`, point the tier at one, or set `mode: passthrough`. A tier whose sides are selectors is judged once they resolve. The tag is a declaration, like an endpoint's own.

**Logging and caching.** Audio and transcript text are never logged at INFO. One INFO line gives counts and latency: `transcribed audio: parts=1 seconds=6.5 segments=2 speakers=2 latency_ms=120 request_id=...`. Transcription runs before routing, so the router, the capability check and the cache key all see the transcript: two clips that transcribe differently never share a cache entry, and a cache hit still costs one transcription. (In `passthrough`, the raw audio is part of the key by value.) A stream request is transcribed first and then streams as usual.

#### Video: preprocess-then-route

Video is turned into frames and a transcript by a separate local service, then routed as an ordinary vision request. RouteLLM is only the client of that service (HTTP contract: `POST <api_base>/v1/video/preprocess`, multipart `file` plus option fields, JSON answer with `scenes[].frames[].image`, `transcript.segments[]`, optional `ocr_text`).

Accepted part shapes; anything else that claims to be video is a 400:

```json
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
{"type": "file", "file": {"file_data": "data:video/mp4;base64,...", "filename": "clip.mp4"}}
```

A remote `http(s)` URL is always refused (400): this server does not fetch client-named URLs. `allow_urls: true` only changes the message; fetching is not implemented. The client's `filename` is never sent to the service.

```yaml
video:                              # global default; a tier overrides key by key under its own `video:`
  mode: preprocess                  # preprocess | passthrough | refuse (default preprocess)
  api_base: http://127.0.0.1:9100   # required for preprocess; the service root, no /v1
  api_key_env: VIDEO_PREPROCESSOR_KEY   # optional; the variable's NAME, read per call
  timeout_ms: 600000
  max_video_bytes: 209715200        # per video, decoded; larger is a 413 before any call
  allow_urls: false
  tags: [local]                     # `local` needs a loopback/private api_base (or allow_public_base: true)
  max_frames: 12                    # 1..64, sent to the service
  layout: frames                    # frames | contact_sheet
  max_side: 768                     # 256..2048 px
  transcribe: true
  language: null                    # per request: X-RouteLLM-Transcribe-Language header
  ocr: false
  scene_detection: true
  require_transcript: false         # true: a failed speech-to-text is a 502, not a warning
  max_transcript_chars: 20000       # bound on the transcript and on the on-screen text
```

Unknown keys are a load error. **Modes.** `preprocess` calls the service. `passthrough` forwards the part unchanged, only to a leaf whose `capabilities.modalities_in` lists `video` (unknown or absent is a refusal, like audio); any other leaf is dropped by the usual capability filtering. `refuse` answers 400. A request with a video part and no `video:` section is a 400 naming the section; video is never silently dropped.

**Replacement.** Each video part is replaced in place, other parts keep their order: a text part `Video <n>: <duration>s, <k> scenes`; per scene a text label `[scene i, start-end s]` followed by its image parts (with `layout: contact_sheet`: one label text and one grid image); then a `Transcript:` text with `[mm:ss SPEAKER] text` lines (cut at `max_transcript_chars` with `[transcript truncated]`; `[no speech detected]`, `[no audio track]` or `[not available]` when there is nothing to show); then `On-screen text:` per scene when `ocr` found any. The request then needs a vision-capable leaf, and the inserted images count in the context estimate like any image part. Only `data:image/` frames are accepted from the service; anything else is a 502.

**Errors.** Each carries a `code` and the request id; what the service said stays in the server log. Service unreachable or 5xx: 502 (`video_unavailable`, or the service's own code). Timeout (`timeout_ms`, or a service 504): 504 `video_timeout`. Service 4xx: 400 with the service's error code (`ocr_unavailable`, `unsupported_media`, ...) and none of its message text; a 401/403 from the service is this server's misconfiguration and is a 502. Over `max_video_bytes`: 413. There is no retry and no fallback: the raw video is never forwarded instead.

**Privacy.** The video is sent to the service before the request's class is known, so it cannot be redacted first. A tier that is local-only, or that intent routing may send to a local-only tier, needs a preprocessor tagged `local`. Under a policy, when any class is `redact` or `block` every active preprocessor must be tagged `local` (startup `PolicyError`), and a request declaring such a class with a non-local preprocessor is refused (403, `video:<class>:not_local`) before upload. The transcript and OCR text then go through the class pipeline like any user text, so a `redact` class redacts names in them. Frames are images and cannot be redacted: with the default `redact.non_text: refuse` a video that yields frames is refused for a `redact` class (500, nothing sent); `non_text: pass` forwards them.

**Logging, audit, caching.** Video, frames and transcript are never logged at INFO; one line gives counts and latency (`preprocessed video: parts=1 seconds=93.4 scenes=2 frames=3 ...`). The audit event gets `routing.video` with `parts`, `scenes` and `frames` counts only, and the prompt digest covers text parts of the original request, never video bytes. The cache key follows the inserted frames and text, so identical frames hit the cache but still cost one preprocessing call.

#### Three outcomes

Each level of the tier walk checks both its sides before running its router:

- **Both sides can serve it** — unchanged, the router decides as always.
- **Exactly one can** — it is taken with **no router call**, and the path entry carries `capability_forced: "strong"` or `"weak"` and `capability_requirement: "<name>"`, the requirement the *other* side failed. `win_rate` is `null`; every other path key keeps its usual value.
- **Neither can** — a `RoutingError`, which the server returns as a 400 with a JSON body, before any upstream request is made:

```json
{"object": "error",
 "message": "Tier 'coding' has no side that can serve this request: vision. Configured sides: cloud_strong (vision), local_fast (vision)."}
```

A tier-valued side passes when **any** reachable leaf passes; that union is computed once at startup, bottom-up, and cached. Flat pairs get the same check — the legacy pair, a traffic rule's pair, a middleware's pair. A fallback sibling that cannot serve the request is dropped from the chain, because falling back to a model that cannot take the request turns one provider error into two.

A request carrying no images, no audio, no tools and no `response_format` demands nothing, so every side passes and the path is byte-identical to what it was before capabilities existed.

#### Seeing the picture

```
python -m routellm.pairing --config config.yaml --capabilities
```

prints one row per endpoint over `vision`, `audio`, `tools`, `structured_output`, `reasoning`, `context`, `max_output`, rendered `yes` / `no` / `?`; then one row per tier with its union and a `via` column naming the leaf that supplied each `yes`; then an `Unknown:` block listing the keys each endpoint cannot answer; and finally the capability terms your own selectors use that some endpoint is unknown on — those are the endpoints a selector drops without saying so.

Add `--explain` to print the selector tables too, selectors first. There is deliberately no `python -m routellm.capabilities`: this command already loads the YAML, builds the registry and resolves the selectors, which is everything the matrix needs, and a second entry point would duplicate all of it and drift from it.

#### Measured quality (optional)

An endpoint's `quality:` is a number written by hand, and it orders `quality_desc` selectors. It can be *measured* instead. A config points at a sidecar:

```yaml
endpoints:
  cloud_strong: {model: gpt-4o}
  local_server: {model: openai/qwen3-coder, api_base: http://127.0.0.1:11800/v1}

quality_from: quality.yaml
```

```yaml
# quality.yaml
version: 1
generated_at: "2026-09-21T10:00:00Z"
min_samples: 50
transform: linear
source: evals
endpoints:
  cloud_strong: {quality: 91, n: 50, by_area: {}}
  local_server: {quality: 78, n: 50, by_area: {}}
```

`quality_from:` is resolved **relative to the config file**, not to the working directory, so a server started from elsewhere reads the sidecar the operator wrote next to the config. An endpoint's own `quality:` always wins over the sidecar — a hand-set number is a decision, and a measurement never overrides a decision silently. A sidecar naming an endpoint the registry does not carry is a warning naming it, not an error: endpoints come and go faster than measurements do. A missing sidecar file is an error naming the resolved path.

`--capabilities` marks each score `(measured)` or `(manual)` so the distinction is visible where the numbers are read.

The harness that writes one:

```
python -m routellm.evals.endpoint_quality --config c.yaml \
    --endpoints cloud_strong,local_server --limit 50 --out quality.yaml
```

It runs the coding benchmark against each named endpoint with `strong_model` and `weak_model` both set to that one endpoint, so no router runs and every call lands where it is meant to. **This spends real tokens on real providers** — `--limit` defaults to 50 prompts per endpoint, so four endpoints is 200 paid completions. Start small.

This whole section is optional and severable: drop it and hand-set `quality:` keeps working exactly as it does today.
### Measured quality

`quality:` on an endpoint is a number someone typed once, and every `quality_desc` selector orders on it. Point the config at a sidecar to order on a measurement instead:

```yaml
quality_from: ./quality.yaml   # the sidecar wins; set quality_from_override: false to flip it
areas:
  coding: [coding, coding_quality, coding_fast]   # per-area numbers, used inside those tiers
```

Traces keep ids, scores, token usage, and a SHA-256 and length of the prompt and of the output; no prompt or response text unless `capture_content: true` (logs a startup WARNING). With a [policy](#policy-file) loaded, `capture_content: true` stores text only for a request whose effective class (the decision's, else the declared one) has treatment `raw`; a `redact`, `block` or `vault` class, or one the policy does not list, keeps digests, and the trace records `metadata.content_captured: false` with `metadata.content_withheld_reason` (`class_treatment:<treatment>` or `class_unknown`). Shadow and comparison traces follow the class of the request they ran for. Scoring reads text, so it needs a capturing store and refuses digest-only traces naming the trace. Trace directory is `0700`, files `0600`. Record traces, score them offline against a rubric, aggregate, and restart: `python -m routellm.quality_scores score --traces .routellm_traces --scorer rubric:./rubric.yaml` then `aggregate --scores scores.jsonl --out quality.yaml`. Scoring needs the `fit` extra; aggregation needs nothing. An endpoint the sidecar does not name keeps the number it has, and each override logs one line naming both. The whole loop is in [docs/runbooks/traces-to-quality.md](docs/runbooks/traces-to-quality.md).

### Comparison routing

Send one request to two or more models at once and record how each did: time to first token, total latency, tokens per second, prompt and completion tokens, cost, error and output. Off by default; nothing below does anything until `comparison.enabled` is `true`.

```yaml
comparison:
  enabled: true
  budget_usd: 5.00            # stop all comparison calls after this much spend
  max_concurrency: 4          # shadow calls in flight at once
  profiles:
    frontier_vs_local: [cloud_strong, local_glm]
  sampling:                   # shadow 5% of requests that enter tier `default`
    default: {rate: 0.05, models: [local_glm]}
  experiments:
    bench:
      models: frontier_vs_local
      rate: 0.1
      budget_usd: 1.00
      max_concurrency: 2
  judge:
    scorer: "rubric:rubric.yaml"

traces:                       # the trace store; off unless enabled
  enabled: true
  trace_dir: .routellm_traces # relative to this file
  max_files: 10000
  max_bytes: 536870912
  capture_content: false      # true stores prompt and response text in plaintext (raw classes only under a policy)
```

`config.example.yaml` carries the same section, commented, with every key.

**Shadow mode.** The normal `/v1/chat/completions` request is answered by its primary, unchanged, streamed or not. The shadow models get the same request concurrently, as background tasks, non-streaming. Their results go to the trace store (turn it on with the `traces:` section below) and nowhere else: a shadow never delays or alters the response, never reads or writes the response cache, and its failures are logged and recorded, never shown to the client. A cache hit on the primary runs no shadow. Primary and shadow traces share a `routellm.comparison.id`, with `role` `primary` or `shadow`.

What triggers a shadow, in priority order, at most one per request:

1. The `X-RouteLLM-Compare` request header: an experiment name, a profile name, or a comma-separated list of endpoint names (`X-RouteLLM-Compare: cloud_strong,local_glm`). Names that are not configured endpoints are ignored. `allow_header: false` turns the header off.
2. A named experiment, sampled at its `rate`. Naming the experiment in the header ignores the rate. `tiers:` limits it to requests that entered those tiers.
3. The `sampling` rule of the tier the request entered.

A shadow is never sent to the endpoint the primary was routed to.

**`POST /v1/compare`.** On whenever comparison is enabled (`endpoint: false` turns it off; see Privacy for who may name which models). The body is a chat request plus `models` (a list of endpoint names, or a profile or experiment name, at least two), and optionally `tier` and `measure_ttft`. Every model answers and the results come back side by side; one model failing never fails the request.

```json
{"object": "comparison", "id": "cmp-...", "results": [
  {"endpoint": "cloud_strong", "model": "gpt-4o", "output": "...", "finish_reason": "stop",
   "error": null, "score": null,
   "metrics": {"ttft_ms": null, "latency_ms": 812, "tokens_per_second": 41.2,
               "prompt_tokens": 12, "completion_tokens": 33, "cost_usd": 0.00041}}]}
```

`error` is a short code (`upstream_error`, `timeout`, `budget_exhausted`, `unsupported: <requirement>`); the upstream's own text stays in the server log. Streaming a compare is not supported. `measure_ttft: true` (or `comparison.measure_ttft`) makes each call stream upstream so `ttft_ms` is measured; the response is still one JSON body. Without it `ttft_ms` is null. `tokens_per_second` is completion tokens over total latency. `cost_usd` is the token usage priced from the endpoint's models.dev record, null when it has none.

**Limits.** `budget_usd` (global and per experiment) is USD, in memory, per process; a restart clears it. A call is charged when it finishes, so calls in flight can overshoot a cap by their own cost, and an unpriced endpoint costs nothing against it. `max_concurrency` (global and per experiment) counts calls in flight; a shadow that does not fit is dropped, never queued. `max_models` bounds one plan, `timeout_ms` one call. Comparison calls are not retried and carry the hop header like every other outbound call.

**Privacy.** A tier is local-only when every endpoint it can reach is tagged `local`. Such a tier may only be shadowed to `local`-tagged endpoints: `sampling` or `experiments` that break this are refused when the config loads, and a header-named model or an untiered experiment is filtered at request time. `POST /v1/compare` is explicit multi-model fan-out, so it is closed by default: a request must either name a `tier` (that tier's local-only rule then applies) or name only models in `comparison.compare_allowed_models` or in a profile or experiment, otherwise 400. `allow_untiered_compare: true` lifts this and lets any caller send any prompt to any configured endpoint, paid and cloud ones included; it is dangerous, leave it off on anything reachable by untrusted clients. `pay: true` endpoints stay refused either way. The replay CLI is an operator tool and is not restricted. Endpoints with `pay: true` can never be comparison targets, so a shadow cannot spend the wallet.

**Judge.** `judge.scorer` is any scorer spec of `python -m routellm.quality_scores` (`rubric:`, `module:`, `composite:`, `judge:<model>`), run on each recorded output and stored as `reward.score` and `routellm.comparison.score`. Only `judge:<model>` calls an LLM, and it counts as local only when `<model>` is an endpoint tagged `local`. A non-local judge is refused at load when a sampled or experiment tier is local-only, and is never run on data from a local-only tier that reached comparison some other way. A failed comparison call is recorded with its `error` and skipped by `quality_scores score`.

**Offline replay.** Run a jsonl dataset of chat requests (`{"messages": [...]}` or `{"prompt": "..."}` per line) through N models, recorded the same way:

```
python -m routellm.compare replay --dataset chats.jsonl \
    --models cloud_strong,local_glm --config config.yaml \
    --trace-dir .routellm_traces --out results.jsonl [--judge SPEC] [--tier T] [--limit N] [--capture-content]
```

Each row gets one comparison id, role `candidate`, and a per-model summary is printed. `comparison.enabled` is not required for replay; the config's `comparison:` section still supplies the timeout, budget and judge. Replay traces hold digests only unless `--capture-content` is passed. Feed the traces to `routellm.quality_scores` as in [Traces to quality](docs/runbooks/traces-to-quality.md).

Shadow mode runs on the async request path the server uses; the synchronous `Controller.completion` is untouched.

### Intents

There are two ways to pick the area of expertise a request is answered from. An app that already knows names the tier itself, in the `model` field. An app that does not sends `default` and lets an intent classifier pick, which is what an `intents:` section turns on:

```yaml
tiers:
  default: {router: mf, threshold: 0.12, strong: premium, weak: local_fast}
  legal:   {strong: cloud_strong, weak: frontier_local}
  coding:  {strong: cloud_strong, weak: local_fast}

intents:
  detector: jev          # or litellm
  model: jev-1           # litellm: a registry endpoint name; jev: the Jev model id
  upstream: jev_host     # jev only: the registry endpoint it calls
  confidence_floor: 0.6  # jev only: below it, no intent is claimed
  descriptions:
    legal: contracts, statutes, and case law
    code: writing or debugging software
  tiers:
    legal: legal
    code: coding
```

Each intent maps to a tier, and `tiers:` must be present and non-empty — an `intents:` section that maps nothing would classify every prompt and act on none. A classified prompt enters its tier; an intent mapped to none, or a prompt the classifier declines to label, leaves the request in the tier it addressed. Every mapped tier must exist, which is checked at startup.

The targets above need no `intent_routing` of their own: a request arriving as `default` is the case the gate below allows, and `legal` and `coding` are where it is sent, not where it came from. A tier needs `intent_routing: true` only when apps address *it* directly and you still want the classifier to move them on.

A classifier never overrules an app that chose a tier by name: an intent-selected tier is honoured only for a request addressed to `default`, or to a tier that opts in with `intent_routing: true`. A tier that is passed over this way says so on the decision path, as `intent_ignored`, and a tier entered by intent carries `tier_from: intent`. `default` can opt out with `intent_routing: false`.

Intent detection never reaches a provider outside the registry, and there is no default model. With `detector: litellm`, `model` must name a registry endpoint (not a tier, not a raw model name: that is a startup error), and the call uses that endpoint's own `api_base` and credential. Leave `model` unset and detection is disabled, with one INFO line at startup. With `detector: jev`, `upstream` must name a registry endpoint whose `api_base` is the Jev host, so the call carries that endpoint's reach and jurisdiction tags like any other. A router that calls a hosted service, such as `jev`, is configured the same way (`jev: {upstream: jev_host}`) and refuses to load without it. The endpoint a detector calls is exposed (`egress_endpoint`) and an [egress gate](#egress-gates) vetoes it: a gated detector endpoint gets no call for a request whose class the gate denies.

`detector: jev` classifies with one TypeSafe Jev question and needs the extension: `pip install -e extensions/typesafe`. `detector: litellm` asks the `model` to classify in a prompt built from the `descriptions`, and needs nothing extra. Both read the same `descriptions`, so the wording that tells the classifier what an intent means lives in one place. `python -m routellm.pairing --config` prints one `intent <label> -> tier <name>` line per mapping.

### Decision step

A `decision:` section labels every routed request with a data class, an intent and optionally a topic area, using a model that is itself a registry endpoint. Nothing is routed on the labels here; the result is the request's `Decision`, recorded on the routing path (`path[0]["decision"]`) and readable during routing through `routellm.decision.current_decision()`, for policy routing and egress gates to consume. Without the section nothing runs and no call is made.

```yaml
decision:
  provider: llm              # built in; others register under the `routellm.decision_providers` entry-point group
  endpoint: decider          # a registry endpoint (not a tier)
  require_tags: [local]      # load error naming endpoint and tag when it lacks one
  data_classes:              # ordered, least to most restrictive; the last is the fail-closed class
    general: nothing sensitive
    private: personal or secret data
  declarable: []             # optional: data_classes only a caller may declare (see below)
  intents:
    code: writing or debugging software
    chat: small talk
  areas: {}                  # optional topic areas, same shape
  input: {messages: last_user, max_chars: 4000, keep: tail}   # last_user | user | all
  timeout_ms: 2000
  min_confidence: 0.6
  on_error: most_restrictive # or refuse (the request fails with a 400)
  recognizers:               # ordered, deterministic, run before the model over the whole conversation
    - {name: key_material, type: regex, class: private, patterns: ['-----BEGIN [A-Z ]*PRIVATE KEY-----']}
    - {name: markers, type: markers, class: private, markers: ["internal use only"]}
  sticky: {enabled: true, ttl_s: 3600}
  eval_min_recall: {private: 0.95}
```

Effective class is the most restrictive of the caller's `declared_class`, the sticky floor, a recognizer hit and the model's answer; no source lowers it. A recognizer hit, or a declared or sticky class already at the strictest level, answers without a model call. The model is asked for a typed answer (a JSON schema whose fields are enums of the configured labels) and any label outside the sets is an error. A timeout, a transport failure, an invalid answer or a confidence under `min_confidence` resolves through `on_error` and is recorded as `error` on the Decision; failure-derived classes are not made sticky. The decision call goes through the endpoint's own resilience and timeout like any leaf, and is awaited, so a slow decision model holds up only its own request.

`declarable` names classes (a subset of `data_classes`, which stays the one ordered list) that only a caller can give, such as consent to training use that no model should infer. The model is never offered them (not in the schema enum or the prompt) and an answer naming one is an `invalid_answer`, resolved through `on_error`. A request may declare one: it is accepted, the model still runs, and it can raise the class to a stricter selectable one but never lower it; the Decision's source is `declared`. An undeclared request never resolves to one, so startup fails when a recognizer, the policy file's `default_class` or a routing policy's `data_class` floor names one, or when the last class (the fail-closed one) is declarable. A declarable class is not made sticky: a later turn that does not declare it is classified afresh.

Sticky keeps a conversation's class from dropping below the strictest seen for `ttl_s`; the key is the `conversation_id` the caller passes, else a hash of the opening message and `user`. A heavier recognizer plugs in as any object with `recognize(text, language=None)`; `routellm.decision.SpanRecognizer` adapts an analyzer exposing `analyze(text, language)` that returns spans.

Measure it on your own prompts with labelled JSONL (`{"text": ..., "data_class": ...}`):

```
python -m routellm.decision.eval --config config.yaml --data labelled.jsonl --min-recall private=0.95
```

It prints recall and precision per class, a confusion table and latency percentiles, and exits 1 when a named class's recall is under its floor (`--model-only` skips the recognizers).

### Egress gates

An `egress_gates:` list holds back a tagged group of endpoints. A gate is keyed on an endpoint tag expression, so it names no provider: the deployment picks the tag.

```yaml
egress_gates:
  - name: gated
    select: "tag:gated"              # tag terms only: tag:x  !tag:x  tag:a|b
    deny_classes: [private]          # classes that never reach a selected endpoint
    require_policy_allow: false      # see below
    models:
      allow:                         # globs on the endpoint's model id
        - "vendor/*"
        - model: "vendor/free-*"     # an entry may require request params
          require: {extra_body.provider.data_collection: deny}
      deny: ["*:legacy"]             # deny wins over allow
    also_check: [extra_body.models]  # dotted paths in `extra` holding more model ids
```

**At load**, every endpoint `select` matches is checked: its own model id, and every id found under `also_check` (a provider-side fallback list the router cannot see at call time), must match `models.allow` and no `models.deny` entry, and each `require` of every matching allow entry must equal the value at that path in the endpoint's `extra`. A violation stops startup naming gate, endpoint and id. Omitting `models.allow` leaves ids unlimited (only `deny` applies); an empty list allows nothing. A `select` matching no endpoint, a `deny_classes` entry the policy does not list, and `deny_classes` or `require_policy_allow` without a `policy:` are load errors too, and so is a `decision.endpoint` that a gate with those rules selects: the decision step reads a request before its class exists.

**At request time** the gate is one more reason at the same point as the class check (`Controller._policy_reason`, behind `_can_serve`), so it holds every tier side, forced side, sibling, candidate-pool member, load-balancer target, shadow, `POST /v1/compare` target and the canary. A selected endpoint whose gate refuses the request's class is not a candidate: it is never called and never replaced by another class. The request goes to the tier's other side, or when none is left fails `403` with `gate:<name>:class:<class>`. A tier is judged by every leaf under it. The reason of a dropped sibling or pool member is listed on the path as `egress_gate_denied`; a gated endpoint that served is listed as `egress_gate: {passed: [<name>]}`.

**Order.** The class is final before any gated call: the gate reads the class the controller put on the request after the decision step (the strictest of the declared class, recognizers and the model), so a decision that raises the class, or fails into `on_error`, is what the gate sees. A request with no class at all (the operator's replay) counts as the most restrictive for `deny_classes` and is refused a gated endpoint (`gate:<name>:class:undecided`); `models` rules apply as at load only.

**Auxiliary calls.** Some calls read a prompt before any leaf does. Each is held to the request's effective class by the same check (`Controller._aux_denial`: the class's reach and `jur`, then any egress gate) on the endpoint it sends content to; a call whose endpoint fails is not made, so the remote takes zero hits and an allowed leaf still answers.

| Step | Endpoint judged | When refused |
|------|-----------------|--------------|
| Router that embeds (`mf`, `sw_ranking`) | the embedding endpoint | router not run; the tier's weak side is taken; the path entry carries `gate: gate:router` and `gate_reason` |
| Hosted router (`requires_upstream`) | its `upstream` endpoint | same |
| Intent detector | `intents.model` / the detector's `upstream` | not consulted; the request stays in the tier it addressed |
| Semantic cache | the embedding endpoint | no lookup and no store (no embedding computed); a class whose treatment is not `raw` never takes part. The exact cache is unaffected |
| Comparison judge | the `judge:<endpoint>` | shadow output recorded unscored |
| Decision classifier | its `decision.endpoint` | it runs before the class is known, so startup fails unless every non-`vault` class may reach it |

Video preprocessing does too: under a `redact` or `block` class every active `video:` preprocessor must be tagged `local`. Audio transcription runs before the class is known too; a tier whose endpoints are all `local` already requires a `local` transcriber at load. An embedding call with no registry endpoint carries no tags, so `--allow-remote-embeddings` is refused at startup under a policy; give the server an embedding endpoint instead.

**`require_policy_allow`** makes the gate depend on the request's [routing policy](#policy-routing): a leaf the gate selects is a candidate only when the selected policy's `allow_tags` lists one of the leaf's tags and its `deny_tags` list none; otherwise the reason is `gate:<name>:policy`. Without `policy_routing:` there is no selected policy, so such a gate refuses every request to its endpoints. The intent detector is not a request with a policy: a gated detector endpoint of such a gate gets no call.

### Policy routing

A `policy_routing:` section routes a request by its attributes instead of a threshold at the entry. Everything in it is config: the header each attribute is read from, the attribute names, the class names, the intent labels and the tier names. It needs a [`policy:`](#policy-file) file (the class order and the class checks come from there).

```yaml
policy_routing:
  attributes:                       # attribute -> header it is read from
    project: {header: X-Route-Project}   # header defaults to X-Route-<Name>
    client: {pattern: '^[a-z]{1,16}$'}   # pattern defaults to ^[a-z0-9][a-z0-9._-]{0,62}$
  key: project                      # the attribute whose value names the policy
  entry: [auto]                     # tiers a client addresses; the rules pick the real tier
  rules:                            # first match wins; every matcher in `when` must hold
    - {when: {class: private}, tier: private}
    - {when: {client: codex}, tier: codex}
    - {when: {intent: hard}, tier: hard}
    - {when: {modality: image}, tier: vision}
  else: code                        # taken when no rule matches
  default_policy: default           # the policy for a missing, invalid or unknown key
  policies:
    default:
      data_class: general           # floor of the effective class
      tiers: [code, private, vision]  # tiers this policy may use
      fallback: code                # taken for a rule target outside `tiers`; none: 403
      allow_tags: []                # tags a gate with `require_policy_allow` admits
      deny_tags: []                 # an endpoint with one of these is never a candidate
    some-project:
      tiers: [code, hard, private]
      allow_tags: [metered]
```

**Attributes.** Each is read off the request headers, validated against its `pattern` and stored on `request.state.attributes` and `routellm.policy_routing.current_attributes`, next to the declared class. A missing, empty or malformed value is an unknown attribute, never an error. No header is copied upstream, so an attribute never reaches a provider, a shadow or comparison call, or the decision classifier.

**Policy.** The value of the `key` attribute names the policy. A missing, invalid or unknown value takes `default_policy` (a policy with that name is required); the path says which (`policy_from`: `attribute`, `absent`, `invalid`, `unknown`).

**Bound to the caller.** The header is a claim, not a right. A value naming a non-default policy is honoured only if the policy file lists it for the authenticated caller: `principals.<name>.projects: [policy, ...]` (a caller with no list may use `default_policy` only). A caller naming a policy it may not use is refused (`403`, naming the policy, `policy_from: forbidden`), never silently moved to the default. With auth off there is no caller, so only `default_policy` applies and a header naming any other policy is a `403`: policy routing with non-default policies requires `auth:`. This is what makes `require_policy_allow` safe: only a policy the caller is authorised for can admit a gated endpoint. A caller listing a policy that `policy_routing:` does not define is a startup error. A missing, invalid or unknown value is still the default policy.

**Effective class.** The most restrictive of the policy's `data_class`, the declared class and the decision step's class, where restrictiveness is the order of the policy file's `classes:` (least restrictive first, the order `decision.data_classes` already uses; startup fails if the two disagree). No source can lower it. Without a `decision:` section the intent is unknown and intent rules never match.

**Rules.** A request addressed to a tier in `entry` goes to the tier of the first matching rule, else to `else`. Matchers are `class` (the effective class), `intent`, `modality` (`text`, `image`, `audio`) and any configured attribute; a list is any-of, a rule with several matchers needs all of them. A request addressed to any other tier is not re-routed. The depth cap and every other tier rule are unchanged.

**Allowed tiers.** A rule target outside the policy's `tiers` goes to its `fallback`, or is refused with `403` naming policy and tier. A tier a client addressed directly that the policy does not allow is refused (`403`), never moved. The fallback is not trusted either: it passes the checks below.

**A policy never widens a class.** The tier the policy picks must still be an entry tier of the effective class (`403` otherwise, naming class and tier) and every candidate under it still passes the class's reach and jurisdiction, the policy's `deny_tags` and the egress gates (`_can_serve`). At startup a rule whose `class` matcher names a class that does not list the rule's tier, a class that none of a policy's selectable tiers can serve, and a selectable tier whose leaves exceed the class's reach are errors.

**No declared class.** With `default_class: null` a request that declares none takes the selected policy's `data_class` floor as its class, checked like any declared class (the caller may send it, it is not vaulted). A policy with no floor keeps the `400`. A declared class, or `default_class`, is never replaced: the effective class is still the strictest of every source.

**`/v1/compare` and auxiliary calls.** Both select the policy the way chat does (bound to the caller; a policy the caller may not use is `403`). `POST /v1/compare` is held to the effective class and to the policy's `deny_tags` and `require_policy_allow` gates on every target: one denied target refuses the request (`403`) with no model called. A shadow comparison already carries the request's policy. Calls that carry request content (hosted routers, the embedding endpoint, middleware detectors, the semantic cache, the judge) pass the same check with the selected policy, so a policy's `deny_tags` and allow set apply to their endpoints too. The decision classifier runs before the policy is selected: its endpoint is held to the startup check only, so keep it local or on tags no policy denies.

**Path.** `path[0]["policy_routing"]` records the attributes, the policy and where it came from, each class source (`policy`, `declared`, `decision`), the effective class, the intent, the addressed tier, the tier chosen, the matched rule (index, or `else`) and, when the fallback replaced a refused target, `refused: {<target>: <fallback>}`.

### Request transforms

A transform rewrites the messages of an upstream call and restores the reply. The server applies a `transforms:` list in order; each entry is a `module:attr` path or an entry point name in the `routellm.transforms` group, optionally with `options:` passed as keyword arguments. Absent: nothing runs.

```yaml
transforms:
  - use: mypkg.transforms:Tokenize
    options: {salt: abc}
```

```python
class Tokenize:                                   # routellm.transform.RequestTransform
    def forward(self, messages, ctx):             # -> (messages, state)
        ...                                       # state None: left this call raw, skip inverse
    def inverse(self, text, state): ...           # one complete piece of reply text
    def selects(self, ctx): ...                   # optional; request-level, ctx.endpoint is None
    def restorer(self, state): ...                # optional; per streamed field: feed(text), finish()
```

`ctx` (`TransformContext`) carries the candidate's endpoint, model and reach labels, the request's effective data class and that class's policy treatment, the request id, and `shadow`. `map_text(messages, fn)` copies the messages and applies `fn` to string content, multimodal text parts, tool calls' and legacy function calls' arguments. `HoldbackRestorer(inverse, holdback)` buffers only a trailing stretch that may still become a token, so a token split across stream chunks restores; the default restorer restores each piece alone, which suits tokens that cannot split. `IdentityTransform` is the no-op default.

Where it runs: at the one point every upstream call passes through, after the candidate is chosen and before the call. That is every attempt: primary, sibling fallback, pool member, retry (`completion` and `acompletion`), and the shadow and `/v1/compare` calls. A transform that raises, or returns no message list, refuses the request: no call, no fallback, a 500 `transform_failed` that names the transform, never the cause. Chain order is forward in list order, inverse in reverse.

The reply is restored after it has been cached or traced, never before:

```
forward -> upstream call -> trace (transformed prompt + reply) -> inverse -> client
```

Traces hold the transformed prompt and the reply as the upstream returned it; streamed traces are built from the chunks as sent, before restore. A request that any transform `selects` is neither served from the cache nor stored in it, since a cached reply cannot be restored without the request's own state. Streams restore per choice and per field (content, reasoning, each tool call's arguments); text a restorer holds back is released on a chunk placed before the `finish_reason`, or at the end of a stream that has none. A shadow call is transformed as the primary is, and its judge and trace see the transformed form; a `/v1/compare` candidate's `output` is restored for the caller. Not covered: the router, decision and judge calls, which make their own egress decisions.

### Audit events

`audit:` writes one event per request recording what was decided and why. An event never holds content: no prompt, no response, no placeholder mapping, no key. The prompt (the last message, as the routers read it) appears only as a keyed HMAC-SHA256, so two events can be tied to one prompt by whoever holds the key and by nobody else.

```yaml
audit:
  sink: file                         # the one built-in sink
  path: /var/log/routellm/audit.jsonl
  hmac_key_env: ROUTELLM_AUDIT_KEY   # or hmac_key_file: a 0600 file; never the key in config
```

With a [`policy:`](#policy-file) loaded, `audit:` is required (absent or `enabled: false` stops startup) and, like the policy, honoured from the system config layer only. Startup also fails when the key is missing or empty, when `hmac_key_file` is wider than 0600, or when the sink cannot be opened.

The file sink appends one JSON object per line with `O_APPEND`, to a 0600 file in a 0700 directory; a wider file or directory, or a symlink, is refused. The path is reopened on every write, so logrotate or newsyslog can rename it with no signal: the next event creates a fresh 0600 file. The server does no rotation of its own.

**Fields** (`v: 1`). Each event has all of them; unknown values are `null`.

| field | meaning |
|---|---|
| `ts`, `request_id` | UTC time; the id in the server log and error bodies |
| `phase` | `complete`, or for a stream `decision` then `outcome` |
| `route` | `chat` or `compare` |
| `principal`, `profile` | the authenticated caller |
| `data_class`, `operation` | as declared; `effective_class` is the class the request was held to |
| `policy`, `entry_tier` | the routing policy selected, the tier addressed |
| `endpoint`, `reach`, `jur` | the endpoint that answered, its reach labels and jurisdiction |
| `treatment` | egress gates the answering endpoint passed |
| `outcome`, `status`, `reason` | `served`, `refused` or `failed`; the HTTP status; a short code (`unauthorized`, `class_refused`, `policy_denied`, `routing_error`, `invalid_request`, `loop_detected`, `endpoints_cooling`, `provider_error`, `stream_error`, ...) |
| `cache` | `hit`, `miss` or `bypass` (streams, uncached classes) |
| `tokens_in`, `tokens_out`, `latency_ms` | usage and time to the event |
| `prompt_hmac` | keyed digest of the prompt |
| `routing` | the decision step's class, source and intent; the routing policy record; the fallback walk; compare targets |

Every request that reaches a route writes an event: served, refused (401, 400, 403, 422, 508) and failed (500, 502, 503), including the refusals made before the handler runs. A non-stream event is written before the response is returned. A stream writes its `decision` event once its route is final and before the first byte, and its `outcome` event as it ends (`failed` with `stream_error` or `client_disconnect` when it does not finish).

**When a write fails.** With a `policy:` loaded the request is refused with a 503 and its answer withheld, since an unaudited decision is not allowed; a stream whose decision event cannot be written is closed before a byte is sent, while a stream's outcome event, written after the response has started, can only be logged. With no policy the failure is logged, counted and the request is served.

A sink is any object with `emit(event)` (raising on failure) and `close()`; `routellm.audit.AuditSink` is the protocol.

### Threshold Calibration

The threshold used for routing controls the cost-quality tradeoff. The range of meaningful thresholds varies depending on the type of router and the queries you receive. Therefore, we recommend calibrating thresholds using a sample of your incoming queries, as well as the % of queries you'd like to route to the stronger model.

By default, we support calibrating thresholds based on the public [Chatbot Arena dataset](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k). For example, to calibrate the threshold for the `mf` router such that 50% of calls are routed to the stronger model:

```
> python -m routellm.calibrate_threshold --task calibrate --routers mf --strong-model-pct 0.5 --config config.example.yaml
For 50.0% strong model calls for mf, threshold = 0.11593
```

This means that the threshold should be set to 0.1881 for the `mf` router so that approximately 50% of calls are routed to the strong model i.e. using a `model` field of `router-mf-0.1159`.

However, note that because we calibrate the thresholds based on an existing dataset, the % of calls routed to each model will differ based on the actual queries received. Therefore, we recommend calibrating on a dataset that closely resembles the types of queries you receive.

## Evaluation

RouteLLM also includes an evaluation framework to measure the performance of different routing strategies on benchmarks.

To evaluate a router on a benchmark, you can use the following command:
```
python -m routellm.evals.evaluate --routers random sw_ranking bert --benchmark gsm8k --config config.example.yaml 
```

- `--routers` specifies the list of routers to evaluate, for instance, `random` and `bert` in this case.
- `--benchmark` specifies the specific benchmark to evaluate the routers on. We currently support: `mmlu`, `gsm8k`, and `mt-bench`.

Evaluation results will be printed to the console. A plot of router performance will also be generated in the current directory (override the path using `--output`). To avoid recomputing results, the results for a router on a given benchmark is cached by default. This behavior can be overridden by using the `--overwrite-cache` flag, which takes in a list of routers to overwrite the cache for.

The results for all our benchmarks have been cached. For MT Bench, we use the precomputed judgements for the desired model pair. For MMLU and GSM8K, we utilized [SGLang](https://github.com/sgl-project/sglang) to compute the results for the desired model pair - the full code for this can be found in the benchmark directories if you would like to evaluate a different model pair.

By default, GPT-4 and Mixtral are used as the model pair for evaluation. To modify the model pair used, set them using the `--strong-model` and `--weak-model` flags.

## Routers

Out of the box, RouteLLM supports 4 routers trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair.

The full list of routers:
1. `mf`: Uses a matrix factorization model trained on the preference data (recommended).
2. `sw_ranking`: Uses a weighted Elo calculation for routing, where each vote is weighted according to how similar it is to the user's prompt.
3. `bert`: Uses a BERT classifier trained on the preference data.
4. `causal_llm`: Uses a LLM-based classifier tuned on the preference data.
5. `random`: Randomly routes to either model.
6. `jev`: Uses TypeSafe's Jev System One model (hosted API, no local weights), shipped as a separate extension; install with `pip install -e extensions/typesafe` and see [extensions/typesafe/README.md](extensions/typesafe/README.md).

While these routers have been trained on the `gpt-4-1106-preview` and `mixtral-8x7b-instruct-v0.1` model pair, we have found that these routers generalize well to other strong and weak model pairs as well. Therefore, you can replace the model pair used for routing without having to retrain these models!

We also provide detailed instructions on how to train the LLM-based classifier in the following [notebook](https://github.com/anyscale/llm-router/blob/main/README.ipynb).

For the full details, refer to our [paper](https://arxiv.org/abs/2406.18665).

## Intent-Based Routing

RouteLLM now supports intent-based routing, which allows you to route queries to specialized models based on detected intents. This is particularly useful when you have domain-specific models that excel at particular types of tasks.

### Quickstart with Intent-Based Routing

```python
from routellm.controller import Controller
from routellm.middleware.intent_model_selector import IntentModelSelector

# Create an intent-based model selector
intent_selector = IntentModelSelector()

# Add mappings for different intents
intent_selector.add_mapping(
    intent="coding",
    strong_model="gpt-4-1106-preview",
    weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"
)
intent_selector.add_mapping(
    intent="math",
    strong_model="gpt-4-1106-preview",
    weak_model="claude-3-opus-20240229"
)

# Initialize controller with the intent selector as middleware
client = Controller(
    routers=["mf"],
    strong_model="gpt-4-1106-preview",
    weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
    middleware=[intent_selector]
)

# Now when you make requests, they'll be routed based on detected intent
response = client.chat.completions.create(
    model="router-mf-0.11593",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence"}
    ]
)
```

### Web UI for Intent-Based Routing

You can also use our web UI to configure and test intent-based routing:

```
python -m routellm.examples.intent_web_ui
```

This will launch a Gradio interface where you can:
- Define intents and add example prompts
- Configure model mappings for each intent
- Test routing with different prompts
- Save and load configurations

### Domain-Specific Intent Detection

For more advanced use cases, you can use our `DomainIntentDetector` to fine-tune intent detection with domain-specific examples:

```python
from routellm.middleware.domain_intent_detector import DomainIntentDetector
from routellm.middleware.intent_model_selector import IntentModelMapping, IntentModelSelector
from routellm.types import ModelPair

intent_mappings = [
    IntentModelMapping(
        intent="coding",
        model_pair=ModelPair(strong="gpt-4-1106-preview", weak="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"),
        description="Coding questions",
    ),
]

# Create a detector with the same mappings used for routing
detector = DomainIntentDetector(intent_mappings=intent_mappings)

# Add examples for each intent
detector.add_examples("coding", [
    "Write a Python function to sort a list",
    "Debug this JavaScript code",
    "How do I implement a binary search tree?"
])

# Use in your intent selector
default_pair = ModelPair(strong="gpt-4-1106-preview", weak="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1")
intent_selector = IntentModelSelector(intent_mappings, default_pair, intent_detector=detector)
```

TypeSafe's Jev-backed detector, `JevIntentDetector`, fits this same
`intent_detector` slot; see the `jev` entry in [Routers](#routers)
and [extensions/typesafe/README.md](extensions/typesafe/README.md).

## Configuration

The server does not take one config file; it discovers a chain of them and merges them, so a global config, a project-local override, and a one-off file all compose. Lowest precedence first, each layer merged onto the ones below it:

| # | Layer | Location | If the file is missing |
|---|---|---|---|
| 1 | defaults | built in, no file | — |
| 2 | system | `/etc/routellm/config.yaml` | skipped |
| 3 | user | `$XDG_CONFIG_HOME/routellm/config.yaml` (`~/.config` when unset) | skipped |
| 4 | project | `./.routellm.yaml` or `./routellm.yaml`, in the CWD or the nearest ancestor below `$HOME` | skipped |
| 5 | env | `ROUTELLM_CONFIG=<path>` | **error** |
| 6 | flag | `--config <path>` | **error** |

Layers 2-4 are discovered, so their absence is silence. Layers 5 and 6 were named by you, so a file that is not there is an error naming the variable or the flag. `Controller` still takes a `config` argument directly for library use.

### Skipping the user layer

An unrelated per-user file can leak endpoints into a pinned deployment. To leave layer 3 out, pass `--no-user-config` (server and `python -m routellm.config`) or set `ROUTELLM_NO_USER_CONFIG=1` (`1`, `true`, `yes`, `on`; anything else keeps the layer). The flag wins over the variable. Only the user layer is dropped: system, project, `ROUTELLM_CONFIG` and `--config` still apply, and the skip does not depend on whether they are set.

`paths` and `show` list the user path as `[skipped]` (`"skipped": true` in JSON), and `show` omits its values. At startup the server logs at INFO which layers loaded and which were skipped, paths only:

```
config layers loaded: project=/srv/app/.routellm.yaml, flag=/etc/app.yaml; skipped: user=/home/u/.config/routellm/config.yaml
```

### Merge rules

- **Mappings merge by key, recursively.** `endpoints:`, `tiers:`, `intents:` and each router's section merge across layers, so a user file can define ten endpoints and a project file add one or retag another without restating the rest.
- **Scalars and lists replace.** A project's `tags:` list replaces the user's for that endpoint; it never concatenates.
- **`null` deletes.** A key whose value is `null` in a higher layer is removed from the merged result. That is the only way to drop a globally defined endpoint locally:

  ```yaml
  # ~/.config/routellm/config.yaml
  endpoints:
    cloud_strong: {model: gpt-4o}
    local_fast:   {model: ollama_chat/qwen3:8b}

  # ./.routellm.yaml — this repo may not talk to the cloud
  endpoints:
    cloud_strong: null
  ```

- **Relative paths resolve against their own file.** A user-layer `prompt_file: prompts/router.yaml` means `~/.config/routellm/prompts/router.yaml` wherever the server is run from, not a path relative to the CWD. This holds for every path key (`policy`, `auth.keys_file`, `audit.path`, `audit.hmac_key_file`, `cache.db_path`, `traces.trace_dir`, `quality_from`, `redact.store.path`, `redact.store.key_file`, `redact.harness_report`): a system-layer `auth.keys_file: keys.yaml` is read from the system file's directory even when `ROUTELLM_CONFIG` names a file elsewhere. A value no file set (built-in default) resolves against the CWD; absolute paths are unchanged and a leading `~` is expanded.

### Governance pinned to the system layer

Layers merge by key, so a project file could otherwise add `local` to a cloud endpoint, or point `policy:` at a file of its own. Governance is honoured only from the system layer (`/etc/routellm/config.yaml`; `ROUTELLM_SYSTEM_CONFIG=<path>` points the layer elsewhere, for containers and tests). Any other layer (user, project, env, flag) that breaks a pin stops startup with an error naming the layer, its file, the key or endpoint, and the system file.

| Pinned | When |
|---|---|
| `policy:` | always system-only, whether or not the system layer sets one |
| `governance`, `strict_registry`, `tag_namespaces`, `egress_gates`, `auth`, `decision`, `audit`, `transcriber`, `video` (whole sections) | while the system layer sets a `policy:` |
| `tiers.<name>.transcriber` and `tiers.<name>.video` (a non-system layer may not add or change them) | while the system layer sets a `policy:` |
| governance tags of every endpoint: reach labels (`local`, `vpn`, `cloud`) and namespaced tags (`jur:ca`, `team:x`) | while the system layer sets a `policy:` |
| every endpoint's `allow_public_base`, including on endpoints a higher layer adds | while the system layer sets a `policy:` |

A higher layer may not add, remove or change a governance tag on any endpoint, including one it defines itself; an endpoint it adds with none stays out of reach of every policy class. Plain tags, every other endpoint key (`api_base` and `model` included; a `local` endpoint on a public host is still refused at load), and `null`-deleting an endpoint remain mergeable. With no system `policy:`, the merge is unchanged.

`paths` marks the system layer `[pins governance]` (`"pins_governance": true` in JSON); `show` marks pinned keys `[pinned]` beside their origin and lists the pins in a closing comment.

### Inspecting the chain

Three commands answer "where does this value come from":

```sh
python -m routellm.config path    # the highest-precedence file that exists; exit 1 if none
python -m routellm.config paths   # every searched location, lowest first, [used] or [absent]
python -m routellm.config show    # the effective merged config, each top-level key commented with its origin
```

`paths` and `show` also take `--format json`. All three honour `--config`, `ROUTELLM_CONFIG` and `--no-user-config`. `python -m routellm.pairing` with no `--config` explains the discovered config the same way. `--format json` on `paths` lists only the entries that actually resolved to a path — an absent project layer has no marker to name, so it is left out rather than represented with a synthetic project line; the text output is where `[absent] project` is always shown for a stable six-layer read.

A project-local `.routellm.yaml` is gitignored: real configuration belongs outside the repo. `config.example.yaml` and `examples/multitier.yaml` are samples to copy from, never the instance config.

### Reserved keys

Most top-level keys are router names mapping to the keyword arguments used for that router's initialization. Two are reserved:

| Key | Holds |
|---|---|
| `endpoints:` | named endpoints, each with its own `model`, optional `api_base`, `api_key_env`, `tags`, `quality`, and `extra` — see [Endpoints](#endpoints) |
| `tiers:` | named strong/weak pairs, each with an optional `router` and `threshold` and a `strong`/`weak` side that names an endpoint, another tier, or a selector — see [Tiers](#tiers) |

Both are read by the endpoint registry and removed before the rest of the file is handed to the routers, so a router can never be shadowed by one of them. A file carrying neither is exactly the router config it always was.

A sample configuration is provided in the `config.example.yaml` file - it provides the configurations for routers that have trained on Arena data augmented using GPT-4 as a judge, plus a worked `endpoints:`/`tiers:` pair spanning a cloud model and two local ones. The models and datasets used are all hosted on Hugging Face under the [RouteLLM](https://huggingface.co/routellm) and [LMSYS](https://huggingface.co/lmsys) organizations.

`routellm/prompts.py` is a core facility: any router or middleware can read a named section from a shared YAML prompt file so its model-facing wording is editable without code changes, without pulling in that adapter's own dependencies. The TypeSafe extension documents its own environment variables in [extensions/typesafe/README.md](extensions/typesafe/README.md).

## Contribution

We welcome contributions! Please feel free to open an issue or a pull request if you have any suggestions or improvements.

### Running the tests

```
pip install -e ".[dev,serve]"
python -m pytest routellm/tests -q -rs
```

Tests that need an optional extra (`pairing`) skip with a reason naming the
extra when it is absent. To run them, install it: `pip install -e ".[dev,serve,pairing]"`.

The suite is hermetic: outbound network is blocked for the whole session,
including subprocesses the tests spawn. Loopback (`127.0.0.0/8`, `::1`,
`localhost`) and Unix sockets work; anything else raises an error naming the
host, and a test that attempted it fails even if the code under test swallowed
the error. A deliberately live test is marked `@pytest.mark.network` and runs
only with `ROUTELLM_TEST_NETWORK=1`. `ROUTELLM_TEST_NO_NETWORK=0` turns the
blocker off. xrr cassettes replay by default; `XRR_MODE=record` re-records.

### Adding a new router

There is only a single method to implement: `calculate_strong_win_rate`, which takes in the user prompt and returns the win rate for the strong model conditioned on that given prompt - if this win rate is great than user-specified cost threshold, then the request is routed to the strong model. Otherwise, it is routed to the weak model.

Two paths to register a router, both backed by the same registry (`routellm/routers/registry.py`):

**In-tree**: subclass `Router` from `routellm/routers/base.py` and call `register_router` in `routers.py`.

**As a package**: subclass `Router` in your own package, publish it as an entry point in your package's `pyproject.toml`, and `pip install` it — the name then appears in `--routers` for the server, calibration, and evals with no core change:

```toml
[project.entry-points."routellm.routers"]
myrouter = "pkg.mod:Cls"
```

```python
from routellm.routers.registry import register_router

register_router("myrouter", MyRouterClass)
```

A plugin that fails to import is skipped with a warning and shown in the error when its name is requested.

### How a router embeds a prompt

A router that embeds the prompt before scoring it — `mf` and `sw_ranking` do — sets `requires_embedding = True` and calls `embed(text)` from `routellm/routers/embeddings.py` inside the method that embeds:

```python
from routellm.routers.embeddings import embed

class MyRouter(Router):
    requires_embedding = True

    def calculate_strong_win_rate(self, prompt):
        vector = embed(prompt)
        ...
```

`embed` sends the text to one registry endpoint: the one `embeddings.endpoint` names, else the endpoint called `embedding`. The endpoint's `model`, `api_base`, `api_key_env` and `extra` are used as written, so the call carries its reach and `jur:` tags like any model call. The semantic cache embeds through the same endpoint. There is no default endpoint and no default model: a server that loads such a router, or enables `cache.semantic_enabled`, with no endpoint refuses to start and names the missing key. The startup log names the endpoint and its reach.

`--allow-remote-embeddings` falls back, for a server with no endpoint, to `OPENAI_BASE_URL` / `OPENAI_API_KEY`, with a WARNING naming the host the prompts go to. No tag describes that call, so no policy can gate it: with `strict_registry: true` or a loaded `policy:` (which forces strict) the flag is refused at startup, naming `embeddings.endpoint` as the fix. Tests that need their own configuration call `configure_embeddings(registry, endpoint)` and `reset_embedding_client()`.

### Adding a new benchmark

To add a new benchmark to RouteLLM, implement the abstract `Benchmark` class in `benchmarks.py` and update the `evaluate.py` module to properly initialize the new benchmark class. Ideally, the results for the benchmark should be precomputed to avoid having to regenerate the results for each evaluation run -- see the existing benchmarks for examples on how to do this.

# Citation

The code in this repository is based on the research from the [paper](https://arxiv.org/abs/2406.18665). Please cite if you find the repository helpful.

```
@misc{ong2024routellmlearningroutellms,
      title={RouteLLM: Learning to Route LLMs with Preference Data},
      author={Isaac Ong and Amjad Almahairi and Vincent Wu and Wei-Lin Chiang and Tianhao Wu and Joseph E. Gonzalez and M Waleed Kadous and Ion Stoica},
      year={2024},
      eprint={2406.18665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.18665},
}
```
