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
- **Auto-Retry**: Exponential backoff for transient failures (e.g., 429, 5xx).
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
- **Trace Collection**: Standardized request/response recording for fine-tuning with **Fit**.
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
    # Caching: Enable semantic caching with 95% similarity threshold
    cache_config=CacheConfig(semantic_enabled=True, semantic_threshold=0.95),
    traffic_manager=traffic_manager,
    # Quality: 5% Canary traffic and record traces for fine-tuning
    quality_manager=QualityManager(
        canary_config=CanaryConfig(enabled=True, canary_model="gpt-4o", weight=0.05),
        fine_tune_config=FineTuneConfig(enabled=True)
    )
)
```

## Installation

**From PyPI**
```
pip install "routellm[serve,eval]"
```

**From source**

```
git clone https://github.com/lm-sys/RouteLLM.git
cd RouteLLM
pip install -e .[serve,eval]
```

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
INFO:     Uvicorn running on http://0.0.0.0:6060 (Press CTRL+C to quit)
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

Note that regardless of the model pair used, an `OPENAI_API_KEY` will currently still be required to generate embeddings for the `mf` and `sw_ranking` routers.

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
python -m routellm.openai_server --routers mf jev --config config.example.yaml
```

- `--routers` specifies the list of routers available to the server. For instance, here, the server is started with two available routers, `mf` and `jev` (see below for the list of routers). The first one is the default for a request that names none.
- `--config` is the single source for the server's endpoints, tiers, and router settings. If unspecified, the server defaults to our best-performing router configuration and routes the flat `--strong-model`/`--weak-model` pair (see [Configuration](#configuration) for details).
- `--default-threshold` is the threshold used by a level that names none and whose request carries none. Default `0.5`.

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

### Tiers

A tier is a named strong/weak pair whose sides may themselves be tiers, so a request addressed to a tier walks a tree, one router call per level on the original prompt:

```yaml
tiers:
  premium:
    router: jev
    threshold: 0.33
    strong: cloud_strong
    weak: colibri_glm
  default:
    router: mf
    threshold: 0.12
    strong: premium
    weak: ollama_qwen
```

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

The `default` tier is what makes the legacy form keep working: an existing OpenAI client sending `router-mf-0.5` reaches the tree without changing its model field. When the config defines no `default` tier and both `--strong-model` and `--weak-model` are given, the server derives an implicit one from them, running `--routers[0]` at `--default-threshold`. A config `default` tier always wins; which one is in effect is logged at startup.

### `GET /v1/models`

Many OpenAI clients probe the model list before their first request. The server answers with one id per configured tier, plus `router-<name>-<default_threshold>` for each loaded router — the legacy form needs a concrete threshold, and only the server's default is a usable id, though any other threshold still routes:

```
> curl -s localhost:6060/v1/models
{"object":"list","data":[
  {"id":"default","object":"model","created":1758326400,"owned_by":"routellm"},
  {"id":"premium","object":"model","created":1758326400,"owned_by":"routellm"},
  {"id":"router-mf-0.5","object":"model","created":1758326400,"owned_by":"routellm"}]}
```

`created` is the moment the process came up: the listing is derived from the config, so it does not change while the server runs.

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

`select` is a space-separated list of terms, all ANDed. A `tag:<label>` term is answered locally from the endpoint's own `tags`; every other `key:value` term is a models.dev fact. Bare free text is rejected — a policy says what a model *is*, never what its name looks like. `order` picks the winner among the matches, one of `cost_asc`, `cost_desc`, `quality_asc`, `quality_desc` (default), `context_desc`.

Selectors need the optional `pairing` extra. Endpoints models.dev does not list, such as anything served by Ollama, are tag-only candidates: they satisfy `tag:` terms and fail every catalog term. Explain a pick without starting a server:

```
python -m routellm.pairing --config config.example.yaml
```

```
-> colibri_glm(model=openai/glm-5.2-colibri, quality=85, cost=-, context=-)
   ollama_qwen(model=ollama_chat/qwen3:8b, quality=None, cost=-, context=-)
```

Every selector is resolved to an endpoint name before the tier graph is validated a second time, so nothing downstream ever sees one.

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

The configuration is specified in either the `config` argument for `Controller` or by passing in the path to a YAML file using the `--config` flag. Most top-level keys are router names mapping to the keyword arguments used for that router's initialization. Two are reserved:

| Key | Holds |
|---|---|
| `endpoints:` | named endpoints, each with its own `model`, optional `api_base`, `api_key_env`, `tags`, `quality`, and `extra` — see [Endpoints](#endpoints) |
| `tiers:` | named strong/weak pairs, each with an optional `router` and `threshold` and a `strong`/`weak` side that names an endpoint, another tier, or a selector — see [Tiers](#tiers) |

Both are read by the endpoint registry and removed before the rest of the file is handed to the routers, so a router can never be shadowed by one of them. A file carrying neither is exactly the router config it always was.

An example configuration is provided in the `config.example.yaml` file - it provides the configurations for routers that have trained on Arena data augmented using GPT-4 as a judge, plus a worked `endpoints:`/`tiers:` pair spanning a cloud model and two local ones. The models and datasets used are all hosted on Hugging Face under the [RouteLLM](https://huggingface.co/routellm) and [LMSYS](https://huggingface.co/lmsys) organizations.

`routellm/prompts.py` is a core facility: any router or middleware can read a named section from a shared YAML prompt file so its model-facing wording is editable without code changes, without pulling in that adapter's own dependencies. The TypeSafe extension documents its own environment variables in [extensions/typesafe/README.md](extensions/typesafe/README.md).

## Contribution

We welcome contributions! Please feel free to open an issue or a pull request if you have any suggestions or improvements.

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

### How a router gets the embedding client

A router that embeds the prompt before scoring it — `mf` and `sw_ranking` do — must not build an OpenAI client at import: that made `import routellm.routers.routers` fail without `OPENAI_API_KEY` even for a server whose routers never embed. Call `get_embedding_client()` from `routellm/routers/embeddings.py` inside the method that embeds instead:

```python
from routellm.routers.embeddings import get_embedding_client

class MyRouter(Router):
    def __init__(self):
        self.embedding_model = "text-embedding-3-small"

    def calculate_strong_win_rate(self, prompt):
        vector = get_embedding_client().embeddings.create(
            input=[prompt], model=self.embedding_model
        )
        ...
```

The client is built on first call and cached. The controller calls `configure_embeddings` with its endpoint registry before constructing any router, so an endpoint named `embedding` in the config points those calls at a different provider than the completions; with no such endpoint, the client falls back to `OPENAI_BASE_URL` / `OPENAI_API_KEY`. That endpoint supplies the base URL and the credential only — the model each router embeds with stays the router's own `embedding_model`. When neither source supplies a key, the first embedding call raises a `RuntimeError` naming both ways to fix it. Tests that need their own client call `reset_embedding_client()`.

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
