# Examples

| File | What it shows |
|---|---|
| [`multitier.yaml`](multitier.yaml) | A multi-tier server config: several providers behind one endpoint, with the model chosen by policy |
| [`python_sdk.md`](python_sdk.md) | Driving the router from Python through the `Controller` class |
| [`router_chat.py`](router_chat.py) | A terminal chat client that talks to the server |
| [`routing_to_local_models.md`](routing_to_local_models.md) | Pairing a cloud model with a local one served by Ollama |

## `multitier.yaml`

One server spanning several providers, where almost every tier side
*selects* an endpoint rather than naming one. Four axes carry the whole
configuration, and each is a tag on an endpoint:

- **governance** — reach, a set an endpoint may hold several of:
  `local` (our own hardware), `vpn` (private network only: a tailscale
  host, or a private cloud endpoint such as AWS Bedrock or Google
  Vertex), `cloud` (public internet). Plus one jurisdiction, `jur:us`
  or `jur:ca`.
- **performance** — `fast` or `slow`.
- **expertise** — `coder`, `writer`, `designer`, `general`.
- **capability** — `tools`, `vision`, `long_context`, written only for
  endpoints [models.dev](https://models.dev) holds no record of. For
  every other endpoint the catalog answers the same question, and a
  hand-written tag would drift from it.

Tiers come in three kinds. Area roots — `coding`, `copywriting`,
`design` — run `jev` on the prompt and hand it to a quality tier or a
fast tier. Governance roots restrict by reach: `private` takes
`tag:local` on both sides, so no cloud endpoint can win either however
good it is; `vpn` widens that to the private network. A performance
root, `fast`, takes the cheapest of everything tagged `fast`. `default`
runs the cheap `mf` router and is the only tier an intent classifier
may move a request out of.

Address a tier by putting its name in the `model` field:

```
python -m routellm.openai_server --config examples/multitier.yaml --routers mf jev
curl -s localhost:6060/v1/chat/completions \
  -d '{"model": "coding", "messages": [{"role": "user", "content": "hi"}]}'
```

The model ids in the file are examples. Check what the selectors
actually pick from your own endpoints before trusting them — this needs
no server and makes one models.dev fetch:

```
python -m routellm.pairing --config examples/multitier.yaml
```

Specialise a copy for a real machine rather than editing this one: swap
in the model ids your servers report, set `quality` on anything
models.dev does not list, and point each `api_base` at the tunnel or
host that actually serves it.
