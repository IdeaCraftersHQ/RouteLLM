# API dialects

routellm serves more than the OpenAI chat API. A client that speaks Anthropic
Messages, OpenAI Responses, Gemini, Ollama or AWS Bedrock points its base URL at
routellm and keeps working: the request is translated to a chat completion, runs
the same pipeline as `/v1/chat/completions` (routing, transforms, cache,
resilience, audit), and the answer is translated back.

Writing a new dialect: [adapters.md](adapters.md).

Contents: [Learn](#learn) · [Do](#do-client-setup) · [Fix](#fix) ·
[Reference](#reference) · [Known gaps](#known-gaps)

## Learn

### What every dialect shares

- **Same pipeline.** Tiers, routers, policies, caches and audit apply unchanged.
  Dialect choice never changes routing.
- **Same keys.** Keys come from `auth.keys_file`
  ([Caller identity](../README.md#caller-identity)); only the header that carries
  the key differs per dialect. Auth off: any value is accepted.
- **Routing rides in the model name.** Same forms as
  [Model names](../README.md#model-names): `premium`, `router-premium`,
  `router-mf-0.5`, `premium:router-mf-0.5`. Where the name sits depends on the
  dialect (body field or URL path). Client-side model names must therefore be
  routellm tiers or routers, not vendor names.
- **Decision path.** Every answer, streamed or not, carries the routing decision in
  the `x-routellm-path` response header (JSON list).
- **Refuse, never drop.** A field routellm cannot carry is a 400 naming the field,
  in the dialect's own error shape. A field with no effect on the answer is accepted
  and ignored. The tables below list both.
- **Stateless.** Nothing is stored: no response ids to fetch, no server-side
  conversation.

### Dialect matrix

| Dialect | Routes | Key header | Model name rides in | Stream format |
|---|---|---|---|---|
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | `x-api-key` or `Authorization: Bearer` | body `model` | SSE (`message_start` ... `message_stop`) |
| OpenAI Responses | `POST /v1/responses` | `Authorization: Bearer` | body `model` | SSE (`response.*` events) |
| OpenAI Completions (legacy) | `POST /v1/completions` | `Authorization: Bearer` | body `model` | SSE (`text_completion` chunks) |
| Gemini | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `:countTokens`; `GET /v1beta/models[/{model}]`; also `/v1` (no listing) | `x-goog-api-key` or `?key=` | URL path `{model}` | `?alt=sse`: SSE; otherwise one JSON array |
| Ollama | `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show`, `GET /api/version` | `Authorization: Bearer` | body `model` | NDJSON (default; `stream` defaults to true) |
| Bedrock Converse | `POST /model/{modelId}/converse`, `/converse-stream` | `Authorization: Bearer` (see [Bedrock auth](#bedrock-auth)) | URL path `{modelId}` | binary `application/vnd.amazon.eventstream` |
| Bedrock InvokeModel (Anthropic body) | `POST /model/{modelId}/invoke`, `/invoke-with-response-stream`, `/count-tokens` | as Converse | URL path `{modelId}` | binary event stream; each frame wraps one Anthropic event |

Error shapes follow the dialect: Anthropic `{"type":"error",...}`, Gemini
`{"error":{"code","message","status"}}`, Ollama `{"error":"..."}`, Bedrock
`{"message":...}` plus `x-amzn-ErrorType`, OpenAI-style for Responses and
Completions.

### Estimates and statelessness

- **Token counts are estimates.** `count_tokens` (Anthropic), `countTokens`
  (Gemini) and `/count-tokens` (Bedrock) translate the request and count with
  litellm's default tokenizer. No model is chosen and nothing goes upstream, so the
  routed model's own tokenizer is not used. Expect a few percent of drift.
- **`previous_response_id` is unsupported** (Responses). The server keeps no
  response state: send the whole conversation in `input`. Also refused:
  `conversation`, `background: true`.
- `store` is accepted and ignored: a response cannot be fetched later.

### Sampling: `top_k`

`top_k` (a whole number from 0) is carried from every dialect that has it:
Anthropic, Gemini (`topK`), Ollama (`options.top_k`), Bedrock
(`additionalModelRequestFields.top_k`). litellm maps it for Anthropic, Gemini,
Ollama and Bedrock leaves, and an OpenAI-compatible server on its own `api_base`
(vLLM, llama.cpp, colibri) takes it as a top-level field. It is dropped, under any
`drop_params` policy except `false`, for leaves whose API rejects unknown fields:
OpenAI itself and Azure OpenAI. `min_p` and `repeat_penalty` are not carried
(refused on Ollama).

### colibri extensions and KV slots

Upstreams that take request hints (colibri, `coli serve`) get them only when the
endpoint opts in; every other endpoint drops them.

- Hints: `cache_slot` (0..15, picks a KV context) and
  `x_colibri_ignore_leading_stop`. Forwarded to endpoints with `kind: colibri` or
  `extensions: [...]`; a dialect request sets them only through the session pin
  below (the dialects have no field for them).
- **Session pinning.** Endpoint with `kv_slots: N` (1..16, match
  `coli serve --kv-slots`): a request naming a session and sending no `cache_slot`
  is pinned to one slot, least recently used first, so turns reuse one KV prefix.
- **Session header.** `X-RouteLLM-Session: <id>` (any dialect); `metadata.session_id`
  where the dialect has `metadata`. Header wins. The header is never forwarded
  upstream. Mapping is in memory: a restart reassigns slots once (one cold turn per
  session).
- The upstream's `x-colibri-queue-wait-ms` is added to the response headers.

Config: [config.example.yaml](../config.example.yaml) (`local_colibri`),
[README](../README.md) "Upstream extensions (colibri)".

### Bedrock auth

Bearer by default. boto3, the Anthropic Bedrock client and Claude Code send
`Authorization: Bearer <AWS_BEARER_TOKEN_BEDROCK>`; the value is looked up as an
ordinary key in the keys file.

- A request signed with SigV4 and carrying no bearer token is a 401
  `UnrecognizedClientException`. The server stores only key hashes, so it cannot
  recompute a signature, and an access key id is not a secret (logs, URLs, error
  messages): trusting it alone would let anyone who has seen one impersonate its
  principal.
- **Opt-in, off by default:** `ROUTELLM_BEDROCK_ALLOW_UNVERIFIED_SIGV4=1` (also
  `true`, `yes`; read on every request). The access key id of a SigV4
  `Authorization` header is then looked up as the key (list
  `sha256(<access key id>)` in the keys file). **The signature is still not
  verified.** Anyone who knows or guesses an access key id can act as that
  principal. Use only on a network where every caller is trusted.
- Credentials are never read from the URL: a presigned `X-Amz-Credential` query
  parameter is ignored in both modes.

## Do: client setup

Examples use `http://127.0.0.1:6060` (default `--host`/`--port`) and `$ROUTELLM_KEY`
for a key from the keys file. Without `auth:`, any non-empty value works. Model
names are routellm tiers or routers, here `default`; list them with
`GET /v1/models`.

Status legend: **verified** = taken from the client's current official docs or CLI
at the time of writing; **unverified** = not confirmed against the client's docs or
a run, check before relying on it.

### Claude Code (Anthropic Messages) — verified

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:6060
export ANTHROPIC_AUTH_TOKEN=$ROUTELLM_KEY      # sent as Authorization: Bearer
# or ANTHROPIC_API_KEY=$ROUTELLM_KEY           # sent as x-api-key; both accepted
export ANTHROPIC_MODEL=default                 # a tier or router name
export ANTHROPIC_DEFAULT_OPUS_MODEL=premium
export ANTHROPIC_DEFAULT_SONNET_MODEL=default
export ANTHROPIC_DEFAULT_HAIKU_MODEL=cheap
claude
```

- `ANTHROPIC_MODEL` and the three `ANTHROPIC_DEFAULT_*_MODEL` variables decide
  what Claude Code puts in the `model` field. Pin all of them: an unpinned alias
  sends a `claude-...` name, which is not a tier and does not route.
- Claude Code also sends `anthropic-beta`, `context_management` and `cache_control`;
  all are accepted and ignored. Hosted tools (`web_search_*`, `bash_*`,
  `computer_*`, ...) are refused: only custom tools are carried.
- `/status` shows the base URL and credential source in use.

### Claude Code, Bedrock mode (InvokeModel) — verified

```bash
export CLAUDE_CODE_USE_BEDROCK=1
export ANTHROPIC_BEDROCK_BASE_URL=http://127.0.0.1:6060
export AWS_BEARER_TOKEN_BEDROCK=$ROUTELLM_KEY
export AWS_REGION=us-east-1                    # region, unused here
export ANTHROPIC_MODEL=default
export ANTHROPIC_DEFAULT_OPUS_MODEL=premium
export ANTHROPIC_DEFAULT_SONNET_MODEL=default
export ANTHROPIC_DEFAULT_HAIKU_MODEL=cheap
claude
```

Claude Code sends InvokeModel with an Anthropic body, served by the
`/model/{modelId}/invoke*` routes, not Converse. With `AWS_BEARER_TOKEN_BEDROCK`
set, Claude Code authenticates with it and skips the AWS credential chain. The
Claude Code docs also list `CLAUDE_CODE_SKIP_BEDROCK_AUTH=1` for gateways that hold
their own credentials; it is not needed with a bearer key. Whether Claude Code needs
`AWS_REGION` here is unverified (the docs default it to `us-east-1`).

### Codex CLI (OpenAI Responses) — provider keys verified, end-to-end run unverified

`~/.codex/config.toml`:

```toml
model_provider = "routellm"
model = "default"                      # a tier or router name

[model_providers.routellm]
name = "routellm"
base_url = "http://127.0.0.1:6060/v1"
env_key = "ROUTELLM_KEY"               # env var holding the key
wire_api = "responses"
```

```bash
export ROUTELLM_KEY=...
codex
```

- Keys `model_provider`, `[model_providers.<id>]`, `name`, `base_url`, `env_key`,
  `wire_api = "responses"` are from the current Codex configuration docs. The
  provider id must be unique and not one of the reserved built-ins (`openai`,
  `ollama`, `lmstudio`, ...).
- Codex always streams with `store: false`, and sends developer messages, prior
  reasoning items, `include: ["reasoning.encrypted_content"]`, `prompt_cache_key`
  and `text.verbosity`: all accepted.
- Codex's hosted `web_search` tool is accepted and ignored: the model is not
  offered it, no need to set `web_search = "disabled"`. Every ignored hosted
  tool is listed in the `x-routellm-ignored` response header (comma-separated
  types, e.g. `web_search`), streamed or not. A Codex setup that uses the
  `local_shell` tool is refused; with the default function `shell` tool it works.

### gemini-cli (Gemini) — variable names verified, end-to-end run unverified

```bash
export GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:6060
export GEMINI_API_KEY=$ROUTELLM_KEY
gemini -m default
```

- `GOOGLE_GEMINI_BASE_URL` overrides the base URL when authenticating with
  `GEMINI_API_KEY` (gemini-cli configuration docs). The docs require HTTPS except
  for `localhost`; a non-localhost plain-HTTP routellm needs TLS in front.
- The model name in the URL path carries routing: `-m premium` or
  `-m premium:router-mf-0.5`.
- That the CLI appends `/v1beta` itself is unverified; if it does not, set the
  base URL accordingly.

### Gemini SDK (`google-genai`) — unverified

```python
from google import genai
client = genai.Client(
    api_key=KEY,
    http_options={"base_url": "http://127.0.0.1:6060", "api_version": "v1beta"},
)
client.models.generate_content(model="default", contents="hi")
```

`api_version` `v1` works for generate, stream, count and get-one, but not for
listing (see [Known gaps](#known-gaps)). Parameter names unverified against the SDK.

### Ollama clients — host variable verified, auth unverified

```bash
export OLLAMA_HOST=http://127.0.0.1:6060
ollama list            # -> GET /api/tags: tiers and routers
ollama run default
```

`OLLAMA_HOST` is the standard Ollama client host variable. If auth is on, the
client must send `Authorization: Bearer <key>`: with the Python client,
`ollama.Client(host=..., headers={"Authorization": "Bearer " + KEY})`. Whether the
`ollama` CLI itself can send a bearer token to a custom host is unverified.
A trailing `:latest` on a model name is dropped before routing.

### boto3 (Bedrock Converse / InvokeModel) — unverified

```python
import boto3
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",                 # required by boto3, unused here
    endpoint_url="http://127.0.0.1:6060",
)
# AWS_BEARER_TOKEN_BEDROCK=$ROUTELLM_KEY in the environment
client.converse(
    modelId="default",
    messages=[{"role": "user", "content": [{"text": "hi"}]}],
)
```

- `endpoint_url` is a standard boto3 client option. Bearer-token auth through
  `AWS_BEARER_TOKEN_BEDROCK` needs a boto3 release that supports Bedrock API keys;
  the minimum version is unverified. Without it, boto3 signs with SigV4 and gets a
  401 unless the [unverified-SigV4 opt-in](#bedrock-auth) is on.
- `invoke_model` / `invoke_model_with_response_stream` / `count_tokens` take an
  Anthropic Messages body without `model` and `stream`.

### OpenAI SDK (chat, Responses, legacy completions) — verified

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:6060/v1", api_key=KEY)
client.chat.completions.create(model="default", messages=[...])
client.responses.create(model="default", input="hi")
client.completions.create(model="default", prompt="hi")
```

`base_url` and `api_key` are the SDK's standard options; the same client covers
all three OpenAI routes.

## Fix

| Symptom | Cause | Fix |
|---|---|---|
| 400 `invalid_request_error` naming a field | field is in the refused list for the dialect | drop it client-side; see the dialect table |
| Claude Code request does not route, or unknown model | `claude-...` model name sent | pin `ANTHROPIC_MODEL` and `ANTHROPIC_DEFAULT_*_MODEL` to tiers or routers |
| Anthropic 400 `max_tokens` | Messages requires `max_tokens` | set it client-side |
| Bedrock 401 `UnrecognizedClientException` | SigV4 request without bearer token | send the bearer key (`AWS_BEARER_TOKEN_BEDROCK`) |
| 400 `previous_response_id` | server keeps no response state | send the whole conversation in `input` |
| Codex refused on `local_shell` | hosted tool unsupported | use the default function `shell` tool |
| Gemini model listing fails with `api_version: v1` | `GET /v1/models` is the OpenAI list | use `v1beta` |
| Token count differs from provider's | counts are estimates | do not use for billing or hard limits |
| 508 | request carried `X-RouteLLM-Hop`: a leaf fronts this server | remove the loop |

## Reference

Condensed from the module docstrings in `routellm/adapters/`. The code is the
source of truth.

### Anthropic Messages

- **Mapped:** `messages` (text, image, document, `tool_use`, `tool_result`,
  `thinking`, `redacted_thinking`), `system`, `max_tokens` (required),
  `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, `metadata.user_id`,
  `tools` (custom), `tool_choice` (incl. `disable_parallel_tool_use`), `thinking`,
  `output_config.format`.
- **Mid-conversation `role: system`** (Claude Code's environment note): sent as a
  chat system message at the same position. OpenAI-compatible and Ollama chat
  upstreams keep it in place; Gemini merges it into `systemInstruction` (position
  lost); Anthropic and Bedrock Converse upstreams turn it into a user turn in place
  marked as an operator note, unless litellm flags the model as accepting system
  messages inside `messages`. The text is never dropped.
- **Tool search** (Claude Code, many MCP tools): `defer_loading` tools are sent
  up front, the `DeferredToolPlaceholder` tool is dropped, and a `tool_reference`
  in a tool result becomes a `<functions>` text block with the tool's name,
  description and schema from `tools`. A reference to a tool not in `tools` is
  refused (Claude Code then sends every tool up front on its own).
- **Ignored:** `cache_control`, `context_management`, `service_tier`,
  `thinking.display`, `tool_result.is_error` (text still reaches the model),
  `eager_input_streaming`, `output_config.effort` without adaptive thinking;
  headers `anthropic-version`, `anthropic-beta`.
- **Refused (400):** any other top-level field (`mcp_servers`, `container`, ...),
  missing `max_tokens`, `metadata` keys other than `user_id`, block types not
  mapped (`server_tool_use`, `search_result`, `web_search_tool_result`, ...),
  hosted tools, `document` sources other than base64/url, roles other than
  `user`/`assistant`/`system`.
- **Answers:** `model` echoes the client's model field; `stop_sequence` is always
  null (the matched sequence is unknown); stream `message_start` reports zero
  input tokens (final usage arrives in `message_delta`); a stream failing after it
  began ends with an `error` event and no `message_stop`.

### OpenAI Responses

- **Mapped:** `model`, `input` (string or items), `instructions`, `tools`,
  `tool_choice`, `parallel_tool_calls`, `temperature`, `top_p`,
  `max_output_tokens`, `stream`, `reasoning.effort`, `text.format`, `metadata`,
  `user`.
- **Ignored:** `store`, `include` of `reasoning.encrypted_content`,
  `prompt_cache_key`, `prompt_cache_retention`, `service_tier`,
  `safety_identifier`, `client_metadata`, `stream_options`, `reasoning.summary`,
  `text.verbosity`, `truncation: "disabled"`, `background: false`.
- **Refused:** `previous_response_id`, `conversation`, `background: true`,
  `max_tool_calls`, `top_logprobs`, `truncation: "auto"`, `prompt`,
  `context_management`, any other `include`, any other field; hosted tools
  (`file_search`, `mcp`, `local_shell`, `computer_use`, `image_generation`); input items `local_shell_call`, `item_reference`,
  `web_search_call`, `computer_call`.
- **Tools carried:** `function`, `custom` (Codex `apply_patch`), `namespace` of
  those. **Ignored:** hosted `web_search` (and `web_search_preview` variants),
  reported in the `x-routellm-ignored` response header.
- **Answers:** ids are `resp_<chat id>`, the same streamed or not; every event
  carries `sequence_number`; a stream failing after it began ends with
  `response.failed`.

### OpenAI Completions (legacy)

- **Mapped:** `model`, `prompt` (string or one-string list, sent as one user
  message), `max_tokens`, `stop`, `temperature`, `top_p`, `n`, `seed`, `user`,
  `presence_penalty`, `frequency_penalty`, `logit_bias`, `stream`,
  `stream_options.include_usage`, `echo`, `best_of` (null or 1).
- **Refused:** several prompts, token-id prompts, empty or missing prompt,
  `suffix`, `logprobs`, `best_of` other than 1, `stream_options` without `stream`,
  any other field.
- **Notes:** absent `max_tokens` means the upstream default, not 16. A reasoning
  trace has no place in `text_completion` and is not sent.

### Gemini

- **Mapped:** `contents[].parts` (text, `inlineData` image/*, `functionCall`,
  `functionResponse`), roles `user`/`model`/`function`, `systemInstruction`,
  `tools[].functionDeclarations`, `toolConfig.functionCallingConfig` (AUTO/ANY/NONE,
  one `allowedFunctionNames`), `generationConfig`: `temperature`, `topP`, `topK`,
  `maxOutputTokens`, `stopSequences`, `responseMimeType`
  (`text/plain`, `application/json`), `responseSchema` / `responseJsonSchema`
  (object), `presencePenalty`, `frequencyPenalty`, `seed`, `candidateCount=1`.
- **Ignored:** `safetySettings`, `generationConfig.thinkingConfig`,
  `generationConfig.mediaResolution`, `thoughtSignature`, `thought: true` parts.
- **Refused (400 `INVALID_ARGUMENT`):** every other field: `cachedContent`,
  `labels`, `fileData`, `executableCode`, built-in tools (`googleSearch`, ...),
  `responseLogprobs`, `responseModalities`, `candidateCount > 1`, non-image
  `inlineData`, several `allowedFunctionNames`, non-object response schemas, `alt`
  other than `sse` or `json`.
- **Limits:** a user turn mixing `functionResponse` and other parts is sent as the
  user message then the tool messages. Model listing ignores `pageSize` and
  `pageToken`.

### Ollama

- **`/api/chat` mapped:** `model`, `messages` (`role`, `content`, `images`,
  `tool_calls`, `tool_name`, `tool_call_id`), `tools`, `format`, `options`,
  `stream`, `think`. **`/api/generate` mapped:** `model`, `prompt`, `system`,
  `images`, `format`, `options`, `stream`, `think`, `raw`.
- **`options` mapped:** `temperature`, `top_p`, `top_k`, `seed`, `stop`,
  `num_predict` (> 0; -1 and -2 mean no limit), `frequency_penalty`,
  `presence_penalty`.
- **Ignored:** `keep_alive`, per-message `thinking`; options `num_ctx`,
  `num_batch`, `num_gpu`, `main_gpu`, `num_thread`, `num_keep`, `use_mmap`,
  `use_mlock`, `numa`, `low_vram`, `f16_kv`, `vocab_only`.
- **Refused:** every other option (`min_p`, `repeat_penalty`, ...), other
  top-level fields (`suffix`, `template`, `context`, `logprobs`), unknown message
  keys, an empty prompt or message list (the "load the model" call), a `tool`
  message with no call to pair.
- **Notes:** `think: true` is `reasoning_effort: medium`; `"low"`, `"medium"`,
  `"high"` pass through; `false` sets nothing. `raw: true` still applies the chat
  template upstream. Durations are best effort: `total_duration` and
  `eval_duration` are wall time, `load_duration` and `prompt_eval_duration` are 0.
  `/api/tags` and `/api/show` list the server's tiers and routers; `/api/show` of
  an unlisted model is a 404. `/api/version` reports the Ollama API level followed,
  not routellm's version.

### Bedrock Converse

- **Mapped:** `system`, `messages[].content[]` (text, image, document, `toolUse`,
  `toolResult`), `toolConfig` (`tools[].toolSpec`, `toolChoice`), `inferenceConfig`
  (`maxTokens`, `temperature`, `topP`, `stopSequences`), `requestMetadata`
  (`data_class`, `operation`), `additionalModelRequestFields` as `top_k` when it is
  the only key.
- **Ignored:** `cachePoint` blocks, `reasoningContent` in assistant history,
  `toolResult.status`, `performanceConfig`.
- **Refused (400 `ValidationException`):** `additionalModelRequestFields` with any
  key but `top_k`, `additionalModelResponseFieldPaths`, `guardrailConfig`,
  `promptVariables`, `outputConfig`, `guardContent`, video/audio/search/citation
  blocks, `s3Location` sources, image or document inside a `toolResult`, `systemTool`
  entries, other roles.
- **Answers:** `stopReason` `stop` -> `end_turn`, `length` -> `max_tokens`,
  `tool_calls` -> `tool_use`, `content_filter` -> `content_filtered`;
  `stop_sequence` never reported. A stream failing after it began ends with an
  `internalServerException` frame and no `messageStop`.

### Bedrock InvokeModel (Anthropic bodies)

Body is an Anthropic Messages request without `model` and `stream` (both refused:
Bedrock rejects them). Fields follow [Anthropic Messages](#anthropic-messages)
above. Differences: `anthropic_version` accepted and ignored; `anthropic_beta`
(list of strings) accepted and ignored. `/invoke` adds the
`x-amzn-bedrock-input-token-count` and `x-amzn-bedrock-output-token-count`
headers. `/count-tokens` takes `{"input": {"invokeModel": {"body": <blob>}}}`
(base64 JSON as boto3 sends, or the JSON itself); the `converse` input form is
refused; answer `{"inputTokens": n}`, an estimate.

## Known gaps

- **`GET /v1/models` is the OpenAI list.** The same path exists in the Gemini API,
  so a `google-genai` client with `api_version: v1` gets the OpenAI shape and cannot
  list models. Gemini listing is served only under `/v1beta` (`GET /v1beta/models`,
  `GET /v1beta/models/{model}`); generate, stream, `countTokens` and get-one work
  under both. Trade-off: the OpenAI list is what most clients probe at startup, and
  serving both shapes would need dialect sniffing on that route, a framework change.
  Use `v1beta`.
- **Cohere is not supported yet.**
- **Non-generation endpoints are out of scope:** embeddings, audio, images. They
  have no chat-completions equivalent, so dialects do not translate them.
- **Hosted and server-side tools** (web search, file search, computer use, MCP
  servers, code execution) are refused in every dialect: only function tools are
  carried.
- **No server-side state:** no stored responses, no `previous_response_id`, no
  Gemini `cachedContent`.
- **Token counts are estimates**, never the routed model's own.
