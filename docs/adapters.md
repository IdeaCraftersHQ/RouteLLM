# Writing a dialect adapter

An adapter lets a client that speaks another LLM API (Anthropic Messages,
OpenAI Responses, Gemini, Ollama, Bedrock, Cohere, ...) use routellm
unchanged. It translates the request into a `ChatCompletionRequest`, the
framework runs it through the same pipeline as `/v1/chat/completions`, and
the adapter translates the answer back.

One dialect is one module in `routellm/adapters/` plus its tests. Nothing
shared is edited: modules are discovered at startup.

## What the framework does

For every route an adapter declares, in order, in process (never over HTTP):

1. `adapter.credential(headers, query)` extracts the caller's key; the same
   resolver as `/v1/chat/completions` maps it to a principal. A missing or
   unknown key is a 401 in the dialect's error shape.
2. Loop refusal (the routing hop header), then the body is read as JSON.
3. `adapter.decode(call)` returns a `ChatCompletionRequest`.
4. `run_chat_completion`: request class, routing, transforms, cache,
   resilience, audit trail. The model field carries routing parameters
   exactly as on `/v1/chat/completions`.
5. The answer is re-encoded: `encode` (non-stream), the stream framing, or
   `encode_error`.

Every adapter response carries the routing decision in the
`x-routellm-path` header (JSON list), streamed or not. A dialect whose body
can carry extra fields may also copy `response["routellm"]` into it.

Upstream hints ride on the same request: `decode` may set `cache_slot` and
`x_colibri_ignore_leading_stop` on the `ChatCompletionRequest`, and a caller's
session comes from the `X-RouteLLM-Session` header or `metadata.session_id`.
Only an endpoint that lists them (`kind: colibri`, `extensions:`) receives
them; any other drops them. An adapter that does nothing leaves them unset.
The upstream's `x-colibri-queue-wait-ms` is added to the response headers.

## Contract

`routellm/adapters/base.py`. Import everything from `routellm.adapters`.

```python
from routellm.adapters import Adapter, Route, ChatCompletionRequest, register

@register
class MyDialect(Adapter):
    name = "mydialect"
    routes = (
        Route("POST", "/v1/mydialect/chat"),                       # runs the pipeline
        Route("GET", "/v1/mydialect/models", operation="models", body="none"),
    )
    def credential(self, headers, query) -> str | None            # default: bearer token
    def decode(self, call) -> ChatCompletionRequest               # required
    def encode(self, response: dict, call) -> body | Reply        # required
    def encode_events(self, events, call) -> AsyncIterator[bytes] # required to stream
    def stream_media_type(self, call) -> str                      # default text/event-stream
    def encode_error(self, status, error: dict, call) -> body | Reply
    def encode_stream_error(self, error: dict, call) -> bytes     # default b""
    async def handle(self, call) -> body | Reply                  # custom routes
```

- `Route(method, path, operation="chat", body="json")`. The path is a
  Starlette path (`{name}`, `{name:path}`); path params are in
  `call.path_params`. `operation="chat"` runs the pipeline; any other
  operation is answered by `handle` (listing models, counting tokens, a
  version probe), after authentication. Two adapters claiming the same
  method and path stop startup.
- `Call` (`call`): `body` (parsed JSON), `headers`, `query_params`,
  `path_params`, `principal`, `controller`, `request` (the decoded chat
  request, after `decode`), `route`, and `state`, a dict that is yours to
  carry things from `decode` to the encoders (the model name the client
  used, a mode chosen by the path). `await call.models()` returns the
  `/v1/models` entries.
- `decode` sets `stream` when the dialect asks for one (a body flag, or the
  route). For streams the framework adds `stream_options.include_usage` so
  usage arrives.
- `encode` gets the OpenAI completion dict, plus `routellm: {path: [...]}`.
- `encode_error(status, error, call)`: `error` is OpenAI-style and always
  has `message`, `type`, `code`, `param`, `request_id` (None when unknown).
  `type` is `invalid_request_error`, `authentication_error`,
  `permission_error`, `not_found_error`, `rate_limit_error` or
  `server_error`: map it to the dialect's error type. The HTTP status is
  kept. It is used for every failure: auth, refusals, routing errors, policy
  denials, upstream failures, and `Retry-After` is preserved.
- Return a bare body (JSON-serializable) or `Reply(body, headers=, status=,
  media_type=)` when you need a header (`x-amzn-ErrorType`), a non-JSON body
  or another status.
- A failure after the stream started: the cause is logged server side, the
  framework calls `encode_stream_error` with a generic message and the
  request id, then ends the stream without a clean terminator (a truncated
  answer must not look complete). Return `b""` if the wire format has no
  error frame.
- Client disconnects and the stream's audit outcome are handled by the
  framework (`routellm/stream_guard.py`); an adapter never closes
  anything.

## Registration

`@register` on the class instantiates it (no constructor arguments) and
adds it by `name`. At startup `mount_adapters` imports every module of
`routellm/adapters/` and mounts all registered routes, hidden from the
OpenAPI schema. Rules:

- a module that fails to import stops startup, naming the module (a dialect
  that silently vanishes is worse than one that refuses to start);
- adapters import only the standard library, pydantic and `routellm`; never
  `routellm.openai_server` (it parses `sys.argv` at import);
- `ROUTELLM_ADAPTER_MODULES=pkg.mod,other.mod` imports extra modules, for
  adapters kept outside this package (the harness uses it for its
  reference adapter).

## Streaming: the event helper

Do not parse OpenAI chunks. Implement `encode_events(events, call)` as an
async generator; `routellm/adapters/stream.py` turns the chunk stream into
ordered events:

| Event | Meaning |
|-------|---------|
| `Start(id, model, created)` | first chunk seen |
| `TextDelta(index, text)` | text |
| `ReasoningDelta(index, text)` | `reasoning_content` |
| `ToolCallStart(index, call_index, id, name)` | a call opened |
| `ToolCallDelta(index, call_index, arguments)` | piece of the arguments JSON |
| `ToolCallEnd(index, call_index, id, name, arguments)` | call closed, whole arguments |
| `Finish(index, reason)` | raw OpenAI finish reason |
| `Usage(prompt_tokens, completion_tokens, total_tokens, cached_tokens, raw)` | once, before `End` |
| `Error(message, code)` | error the upstream reported inside a 200 stream |
| `End(text, tool_calls, finish_reason, usage)` | stream over; choice 0 summary |

The helper owns tool-call accumulation by index, closing a call before the
next block (next call, text, finish, or stream end), whole calls arriving
in one chunk, and upstreams that reuse index 0 for each call. Usage comes
after the finish chunk, so a dialect that reports usage in a final event
(`message_delta`, `metadata`) emits it on `Usage` or `End`. `index` is the
choice index; without `n` it is always 0. Override `encode_stream(chunks,
call)` instead only if you need raw chunks.

## Lossless or refused

A field the dialect has and chat completions lacks is either carried by a
documented chat field (`metadata`, `reasoning_effort`, `response_format`,
`tools`, ...) or refused. Never silently dropped: answering a different
question than the client asked is worse than a 400.

- `refuse_unknown(body, supported, ignored=..., where=...)` refuses every
  key that is neither mapped nor documented as having no effect.
- `refuse(message, param=...)` refuses one value (an unsupported block
  type, an unsupported option). It raises `AdapterRefusal`, which becomes
  the dialect's own 400 shape through `encode_error`; nothing reaches the
  pipeline.
- A Pydantic `ValidationError` from building the request is a 400 naming
  the field. Any other exception from `decode` is a generic 400 ("Malformed
  request body", the exception type is logged, never the body).
- List what is accepted-and-ignored and what is refused in the adapter's
  module docstring; it is the dialect's unsupported-field table.

## Tests: the conformance harness

`routellm/tests/adapters/`. A real server runs in a child process (the
server parses `sys.argv` at import) with one fake upstream on loopback
(`routellm/tests/fake_upstream.py`, no network), auth on, discovery on.
Write a `Driver` (the client side of the dialect) and subclass the suite:

```python
from routellm.tests.adapters.conformance import ConformanceSuite, Driver, Http, Outcome

class MyDriver(Driver):
    def build(self, scenario) -> Http: ...        # neutral Scenario -> wire request
    def authorize(self, http, key) -> Http: ...   # put the key where the dialect does
    def parse(self, response, scenario) -> Outcome: ...   # JSON body or whole stream
    def parse_error(self, response) -> str: ...   # message from the dialect error shape
    def unsupported(self) -> Http | None: ...     # a request with a refused field

class TestMyDialect(ConformanceSuite):
    driver = MyDriver()
    skip = {"image_input": "dialect has no image blocks"}
```

Cases (`test_<case>`; `skip` is keyed by `<case>`): `plain_text`,
`system_prompt`, `multi_turn`, `image_input`, `tool_call_response`,
`tool_result_request`, `streaming_text`, `streaming_tool_call`, `usage`,
`streaming_usage`, `decision_path_header`, `upstream_error`,
`unauthorized`, `refuses_unsupported_field`. They assert what the upstream
received (system message, history order, `image_url` data URL, assistant
`tool_calls`, `role: tool` results, the tools list) and what the `Outcome`
read from the answer holds. `KEY` is the harness key; `Scenario`,
`Message`, `ToolUse`, `ToolResult`, `ToolDef` are the neutral inputs.

Add dialect-specific tests beside the suite, using the `harness` fixture:
`harness.base` (server URL), `harness.upstream` (the fake: set `.mode` to
`ok`, `echo`, `tools`, `500`, `die_after`, ...; read `.hits`),
`harness.send(Http)`. An official-SDK test points the SDK's base URL at
`harness.base` and sends `KEY`; skip it when the SDK is not installed
(`pytest.importorskip`). Mark module `pytest.mark.slow`.

The adapter-free logic (a stub pipeline, no server) is covered in
`test_dispatch.py`; copy its approach for unit-testing a mapping.

## Reference adapter

`routellm/tests/adapters/reference_adapter.py` is the smallest complete
adapter: a non-bearer credential, a custom route, NDJSON streaming from
events, a dialect error shape, refusal of unknown fields. It is test-only
and loaded through `ROUTELLM_ADAPTER_MODULES`; copy it as a starting point.
