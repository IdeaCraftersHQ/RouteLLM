# US-0010: OpenAI-compatible server

**Status:** shipped, no-e2e
**Persona:** API consumer

## User goal

As a developer, I want to point any OpenAI client at a routellm
server (instead of api.openai.com) so I can swap routing in without
SDK changes.

## Context

`python -m routellm.openai_server` exposes an OpenAI-compatible REST
surface (chat completions, /v1/models, /health). Clients specify
the routing strategy via the `model` field (e.g.,
`model=router-mf-0.5`). `--routers` registers available routers;
`--config` points at a router config file. `--strong-model` and
`--weak-model` set the default pair.

## Acceptance criteria

- [ ] Server starts and listens on configured port (default 6060)
- [ ] `POST /v1/chat/completions` accepts OpenAI-format requests
- [ ] Response shape matches OpenAI completions schema
- [ ] `model=router-<name>-<threshold>` triggers routed call
- [ ] Unknown router name → 4xx
- [ ] `/health` returns 200 when server healthy
- [ ] Streaming responses supported (`stream: true`)
- [ ] Drop-in for OpenAI Python client (`openai.OpenAI(base_url=…)`)

## Implementation

- Module: `routellm/openai_server.py`

## E2E tests

- planned: `routellm/tests/test_openai_server.py::test_chat_completions_routed`
- planned: `routellm/tests/test_openai_server.py::test_health_endpoint`
- planned: `routellm/tests/test_openai_server.py::test_streaming_completion`

(test file exists but is empty as of 2026-04-30; populating tests
is a follow-up task)

## Related

- US-0001 — cost-threshold router (consumed via model field)
- US-0009 — intent-based routing middleware (mountable in server)
