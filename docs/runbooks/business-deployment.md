# Business deployment — LAN + VPN

Last run: 2026-04-29 (paper)
Author: jadb
Task: T-0105
Slug: RLM-NEW-DEPLOY

## Use case

Single LLM gateway for a business. Office employees on LAN, remote
employees over VPN. Hybrid egress: local Ollama (weak) + cloud
provider (strong, e.g. OpenAI / Anthropic). One bearer token per
employee. Per-tenant cache scoping. Per-request audit row.

Targets:

- one host on the office LAN, reachable from VPN subnet
- TLS-only ingress; no plaintext bearer tokens on wire
- weak path stays on-host (Ollama, loopback) — zero egress for
  cheap / repetitive prompts
- strong path egresses to upstream provider only when router or
  policy says so
- every request audited (who, model, cache hit/miss, cost)

## Status (paper vs shipped)

| Capability                | Story   | Status   | Blocker for deploy?      |
|---------------------------|---------|----------|--------------------------|
| OpenAI-compatible server  | US-0010 | shipped  | no                       |
| Strong/weak routing (`mf`)| US-0001 | shipped  | no                       |
| Exact-match cache         | US-0002 | shipped  | no                       |
| Semantic cache            | US-0003 | shipped  | no                       |
| Resilience (retry/CB)     | US-0004 | shipped  | no                       |
| Load balancing            | US-0005 | shipped  | no                       |
| Per-employee bearer auth  | US-0100 | paper    | YES — single-key today   |
| Audit log                 | US-0101 | paper    | partial — no per-req row |
| Class-aware cache hooks   | US-0102 | paper    | no — defaults work       |

Until US-0100 ships: the gateway accepts any caller holding the
upstream provider key. LAN + VPN deploy is feasible today only with
a shared key (org-level secret), no per-employee attribution. Treat
this runbook's auth section as the target shape; current shape =
shared key + IP allowlist on the proxy.

## Prerequisites

- Bare LAN host (Linux). Static IP on office subnet. Hostname
  resolvable from LAN + VPN (e.g. `gateway.lan`).
- TLS cert + key for the hostname. Internal CA OK; self-signed
  workable but pushes cert pinning to clients (see Gotchas).
- Reverse proxy: caddy (auto-TLS for internal CA) or nginx
  (manual cert files). Pick caddy unless ops standard says nginx.
- Ollama installed on the host. Weak model pulled
  (`ollama pull llama3` or equivalent).
- Upstream provider keys: `OPENAI_API_KEY`, optionally
  `ANTHROPIC_API_KEY`. Stored in routellm config / env, never
  exposed to clients.
- VPN already terminating into the LAN with a known subnet (e.g.
  `10.8.0.0/24`). Routable to the gateway host.
- Python 3.10+ on the host. `pip install routellm`.
- (US-0100 shipped) `routellm token` CLI available.

## Components

| Component               | Bind          | Purpose                          |
|-------------------------|---------------|----------------------------------|
| Reverse proxy           | `0.0.0.0:443` | TLS termination, ingress filter  |
| `routellm.openai_server`| `127.0.0.1:6060` | OpenAI-compat REST router     |
| Ollama                  | `127.0.0.1:11434` | weak local model            |
| Token store (US-0100)   | SQLite (host) | per-employee bearer tokens       |
| Audit log (US-0101)     | SQLite (host) | per-request rows                 |
| Cache (US-0002/US-0003) | SQLite (host) | exact + semantic                 |
| Identity provider       | external/CLI  | issues bearer tokens (US-0100)   |

Routellm + Ollama bind to loopback. Only the proxy is reachable
from the network. Segregates blast radius.

## Network topology

See [topology-v1.mmd](business-deployment/topology-v1.mmd).

LAN clients hit `https://gateway.lan/`; VPN clients hit the same
hostname over the VPN tunnel; proxy forwards to loopback router;
router fans out to Ollama (loopback) or cloud (egress). Token
store, audit, cache stay on-host SQLite.

## Steps

### 1. Install on bare LAN host

```
# OS deps
sudo apt-get install -y python3-venv ca-certificates curl

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
ollama pull llama3

# routellm
python3 -m venv /opt/routellm/venv
source /opt/routellm/venv/bin/activate
pip install --upgrade pip
pip install routellm
```

Place `config.yaml` at `/opt/routellm/config.yaml` (see step 5).

### 2. Set up TLS

Caddy (recommended for internal CA / Let's Encrypt internal):

```
# /etc/caddy/Caddyfile
gateway.lan {
    tls /etc/ssl/gateway.lan.crt /etc/ssl/gateway.lan.key
    encode gzip
    reverse_proxy 127.0.0.1:6060
}
```

nginx alternative:

```
# /etc/nginx/conf.d/gateway.conf
server {
    listen 443 ssl http2;
    server_name gateway.lan;
    ssl_certificate     /etc/ssl/gateway.lan.crt;
    ssl_certificate_key /etc/ssl/gateway.lan.key;
    location / {
        proxy_pass http://127.0.0.1:6060;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_read_timeout 600s;   # streaming completions
        proxy_buffering off;        # SSE / chunked
    }
}
```

Reload proxy. `curl -v https://gateway.lan/health` from the host.

### 3. Configure reverse-proxy ingress rules

LAN + VPN allow-list. Example for nginx (caddy has `@allowed`
matchers):

```
allow 10.0.0.0/24;     # LAN
allow 10.8.0.0/24;     # VPN
deny all;
```

Belt + braces with a host firewall (ufw / firewalld) restricting
`:443` to LAN+VPN ranges. Block `:6060` and `:11434` at the host
firewall (loopback-only is enforced by bind, not just firewall).

### 4. Set up Ollama (weak path)

Already running from step 1. Verify:

```
curl http://127.0.0.1:11434/api/tags
ollama list
```

Pin model name in routellm config (next step) as
`ollama_chat/<model>`.

### 5. Configure routellm

`/opt/routellm/config.yaml` — minimum viable strong+weak pair:

```yaml
mf:
  checkpoint_path: routellm/mf_gpt4_augmented

# Cache (US-0002 / US-0003)
cache:
  backend: sqlite
  path: /var/lib/routellm/cache.db
  exact: true
  semantic: true
  ttl_seconds: 86400

# Resilience (US-0004)
resilience:
  retry: { max_attempts: 3, backoff_ms: 250 }
  circuit_breaker: { failure_threshold: 5, reset_seconds: 60 }
  fallback_to_weak_on_strong_failure: true
  request_timeout_seconds: 60

# Auth (US-0100 — paper; section enforced once shipped)
auth:
  enabled: false       # flip to true post-US-0100
  store: sqlite
  store_path: /var/lib/routellm/tokens.db

# Audit (US-0101 — paper; ditto)
audit:
  enabled: false
  store: sqlite
  store_path: /var/lib/routellm/audit.db
```

Strong/weak pair via CLI flags (or env):

```
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...     # optional

python -m routellm.openai_server \
  --routers mf \
  --strong-model gpt-4-1106-preview \
  --weak-model  ollama_chat/llama3 \
  --config /opt/routellm/config.yaml \
  --host 127.0.0.1 --port 6060
```

Wrap as a systemd unit (`/etc/systemd/system/routellm.service`)
binding to loopback, restart-on-failure, env-file for secrets.

### 6. Enable per-employee auth — US-0100 (paper, flagged)

Once shipped:

```
routellm token issue --tenant acme --user alice --expires 90d
routellm token issue --tenant acme --user bob   --expires 90d
routellm token list  --tenant acme
routellm token revoke <token>
```

Flip `auth.enabled: true` in config; restart server. Distribute
tokens to employees out-of-band (1Password, password manager,
encrypted email). Never embed in shared scripts.

Until then: place a single shared bearer behind the proxy
(proxy-enforced `Authorization` header check) as a stopgap. This
gives no per-employee attribution and is NOT a substitute for
US-0100; flag as tech debt.

### 7. Enable audit log — US-0101 (paper, flagged)

Once shipped: flip `audit.enabled: true`; restart. Verify rows
land:

```
routellm audit query --tenant acme --since 1h
routellm audit spend --tenant acme --since 7d
```

Until then: rely on proxy access logs (no model / cache / cost
fields). Spend reporting impossible without US-0101.

### 8. (Optional) Class-aware cache — US-0102 (paper)

Once shipped: install `extensions/policy_routing/`, point
`cache_policy` hook at the c12n classifier. Per scenario 3 spec
table: PII bypass, code exact-match, math global, etc. Default
config (no hook) = today's behavior — safe to deploy without it.

### 9. VPN ingress

VPN concentrator routes `10.8.0.0/24` → office subnet. Add
`10.8.0.0/24` to proxy allow-list (step 3) and host firewall.
DNS: ensure `gateway.lan` resolves over VPN — split-horizon DNS
or VPN-pushed search domain.

### 10. Smoke test

LAN client:

```
curl -sS https://gateway.lan/health
curl -sS https://gateway.lan/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router-mf-0.11593",
    "messages": [{"role":"user","content":"hello"}]
  }'
```

VPN client (same hostname; tunnel up):

```
curl -sS https://gateway.lan/v1/chat/completions ...
```

OpenAI Python client (drop-in, US-0010):

```python
from openai import OpenAI
c = OpenAI(base_url="https://gateway.lan/v1", api_key=TOKEN)
c.chat.completions.create(model="router-mf-0.11593",
                          messages=[{"role":"user","content":"hi"}])
```

## Verification

| Check                                  | Expected                              |
|----------------------------------------|---------------------------------------|
| `GET /health` from LAN                 | 200                                   |
| `GET /health` from VPN                 | 200                                   |
| `GET /health` from outside LAN+VPN     | refused (proxy 403 / connect timeout) |
| `POST /v1/chat/completions` no bearer  | 401 (post-US-0100); else upstream     |
| Same prompt twice, identical token     | 2nd → cache hit, sub-100ms (US-0002)  |
| Weak-class prompt ("2+2")              | served by Ollama; no cloud egress     |
| Strong-class prompt (long reasoning)   | egress to OpenAI/Anthropic            |
| Audit row per request (post-US-0101)   | `routellm audit query` returns 1 row  |
| Cross-tenant cache leak (post-US-0100) | tenant B's prompt → miss, not A's hit |

Egress check: `tcpdump -i <wan-iface> host api.openai.com` while
firing weak-class prompts; expect zero packets.

## Gotchas

- **Cert pinning**: internal-CA certs require pushing the CA root
  to every client (LAN + VPN laptops, IDE plugins, CI runners).
  Self-signed → `verify=False` everywhere → token leak risk.
  Strongly prefer internal CA (smallstep, step-ca) over
  per-host self-signed.
- **Cookie-bridge / browser clients**: browser-side OpenAI clients
  need CORS on the proxy. Add `Access-Control-Allow-Origin` for
  known LAN dev origins; do NOT use `*` with credentialed
  bearers.
- **Streaming responses**: SSE / chunked. nginx default
  `proxy_buffering on` breaks streaming → set `off`. caddy
  handles by default.
- **Key rotation (upstream)**: rotate `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` quarterly. Reload server (systemd
  reload) — no client restart needed; tokens are gateway-scoped
  (post-US-0100).
- **Token rotation (per-employee, post-US-0100)**: revoke before
  reissue. Race: in-flight requests with old token complete;
  next request 401s. Document offboarding steps in HR runbook.
- **Quota by tenant (post-US-0100)**: not in US-0100 scope.
  Coarse rate-limit at proxy level today (`limit_req` /
  `rate_limit` per source IP). Real per-tenant quota waits on
  follow-up story.
- **Cache cross-tenant leak (post-US-0100)**: default namespace
  must be `tenant_id`. Verify before opening to second tenant —
  single-tenant deploys mask the bug. US-0102 hooks let
  classified-safe prompts opt into a global namespace; do not
  flip global on by default.
- **Loopback bind**: confirm `ss -ltnp | grep -E ':(6060|11434)'`
  shows `127.0.0.1`, not `0.0.0.0`. A bind misconfig exposes the
  router unauthenticated to the LAN.
- **VPN MTU**: VPN tunnels often drop MTU to 1380 or lower.
  Streaming completions can hang at the boundary if proxy /
  upstream renegotiate large TLS records. Set `proxy-bufsize`
  conservatively.
- **PII in prompts (pre-US-0102)**: without class-aware policy,
  PII-bearing prompts hit the cache and may egress to cloud.
  Communicate this to employees; consider regex-based prefilter
  at the proxy as stopgap.
- **Audit retention**: SQLite grows unbounded. Add a rotation
  cron (`routellm audit prune --before <ts>`) once US-0101 ships.

## References (stories)

- [US-0001](../stories/US-0001-strong-weak-routing.md) —
  strong/weak routing (`mf`)
- [US-0002](../stories/US-0002-exact-match-cache.md) — exact cache
- [US-0003](../stories/US-0003-semantic-cache.md) — semantic cache
- [US-0004](../stories/US-0004-resilience.md) — retry/CB/fallback
- [US-0005](../stories/US-0005-load-balancing.md) — LB
- [US-0010](../stories/US-0010-openai-compatible-server.md) —
  OpenAI-compat server (entry point)
- [US-0100](../stories/US-0100-multi-tenant-auth.md) —
  per-employee auth (paper; deploy blocker for attribution)
- [US-0101](../stories/US-0101-audit-log.md) — audit log
  (paper; deploy blocker for spend reports)
- [US-0102](../stories/US-0102-cache-policy-hooks.md) —
  class-aware cache hooks (paper; non-blocking)

Showcase context:
`~/.ops/.tlc/tracks/tools-showcase-scenarios/scenarios/3-class-aware-llm-gateway.md`
