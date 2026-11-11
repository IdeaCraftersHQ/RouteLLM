"""Trusted-proxy mode through the real app: policy, audit and cache see the client."""

import json

import pytest

from routellm.tests.test_audit_server import _run, sha

PROXY, PLAIN = "sk-proxy-0001", "sk-plain-0002"
HDR = "X-Route-Client"

KEYS = f"""
keys:
  - key_sha256: "{sha(PROXY)}"
    principal: proxy
    client_header: {HDR}
    clients:
      web: {{}}
      batch: {{}}
  - key_sha256: "{sha(PLAIN)}"
    principal: plain
"""

# `proxy/web` may send `open` and `internal`; `proxy/batch` only `open`;
# the bare entry and `plain` nothing the policy lists.
POLICY = {
    "principals": {
        "proxy/web": {"classes": ["open", "internal"]},
        "proxy/batch": {"classes": ["open"]},
        "proxy": {"classes": ["open"]},
        "plain": {"classes": ["open"]},
    }
}


def run(tmp_path, body):
    return _run(tmp_path, body, keys_text=KEYS, policy_extra=POLICY)


@pytest.mark.slow
def test_policy_audit_and_cache_follow_the_resolved_principal(tmp_path):
    out = run(
        tmp_path,
        f"""
def post(key, cls, **hdr):
    h = H(key, **{{"X-Data-Class": cls}}, **hdr)
    return client.post("/v1/chat/completions", json=BODY, headers=h).status_code
sent = {{
  "web_internal": post("{PROXY}", "internal", **{{"{HDR}": "web"}}),
  "batch_internal": post("{PROXY}", "internal", **{{"{HDR}": "batch"}}),
  "bare_internal": post("{PROXY}", "internal"),
  "web_open_1": post("{PROXY}", "open", **{{"{HDR}": "web"}}),
  "web_open_2": post("{PROXY}", "open", **{{"{HDR}": "web"}}),
  "batch_open": post("{PROXY}", "open", **{{"{HDR}": "batch"}}),
  "bare_open": post("{PROXY}", "open"),
  "plain_spoof_internal": post("{PLAIN}", "internal", **{{"{HDR}": "web"}}),
  "plain_spoof_open": post("{PLAIN}", "open", **{{"{HDR}": "web"}}),
}}
emit({{"sent": sent, "events": events()}})
""",
    )
    assert out["sent"] == {
        "web_internal": 200,
        "batch_internal": 403,  # policy principals.proxy/batch lacks `internal`
        "bare_internal": 403,
        "web_open_1": 200,
        "web_open_2": 200,
        "batch_open": 200,
        "bare_open": 200,
        "plain_spoof_internal": 403,  # header ignored: `plain` is not proxy/web
        "plain_spoof_open": 200,
    }
    served = [e for e in out["events"] if e["outcome"] == "served"]
    assert [e["principal"] for e in served] == [
        "proxy/web",
        "proxy/web",
        "proxy/web",
        "proxy/batch",
        "proxy",
        "plain",
    ]
    # Cache isolation per resolved principal: web's second call hits, and
    # batch / bare / plain do not read web's entry.
    caches = [(e["principal"], e["cache"]) for e in served if e["data_class"] == "open"]
    assert caches == [
        ("proxy/web", "miss"),
        ("proxy/web", "hit"),
        ("proxy/batch", "miss"),
        ("proxy", "miss"),
        ("plain", "miss"),
    ]
    refused = [e for e in out["events"] if e["outcome"] == "refused"]
    assert sorted(e["principal"] for e in refused) == ["plain", "proxy", "proxy/batch"]


@pytest.mark.slow
def test_unlisted_client_is_403_with_an_audit_event_and_no_echo(tmp_path):
    out = run(
        tmp_path,
        f"""
big = "Z" * 5000
r = client.post("/v1/chat/completions", json=BODY,
                headers=H("{PROXY}", **{{"X-Data-Class": "open", "{HDR}": big}}))
m = client.get("/v1/models", headers=H("{PROXY}", **{{"{HDR}": "ghost"}}))
bad = client.post("/v1/chat/completions", json=BODY, headers=H("sk-nope", **{{"{HDR}": "web"}}))
emit({{"s": [r.status_code, m.status_code, bad.status_code], "text": r.text, "events": events(),
      "hits": len(A.hits + B.hits)}})
""",
    )
    assert out["s"] == [403, 403, 401]
    assert "Z" * 20 not in out["text"] and "X-Route-Client" in out["text"]
    assert json.loads(out["text"])["object"] == "error"
    assert out["hits"] == 0
    reasons = [(e["status"], e["reason"]) for e in out["events"]]
    assert (403, "client_not_listed") in reasons and (401, "unauthorized") in reasons


@pytest.mark.slow
def test_client_header_is_not_forwarded_upstream(tmp_path):
    out = run(
        tmp_path,
        f"""
r = client.post("/v1/chat/completions", json=BODY,
                headers=H("{PROXY}", **{{"X-Data-Class": "open", "{HDR}": "web"}}))
sent = [json.dumps({{k.lower(): v for k, v in h.items()}}) for _, h, _ in A.hits + B.hits]
emit({{"status": r.status_code, "n": len(sent), "sent": sent}})
""",
    )
    assert out["status"] == 200 and out["n"] == 1
    blob = out["sent"][0].lower()
    assert "x-route-client" not in blob and "proxy/web" not in blob and PROXY not in blob
