"""The reference adapter, through the full conformance suite.

It is also the contract's own test: auth outside the bearer header, a
custom route, NDJSON streaming built from normalized events, a dialect
error shape and a refusal for a field the chat pipeline lacks.
"""

import json

import httpx
import pytest

from routellm.tests.adapters.conformance import (
    KEY,
    ConformanceSuite,
    Driver,
    Http,
    Outcome,
    Scenario,
    ToolUse,
)

pytestmark = pytest.mark.slow


class ReferenceDriver(Driver):
    def build(self, scenario: Scenario) -> Http:
        messages = []
        if scenario.system:
            messages.append({"role": "system", "content": scenario.system})
        for m in scenario.messages:
            if m.tool_results:
                messages += [
                    {"role": "tool", "tool_call_id": r.id, "content": r.content}
                    for r in m.tool_results
                ]
                continue
            message = {"role": m.role, "content": m.text}
            if m.image_png_b64:
                message["content"] = [
                    {"type": "text", "text": m.text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{m.image_png_b64}"},
                    },
                ]
            if m.tool_uses:
                message["tool_calls"] = [
                    {
                        "id": u.id,
                        "type": "function",
                        "function": {"name": u.name, "arguments": json.dumps(u.arguments)},
                    }
                    for u in m.tool_uses
                ]
            messages.append(message)
        body = {"model": scenario.model, "messages": messages, "stream": scenario.stream}
        if scenario.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in scenario.tools
            ]
        return Http("POST", "/reference/v1/chat", json=body)

    def authorize(self, http: Http, key: str | None) -> Http:
        if key is not None:
            http.headers["x-reference-key"] = key
        return http

    def parse(self, response: httpx.Response, scenario: Scenario) -> Outcome:
        if not scenario.stream:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            return Outcome(
                text=message.get("content") or "",
                tool_uses=[
                    ToolUse(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                    for c in message.get("tool_calls") or []
                ],
                finish=choice["finish_reason"],
                input_tokens=body["usage"]["prompt_tokens"],
                output_tokens=body["usage"]["completion_tokens"],
            )
        out = Outcome()
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1] == {"event": "end"}
        for event in events:
            if event["event"] == "text":
                out.text += event["text"]
            elif event["event"] == "tool_call":
                out.tool_uses.append(
                    ToolUse(event["id"], event["name"], json.loads(event["arguments"]))
                )
            elif event["event"] == "finish":
                out.finish = event["reason"]
            elif event["event"] == "usage":
                out.input_tokens, out.output_tokens = event["input"], event["output"]
        return out

    def parse_error(self, response: httpx.Response) -> str:
        return response.json()["reference_error"]["message"]

    def unsupported(self) -> Http:
        return Http(
            "POST",
            "/reference/v1/chat",
            json={"model": "default", "messages": "hi", "safety_settings": []},
        )


class TestReference(ConformanceSuite):
    driver = ReferenceDriver()


def test_custom_route_lists_models_behind_the_same_auth(harness):
    url = f"{harness.base}/reference/v1/models"
    assert httpx.get(url).status_code == 401
    response = httpx.get(url, headers={"x-reference-key": KEY})
    assert response.status_code == 200
    assert "default" in response.json()["models"]


def test_bearer_token_is_not_this_dialects_credential(harness):
    response = httpx.post(
        f"{harness.base}/reference/v1/chat",
        headers={"authorization": f"Bearer {KEY}"},
        json={"model": "default", "messages": "hi"},
    )
    assert response.status_code == 401
    assert response.json()["reference_error"]["code"] == "unauthorized"


def test_chat_completions_route_is_untouched(harness):
    harness.upstream.mode = "echo"
    response = httpx.post(
        f"{harness.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {KEY}"},
        json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo: hi"
    assert "x-routellm-path" not in response.headers
    assert httpx.post(f"{harness.base}/v1/chat/completions", json={}).json()["object"] == "error"
