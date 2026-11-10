"""Client-path privacy cases: proxy-shaped bodies addressed to `private`.

Agent clients reach the server through a proxy that translates their
Messages or Responses requests into chat completions. These are those
translated bodies, as the proxy sends them, with the model field naming
the private tier.
"""

from routellm.tests.privacy_fixture import IMAGE, TOOLS, case

#: Messages-translated: system as parts, a tool round trip, streamed, usage asked for.
MESSAGES_BODY = {
    "model": "private",
    "max_tokens": 256,
    "stream": True,
    "stream_options": {"include_usage": True},
    "reasoning_effort": "medium",
    "tool_choice": "auto",
    "tools": TOOLS,
    "messages": [
        {"role": "system", "content": [{"type": "text", "text": "You are an agent"}]},
        {"role": "user", "content": [{"type": "text", "text": "read it"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"p":"a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "file text"},
    ],
}

#: Responses-translated: no content on tool-call turns, reasoning content, parallel calls.
RESPONSES_BODY = {
    "model": "private",
    "stream": True,
    "parallel_tool_calls": True,
    "reasoning_effort": "medium",
    "tools": TOOLS,
    "messages": [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "reasoning_content": "thinking",
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_9", "content": "out"},
    ],
}

IMAGE_BODY = {"model": "private", "max_tokens": 64, "messages": IMAGE}


def _send(w, body):
    if body.get("stream"):
        with w.client.stream("POST", "/v1/chat/completions", json=body) as r:
            events = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
            return {"status": r.status_code, "done": bool(events) and events[-1] == "data: [DONE]"}
    return {"status": w.client.post("/v1/chat/completions", json=body).status_code}


@case("client_messages", 200, {"local_main": 1})
def client_messages(w):
    return _send(w, MESSAGES_BODY)


@case("client_responses", 200, {"local_main": 1})
def client_responses(w):
    return _send(w, RESPONSES_BODY)


@case("client_image", 200, {"local_main": 1})
def client_image(w):
    return _send(w, IMAGE_BODY)


@case("client_messages_engines_down", None, {"local_main": 1, "local_small": 1})
def client_messages_down(w):
    w.fail()
    return _send(w, MESSAGES_BODY)


@case("client_responses_engines_down", None, {"local_main": 1, "local_small": 1})
def client_responses_down(w):
    w.fail()
    return _send(w, RESPONSES_BODY)
