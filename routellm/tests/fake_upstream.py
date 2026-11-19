"""A fake OpenAI-compatible upstream on 127.0.0.1, for server tests.

It records every request it receives and answers according to `mode`,
which is either a string or a callable taking the parsed body:

- `ok`: a normal answer, streamed when the body says `stream`.
- `tools`: the answer is a tool call (streamed as deltas when asked).
- `echo`: the answer repeats the last message's text (`echo: <text>`); a
  stream carries it in 4-character pieces, so any placeholder in it is
  split over several chunks.
- `500`: HTTP 500 before any body.
- `die_first`: HTTP 200, then the connection drops before any chunk.
- `die_after`: one content chunk, then the connection drops mid-body.
- `429`: HTTP 429; `Retry-After: <retry_after>` when that is set.
- `429_daily`: HTTP 429 whose message names a daily limit; carries
  `X-RateLimit-Reset: <reset_ms>` (epoch milliseconds) when that is set.
- `stream_error_first` / `stream_error_after`: HTTP 200 stream whose
  first / second event is an in-stream rate-limit error event
  (`finish_reason: error` with an `error` object).

`reply` replaces the default answer text of `ok`.
`delay` holds every answer back, before the first byte.
`headers` adds response headers to every successful answer.

Nothing here leaves the loopback interface.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _last_text(body) -> str:
    """The text of the request's last message, parts joined."""
    content = (body.get("messages") or [{}])[-1].get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class FakeUpstream:
    def __init__(self, name, mode="ok"):
        self.name = name
        self.mode = mode
        self.delay = 0.0  # seconds to wait before answering anything
        self.retry_after = None  # `Retry-After` value on a 429, as sent
        self.reset_ms = None  # `X-RateLimit-Reset` (epoch ms) on a 429_daily
        self.reply = None  # answer text (or callable(body) -> text) in place of the default
        self.headers = {}  # extra response headers on a 200
        self.hits = []  # (path, headers, body)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _chunk(self, data: bytes):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def _event(self, payload):
                self._chunk(f"data: {json.dumps(payload)}\n\n".encode())

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.hits.append((self.path, dict(self.headers), body))
                mode = fake.mode(body) if callable(fake.mode) else fake.mode
                if fake.delay:
                    time.sleep(fake.delay)
                model = body.get("model", "m")
                reply = f"hello from {fake.name}"
                if fake.reply is not None:
                    reply = fake.reply(body) if callable(fake.reply) else fake.reply
                if mode == "echo":
                    reply = "echo: " + _last_text(body)

                if mode in ("429", "429_daily"):
                    daily = mode == "429_daily"
                    message = (
                        "Rate limit exceeded: free-models-per-day"
                        if daily
                        else "Rate limit exceeded"
                    )
                    data = json.dumps({"error": {"message": message, "code": 429}}).encode()
                    self.send_response(429)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    if fake.retry_after is not None and not daily:
                        self.send_header("Retry-After", str(fake.retry_after))
                    if fake.reset_ms is not None and daily:
                        self.send_header("X-RateLimit-Reset", str(fake.reset_ms))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                if mode == "500":
                    data = b'{"error":{"message":"boom"}}'
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("transfer-encoding", "chunked")
                    for key, value in fake.headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    if mode == "die_first":
                        self.close_connection = True
                        return

                    def chunk(delta, finish=None, usage=None, choices=True):
                        out = {
                            "id": "c1",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": model,
                            "choices": (
                                [{"index": 0, "delta": delta, "finish_reason": finish}]
                                if choices
                                else []
                            ),
                        }
                        if usage:
                            out["usage"] = usage
                        return out

                    if mode in ("stream_error_first", "stream_error_after"):
                        error_event = {
                            "id": "c1",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": model,
                            "error": {"code": 429, "message": "Rate limit exceeded"},
                            "choices": [
                                {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
                            ],
                        }
                        if mode == "stream_error_after":
                            self._event(chunk({"role": "assistant", "content": reply}))
                        self._event(error_event)
                        self._chunk(b"data: [DONE]\n\n")
                        self._chunk(b"")
                        return

                    if mode == "tools":
                        call = {"index": 0, "id": "call_1", "type": "function"}
                        self._event(
                            chunk(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {**call, "function": {"name": "read_file", "arguments": ""}}
                                    ],
                                }
                            )
                        )
                        for piece in ('{"p"', ': "a.txt"}'):
                            self._event(
                                chunk(
                                    {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}
                                )
                            )
                        self._event(chunk({}, finish="tool_calls"))
                    elif mode == "echo":
                        self._event(chunk({"role": "assistant", "content": ""}))
                        for i in range(0, len(reply), 4):
                            self._event(chunk({"content": reply[i : i + 4]}))
                        self._event(chunk({}, finish="stop"))
                    else:
                        self._event(chunk({"role": "assistant", "content": reply}))
                        if mode == "die_after":
                            self.close_connection = True
                            return
                        self._event(chunk({}, finish="stop"))

                    if (body.get("stream_options") or {}).get("include_usage"):
                        self._event(
                            chunk(
                                {},
                                usage={
                                    "prompt_tokens": 3,
                                    "completion_tokens": 3,
                                    "total_tokens": 6,
                                },
                                choices=False,
                            )
                        )
                    self._chunk(b"data: [DONE]\n\n")
                    self._chunk(b"")
                    return

                message = {"role": "assistant", "content": reply}
                finish = "stop"
                if mode == "tools":
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"p": "a.txt"}',
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                out = {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
                }
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for key, value in fake.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
