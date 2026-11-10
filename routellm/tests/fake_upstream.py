"""A fake OpenAI-compatible upstream on 127.0.0.1, for server tests.

It records every request it receives and answers according to `mode`,
which is either a string or a callable taking the parsed body:

- `ok`: a normal answer, streamed when the body says `stream`.
- `tools`: the answer is a tool call (streamed as deltas when asked).
- `500`: HTTP 500 before any body.
- `die_first`: HTTP 200, then the connection drops before any chunk.
- `die_after`: one content chunk, then the connection drops mid-body.

Nothing here leaves the loopback interface.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeUpstream:
    def __init__(self, name, mode="ok"):
        self.name = name
        self.mode = mode
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
                model = body.get("model", "m")

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
                    else:
                        self._event(
                            chunk({"role": "assistant", "content": f"hello from {fake.name}"})
                        )
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

                message = {"role": "assistant", "content": f"hello from {fake.name}"}
                finish = "stop"
                if mode == "tools":
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
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
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
