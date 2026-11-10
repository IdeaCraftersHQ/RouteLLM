"""A fake Ollama `/api/chat` upstream on 127.0.0.1, for server tests.

litellm's `ollama_chat/` transport posts to `<api_base>/api/chat` and
reads either one JSON object or newline-delimited JSON when streaming.
Like `FakeUpstream`, it records every request as `(path, headers, body)`
in `hits`. Nothing here leaves the loopback interface.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, name="O"):
        self.name = name
        self.hits = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.hits.append((self.path, dict(self.headers), body))
                reply = {
                    "model": body.get("model", "m"),
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": f"hello from {fake.name}"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3,
                    "eval_count": 3,
                }
                if body.get("stream") is False:
                    data = json.dumps(reply).encode()
                else:
                    data = (
                        json.dumps({**reply, "done": False, "done_reason": None}).encode()
                        + b"\n"
                        + json.dumps(
                            {**reply, "message": {"role": "assistant", "content": ""}}
                        ).encode()
                        + b"\n"
                    )
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
