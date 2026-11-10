"""A loopback server that answers `/embeddings` and counts what it is asked."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeEmbeddingServer:
    """OpenAI-shaped embeddings endpoint on 127.0.0.1; `hits` holds each request body."""

    def __init__(self, width: int = 8):
        self.hits: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.hits.append({"path": self.path, **body})
                texts = body.get("input") or [""]
                if isinstance(texts, str):
                    texts = [texts]
                data = [
                    {"object": "embedding", "index": i, "embedding": [1.0] * width}
                    for i, _ in enumerate(texts)
                ]
                out = json.dumps(
                    {
                        "object": "list",
                        "model": body.get("model", "m"),
                        "data": data,
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._server.server_port}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
