"""A fake OpenAI-compatible `/v1/audio/transcriptions` service on 127.0.0.1.

It records every request (path, headers, form fields, uploaded file) and
answers according to `mode`:

- `ok`: a `verbose_json` answer built by `answer(fields, filename, audio)`;
  by default two segments from two speakers whose text is the uploaded
  bytes, so different audio gives a different transcript.
- `500`: HTTP 500 with a body that names a secret, to prove it never
  reaches a client.
- `garbage`: HTTP 200 with a body that is not JSON.

`delay` holds the answer back, for timeout tests. Nothing here leaves
the loopback interface.
"""

import json
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LEAK = "UPSTREAM-SECRET-TRANSCRIBER-DETAIL"


def default_answer(fields, filename, audio):
    text = audio.decode(errors="replace")
    return {
        "task": "transcribe",
        "language": fields.get("language", "en"),
        "duration": 6.5,
        "text": f"{text} one {text} two",
        "segments": [
            {"id": 0, "start": 1.2, "end": 4.8, "text": f" {text} one", "speaker": "SPEAKER_00"},
            {"id": 1, "start": 5.0, "end": 6.5, "text": f"{text} two ", "speaker": "SPEAKER_01"},
        ],
    }


class FakeTranscriber:
    def __init__(self, mode="ok", answer=default_answer):
        self.mode = mode
        self.answer = answer
        self.delay = 0.0
        self.media = b"fetched"  # served at GET /media/<name>
        self.gets = []
        self.hits = []  # dicts: path, headers, fields, filename, audio
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, data, ctype="application/json"):
                self.send_response(status)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.gets.append(self.path)
                self._send(200, fake.media, "audio/wav")

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                head = f"Content-Type: {self.headers['content-type']}\r\n\r\n".encode()
                message = BytesParser(policy=HTTP).parsebytes(head + raw)
                fields, filename, audio = {}, None, b""
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        filename, audio = part.get_filename(), part.get_payload(decode=True)
                    else:
                        fields.setdefault(name, []).append(part.get_content())
                fields = {k: v[0] if len(v) == 1 else v for k, v in fields.items()}
                fake.hits.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "fields": fields,
                        "filename": filename,
                        "audio": audio,
                    }
                )
                if fake.delay:
                    time.sleep(fake.delay)
                if fake.mode == "500":
                    self._send(500, json.dumps({"error": LEAK}).encode())
                elif fake.mode == "garbage":
                    self._send(200, b"<html>not json</html>", "text/html")
                else:
                    self._send(200, json.dumps(fake.answer(fields, filename, audio)).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
