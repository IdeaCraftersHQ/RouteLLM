"""A fake video preprocessor on 127.0.0.1 implementing the HTTP contract.

`POST /v1/video/preprocess` takes multipart `file` plus option fields and
answers per `mode`:

- `ok`: two scenes, frames (or a contact sheet), a transcript whose text
  is `TRANSCRIPT` plus the uploaded bytes, and optional OCR text.
- `500` / `502_code`: a 5xx whose message names a secret (`LEAK`);
  `400`: a 4xx with code `ocr_unavailable`; `401`: bad key.
- `garbage`: 200 with a body that is not JSON; `remote`: a frame that
  is an http URL, which a client must refuse to forward.

Every request is recorded in `hits`. `delay` holds the answer back.
Nothing here leaves the loopback interface.
"""

import json
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LEAK = "UPSTREAM-SECRET-VIDEO-DETAIL"
FRAME = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
SHEET = "data:image/jpeg;base64,/9j/SHEETSHEET=="


def default_answer(fields, filename, video, transcript="said:"):
    sheet = fields.get("layout") == "contact_sheet"
    frame = lambda t: {"t_s": t, **({} if sheet else {"image": FRAME + f"{t}"})}  # noqa: E731
    text = video.decode(errors="replace")
    no_stt = fields.get("transcribe") == "false"
    return {
        "duration_s": 93.4,
        "width": 640,
        "height": 360,
        "fps": 25.0,
        "has_audio": True,
        "scenes": [
            {
                "index": 0,
                "start_s": 0.0,
                "end_s": 12.3,
                "frames": [frame(1.2), frame(6.0)],
                "ocr_text": "OCR-ONE" if fields.get("ocr") == "true" else "",
            },
            {"index": 1, "start_s": 12.3, "end_s": 93.4, "frames": [frame(40.0)]},
        ],
        "contact_sheet": SHEET if sheet else None,
        "transcript": None
        if no_stt
        else {
            "text": f"{transcript} {text}",
            "language": fields.get("language", "en"),
            "segments": [
                {
                    "start_s": 0.0,
                    "end_s": 4.1,
                    "speaker": "SPEAKER_00",
                    "text": f"{transcript} {text}",
                },
                {"start_s": 65.0, "end_s": 70.0, "text": "no speaker here"},
            ],
        },
        "warnings": [],
    }


class FakeVideoService:
    def __init__(self, mode="ok", answer=default_answer):
        self.mode = mode
        self.answer = answer
        self.delay = 0.0
        self.transcript = "said:"
        self.hits = []  # dicts: path, headers, fields, filename, video
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

            def _error(self, status, code, message):
                body = {"error": {"code": code, "message": message}}
                self._send(status, json.dumps(body).encode())

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                head = f"Content-Type: {self.headers['content-type']}\r\n\r\n".encode()
                message = BytesParser(policy=HTTP).parsebytes(head + raw)
                fields, filename, video = {}, None, b""
                for part in message.iter_parts():
                    if part.get_filename():
                        filename, video = part.get_filename(), part.get_payload(decode=True)
                    else:
                        name = part.get_param("name", header="content-disposition")
                        fields[name] = part.get_content()
                fake.hits.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "fields": fields,
                        "filename": filename,
                        "video": video,
                    }
                )
                if fake.delay:
                    time.sleep(fake.delay)
                if fake.mode == "500":
                    self._error(500, "internal", LEAK)
                elif fake.mode == "stt_failed":
                    self._error(502, "stt_failed", LEAK)
                elif fake.mode == "400":
                    self._error(400, "ocr_unavailable", LEAK)
                elif fake.mode == "401":
                    self._error(401, "unauthorized", LEAK)
                elif fake.mode == "garbage":
                    self._send(200, b"<html>no</html>", "text/html")
                else:
                    out = fake.answer(fields, filename, video, fake.transcript)
                    if fake.mode == "remote":
                        out["scenes"][0]["frames"][0]["image"] = "http://127.0.0.1:1/x.jpg"
                    self._send(200, json.dumps(out).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
