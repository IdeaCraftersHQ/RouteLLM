"""The free-roster report, against a local fake models endpoint."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from routellm import roster


@pytest.fixture
def models_url():
    payload = {
        "data": [
            {"id": "vendor/a:free", "expiration_date": None},
            {"id": "vendor/b:free", "expiration_date": "2026-12-01"},
            {"id": "vendor/new:free"},
            {"id": "vendor/paid"},
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/models"
    server.shutdown()
    server.server_close()


def _config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "endpoints:\n"
        "  a: {model: 'openrouter/vendor/a:free'}\n"
        "  b: {model: 'openrouter/vendor/b:free'}\n"
        "  gone: {model: 'openrouter/vendor/gone:free'}\n"
        "  paid: {model: 'openrouter/vendor/paid'}\n"
        "  local: {model: 'ollama_chat/q'}\n"
    )
    return str(path)


def test_report_lists_new_gone_and_expiring(models_url, tmp_path, capsys):
    code = roster.main(
        ["--config", _config(tmp_path), "--url", models_url, "--prefix", "openrouter/"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "not in the config" in out and "vendor/new:free" in out
    assert "no longer listed" in out and "gone" in out
    assert "expires" in out and "vendor/b:free" in out and "2026-12-01" in out
    assert "vendor/paid" not in out.split("not in the config")[1].split("no longer")[0]


def test_unreachable_endpoint_is_an_error_exit(tmp_path, capsys):
    code = roster.main(
        ["--config", _config(tmp_path), "--url", "http://127.0.0.1:9/models", "--prefix", "or/"]
    )
    assert code == 1
    assert capsys.readouterr().err
