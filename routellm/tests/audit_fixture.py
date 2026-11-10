"""An `audit:` config section for server tests that load a policy."""

import os
from pathlib import Path

HMAC_KEY = "audit-hmac-key-0123456789"


def audit_section(tmp_path: Path) -> dict:
    """Return the `audit:` section: a file sink and a 0600 HMAC key file under `tmp_path`."""
    key = tmp_path / "audit.key"
    key.write_text(HMAC_KEY)
    os.chmod(key, 0o600)
    return {
        "sink": "file",
        "path": str(tmp_path / "audit" / "audit.jsonl"),
        "hmac_key_file": str(key),
    }
