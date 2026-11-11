"""A stamped harness report written without running the harness, for tests of the gate."""

import json
from pathlib import Path

from routellm.redact.gate import REPORT_SCHEMA, REQUIRED_CHECKS, config_digest


def write_report(raw: dict, path: Path, *, passed: bool = True, **over) -> str:
    """Write a report for `raw` (a `redact:` section) to `path`; `over` replaces fields."""
    raw = {k: v for k, v in raw.items() if k != "harness_report"}
    report = {
        "schema": REPORT_SCHEMA,
        "digest": config_digest(raw),
        "passed": passed,
        "categories": {"sin": {"passed": passed}},
        "checks": {name: {"passed": passed} for name in REQUIRED_CHECKS},
        **over,
    }
    Path(path).write_text(json.dumps(report))
    return str(path)
