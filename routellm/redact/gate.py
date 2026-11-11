"""The harness gate: a report stamped with the recognizer config digest.

`python -m routellm.redact.harness` measures the analyzers a `redact:`
section describes and writes a JSON report stamped with `config_digest`.
A policy that lets a redact class reach the cloud loads only when
`redact.harness_report` names a report whose stamp is the digest of the
config now in effect and whose results all pass (`verify_report`).

The digest covers what changes what the recognizers find: the engine, the
languages, the client-id patterns, the spaCy models and their installed
versions (presidio engine), the presidio and spaCy versions, the routellm
version and a hash of the recognizer sources. Anything else in `redact:`
(store, `non_text`, `harness_report`) is not an input to recall and is not
in it.
"""

import hashlib
import json
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from routellm.config import ConfigError

__all__ = [
    "REPORT_SCHEMA",
    "REQUIRED_CHECKS",
    "Gate",
    "config_digest",
    "digest_material",
    "verify_report",
]

REPORT_SCHEMA = 1
#: Checks a passing report must carry, each with `passed: true`.
REQUIRED_CHECKS = ("leak", "reidentification", "linkability")
_SOURCES = ("recognizers.py", "presidio_adapter.py")


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "absent"


def _sources_hash() -> str:
    h = hashlib.sha256()
    for name in _SOURCES:
        h.update(name.encode())
        h.update((Path(__file__).with_name(name)).read_bytes())
    return h.hexdigest()


def digest_material(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """The recognizer-relevant facts of a `redact:` section, canonicalised."""
    raw = raw or {}
    engine = raw.get("engine", "presidio")
    languages = sorted(raw.get("languages") or ["fr", "en"])
    material: dict[str, Any] = {
        "engine": engine,
        "languages": languages,
        "client_id_patterns": sorted(raw.get("client_id_patterns") or []),
        "models": {},
        "routellm": _version("routellm"),
        "recognizers": _sources_hash(),
    }
    if engine == "presidio":
        from routellm.redact.presidio_adapter import DEFAULT_MODELS

        names = {**DEFAULT_MODELS, **(raw.get("models") or {})}
        material["models"] = {
            lang: {"name": names[lang], "version": _version(names[lang].replace("_", "-"))}
            for lang in languages
            if lang in names
        }
        material["presidio-analyzer"] = _version("presidio-analyzer")
        material["spacy"] = _version("spacy")
    return material


def config_digest(raw: Mapping[str, Any] | None) -> str:
    """`sha256:<hex>` over `digest_material(raw)`."""
    body = json.dumps(digest_material(raw), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


class Gate:
    """What a policy enabling redact + cloud asks of the `redact:` section it runs on."""

    def __init__(self, raw: Mapping[str, Any], config_path: str | None = None):
        self._raw = dict(raw)
        report = self._raw.pop("harness_report", None)
        if report is not None and (not isinstance(report, str) or not report):
            raise ConfigError("redact.harness_report must be a path")
        if report and not Path(report).expanduser().is_absolute() and config_path:
            report = str(Path(config_path).resolve().parent / Path(report).expanduser())
        self.report = report

    def check(self) -> None:
        verify_report(self.report, self._raw)


def verify_report(path: str | None, raw: Mapping[str, Any] | None) -> None:
    """Raise `ConfigError` unless `path` holds a passing report for `raw`.

    Checked, in order: a path is configured and readable JSON; the schema
    is known; the stamped digest is the current one; every recall row and
    every required check passed (the report's own `passed` flag is not
    trusted alone).
    """
    why = "redact + cloud needs a passing harness report"
    if not path:
        raise ConfigError(f"{why}: set `redact.harness_report`")
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{why}: cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"{why}: {path} is not JSON: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise ConfigError(f"{why}: {path} is not a schema {REPORT_SCHEMA} report")
    current = config_digest(raw)
    if report.get("digest") != current:
        raise ConfigError(
            f"{why}: {path} is stamped {report.get('digest')}, the recognizer config "
            f"is {current}; rerun `python -m routellm.redact.harness`"
        )
    failed = [
        name
        for name, row in (report.get("categories") or {}).items()
        if not isinstance(row, dict) or row.get("passed") is not True
    ]
    if not report.get("categories"):
        failed.append("categories")
    checks = report.get("checks") or {}
    failed += [
        name
        for name in REQUIRED_CHECKS
        if not isinstance(checks.get(name), dict) or checks[name].get("passed") is not True
    ]
    if report.get("passed") is not True or failed:
        raise ConfigError(f"{why}: {path} failed ({', '.join(failed) or 'passed flag false'})")
