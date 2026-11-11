"""The `redact:` config section, built into a `RedactTransform`.

```yaml
redact:
  engine: presidio            # presidio (default; needs the `redact` extra) | patterns
  languages: [fr, en]         # analysed on every text, spans merged
  client_id_patterns: ['ACME-\\d{5}']
  models: {fr: fr_core_news_md, en: en_core_web_md}
  non_text: refuse            # refuse (default) | pass: parts that are not text
  store:
    path: state/tokens.db     # default: $XDG_STATE_HOME/routellm/redact/tokens.db
    ttl_s: 86400
    encryption: none          # none | aes-gcm (see routellm.redact.cipher)
    key_file: /etc/routellm/redact.key
  harness_report: redact-report.json   # required to let a redact class reach the cloud
```

Everything is checked at startup; nothing is left for the first request
to discover. With `engine: presidio` the analysis engine and its models
are loaded here, so a missing extra or model stops the server rather than
refusing every request later.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from routellm.config import ConfigError
from routellm.redact.cipher import build_cipher
from routellm.redact.gate import Gate
from routellm.redact.placeholders import Redactor
from routellm.redact.tokens import DEFAULT_TTL_S, TokenStore
from routellm.redact.transform import RedactTransform, union_languages

__all__ = ["build_redact_transform"]

_KEYS = {
    "engine",
    "languages",
    "client_id_patterns",
    "models",
    "non_text",
    "store",
    "harness_report",
}
_STORE_KEYS = {"path", "ttl_s", "encryption", "key_file", "key_env"}
_LANGUAGES = ("fr", "en")


def _store_path(raw: Any, resolve) -> Path:
    if raw is None:
        state = os.environ.get("XDG_STATE_HOME") or os.path.join(Path.home(), ".local", "state")
        return Path(state) / "routellm" / "redact" / "tokens.db"
    if not isinstance(raw, str) or not raw:
        raise ConfigError("redact.store.path must be a path")
    return resolve("redact.store.path", raw)


def _analyzer(raw: Mapping[str, Any], languages: tuple[str, ...]):
    patterns = raw.get("client_id_patterns") or []
    if not isinstance(patterns, list):
        raise ConfigError("redact.client_id_patterns must be a list of regular expressions")
    engine = raw.get("engine", "presidio")
    if engine == "patterns":
        from routellm.redact import analyze_patterns, compile_client_id_patterns

        try:
            compiled = compile_client_id_patterns(patterns)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return union_languages(
            lambda text, language: [
                s.to_dict() for s in analyze_patterns(text, language, compiled)
            ],
            languages,
        )
    if engine != "presidio":
        raise ConfigError(f"redact.engine must be presidio or patterns, got {engine!r}")
    from routellm import redact

    try:
        redact.compile_client_id_patterns(patterns)  # fail loud on a bad regex
        engine_obj = redact.build_analyzer(
            {"client_id_patterns": patterns, "models": raw.get("models") or {}}
        )
        for language in languages:  # load the models now
            engine_obj.analyze("warm up", language, 0.4)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    except Exception as exc:
        if type(exc).__name__ == "RedactUnavailable":
            raise ConfigError(f"redact.engine presidio is unavailable: {exc}") from exc
        raise
    return union_languages(
        lambda text, language: [s.to_dict() for s in engine_obj.analyze(text, language, 0.4)],
        languages,
    )


def build_redact_transform(
    raw: Mapping[str, Any] | None, config_path: str | None, resolve_path=None
) -> RedactTransform | None:
    """The transform a `redact:` section describes, or None without one.

    Raises
    ------
    ConfigError
        On any unknown or invalid key, an unavailable engine, a store
        that cannot be opened or a cipher that cannot be built.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"`redact:` must be a mapping, got {type(raw).__name__}")
    if "stable_across_requests" in raw:
        raise ConfigError("redact.stable_across_requests is not supported yet")
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ConfigError(f"`redact:` unknown key(s) {', '.join(unknown)}")
    languages = raw.get("languages", list(_LANGUAGES))
    if (
        not isinstance(languages, list)
        or not languages
        or any(lang not in _LANGUAGES for lang in languages)
    ):
        raise ConfigError(f"redact.languages must be a non-empty list of {', '.join(_LANGUAGES)}")
    store = raw.get("store") or {}
    if not isinstance(store, Mapping):
        raise ConfigError("redact.store must be a mapping")
    unknown = sorted(set(store) - _STORE_KEYS)
    if unknown:
        raise ConfigError(f"redact.store: unknown key(s) {', '.join(unknown)}")
    ttl = store.get("ttl_s", DEFAULT_TTL_S)
    if isinstance(ttl, bool) or not isinstance(ttl, int | float) or ttl <= 0:
        raise ConfigError("redact.store.ttl_s must be a positive number of seconds")
    non_text = raw.get("non_text", "refuse")
    if non_text not in ("refuse", "pass"):
        raise ConfigError(f"redact.non_text must be refuse or pass, got {non_text!r}")

    from routellm.config import path_resolver

    resolve = path_resolver(config_path, resolve_path)
    analyze = _analyzer(raw, tuple(languages))
    cipher_cfg = {k: store[k] for k in ("encryption", "key_file", "key_env") if k in store}
    if isinstance(cipher_cfg.get("key_file"), str) and cipher_cfg["key_file"]:
        cipher_cfg["key_file"] = os.fspath(resolve("redact.store.key_file", cipher_cfg["key_file"]))
    cipher = build_cipher(cipher_cfg)
    try:
        token_store = TokenStore(
            os.fspath(_store_path(store.get("path"), resolve)), ttl_s=ttl, cipher=cipher
        )
    except OSError as exc:
        raise ConfigError(f"redact.store: cannot open the token store: {exc}") from exc
    transform = RedactTransform(Redactor(token_store, analyze=analyze), non_text=non_text)
    # Asked by the controller when a policy lets a redact class leave the host.
    transform.harness_gate = Gate(raw, config_path, resolve_path)
    return transform
