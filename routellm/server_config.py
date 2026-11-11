"""Server settings read from the merged config file.

Top-level sections configure what the server builds around the
router: `cache:` (the response cache), `resilience:` (retries,
timeouts, circuit breaker), `embeddings:` (which registry endpoint
embeds prompts) and `transforms:` (request transforms). All are optional. An absent `cache:` means
no cache at all; an absent `resilience:` keeps the `ResilienceConfig`
defaults.

Unknown keys are refused rather than ignored: a mistyped `ttl:` that
silently kept the default would leave the operator believing a limit is
in force.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from routellm.caching import CacheConfig
from routellm.comparison import ComparisonConfig
from routellm.config import ConfigError
from routellm.quality import FineTuneConfig, QualityManager
from routellm.resilience import ResilienceConfig
from routellm.transform import load_transforms

SERVER_SECTIONS = (
    "cache",
    "resilience",
    "comparison",
    "traces",
    "auth",
    "embeddings",
    "transforms",
    "redact",
)


class _StrictCache(CacheConfig):
    model_config = ConfigDict(extra="forbid")


class _StrictTraces(FineTuneConfig):
    model_config = ConfigDict(extra="forbid")


class _StrictResilience(ResilienceConfig):
    model_config = ConfigDict(extra="forbid")


def _section(file_config: dict | None, name: str) -> dict[str, Any]:
    raw = (file_config or {}).get(name)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"`{name}:` must be a mapping, got {type(raw).__name__}")
    return raw


def _validate(model: type[BaseModel], name: str, raw: dict[str, Any]):
    try:
        return model(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<section>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"invalid `{name}:` section: {problems}") from exc


def build_cache_config(
    file_config: dict | None, config_path: str | None, resolve_path=None
) -> CacheConfig:
    """Return the cache settings; disabled unless the config enables it.

    Parameters
    ----------
    file_config : dict, optional
        The merged config.
    config_path : str, optional
        The file the config was read from. A relative `db_path` is
        resolved against its directory, never the process CWD, so a
        server started from elsewhere keeps its cache beside its config.
    """
    raw = dict(_section(file_config, "cache"))
    raw.setdefault("enabled", False)
    config = _validate(_StrictCache, "cache", raw)
    from routellm.config import path_resolver

    path = path_resolver(config_path, resolve_path)("cache.db_path", config.db_path)
    return CacheConfig(**{**config.model_dump(), "db_path": os.fspath(path)})


def build_embedding_endpoint(file_config: dict | None) -> str | None:
    """Return the `embeddings.endpoint` name, else None (the endpoint named `embedding`)."""
    raw = _section(file_config, "embeddings")
    unknown = sorted(set(raw) - {"endpoint"})
    if unknown:
        raise ConfigError(f"invalid `embeddings:` section: unknown key(s) {', '.join(unknown)}")
    name = raw.get("endpoint")
    if name is not None and (not isinstance(name, str) or not name):
        raise ConfigError("`embeddings.endpoint` must name a registry endpoint")
    return name


def build_transforms(
    file_config: dict | None, config_path: str | None = None, resolve_path=None
) -> list:
    """Return the request transforms, in order: the redactor, then `transforms:`.

    The redactor is built from a `redact:` section (see
    `routellm.redact.config`) and runs first, so a later transform sees
    placeholders, never the values. Each `transforms:` entry is
    `module:attr`, an entry point name in the `routellm.transforms`
    group, or a mapping with `use:` and `options:`. Neither present:
    none, and the request path is untouched.
    """
    from routellm.redact.config import build_redact_transform

    redactor = build_redact_transform((file_config or {}).get("redact"), config_path, resolve_path)
    chain = load_transforms((file_config or {}).get("transforms"))
    return [redactor, *chain] if redactor is not None else chain


def build_resilience_config(file_config: dict | None) -> ResilienceConfig:
    """Return the resilience settings, field names as in `ResilienceConfig`."""
    config = _validate(_StrictResilience, "resilience", _section(file_config, "resilience"))
    return ResilienceConfig(**config.model_dump())


def build_comparison_config(file_config: dict | None) -> ComparisonConfig:
    """Return the comparison settings; disabled unless the config enables it.

    The model refuses unknown keys itself, so a mistyped `rate:` is an
    error rather than a limit that quietly never applied.
    """
    return _validate(ComparisonConfig, "comparison", _section(file_config, "comparison"))


def build_quality_manager(
    file_config: dict | None, config_path: str | None, resolve_path=None
) -> QualityManager | None:
    """Return the trace recorder the `traces:` section asks for, else None.

    Traces are off unless the section enables them. The keys are those
    of `FineTuneConfig` (`enabled`, `trace_dir`, `max_files`,
    `max_bytes`). A relative `trace_dir` resolves against the config
    file's directory, never the process CWD.

    Parameters
    ----------
    file_config : dict, optional
        The merged config.
    config_path : str, optional
        The file the config was read from.

    Returns
    -------
    QualityManager or None
        None leaves the controller its default, which records nothing.
    """
    raw = _section(file_config, "traces")
    config = _validate(_StrictTraces, "traces", dict(raw))
    if not config.enabled:
        return None
    from routellm.config import path_resolver

    path = path_resolver(config_path, resolve_path)("traces.trace_dir", config.trace_dir)
    values = {**config.model_dump(), "trace_dir": os.fspath(path)}
    return QualityManager(fine_tune_config=FineTuneConfig(**values))
