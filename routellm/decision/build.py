"""Build the classifier from the merged config and the endpoint registry."""

from typing import Any

from routellm.decision.classifier import DecisionClassifier
from routellm.decision.config import DecisionConfig
from routellm.decision.llm import create as create_llm
from routellm.decision.plugins import PROVIDER_GROUP, load_entry_point
from routellm.decision.recognizers import build_recognizer

__all__ = ["build_decision"]

BUILTIN_PROVIDERS = {"llm": create_llm}


def build_decision(file_config: dict | None, registry: Any) -> DecisionClassifier | None:
    """Return the classifier for the `decision:` section, or None without one.

    Raises
    ------
    ValueError
        If the section is malformed, the endpoint is unknown or a tier,
        lacks a `require_tags` tag, or the provider is unknown.
    """
    spec = (file_config or {}).get("decision")
    if not spec:
        return None
    try:
        config = DecisionConfig.model_validate(spec)
    except ValueError as exc:
        raise ValueError(f"decision: {exc}") from exc

    name = config.endpoint
    if registry.has_tier(name):
        raise ValueError(
            f"decision.endpoint: {name!r} is a tier; name a registry endpoint instead."
        )
    if name not in registry.names():
        known = ", ".join(registry.names()) or "<none>"
        raise ValueError(
            f"decision.endpoint: {name!r} is not a registry endpoint. Configured endpoints: {known}"
        )
    missing = [t for t in config.require_tags if t not in registry.get(name).tags]
    if missing:
        raise ValueError(
            f"decision.endpoint {name!r} lacks required tag(s) {', '.join(map(repr, missing))} "
            f"(decision.require_tags); it carries {registry.get(name).tags or 'none'}."
        )

    factory = BUILTIN_PROVIDERS.get(config.provider) or load_entry_point(
        PROVIDER_GROUP, config.provider
    )
    if factory is None:
        raise ValueError(
            f"decision.provider: unknown {config.provider!r} "
            f"(built in: {', '.join(BUILTIN_PROVIDERS)}; others register under {PROVIDER_GROUP})"
        )
    return DecisionClassifier(
        config,
        recognizers=[build_recognizer(r) for r in config.recognizers],
        provider_factory=factory,
    )
