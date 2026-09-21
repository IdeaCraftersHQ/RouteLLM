"""Tests for the typed capability record merged per endpoint.

Covers the three sources and their precedence (an explicit
`capabilities:` block, the deprecated tag aliases, then the models.dev
record), `None` as the only unknown marker, the one-warning-per-alias
deprecation, and the `ModelRecord` fields the catalog now carries.

No catalog is ever fetched: every record here is built by hand.
"""

import logging

import pytest

from routellm.capabilities import (
    LONG_CONTEXT_TOKENS,
    Capabilities,
    capabilities_for,
    from_tags,
    merge,
)
from routellm.endpoints import Endpoint, EndpointRegistry
from routellm.pairing import ModelRecord


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    """Reset the per-process warned set so each test starts quiet."""
    from routellm import capabilities

    monkeypatch.setattr(capabilities, "_warned", set())


def _record(**kw):
    """Build one catalog record with sane defaults."""
    return ModelRecord(
        provider=kw.pop("provider", "openai"),
        id=kw.pop("id", "gpt-4o"),
        **kw,
    )


def test_explicit_block_beats_the_catalog_record():
    explicit = Capabilities(vision=False, context=8192)
    record = _record(context=128000)

    merged = merge(explicit, record, None)

    assert merged.vision is False
    assert merged.context == 8192


def test_catalog_record_fills_what_the_config_omits():
    explicit = Capabilities(vision=False)
    record = _record(tool_call=True, reasoning=True, context=128000)

    merged = merge(explicit, record, None)

    assert merged.vision is False
    assert merged.tools is True
    assert merged.reasoning is True
    assert merged.context == 128000


def test_unknown_stays_none_when_neither_source_knows():
    merged = merge(None, None, None)

    assert merged.vision is None
    assert merged.context is None
    assert set(merged.unknown_fields()) == {
        "vision",
        "tools",
        "structured_output",
        "reasoning",
        "open_weights",
        "context",
        "max_output",
        "modalities_in",
    }


def test_known_false_is_distinct_from_unknown():
    known_false = merge(Capabilities(vision=False), None, None)
    unknown = merge(Capabilities(), None, None)

    assert known_false.vision is False
    assert unknown.vision is None
    assert "vision" not in known_false.unknown_fields()
    assert "vision" in unknown.unknown_fields()


def test_tools_tag_populates_the_typed_block_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="routellm.capabilities"):
        first = from_tags(["tools"])
        second = from_tags(["tools"])

    assert first.tools is True
    assert second.tools is True

    messages = [r.message for r in caplog.records]
    deprecations = [m for m in messages if "'tools' is deprecated" in m]
    assert len(deprecations) == 1
    assert "capabilities: {tools: true}" in deprecations[0]


def test_long_context_tag_sets_a_context_floor():
    caps = from_tags(["long_context"])

    assert caps.context == LONG_CONTEXT_TOKENS
    assert LONG_CONTEXT_TOKENS == 200_000


def test_tags_survive_the_alias_and_stay_selectable():
    endpoint = Endpoint(
        name="local_server",
        model="ollama_chat/qwen3:8b",
        tags=["local", "vision", "private"],
    )

    caps = capabilities_for(endpoint, None)

    assert caps.vision is True
    assert endpoint.tags == ["local", "vision", "private"]


def test_anonymous_endpoint_carries_capabilities_none():
    registry = EndpointRegistry()

    endpoint = registry.resolve("gpt-4o-mini")

    assert endpoint.capabilities is None
    assert endpoint.strict is False


def test_old_snapshot_without_the_new_fields_still_reads():
    entry = {
        "provider": "openai",
        "id": "gpt-4o",
        "cost_input": 2.5,
        "cost_output": 10.0,
        "tool_call": True,
        "reasoning": False,
        "context": 128000,
        "release_date": "2024-05-13",
    }

    record = ModelRecord(**entry)

    assert record.structured_output is False
    assert record.open_weights is False
    assert record.modalities_in == []
    assert record.max_output is None


def test_explicit_block_beats_a_tag_alias():
    endpoint = Endpoint(
        name="local_server",
        model="ollama_chat/qwen3:8b",
        tags=["vision"],
        capabilities=Capabilities(vision=False),
    )

    caps = capabilities_for(endpoint, None)

    assert caps.vision is False


def test_a_tag_alias_beats_the_catalog_record():
    endpoint = Endpoint(name="cloud", model="gpt-4o", tags=["tools"])
    record = _record(tool_call=False)

    caps = capabilities_for(endpoint, record)

    assert caps.tools is True


def test_record_modalities_and_max_output_reach_the_block():
    record = _record(
        structured_output=True,
        open_weights=True,
        modalities_in=["text", "image"],
        max_output=16384,
    )

    caps = merge(None, record, None)

    assert caps.structured_output is True
    assert caps.open_weights is True
    assert caps.modalities_in == ["text", "image"]
    assert caps.max_output == 16384
