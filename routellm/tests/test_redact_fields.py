"""The redact transform on the fields outside `messages`: redact, refuse, allow.

Every field of a request is in one of three lists (see docs/redact.md);
one the lists do not name is refused, so a field added later never goes
upstream unseen.
"""

import ast
import json
from pathlib import Path

import pytest

from routellm.redact.fields import ALLOWED, REDACTED, REFUSED
from routellm.redact.placeholders import Redactor
from routellm.redact.tokens import TokenStore
from routellm.redact.transform import RedactTransform
from routellm.tests.test_redact_transform import analyzer as _analyzer
from routellm.tests.test_redact_transform import ctx, leaked
from routellm.transform import TransformError, TransformRunner

NAMES = ("Jean Tremblay", "Marie Gagnon", "Alice Smith", "jean.t@example.com", "046 454 286")


def request_fields() -> set[str]:
    """The fields of `ChatCompletionRequest`, read from source (the module parses argv on import)."""
    source = Path(__file__).parents[1] / "openai_server.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "ChatCompletionRequest":
            return {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
    raise AssertionError("ChatCompletionRequest not found")


@pytest.fixture
def transform(tmp_path):
    return RedactTransform(
        Redactor(TokenStore(str(tmp_path / "t.db")), analyze=_analyzer), non_text="pass"
    )


def run(transform, **fields):
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    return transform.forward_kwargs({**base, **fields}, ctx())


def refused(transform, **fields):
    with pytest.raises(Exception) as caught:
        run(transform, **fields)
    return caught.value


SCHEMA = {
    "type": "object",
    "description": "About Jean Tremblay",
    "properties": {
        "who": {
            "type": "string",
            "description": "e.g. Marie Gagnon",
            "enum": ["Alice Smith", "other"],
            "default": "jean.t@example.com",
            "examples": [{"nas": "046 454 286"}],
        }
    },
    "required": ["who"],
    "additionalProperties": False,
}
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Jean Tremblay's file",
        "parameters": SCHEMA,
        "strict": True,
    },
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user", "jean.t@example.com"),
        ("tools", [TOOL]),
        ("functions", [TOOL["function"]]),
        (
            "response_format",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "a",
                    "description": "Jean Tremblay",
                    "schema": SCHEMA,
                    "strict": True,
                },
            },
        ),
        ("prediction", {"type": "content", "content": "Jean Tremblay"}),
        ("prediction", {"type": "content", "content": [{"type": "text", "text": "Marie Gagnon"}]}),
        ("metadata", {"owner": "Alice Smith", "n": {"deep": ["046 454 286"]}}),
    ],
)
def test_free_text_fields_are_redacted(transform, field, value):
    out, state = run(transform, **{field: value})
    assert leaked(out) == [] and state is not None
    assert out[field] != value


def test_names_and_structure_of_a_tool_are_kept(transform):
    out, _ = run(
        transform, tools=[TOOL], tool_choice={"type": "function", "function": {"name": "lookup"}}
    )
    fn = out["tools"][0]["function"]
    assert fn["name"] == "lookup" and fn["strict"] is True
    params = fn["parameters"]
    assert params["type"] == "object" and params["required"] == ["who"]
    assert set(params["properties"]) == {"who"} and params["additionalProperties"] is False
    assert params["properties"]["who"]["type"] == "string"
    assert out["tool_choice"]["function"]["name"] == "lookup"


def test_message_fields_beyond_content_are_redacted(transform):
    msgs = [
        {"role": "user", "name": "Jean Tremblay", "content": "x"},
        {
            "role": "assistant",
            "content": None,
            "refusal": "no, Marie Gagnon",
            "annotations": [],
            "audio": None,
            "reasoning_content": "Alice Smith thinks",
        },
    ]
    out, _ = run(transform, messages=msgs)
    assert leaked(out) == []
    assert out["messages"][1]["annotations"] == []


@pytest.mark.parametrize(
    "fields",
    [
        {"tools": [{"type": "function", "function": {"name": "jean.t@example.com"}}]},
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"properties": {"Jean Tremblay": {"type": "string"}}},
                    },
                }
            ]
        },
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "required": ["Jean Tremblay"]},
                    },
                }
            ]
        },
        {"tool_choice": {"type": "function", "function": {"name": "Jean Tremblay"}}},
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "Jean Tremblay",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ]
        },
        {"messages": [{"role": "tool", "tool_call_id": "Jean Tremblay", "content": "x"}]},
    ],
)
def test_a_name_that_carries_a_value_is_refused(transform, fields):
    refused(transform, **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"frobnicate": "x"},
        {"web_search_options": {"user_location": "Québec"}},
        {"audio": {"voice": "alloy"}},
        {"tools": [{"type": "custom", "custom": {"name": "f"}}]},
        {"response_format": {"type": "grammar", "grammar": "x"}},
        {"messages": [{"role": "user", "content": "x", "weird": "Jean Tremblay"}]},
        {"stream_options": {"include_usage": True, "extra": "Jean Tremblay"}},
    ],
)
def test_an_unknown_field_is_refused(transform, fields):
    refused(transform, **fields)


def test_an_unset_unknown_field_carries_nothing(transform):
    out, _ = run(transform, frobnicate=None, other=[], more={})
    assert out["frobnicate"] is None


def test_the_allowlisted_fields_pass_unchanged(transform):
    allowed = {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 5,
        "max_completion_tokens": 6,
        "n": 1,
        "seed": 3,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "logit_bias": {"50256": -100},
        "logprobs": True,
        "top_logprobs": 2,
        "stream": False,
        "stream_options": {"include_usage": True},
        "parallel_tool_calls": True,
        "reasoning_effort": "low",
        "tool_choice": "auto",
        "extra_headers": {"x-a": "b"},
        "timeout": 5,
    }
    out, _ = run(transform, **allowed)
    assert {k: out[k] for k in allowed} == allowed


def test_a_local_leaf_gets_every_field_raw(transform):
    fields = {
        "model": "m",
        "messages": [{"role": "user", "content": "Jean Tremblay"}],
        "user": "Jean Tremblay",
        "frobnicate": "x",
    }
    out, state = transform.forward_kwargs(fields, ctx(reach=("local",)))
    assert out is fields and state is None


def test_the_runner_hands_the_transform_the_whole_call(transform):
    runner = TransformRunner([transform])
    out, applied = runner.forward_kwargs(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "user": "Jean Tremblay"},
        ctx(),
    )
    assert leaked(out) == [] and applied.active
    with pytest.raises(TransformError):
        runner.forward_kwargs({"model": "m", "messages": [], "frobnicate": "x"}, ctx())


def test_every_field_the_server_forwards_is_classified():
    """A field added to the request model without a row here would be refused for a redact class."""
    forwarded = request_fields() - {"messages", "model", "metadata"}
    assert forwarded <= ALLOWED | REDACTED
    assert not (ALLOWED & REDACTED) and not (ALLOWED & REFUSED) and not (REDACTED & REFUSED)
    assert "metadata" in REDACTED and "tools" in REDACTED and "user" in REDACTED


def test_an_unclassified_field_added_upstream_is_refused(transform):
    new = "shiny_new_field"
    assert new not in request_fields() | ALLOWED | REDACTED
    refused(transform, **{new: "Jean Tremblay"})


def test_the_json_blob_of_a_call_never_holds_a_canary_after_forward(transform):
    out, _ = run(transform, user="Jean Tremblay", tools=[TOOL], metadata={"a": "Alice Smith"})
    assert not any(n in json.dumps(out, ensure_ascii=False) for n in NAMES)
