"""Tests for what one OpenAI request needs, read off the request.

Covers vision from image parts, tools from `tools` or `functions`,
structured output from `response_format`, the token count plus
`max_tokens`, its chars/4 fallback, and the fields that are deliberately
never read.

No router runs and no network is touched: `derive` reads the request
and nothing else.
"""

import logging

from routellm import requirements
from routellm.requirements import Requirements, derive


def _text(content="hello"):
    """One plain user message."""
    return [{"role": "user", "content": content}]


def _image(part_type="image_url"):
    """One user message carrying a text part and an image part."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": part_type, part_type: {"url": "https://x/y.png"}},
            ],
        }
    ]


def test_plain_text_request_has_no_requirements():
    reqs = derive(_text(), {}, "default")

    assert reqs.vision is False
    assert reqs.tools is False
    assert reqs.structured_output is False


def test_image_url_part_sets_vision():
    assert derive(_image("image_url"), {}, "default").vision is True


def test_input_image_part_sets_vision():
    assert derive(_image("input_image"), {}, "default").vision is True


def test_string_content_never_sets_vision():
    messages = _text("here is an image_url for you")

    assert derive(messages, {}, "default").vision is False


def test_malformed_content_parts_do_not_raise():
    messages = [
        {"role": "user", "content": ["a bare string", 7, None]},
        {"role": "user"},
        {"content": {"type": "image_url"}},
        "not a dict at all",
    ]

    reqs = derive(messages, {}, "default")

    assert reqs.vision is False


def test_tools_key_sets_tools():
    kwargs = {"tools": [{"type": "function", "function": {"name": "f"}}]}

    assert derive(_text(), kwargs, "default").tools is True


def test_functions_key_sets_tools():
    kwargs = {"functions": [{"name": "f"}]}

    assert derive(_text(), kwargs, "default").tools is True


def test_empty_tools_list_does_not_set_tools():
    assert derive(_text(), {"tools": []}, "default").tools is False
    assert derive(_text(), {"functions": []}, "default").tools is False


def test_json_schema_response_format_sets_structured_output():
    kwargs = {"response_format": {"type": "json_schema", "json_schema": {}}}

    assert derive(_text(), kwargs, "default").structured_output is True


def test_json_object_response_format_sets_structured_output():
    kwargs = {"response_format": {"type": "json_object"}}

    assert derive(_text(), kwargs, "default").structured_output is True


def test_text_response_format_sets_nothing():
    kwargs = {"response_format": {"type": "text"}}

    assert derive(_text(), kwargs, "default").structured_output is False


def test_context_needed_counts_prompt_plus_max_tokens():
    messages = _text("a somewhat longer prompt than a single word")

    without = derive(messages, {}, "default").context_needed
    with_max = derive(messages, {"max_tokens": 512}, "default").context_needed

    assert without is not None
    assert with_max == without + 512


def test_token_counter_failure_falls_back_to_chars_over_four(monkeypatch, caplog):
    def _boom(**kwargs):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(requirements.litellm, "token_counter", _boom)

    content = "x" * 400
    with caplog.at_level(logging.DEBUG, logger="routellm.requirements"):
        reqs = derive(_text(content), {"max_tokens": 10}, "default")

    assert reqs.context_needed == 100 + 10


def test_streaming_and_sampling_fields_change_nothing():
    kwargs = {
        "stream": True,
        "n": 4,
        "temperature": 0.9,
        "seed": 7,
        "user": "someone",
        "top_p": 0.1,
        "stop": ["\n"],
    }

    plain = derive(_text(), {}, "default")
    noisy = derive(_text(), kwargs, "default")

    assert plain == noisy


def test_is_empty_is_true_only_for_a_plain_request():
    assert Requirements().is_empty() is True
    assert Requirements(vision=True).is_empty() is False
    assert Requirements(context_needed=10).is_empty() is False


def test_a_plain_request_needs_no_capability():
    # `context_needed` is always counted, so a derived Requirements is
    # never literally empty; what makes a plain request a no-op is that
    # it demands no capability and its context is judged only against
    # an endpoint that declares one.
    reqs = derive(_text(), {}, "default")

    assert (reqs.vision, reqs.tools, reqs.structured_output) == (
        False,
        False,
        False,
    )
    assert derive(_image(), {}, "default").vision is True


def test_prompt_text_joins_list_content_and_passes_a_string_through():
    assert requirements._prompt_text(_text("plain")) == "plain"

    joined = requirements._prompt_text(_image())
    assert "what is this" in joined
    assert isinstance(joined, str)


def test_unknown_model_name_still_counts():
    reqs = derive(_text("hello there"), {}, "zzz/not-a-model")

    assert reqs.context_needed is not None
    assert reqs.context_needed > 0
