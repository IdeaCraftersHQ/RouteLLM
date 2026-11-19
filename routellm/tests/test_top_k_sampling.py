"""`top_k` is carried through chat completions and reaches each leaf as it takes it.

litellm sends a sampling value it does not list for a provider as a
top-level body field, so an OpenAI or Azure upstream, which rejects an
unknown field, would answer 400. Routing picks the leaf, so the drop is
made per leaf (`routellm.leaf_params`). Every server case drives the real
app and the real litellm against local fakes, one subprocess each.
"""

import json
import os
from unittest import mock

import pytest
from pydantic import ValidationError

from routellm.chat_request import ChatCompletionRequest
from routellm.leaf_params import drop_kwargs
from routellm.tests.test_drop_params import _call, _run

BASE = {"model": "m", "messages": "hi"}


def test_top_k_is_forwarded_when_sent_and_absent_otherwise():
    sent = ChatCompletionRequest(**BASE, top_k=5).model_dump(exclude_unset=True, exclude_none=True)
    assert sent["top_k"] == 5
    assert "top_k" not in ChatCompletionRequest(**BASE).model_dump(
        exclude_unset=True, exclude_none=True
    )


@pytest.mark.parametrize("value", [0, 1, 40])
def test_top_k_takes_a_whole_number_from_zero(value):
    assert ChatCompletionRequest(**BASE, top_k=value).top_k == value


@pytest.mark.parametrize("value", [-1, 2.5, True, "5"])
def test_top_k_rejects_anything_else(value):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**BASE, top_k=value)


# -- per leaf, against the fakes ------------------------------------------------

_REAL_OPENAI = '"api_base": None'  # litellm falls back to OPENAI_API_BASE, here the fake
_POINT_OPENAI_AT_FAKE = 'import os; os.environ["OPENAI_API_BASE"] = A.base\n'


def _openai_proper(extra: str, a_extra: str = _REAL_OPENAI) -> tuple[str, str]:
    body = _POINT_OPENAI_AT_FAKE + _call("to_a", "A", extra)
    return body, a_extra


@pytest.mark.slow
def test_openai_upstream_is_served_with_top_k_dropped_not_refused(tmp_path):
    body, a_extra = _openai_proper('"top_k": 5, "temperature": 0.2')
    out = _run(tmp_path, body, a_extra=a_extra)

    assert out["status"] == 200, out
    assert out["seen"] is not None  # the OpenAI-shaped leaf itself served it
    assert "top_k" not in out["seen"]
    assert out["seen"]["temperature"] == 0.2
    assert any("top_k" in line for line in out["logs"])  # a drop is visible, by name


@pytest.mark.slow
def test_openai_upstream_drops_top_k_under_drop_params_true_too(tmp_path):
    body, a_extra = _openai_proper('"top_k": 5')
    out = _run(tmp_path, body, a_extra=f'{a_extra}, "drop_params": True')

    assert out["status"] == 200 and out["seen"] is not None, out
    assert "top_k" not in out["seen"]


@pytest.mark.slow
def test_drop_params_false_sends_top_k_to_openai_as_the_operator_asked(tmp_path):
    body, a_extra = _openai_proper('"top_k": 5')
    out = _run(tmp_path, body, a_extra=f'{a_extra}, "drop_params": False')

    assert out["seen"]["top_k"] == 5, out


@pytest.mark.slow
def test_openai_compatible_server_on_its_own_base_receives_top_k(tmp_path):
    """vLLM, llama.cpp and colibri behind `openai/` take `top_k` at the top level."""
    out = _run(tmp_path, _call("to_a", "A", '"top_k": 5'))

    assert out["status"] == 200, out
    assert out["seen"]["top_k"] == 5


@pytest.mark.slow
def test_ollama_leaf_receives_top_k_as_an_option(tmp_path):
    # The last hit can be litellm's `/api/show` model lookup, not the chat call.
    call = _call("to_o", "O", '"top_k": 5').replace(
        "fake.hits[-1][2] if fake.hits else None",
        "next((h[2] for h in reversed(fake.hits) if h[0].endswith('/chat')), None)",
    )
    out = _run(tmp_path, call)

    assert out["status"] == 200, out
    assert out["seen"] is not None
    assert out["seen"]["options"]["top_k"] == 5


@pytest.mark.slow
def test_endpoint_drop_list_drops_top_k_on_a_compatible_server(tmp_path):
    out = _run(tmp_path, _call("to_a", "A", '"top_k": 5'), a_extra='"drop_params": ["top_k"]')

    assert out["status"] == 200 and out["seen"] is not None, out
    assert "top_k" not in out["seen"]


# -- the policy, unit --------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "api_base", "policy", "dropped"),
    [
        ("openai/gpt-4o", None, None, True),
        ("openai/gpt-4o", "https://api.openai.com/v1", None, True),
        ("openai/gpt-4o", None, True, True),
        ("openai/gpt-4o", None, ["seed"], True),
        ("openai/gpt-4o", None, False, False),
        ("azure/dep", "https://x.openai.azure.com", None, True),
        ("openai/m", "http://127.0.0.1:9", None, False),
        ("hosted_vllm/m", "http://127.0.0.1:9", None, False),
        ("ollama_chat/m", "http://127.0.0.1:9", None, False),
        ("anthropic/claude-3-5-sonnet-20241022", None, None, False),
        ("gemini/gemini-1.5-pro", None, None, False),
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", None, None, False),
    ],
)
def test_top_k_is_dropped_only_for_providers_that_reject_unknown_fields(
    model, api_base, policy, dropped
):
    with mock.patch.dict(os.environ, clear=False):
        out = drop_kwargs(model, api_base, policy, {})
    assert ("top_k" in out.get("additional_drop_params", [])) is dropped, out


def test_top_k_json_roundtrip_is_plain_number():
    assert json.loads(ChatCompletionRequest(**BASE, top_k=3).model_dump_json())["top_k"] == 3
