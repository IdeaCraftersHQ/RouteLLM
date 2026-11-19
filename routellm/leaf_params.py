"""Which request params a routed leaf's provider can take.

Agent clients send advisory flags such as `parallel_tool_calls` on every
request, whatever the routed provider supports. litellm's `ollama_chat/`
transport answers an unsupported one with `UnsupportedParamsError`,
which would fail a request the leaf could have served. Routing picks the
leaf, so only here is it known which provider the flag is for.

An endpoint's `drop_params:` picks the policy:

- unset: drop the advisory params the provider does not list as
  supported, and nothing else. A param that changes what the model is
  asked to do (`tools`, `response_format`, a sampling value) still
  fails loudly rather than being silently ignored.
- `true`: litellm's `drop_params`: drop every param the provider rejects.
- `false`: drop nothing; an unsupported param fails the call.
- a list of names: drop those names for this endpoint, supported or not,
  on top of the unset-policy default.

`top_k` is the one sampling value dropped by default, and only for a leaf
whose API rejects a field it does not know (OpenAI itself, Azure OpenAI,
under any policy but `false`): litellm maps it for Anthropic, Gemini,
Ollama and Bedrock, and sends it as a top-level field to an
OpenAI-compatible server on its own `api_base` (vLLM, llama.cpp,
colibri), which takes it. It never reaches a leaf that would answer 400.

litellm applies the result: `additional_drop_params` removes the named
params before its support check, and `drop_params` relaxes the check.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

import litellm

logger = logging.getLogger(__name__)

#: Flags that tune how a provider answers without changing what it is
#: asked, so a provider that lacks one can serve the request without it.
ADVISORY_PARAMS = ("parallel_tool_calls", "reasoning_effort", "stream_options")


#: Sampling extras litellm sends as a top-level body field to a provider it
#: has no mapping for, which an API that rejects unknown fields answers 400.
UNKNOWN_FIELD_PARAMS = ("top_k",)

_OPENAI_HOSTS = frozenset({"api.openai.com"})


def rejects_unknown_fields(model: str, api_base: str | None) -> bool:
    """Whether the leaf is OpenAI itself or Azure OpenAI, which 400 an unknown field.

    An `openai/` model on any other `api_base` is an OpenAI-compatible
    server, which is not held to that. Unknown is not rejecting.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model, api_base=api_base)
    except Exception:
        return False
    if provider == "azure":
        return True
    if provider != "openai":
        return False
    return api_base is None or (urlsplit(api_base).hostname or "") in _OPENAI_HOSTS


def unsupported_by(model: str, api_base: str | None, names: Iterable[str]) -> list[str]:
    """Return those of `names` the model's provider does not list as supported.

    Unknown is not unsupported: a provider litellm cannot identify, or
    one that publishes no list, drops nothing.
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model, api_base=api_base)
        supported = litellm.get_supported_openai_params(model=model, custom_llm_provider=provider)
    except Exception:
        return []
    if supported is None:
        return []
    return [name for name in names if name not in supported]


def drop_kwargs(
    model: str,
    api_base: str | None,
    drop_params: bool | list[str] | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the litellm kwargs that carry an endpoint's drop policy.

    Parameters
    ----------
    model : str
        The litellm model the call is made to, after load balancing.
    api_base : str, optional
        The base URL of the call.
    drop_params : bool or list[str], optional
        The endpoint's `drop_params:` setting.
    extra : Mapping
        The endpoint's own `extra:`; its `drop_params` and
        `additional_drop_params` win over the policy and are merged.

    Returns
    -------
    dict
        `drop_params` and/or `additional_drop_params`; empty when the
        policy drops nothing.
    """
    if drop_params is False:
        return {}

    out: dict[str, Any] = {}
    names: list[str] = []
    if drop_params is True:
        out["drop_params"] = True
    else:
        names = list(drop_params or [])
        names += [n for n in unsupported_by(model, api_base, ADVISORY_PARAMS) if n not in names]
    # litellm's own `drop_params` does not cover a field it has no mapping for.
    if rejects_unknown_fields(model, api_base):
        names += [n for n in UNKNOWN_FIELD_PARAMS if n not in names]
    if names:
        out["additional_drop_params"] = names

    # An `extra:` naming litellm's own drop settings is explicit: its
    # `drop_params` wins, its `additional_drop_params` is merged.
    if "drop_params" in extra:
        out.pop("drop_params", None)
    if "additional_drop_params" in out and "additional_drop_params" in extra:
        merged = list(extra["additional_drop_params"])
        out["additional_drop_params"] = merged + [
            n for n in out["additional_drop_params"] if n not in merged
        ]
    return out


def log_dropped(endpoint: str, model: str, api_base: str | None, kwargs: Mapping[str, Any]) -> None:
    """Log at DEBUG which params of this call the leaf will not receive.

    Names only, never values: a param can carry content. Silent loss of
    a param is the cost of dropping it, so it is made visible here.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    dropped = [n for n in kwargs.get("additional_drop_params") or () if n in kwargs]
    if kwargs.get("drop_params") is True:
        known = [k for k in kwargs if k in litellm.OPENAI_CHAT_COMPLETION_PARAMS]
        dropped += [n for n in unsupported_by(model, api_base, known) if n not in dropped]
    if dropped:
        logger.debug(
            "endpoint %s (%s) drops params its provider rejects: %s", endpoint, model, dropped
        )
