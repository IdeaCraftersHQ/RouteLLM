"""The chat completion body the server accepts.

Kept apart from `openai_server` so dialect adapters can build one without
importing the server (which parses `sys.argv` at import).
"""

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["ChatCompletionRequest"]


class ChatCompletionRequest(BaseModel):
    """The chat completion body this server accepts.

    Agent clients send the full OpenAI chat shapes: nested `tools`,
    `tool_choice` as a string or an object, messages with `tool_calls`,
    `role: "tool"` turns, null or absent `content`, content parts
    (text, image_url, input_audio) and `response_format` of type `json_schema`. The
    nested shapes belong to the provider, so they are typed as open
    dicts and forwarded untouched.

    Only the fields named here are forwarded. Anything else in a body is
    dropped, which is deliberate: a parameter this server does not know
    may be one a routed provider rejects. The allowlist is this class.
    """

    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: str | list[dict[str, Any]]
    # Every optional field defaults to None so an unset one stays unset.
    # A default value here would be manufactured into the litellm call,
    # and a provider that does not accept the parameter rejects the whole
    # request over a value the client never sent.
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    # { "type": "json_object" } for json mode, or a `json_schema` object.
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Not an OpenAI field: a sampling extra that Anthropic, Gemini, Ollama,
    # Bedrock and OpenAI-compatible local runtimes (vLLM, llama.cpp, colibri)
    # take. A whole number from 0 (0 turns it off on the runtimes that
    # accept it; Gemini wants 1 or more and says so itself). litellm maps it
    # per provider, and `routellm.leaf_params` drops it for the leaves that
    # reject unknown fields (OpenAI, Azure). `min_p` and `repeat_penalty`
    # are not carried: litellm maps them for Ollama only and sends them
    # raw, and so wrongly, to every other provider.
    top_k: int | None = Field(default=None, strict=True, ge=0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    user: str | None = None
    # Read for `data_class` and `operation` only, then dropped: never forwarded.
    metadata: dict[str, Any] | None = None
    # Upstream hints (see `routellm.kv_slots`). Accepted here, then sent
    # only to an endpoint whose `extensions:` lists them; every other
    # endpoint drops them. `cache_slot` is a colibri KV context, 0..15.
    cache_slot: int | None = Field(default=None, strict=True, ge=0, le=15)
    x_colibri_ignore_leading_stop: bool | None = None
