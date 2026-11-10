"""Standalone model-call seam for callers with no Controller (the eval CLI)."""

from typing import Any


def litellm_transport(registry: Any):
    """Return a transport calling registry endpoints through litellm directly.

    No resilience or breaker: the eval CLI wants raw latency and raw failures.
    """

    async def transport(endpoint, messages, response_format, timeout_ms):
        from litellm import acompletion

        ep = registry.resolve(endpoint)
        api_base, api_key = ep.credentials(None, None)
        kwargs = {**ep.extra}
        kwargs.update(
            model=ep.model,
            messages=messages,
            response_format=response_format,
            temperature=0,
            stream=False,
            timeout=timeout_ms / 1000,
            num_retries=0,
        )
        if api_base is not None:
            kwargs["api_base"] = api_base
        if api_key is not None:
            kwargs["api_key"] = api_key
        res = await acompletion(**kwargs)
        return res.choices[0].message.content

    return transport
