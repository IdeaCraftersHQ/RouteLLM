"""One comparison call: the upstream request, its metrics, cost and judge.

Split from `routellm.comparison`: `CallMixin` holds the methods that
run and measure a single model call; `ComparisonManager` mixes it in.
"""

import asyncio
import logging
import time
from typing import Any

from litellm import acompletion, stream_chunk_builder

from routellm.leaf_params import drop_kwargs, log_dropped

logger = logging.getLogger(__name__)

#: Roles a recorded trace can have in a comparison.
ROLE_PRIMARY = "primary"
ROLE_SHADOW = "shadow"
ROLE_CANDIDATE = "candidate"


class CallMixin:
    """Calls one endpoint and records the result. Needs `host`, `config`,
    `registry`, `over_budget`, `_charge`, `_scorer`, `_scorer_failed`.
    """

    def policy_reasons(self, models: list[str], reqs) -> dict[str, str]:
        """Return `{endpoint: policy reason}` for every model the request's class may not use.

        Empty without a loaded policy or a class on `reqs`. Asked of the
        host, so a comparison target is held to the one check every
        routed candidate passes.
        """
        reason_of = getattr(self.host, "_policy_reason", None)
        if reason_of is None:
            return {}
        out: dict[str, str] = {}
        for name in models:
            reason = reason_of(name, reqs)
            if reason is not None:
                out[name] = reason
        return out

    async def _one(
        self,
        name,
        kwargs,
        prompt,
        *,
        plan,
        role,
        request_model,
        path,
        session_id,
        reqs,
        measure_ttft,
    ) -> dict[str, Any]:
        """Call one endpoint, judge, record, and return its result."""
        result = await self._call(name, kwargs, reqs, measure_ttft, plan)
        self._charge(result["metrics"]["cost_usd"], plan.experiment)

        if result["error"] is None:
            result["score"] = await self._judge(prompt, result["output"], plan.local_only, reqs)

        comparison = {
            "id": plan.id,
            "role": role,
            "trigger": plan.trigger,
            "experiment": plan.experiment,
            "error": result["error"],
            "metrics": result["metrics"],
            "score": result["score"],
        }
        try:
            self.host.quality_manager.record_trace(
                prompt,
                name,
                result.pop("_response"),
                None,
                path=path,
                request_model=request_model,
                endpoint=name,
                session_id=session_id,
                latency_ms=result["metrics"]["latency_ms"],
                comparison=comparison,
                score=result["score"],
            )
        except Exception as exc:
            logger.warning("recording comparison trace failed: %s", exc)
        result.pop("_response", None)
        return result

    async def _call(self, name, kwargs, reqs, measure_ttft, plan) -> dict[str, Any]:
        endpoint = self.registry.get(name)
        metrics: dict[str, Any] = {
            "ttft_ms": None,
            "latency_ms": None,
            "tokens_per_second": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
        }
        result: dict[str, Any] = {
            "endpoint": name,
            "model": endpoint.model,
            "output": "",
            "finish_reason": None,
            "error": None,
            "score": None,
            "metrics": metrics,
            "_response": {},
        }

        can_serve = getattr(self.host, "_can_serve", None)
        if reqs is not None and can_serve is not None:
            try:
                reason = can_serve(name, reqs)
            except Exception:
                # A check that cannot answer serves nothing for a classed
                # request: the data class fails closed, a capability
                # guess stays open as it always was.
                reason = "policy" if getattr(reqs, "data_class", None) else None
            if reason:
                result["error"] = f"unsupported: {reason}"
                return result

        if self.over_budget():
            result["error"] = "budget_exhausted"
            return result

        started = time.monotonic()
        try:
            res, ttft = await asyncio.wait_for(
                self._request(name, endpoint, kwargs, measure_ttft),
                timeout=self.config.timeout_ms / 1000,
            )
        except TimeoutError:
            result["error"] = "timeout"
            metrics["latency_ms"] = int((time.monotonic() - started) * 1000)
            return result
        except Exception as exc:
            logger.warning("comparison call to %s failed: %s: %s", name, type(exc).__name__, exc)
            result["error"] = "upstream_error"
            metrics["latency_ms"] = int((time.monotonic() - started) * 1000)
            return result

        elapsed = time.monotonic() - started
        body = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        result["output"] = message.get("content") or ""
        result["finish_reason"] = choice.get("finish_reason")
        result["_response"] = body
        metrics.update(
            ttft_ms=None if ttft is None else int(ttft * 1000),
            latency_ms=int(elapsed * 1000),
            tokens_per_second=(
                round(completion_tokens / elapsed, 3) if completion_tokens and elapsed > 0 else None
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(name, prompt_tokens, completion_tokens),
        )
        return result

    async def _request(self, name, endpoint, kwargs, measure_ttft):
        """One non-cached, non-retried upstream call; returns (response, ttft)."""
        host = self.host
        api_base, api_key = endpoint.credentials(host.api_base, host.api_key)
        extra = {
            **endpoint.extra,
            **drop_kwargs(endpoint.model, api_base, endpoint.drop_params, endpoint.extra),
        }
        from routellm.controller import outbound_headers

        call = {**extra, **kwargs}
        call.pop("stream", None)
        call.pop("stream_options", None)
        call.pop("extra_headers", None)
        call["model"] = endpoint.model
        log_dropped(name, endpoint.model, api_base, call)
        headers = outbound_headers(extra.get("extra_headers"), kwargs.get("extra_headers"))

        if not measure_ttft:
            res = await acompletion(
                api_base=api_base, api_key=api_key, extra_headers=headers, **call
            )
            return res, None

        started = time.monotonic()
        stream = await acompletion(
            api_base=api_base,
            api_key=api_key,
            extra_headers=headers,
            stream=True,
            stream_options={"include_usage": True},
            **call,
        )
        chunks = []
        ttft = None
        async for chunk in stream:
            if ttft is None and _has_content(chunk):
                ttft = time.monotonic() - started
            chunks.append(chunk)
        res = stream_chunk_builder(chunks, messages=kwargs.get("messages"))
        if res is None:
            raise RuntimeError("empty stream")
        return res, ttft

    def _cost(self, name: str, prompt_tokens, completion_tokens) -> float | None:
        record = (getattr(self.host, "_catalog_records", None) or {}).get(name)
        if record is None:
            return None
        cost_in = getattr(record, "cost_input", None)
        cost_out = getattr(record, "cost_output", None)
        if cost_in is None and cost_out is None:
            return None
        return (
            (prompt_tokens or 0) * (cost_in or 0.0) + (completion_tokens or 0) * (cost_out or 0.0)
        ) / 1e6

    # ------------------------------------------------------------- judge

    def _judge_denial(self, spec: str, reqs) -> str | None:
        """Return why the request's class may not be judged by `spec`, or None.

        Only `judge:<model>` calls a model; it is held to the same check
        as any call carrying request content. A judge that names no
        registry endpoint carries no tags, so a request with a class is
        refused it.
        """
        data_class = getattr(reqs, "data_class", None)
        kind, _, model = spec.partition(":")
        denial = getattr(self.host, "_aux_denial", None)
        if kind != "judge" or denial is None or data_class is None:
            return None
        names = self.registry.names()
        match = (
            model
            if model in names
            else next((n for n in names if self.registry.get(n).model == model), None)
        )
        if match is None:
            return None if getattr(self.host, "policy", None) is None else "unregistered"
        return denial(match, data_class, getattr(reqs, "route_policy", None))

    async def _judge(self, prompt: str, output: str, local_only: bool, reqs=None) -> float | None:
        judge = self.config.judge
        if judge is None or self._scorer_failed:
            return None
        if local_only and not self._judge_local:
            logger.warning("not judging local-only data: the judge is not local")
            return None
        refused = self._judge_denial(judge.scorer, reqs)
        if refused is not None:
            logger.info("comparison judge skipped: %s", refused)
            return None

        loop = asyncio.get_running_loop()
        try:
            if self._scorer is None:
                from routellm.quality_scores import build_scorer

                self._scorer = await loop.run_in_executor(
                    None, lambda: build_scorer(judge.scorer, allow_llm=True)
                )
            reward = await loop.run_in_executor(
                None, lambda: self._scorer.score(output, {"prompt": prompt})
            )
            return None if reward.score is None else float(reward.score)
        except Exception as exc:
            if self._scorer is None:
                self._scorer_failed = True
            logger.warning("comparison judge failed: %s", exc)
            return None


def _has_content(chunk) -> bool:
    try:
        return bool(chunk.choices[0].delta.content)
    except (AttributeError, IndexError, TypeError):
        return False
