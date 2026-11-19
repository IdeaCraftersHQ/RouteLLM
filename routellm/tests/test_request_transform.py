"""Request transforms: rewrite before the call, restore after, every call path.

The provider is a patched `acompletion` / `completion` that records what
it is sent and answers with text built from it, so each assertion is about
what crossed the call boundary. The test transform swaps the word SECRET
for the placeholder `[[S_1]]` and back; a placeholder is exactly what a
stream can split, so the stream cases cut it mid-token.
"""

import asyncio
import json
from pathlib import Path

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

import routellm.controller as controller_module
from routellm.caching import CacheConfig
from routellm.controller import Controller
from routellm.endpoints import EndpointRegistry
from routellm.quality import FineTuneConfig, QualityManager
from routellm.routers.routers import ROUTER_CLS
from routellm.transform import (
    HoldbackRestorer,
    IdentityTransform,
    TransformError,
    map_text,
)

TOKEN = "[[S_1]]"


class _PinStrong:
    def route(self, prompt, threshold, pair):
        return pair.strong


ROUTER_CLS["pin_strong"] = lambda **_: _PinStrong()


def _hold(buffer: str) -> int:
    """Trailing characters that may still grow into TOKEN."""
    for n in range(min(len(TOKEN) - 1, len(buffer)), 0, -1):
        if TOKEN.startswith(buffer[-n:]):
            return n
    return 0


class Swap:
    """SECRET <-> [[S_1]]; leaves `local` leaves raw (state None)."""

    def __init__(self):
        self.seen: list[str | None] = []

    def forward(self, messages, ctx):
        self.seen.append(ctx.endpoint)
        if "local" in ctx.reach:
            return messages, None
        return map_text(messages, lambda t: t.replace("SECRET", TOKEN)), "swapped"

    def inverse(self, text, state):
        return text.replace(TOKEN, "SECRET")

    def restorer(self, state):
        return HoldbackRestorer(lambda t: self.inverse(t, state), _hold)


class Always(Swap):
    """Rewrites on every leaf, local or not."""

    def forward(self, messages, ctx):
        return map_text(messages, lambda t: t.replace("SECRET", TOKEN)), "swapped"


class Boom:
    def forward(self, messages, ctx):
        raise RuntimeError("redactor exploded with SECRET inside")

    def inverse(self, text, state):
        return text


class Provider:
    """Records every call; `fail` names models that raise, `reply` builds the text."""

    def __init__(self):
        self.calls: list[dict] = []
        self.fail: set[str] = set()
        self.reply = lambda messages: "got: " + str(messages[-1]["content"])
        self.stream_cuts: list[int] | None = None
        self.tool_args: str | None = None
        self.no_finish = False

    def _record(self, kwargs):
        self.calls.append(
            {
                "model": kwargs["model"],
                "messages": json.loads(json.dumps(kwargs["messages"])),
            }
        )
        if kwargs["model"] in self.fail:
            raise RuntimeError("upstream down")

    def _message(self, kwargs):
        text = self.reply(kwargs["messages"])
        if self.tool_args is not None:
            call = {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": self.tool_args},
            }
            return Message(content=text, tool_calls=[call]), "tool_calls"
        return Message(content=text), "stop"

    async def acompletion(self, **kwargs):
        self._record(kwargs)
        if kwargs.get("stream"):
            return self._stream(kwargs)
        message, finish = self._message(kwargs)
        return ModelResponse(choices=[Choices(index=0, message=message, finish_reason=finish)])

    def completion(self, **kwargs):
        self._record(kwargs)
        if kwargs.get("stream"):
            return iter(list(self._chunks(kwargs)))
        message, finish = self._message(kwargs)
        return ModelResponse(choices=[Choices(index=0, message=message, finish_reason=finish)])

    def _chunks(self, kwargs):
        def make(delta, finish=None):
            return ModelResponseStream(
                id="c",
                model="m",
                choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish)],
            )

        yield make(Delta(role="assistant", content=""))
        if self.tool_args is not None:
            args = self.tool_args
            cuts = self.stream_cuts or [len(args)]
            lo = 0
            for i, hi in enumerate([*cuts, len(args)]):
                if hi <= lo:
                    continue
                call = ChatCompletionDeltaToolCall(
                    index=0,
                    id="c1" if i == 0 else None,
                    type="function",
                    function=Function(name="f" if i == 0 else None, arguments=args[lo:hi]),
                )
                yield make(Delta(tool_calls=[call]))
                lo = hi
            yield make(Delta(), "tool_calls")
            return
        text = self.reply(kwargs["messages"])
        cuts = self.stream_cuts or [len(text)]
        lo = 0
        for hi in [*cuts, len(text)]:
            if hi > lo:
                yield make(Delta(content=text[lo:hi]))
                lo = hi
        if not self.no_finish:
            yield make(Delta(), "stop")

    async def _stream(self, kwargs):
        for chunk in self._chunks(kwargs):
            yield chunk


@pytest.fixture
def provider(monkeypatch, tmp_path):
    from routellm.pairing import CATALOG_CACHE_ENV, write_snapshot

    catalog = tmp_path / "models_dev.json"
    write_snapshot(catalog, [])
    monkeypatch.setenv(CATALOG_CACHE_ENV, str(catalog))
    fake = Provider()
    monkeypatch.setattr(controller_module, "acompletion", fake.acompletion)
    monkeypatch.setattr(controller_module, "completion", fake.completion)
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    return fake


def build(
    transforms, tmp_path, *, cache=False, tags=None, tiers=None, endpoint_fields=None, **kwargs
):
    tags = tags or {}
    endpoints = {
        name: {
            "model": f"openai/{name}",
            "api_base": "http://127.0.0.1:1/v1",
            "api_key_env": "FAKE_KEY",
            "tags": tags.get(name, ["cloud"]),
            "quality": {"cld": 90, "cld2": 80, "loc": 10}[name],
            **(endpoint_fields or {}).get(name, {}),
        }
        for name in ("cld", "cld2", "loc")
    }
    controller = Controller(
        routers=["random"],
        endpoints=EndpointRegistry.from_config(
            {
                "endpoints": endpoints,
                "tiers": tiers
                or {
                    "default": {
                        "router": "pin_strong",
                        "threshold": 0.0,
                        "strong": "cld",
                        "weak": "cld2",
                    }
                },
            }
        ),
        cache_config=CacheConfig(enabled=cache, db_path=str(tmp_path / "cache.db")),
        quality_manager=QualityManager(
            fine_tune_config=FineTuneConfig(
                enabled=True, trace_dir=str(tmp_path / "traces"), capture_content=True
            )
        ),
        transforms=transforms,
        **kwargs,
    )
    controller.resilience.config.max_retries = 0
    return controller


def messages(text="tell SECRET now"):
    return [{"role": "user", "content": text}]


def ask(controller, msgs=None, **kwargs):
    return asyncio.run(
        controller.acompletion(model="default", messages=msgs or messages(), **kwargs)
    )


def drain(controller, msgs=None):
    async def go():
        res = await controller.acompletion(
            model="default", messages=msgs or messages(), stream=True
        )
        return [c async for c in res]

    return asyncio.run(go())


def text_of(chunks):
    return "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)


def args_of(chunks):
    out = ""
    for c in chunks:
        for call in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            out += call.function.arguments or ""
    return out


def sent(provider):
    return json.dumps([c["messages"] for c in provider.calls])


# --- non-stream -------------------------------------------------------------


def test_provider_sees_the_rewrite_and_client_gets_the_restored_reply(provider, tmp_path):
    res = ask(build([Swap()], tmp_path))
    assert provider.calls[0]["messages"][0]["content"] == f"tell {TOKEN} now"
    assert "SECRET" not in sent(provider)
    assert res.choices[0].message.content == "got: tell SECRET now"


def test_no_transform_changes_nothing(provider, tmp_path):
    res = ask(build([], tmp_path))
    assert provider.calls[0]["messages"][0]["content"] == "tell SECRET now"
    assert res.choices[0].message.content == "got: tell SECRET now"


def test_the_identity_default_changes_nothing(provider, tmp_path):
    controller = build([IdentityTransform()], tmp_path)
    assert not controller.transforms.active
    assert ask(controller).choices[0].message.content == "got: tell SECRET now"


def test_a_transform_may_leave_a_local_leaf_raw(provider, tmp_path):
    tiers = {"default": {"router": "pin_strong", "threshold": 0.0, "strong": "loc", "weak": "cld"}}
    res = ask(build([Swap()], tmp_path, tags={"loc": ["local"]}, tiers=tiers))
    assert provider.calls[0]["messages"][0]["content"] == "tell SECRET now"
    assert res.choices[0].message.content == "got: tell SECRET now"


def test_the_caller_s_messages_are_not_mutated(provider, tmp_path):
    msgs = messages()
    ask(build([Swap()], tmp_path), msgs)
    assert msgs == messages()


def test_transforms_run_in_order_and_undo_in_reverse(provider, tmp_path):
    class Wrap:
        def __init__(self, tag):
            self.tag = tag

        def forward(self, msgs, ctx):
            return map_text(msgs, lambda t: f"<{self.tag}>{t}"), self.tag

        def inverse(self, text, state):
            assert text.startswith(f"<{state}>"), (text, state)
            return text[len(state) + 2 :]

    provider.reply = lambda m: m[-1]["content"]
    res = ask(build([Wrap("a"), Wrap("b")], tmp_path))
    assert provider.calls[0]["messages"][0]["content"] == "<b><a>tell SECRET now"
    assert res.choices[0].message.content == "tell SECRET now"


# --- every attempt ----------------------------------------------------------


def test_the_sibling_fallback_is_transformed_too(provider, tmp_path):
    provider.fail = {"openai/cld"}
    res = ask(build([Swap()], tmp_path))
    assert [c["model"] for c in provider.calls] == ["openai/cld", "openai/cld2"]
    assert all(c["messages"][0]["content"] == f"tell {TOKEN} now" for c in provider.calls)
    assert res.choices[0].message.content == "got: tell SECRET now"


def test_a_retry_of_the_same_leaf_is_transformed(provider, tmp_path):
    controller = build([Swap()], tmp_path)
    controller.resilience.config.max_retries = 1
    controller.resilience.config.initial_backoff_ms = 1
    first = provider.acompletion
    state = {"n": 0}

    class Flaky(RuntimeError):
        status_code = 503

    async def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            provider._record(kwargs)
            raise Flaky("503")
        return await first(**kwargs)

    controller_module.acompletion = flaky
    ask(controller)
    assert len(provider.calls) == 2
    assert all(c["messages"][0]["content"] == f"tell {TOKEN} now" for c in provider.calls)


def test_a_pool_member_is_transformed(provider, tmp_path):
    tiers = {
        "default": {
            "router": "pin_strong",
            "threshold": 0.0,
            "strong": {"select": "tag:cloud", "order": "quality_desc", "pool": 2},
            "weak": "loc",
        }
    }
    provider.fail = {"openai/cld"}
    res = ask(build([Swap()], tmp_path, tiers=tiers))
    assert len(provider.calls) >= 2
    assert all(c["messages"][0]["content"] == f"tell {TOKEN} now" for c in provider.calls)
    assert res.choices[0].message.content == "got: tell SECRET now"


def test_a_raising_transform_means_no_call_and_no_fallback(provider, tmp_path):
    with pytest.raises(TransformError) as exc:
        ask(build([Boom()], tmp_path))
    assert provider.calls == []
    assert "SECRET" not in str(exc.value)


def test_a_raising_transform_refuses_a_stream_before_it_starts(provider, tmp_path):
    with pytest.raises(TransformError):
        drain(build([Boom()], tmp_path))
    assert provider.calls == []


def test_a_transform_returning_no_messages_is_refused(provider, tmp_path):
    class Bad:
        def forward(self, msgs, ctx):
            return None, "x"

        def inverse(self, text, state):
            return text

    with pytest.raises(TransformError):
        ask(build([Bad()], tmp_path))
    assert provider.calls == []


# --- tool calls and multimodal ---------------------------------------------------


def test_tool_call_arguments_are_transformed_out_and_restored_back(provider, tmp_path):
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"k": "SECRET"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "SECRET result"},
    ]
    provider.tool_args = '{"k": "[[S_1]]"}'
    res = ask(build([Swap()], tmp_path), msgs)
    assert "SECRET" not in sent(provider)
    assert provider.calls[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"k": "[[S_1]]"}'
    )
    assert res.choices[0].message.tool_calls[0].function.arguments == '{"k": "SECRET"}'


def test_multimodal_text_parts_are_transformed_and_images_left_alone(provider, tmp_path):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "see SECRET"}, image]}]
    provider.reply = lambda m: "ok"
    ask(build([Swap()], tmp_path), msgs)
    parts = provider.calls[0]["messages"][0]["content"]
    assert parts[0]["text"] == f"see {TOKEN}"
    assert parts[1] == image


# --- streams -------------------------------------------------------------------


def test_stream_restores_a_placeholder_split_over_three_chunks(provider, tmp_path):
    provider.reply = lambda m: f"a {TOKEN} b"
    provider.stream_cuts = [4, 7]  # "a [[S" | "_1]" | "] b"
    chunks = drain(build([Swap()], tmp_path))
    assert text_of(chunks) == "a SECRET b"
    assert "[[" not in text_of(chunks)
    assert "SECRET" not in sent(provider)


def test_stream_holds_only_a_possible_partial_token(provider, tmp_path):
    provider.reply = lambda m: "plain words here"
    provider.stream_cuts = [5, 11]
    chunks = drain(build([Swap()], tmp_path))
    pieces = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
    assert pieces == ["plain", " words", " here"]


def test_stream_releases_a_held_tail_before_the_finish(provider, tmp_path):
    provider.reply = lambda m: "cut off [[S_"
    chunks = drain(build([Swap()], tmp_path))
    assert text_of(chunks) == "cut off [[S_"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(c.choices[0].finish_reason is None for c in chunks[:-1])


def test_stream_ending_without_a_finish_still_releases_a_held_tail(provider, tmp_path):
    provider.reply = lambda m: "cut off [[S_"
    provider.no_finish = True
    assert text_of(drain(build([Swap()], tmp_path))) == "cut off [[S_"


def test_stream_restores_tool_call_arguments_split_across_chunks(provider, tmp_path):
    provider.tool_args = '{"k": "[[S_1]]"}'
    provider.stream_cuts = [9, 13]
    chunks = drain(build([Swap()], tmp_path))
    assert args_of(chunks) == '{"k": "SECRET"}'
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_stream_trace_holds_the_transformed_form_and_client_the_restored(provider, tmp_path):
    provider.reply = lambda m: f"a {TOKEN} b"
    provider.stream_cuts = [4, 7]
    controller = build([Swap()], tmp_path)
    seen = []
    controller.quality_manager.record_trace = lambda prompt, model, resp, *a, **k: seen.append(
        (prompt, json.dumps(resp))
    )

    async def go():
        res = await controller.acompletion(model="default", messages=messages(), stream=True)
        out = [c async for c in res]
        await asyncio.gather(*list(controller._trace_writes))
        return out

    chunks = asyncio.run(go())
    assert text_of(chunks) == "a SECRET b"
    assert len(seen) == 1
    prompt, response = seen[0]
    assert "SECRET" not in prompt and "SECRET" not in response
    assert TOKEN in prompt and TOKEN in response


def test_sync_completion_restores_non_stream_and_stream(provider, tmp_path):
    controller = build([Swap()], tmp_path)
    res = controller.completion(model="default", messages=messages())
    assert res.choices[0].message.content == "got: tell SECRET now"
    assert provider.calls[0]["messages"][0]["content"] == f"tell {TOKEN} now"

    provider.reply = lambda m: f"a {TOKEN} b"
    provider.stream_cuts = [4, 7]
    stream = controller.completion(model="default", messages=messages(), stream=True)
    stream._hidden_params["x"] = 1
    assert text_of(list(stream)) == "a SECRET b"
    assert "SECRET" not in sent(provider)


def test_sync_completion_refuses_when_the_transform_raises(provider, tmp_path):
    with pytest.raises(TransformError):
        build([Boom()], tmp_path).completion(model="default", messages=messages())
    assert provider.calls == []


# --- cache and trace order ---------------------------------------------------------


def test_trace_holds_the_transformed_prompt_and_reply_client_the_restored(provider, tmp_path):
    controller = build([Swap()], tmp_path)
    seen = []
    controller.quality_manager.record_trace = lambda prompt, model, resp, *a, **k: seen.append(
        (prompt, json.dumps(resp))
    )
    res = ask(controller)
    assert res.choices[0].message.content == "got: tell SECRET now"
    prompt, response = seen[0]
    assert TOKEN in prompt and "SECRET" not in prompt
    assert TOKEN in response and "SECRET" not in response


def test_a_selected_request_is_never_cached(provider, tmp_path):
    controller = build([Swap()], tmp_path, cache=True)
    ask(controller)
    ask(controller)
    assert len(provider.calls) == 2
    assert not list(Path(tmp_path).glob("cache.db*")) or "SECRET" not in b"".join(
        p.read_bytes() for p in Path(tmp_path).glob("cache.db*")
    ).decode("latin-1")


def test_an_unselected_request_is_still_cached(provider, tmp_path):
    class Never(Swap):
        def selects(self, ctx):
            return False

    controller = build([Never()], tmp_path, cache=True)
    ask(controller)
    ask(controller)
    assert len(provider.calls) == 1
