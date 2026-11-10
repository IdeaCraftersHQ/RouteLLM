"""Request transforms: rewrite the messages before a call, restore the reply after.

A transform sees the messages of one upstream call, returns the messages
to send, and keeps a `state` it needs to undo the rewrite on what comes
back. The controller runs the chain at the one point every upstream call
passes through: after the candidate is chosen (primary, fallback, sibling,
pool member, shadow or compare target alike) and before the call, on every
attempt. Nothing is sent that a transform has not seen, and a transform
that raises ends the request (`TransformError`): the untransformed
messages are never sent as a fallback.

Order on a request (per attempt)::

    forward(messages)  ->  upstream call  ->  cache write, trace write
                                         ->  inverse(response)  ->  client

The cache and the trace hold the response as the upstream returned it,
the transformed form; only the copy that goes back to the client is
restored. For the same reason a request that a transform `selects` is
neither served from the cache nor stored in it: a cached reply could not
be restored without the per-request state. Traces record the transformed
prompt, never the original.

With no transform configured the chain is empty and the request path is
untouched.
"""

import copy
import importlib
import importlib.metadata
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from routellm.config import ConfigError

ENTRY_POINT_GROUP = "routellm.transforms"


class TransformError(Exception):
    """A transform failed; the request is refused, never sent as it was.

    Attributes
    ----------
    reason : str
        Short, client-safe cause. The underlying exception stays in the
        server log and on `__cause__`.
    """

    def __init__(self, reason: str, *, transform: str | None = None):
        self.reason = reason
        self.transform = transform
        super().__init__(f"request transform failed: {reason}")


@dataclass(frozen=True)
class TransformContext:
    """What a transform may read about the call it is about to rewrite.

    Attributes
    ----------
    endpoint : str or None
        The candidate's registry name; None in the request-level `selects`
        check, which runs before any candidate is chosen.
    model : str or None
        The upstream model id the call carries.
    reach : frozenset[str]
        The candidate's reach labels (`local`, `vpn`, `cloud`); empty for
        an anonymous endpoint, which carries none.
    data_class : str or None
        The request's effective class: the one cache, compare and policy
        read.
    treatment : str or None
        That class's treatment in the loaded policy (`raw`, `redact`...),
        None without a policy or class.
    request_id : str or None
        The caller's request id, for correlation only.
    shadow : bool
        True for a shadow or compare call, whose reply no client reads.
    """

    endpoint: str | None = None
    model: str | None = None
    reach: frozenset[str] = frozenset()
    data_class: str | None = None
    treatment: str | None = None
    request_id: str | None = None
    shadow: bool = False


@runtime_checkable
class StreamRestorer(Protocol):
    """Restores one text field of a stream, a piece at a time."""

    def feed(self, text: str) -> str:
        """Take the next piece; return the text that is safe to release."""
        ...

    def finish(self) -> str:
        """The stream ended: release whatever is still held."""
        ...


@runtime_checkable
class RequestTransform(Protocol):
    """One rewrite of a call's messages and its inverse on the reply.

    Only `forward` and `inverse` are required. `selects` defaults to
    True; `restorer` defaults to restoring each streamed piece on its
    own, which is right only when a rewritten token can never be split
    across pieces (see `HoldbackRestorer` for when it can).
    """

    def forward(self, messages: list[dict], ctx: TransformContext) -> tuple[list[dict], Any]:
        """Return the messages to send and the state `inverse` needs.

        Must not mutate `messages`. Reach, class and endpoint are on
        `ctx`: a transform that leaves a leaf raw returns the messages
        unchanged and a state of None, which skips its inverse.
        Anything raised refuses the request.
        """
        ...

    def inverse(self, text: str, state: Any) -> str:
        """Restore one complete piece of reply text (message content,
        a tool call's arguments)."""
        ...

    def selects(self, ctx: TransformContext) -> bool:
        """Optional. Whether this transform may rewrite requests like
        this one at all (asked once per request, with `endpoint` None).
        A request it selects bypasses the response cache."""
        ...

    def restorer(self, state: Any) -> StreamRestorer:
        """Optional. A fresh restorer for one streamed text field."""
        ...


class IdentityTransform:
    """The default: rewrites nothing, so there is nothing to restore."""

    def selects(self, ctx: TransformContext) -> bool:
        return False

    def forward(self, messages, ctx):
        return messages, None

    def inverse(self, text, state):
        return text


class HoldbackRestorer:
    """A stream restorer for tokens that may arrive split across pieces.

    Buffers only what could still turn into a token: `holdback(buffer)`
    says how many trailing characters might be the start of one (0 when
    none). The rest of the buffer is restored with `inverse` and
    released. A held tail is released, restored, by `finish`.

    Parameters
    ----------
    inverse : callable
        `str -> str`, applied to every released stretch.
    holdback : callable
        `str -> int`, the length of the suffix to keep back.
    """

    def __init__(self, inverse: Callable[[str], str], holdback: Callable[[str], int]):
        self._inverse = inverse
        self._holdback = holdback
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        keep = max(0, min(self._holdback(self._buffer), len(self._buffer)))
        cut = len(self._buffer) - keep
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._inverse(ready) if ready else ""

    def finish(self) -> str:
        tail, self._buffer = self._buffer, ""
        return self._inverse(tail) if tail else ""


class _PerPiece:
    """The default restorer: each piece restored on its own."""

    def __init__(self, inverse: Callable[[str], str]):
        self._inverse = inverse

    def feed(self, text: str) -> str:
        return self._inverse(text) if text else ""

    def finish(self) -> str:
        return ""


def map_text(messages: list[dict], fn: Callable[[str], str]) -> list[dict]:
    """Return a copy of `messages` with `fn` applied to every text it carries.

    Covers string content, the `text` of text parts in a multimodal
    content list (other parts are left as they are), tool calls'
    function arguments and a legacy `function_call`'s arguments, on
    every role. Does not mutate the input; a transform's `forward` is
    the intended caller.
    """
    out = copy.deepcopy(messages)
    for message in out:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = fn(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = fn(part["text"])
        for call in message.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                function["arguments"] = fn(function["arguments"])
        legacy = message.get("function_call")
        if isinstance(legacy, dict) and isinstance(legacy.get("arguments"), str):
            legacy["arguments"] = fn(legacy["arguments"])
    return out


def _chain_restorers(restorers: list[StreamRestorer]) -> StreamRestorer:
    """One restorer running `restorers` in order, each fed by the last."""

    class _Chain:
        def feed(self, text: str) -> str:
            for restorer in restorers:
                text = restorer.feed(text) if text else text
            return text

        def finish(self) -> str:
            carry = ""
            for restorer in restorers:
                carry = restorer.feed(carry) if carry else ""
                carry += restorer.finish()
            return carry

    return _Chain()


@dataclass
class Applied:
    """The transforms that ran on one call, with the state of each.

    Built by `TransformRunner.forward`; undoes them in reverse order.
    """

    steps: list[tuple[Any, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """True when at least one transform rewrote this call."""
        return bool(self.steps)

    def restore_text(self, text: str) -> str:
        for transform, state in reversed(self.steps):
            text = transform.inverse(text, state)
        return text

    def restorer(self) -> StreamRestorer:
        """A fresh restorer for one streamed text field."""
        made = []
        for transform, state in reversed(self.steps):
            make = getattr(transform, "restorer", None)
            if make is not None:
                made.append(make(state))
            else:
                made.append(_PerPiece(lambda t, tr=transform, st=state: tr.inverse(t, st)))
        return _chain_restorers(made)

    def restore_response(self, response):
        """Restore `response` in place and return it.

        Rewrites message content, reasoning, tool calls' arguments and a
        legacy function call's arguments on every choice. Call it after
        the cache and trace writes, never before.
        """
        if not self.active:
            return response
        for choice in getattr(response, "choices", None) or ():
            message = getattr(choice, "message", None)
            if message is None:
                continue
            for name in ("content", "reasoning_content"):
                value = getattr(message, name, None)
                if isinstance(value, str):
                    setattr(message, name, self.restore_text(value))
            for call in getattr(message, "tool_calls", None) or ():
                function = getattr(call, "function", None)
                if function is not None and isinstance(function.arguments, str):
                    function.arguments = self.restore_text(function.arguments)
            legacy = getattr(message, "function_call", None)
            if legacy is not None and isinstance(getattr(legacy, "arguments", None), str):
                legacy.arguments = self.restore_text(legacy.arguments)
        return response

    def stream_restore(self):
        """The per-stream restorer of chunks (see `routellm.transform_stream`)."""
        from routellm.transform_stream import StreamRestore

        return StreamRestore(self.restorer)


NOT_APPLIED = Applied()


class TransformRunner:
    """The configured chain of transforms, in order.

    Parameters
    ----------
    transforms : list, optional
        Instances of `RequestTransform`. Empty or None: the runner is
        inactive and every method is a pass-through.
    """

    def __init__(self, transforms: list[Any] | None = None):
        self.transforms = [t for t in (transforms or []) if not isinstance(t, IdentityTransform)]

    @property
    def active(self) -> bool:
        return bool(self.transforms)

    def selects(self, ctx: TransformContext) -> bool:
        """Whether any transform may rewrite a request like this one."""
        for transform in self.transforms:
            check = getattr(transform, "selects", None)
            if check is None or check(ctx):
                return True
        return False

    def forward(self, messages: list[dict], ctx: TransformContext) -> tuple[list[dict], Applied]:
        """Run the chain over `messages`.

        Raises
        ------
        TransformError
            When any transform raises or returns something that is not a
            message list. Nothing was sent.
        """
        if not self.transforms:
            return messages, NOT_APPLIED
        applied = Applied()
        current = messages
        for transform in self.transforms:
            name = type(transform).__name__
            try:
                current, state = transform.forward(current, ctx)
            except TransformError:
                raise
            except Exception as exc:
                raise TransformError(f"{name} failed", transform=name) from exc
            if not isinstance(current, list):
                raise TransformError(f"{name} returned no message list", transform=name)
            if state is not None:
                applied.steps.append((transform, state))
        return current, applied

    def __iter__(self) -> Iterator[Any]:
        return iter(self.transforms)


def context_for(
    host, reqs, endpoint: str | None, request_id=None, shadow=False
) -> TransformContext:
    """Build the context of one call from the controller's own state."""
    data_class = getattr(reqs, "data_class", None)
    policy = getattr(host, "policy", None)
    spec = policy.classes.get(data_class) if policy is not None and data_class else None
    model, reach = None, frozenset()
    if endpoint is not None:
        resolved = host.endpoints.resolve(endpoint)
        model, reach = resolved.model, frozenset(resolved.reach)
    return TransformContext(
        endpoint=endpoint,
        model=model,
        reach=reach,
        data_class=data_class,
        treatment=spec.treatment if spec is not None else None,
        request_id=request_id,
        shadow=shadow,
    )


def forward_call(host, kwargs: dict, endpoint: str, reqs, request_id=None, shadow=False):
    """Apply the host's transforms to one call's kwargs.

    Returns `(kwargs, applied)`; `kwargs` is a shallow copy with the
    transformed messages when a transform ran, else the input itself.
    """
    runner = getattr(host, "transforms", None)
    if runner is None or not runner.active:
        return kwargs, NOT_APPLIED
    ctx = context_for(host, reqs, endpoint, request_id, shadow)
    messages, applied = runner.forward(kwargs["messages"], ctx)
    if not applied.active and messages is kwargs["messages"]:
        return kwargs, applied
    return {**kwargs, "messages": messages}, applied


def load_transforms(spec: Any) -> list[Any]:
    """Build the chain a `transforms:` config list names.

    Each entry is a string or a mapping with `use` and optional
    `options`. `use` is a `module:attr` import path or the name of an
    entry point in the `routellm.transforms` group. A class or factory
    is called with `options` as keyword arguments; an object that
    already has `forward` is used as it is.

    Raises
    ------
    ConfigError
        On a malformed entry, an unknown name, an import that fails or
        an object that is not a transform. Startup stops.
    """
    if spec is None:
        return []
    if not isinstance(spec, list):
        raise ConfigError(f"`transforms:` must be a list, got {type(spec).__name__}")
    built = []
    for index, entry in enumerate(spec):
        where = f"transforms[{index}]"
        if isinstance(entry, str):
            entry = {"use": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("use"), str):
            raise ConfigError(f"`{where}` needs `use:` (a module:attr path or an entry point name)")
        unknown = sorted(set(entry) - {"use", "options"})
        if unknown:
            raise ConfigError(f"`{where}`: unknown key(s) {', '.join(unknown)}")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            raise ConfigError(f"`{where}.options` must be a mapping")
        target = _resolve(entry["use"], where)
        try:
            is_instance = not inspect.isclass(target) and hasattr(target, "forward")
            obj = target if is_instance else target(**options)
        except Exception as exc:
            raise ConfigError(f"`{where}`: building {entry['use']!r} failed: {exc}") from exc
        if not (
            callable(getattr(obj, "forward", None)) and callable(getattr(obj, "inverse", None))
        ):
            raise ConfigError(f"`{where}`: {entry['use']!r} has no forward() and inverse()")
        built.append(obj)
    return built


def _resolve(use: str, where: str):
    if ":" in use:
        module, _, attr = use.partition(":")
        try:
            target = importlib.import_module(module)
            for part in attr.split("."):
                target = getattr(target, part)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"`{where}`: cannot import {use!r}: {exc}") from exc
        return target
    if use == "identity":
        return IdentityTransform
    for point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if point.name == use:
            try:
                return point.load()
            except Exception as exc:
                raise ConfigError(f"`{where}`: entry point {use!r} failed to load: {exc}") from exc
    raise ConfigError(
        f"`{where}`: {use!r} is neither a module:attr path nor an entry point in "
        f"{ENTRY_POINT_GROUP!r}"
    )
