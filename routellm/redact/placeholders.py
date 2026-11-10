"""Reversible placeholders: `[[PERSON_1]]` out, the value back on return.

    redactor = Redactor(TokenStore("state/tokens.db"))
    ctx = RedactContext(principal="team-a", request_id="r1", language="fr")
    text, mapping_id = redactor.redact("Écrire à Jean Tremblay", ctx)
    # "Écrire à [[PERSON_1]]"
    redactor.restore(model_reply, mapping_id, ctx)

Pure Python: spans come from `routellm.redact.analyze` (needs the extra)
or from any `analyze(text, language) -> [{start, end, category, score}]`
you inject.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from routellm.redact.tokens import (
    MappingError,
    MappingExpired,
    RedactError,
    TokenStore,
    format_placeholder,
)

__all__ = [
    "MappingError",
    "MappingExpired",
    "RedactContext",
    "RedactError",
    "Redactor",
    "Restorer",
    "format_placeholder",
    "split_incomplete",
]

# Category, then `_<n>`; case and padding inside the brackets are forgiven
# on the way back because models retype what they are given.
_PLACEHOLDER = re.compile(r"\[\[\s*([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*)_(\d+)\s*\]\]")
# A suffix that may still become a placeholder once more text arrives.
_OPEN = re.compile(r"\[(?:\[[A-Za-z0-9_ ]{0,64}\]?)?$")


def _canonical(m: re.Match) -> str:
    return format_placeholder(m.group(1), int(m.group(2)))


def split_incomplete(text: str) -> tuple[str, str]:
    """`(safe, pending)`: hold `pending` back until the next chunk arrives.

    `pending` is a trailing fragment that could still grow into a
    placeholder (`[`, `[[PERS`, `[[PERSON_1]`); empty when there is none.
    """
    m = _OPEN.search(text)
    return (text, "") if m is None else (text[: m.start()], text[m.start() :])


@dataclass(frozen=True)
class RedactContext:
    """Who is asking. Carries no text.

    `principal` and `request_id` are required; `conversation_id` only
    when `stable_across_requests` is on; `language` is `fr` or `en`.
    """

    principal: str
    request_id: str
    conversation_id: str | None = None
    language: str = "en"


class Redactor:
    """Swap detected spans for placeholders and back.

    By default placeholders are scoped to one request and their numbers
    are never reused for the same principal, so two requests cannot be
    linked through them. With `stable_across_requests`, the scope is the
    conversation: the same value keeps the same placeholder for as long
    as the conversation's mappings live.
    """

    def __init__(
        self,
        store: TokenStore,
        analyze: Callable[[str, str], list[dict[str, Any]]] | None = None,
        *,
        stable_across_requests: bool = False,
    ):
        self.store = store
        self._analyze = analyze
        self.stable = stable_across_requests

    def _scope(self, ctx: RedactContext) -> tuple[str, str]:
        """`(scope, counter_key)` for `ctx`."""
        if self.stable:
            if not ctx.conversation_id:
                raise RedactError("stable_across_requests needs a conversation id")
            scope = self.store.scope_id("conversation", ctx.principal, ctx.conversation_id)
            return scope, scope
        return (
            self.store.scope_id("request", ctx.principal, ctx.request_id),
            self.store.scope_id("principal", ctx.principal, ""),
        )

    def _spans(self, text: str, ctx: RedactContext) -> list[dict[str, Any]]:
        if self._analyze is None:
            from routellm.redact import analyze

            self._analyze = analyze
        spans = []
        for s in sorted(self._analyze(text, ctx.language), key=lambda s: (s["start"], -s["end"])):
            if not 0 <= s["start"] < s["end"] <= len(text):
                raise RedactError("span outside the text")
            if spans and s["start"] < spans[-1]["end"]:
                continue  # overlap: the earlier, longer span wins
            spans.append(s)
        return spans

    def redact(
        self, text: str, ctx: RedactContext, mapping_id: str | None = None
    ) -> tuple[str, str]:
        """`(redacted_text, mapping_id)`.

        Placeholder-shaped text already in `text` is left alone and never
        assigned to a value, so a restore cannot mistake it for one.
        `mapping_id`, from an earlier call for the same `ctx`, makes the
        texts of one request share a mapping instead of opening a new one.
        """
        scope, counter_key = self._scope(ctx)
        spans = self._spans(text, ctx)
        reserved = frozenset(_canonical(m) for m in _PLACEHOLDER.finditer(text))
        found = self.store.assign(
            scope,
            counter_key,
            [(s["category"], text[s["start"] : s["end"]]) for s in spans],
            reserved,
            counter_expires=self.stable,
        )
        parts, pos = [], 0
        for s, ph in zip(spans, found, strict=True):
            parts += [text[pos : s["start"]], ph]
            pos = s["end"]
        parts.append(text[pos:])
        if mapping_id is None:
            mapping_id = self.store.new_mapping(ctx.principal, scope)
        return "".join(parts), mapping_id

    def restore(self, text: str, mapping_id: str, ctx: RedactContext) -> str:
        """`text` with each known placeholder replaced by its value.

        One pass; a placeholder with no mapping (mangled number, partial
        echo, never issued) is left exactly as the model wrote it.
        Raises `MappingExpired` / `MappingError` rather than return text
        with placeholders the caller cannot trust.
        """
        scope, _ = self._scope(ctx)
        values = self.store.resolve(mapping_id, ctx.principal, scope)
        return _PLACEHOLDER.sub(lambda m: values.get(_canonical(m), m.group(0)), text)

    def restorer(self, mapping_id: str, ctx: RedactContext) -> "Restorer":
        """A `Restorer` holding the mapping's values, read once.

        Raises `MappingExpired` / `MappingError` now, before anything is
        sent, rather than on a reply that is already on its way back.
        """
        scope, _ = self._scope(ctx)
        return Restorer(self.store.resolve(mapping_id, ctx.principal, scope))


class Restorer:
    """Restores placeholders from values already read; touches no store.

    `text` puts values back as they are. `json_string` is for text inside
    a JSON document (tool-call arguments): each value is escaped so the
    document stays valid.
    """

    def __init__(self, values: dict[str, str]):
        self._values = values

    def text(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self._values.get(_canonical(m), m.group(0)), text)

    def json_string(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            value = self._values.get(_canonical(m))
            return m.group(0) if value is None else json.dumps(value)[1:-1]

        return _PLACEHOLDER.sub(sub, text)
