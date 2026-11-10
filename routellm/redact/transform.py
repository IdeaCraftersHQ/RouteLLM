"""The `redact` treatment as a request transform.

For a request whose class has treatment `redact`, a call to a leaf that
is not local-only goes out with every text swapped for placeholders
(`[[PERSON_1]]`): system, user, assistant and tool messages, the text
parts of multimodal content, and tool calls' arguments. The reply is
restored here, locally, on its way back to the client, a stream too (a
placeholder split across chunks is held until it completes).

    transforms live in `routellm.transform`; this one is added by a
    `redact:` config section (see docs/redact.md).

What it never does:

- send text it could not redact. Any failure (analyzer, store, a mapping
  that cannot be read back) raises, and the controller refuses the
  request before a provider is called;
- touch a leaf whose reach is exactly `local`: redaction is for egress;
- restore from the store mid-reply: the mapping is read once, before the
  call, so a reply is restored from memory or not sent at all.

The controller keeps such a request out of the response cache and
traces hold digests only for a non-`raw` class, so the raw values live
in the token store and nowhere else.
"""

import json
import logging
import secrets
from collections.abc import Callable
from typing import Any

from routellm.auth import current_principal
from routellm.redact.placeholders import RedactContext, Redactor, Restorer, split_incomplete
from routellm.redact.tokens import RedactError
from routellm.transform import LOCAL_ONLY, HoldbackRestorer, TransformContext, map_text

__all__ = ["RedactTransform", "union_languages"]

TREATMENT = "redact"
#: litellm's loggers; at DEBUG they print the messages of every request.
_LITELLM_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
_TEXT_PARTS = {"text"}


class _NoDebug(logging.Filter):
    """Drops records below INFO: litellm's debug lines carry message text."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO


_NO_DEBUG = _NoDebug()


def scrub_litellm_logs() -> None:
    """Keep litellm's debug records, which hold raw prompts, out of every log.

    Idempotent. The raw request is seen by litellm (token counting) before
    the class is known, so the only safe place to stop it is the logger.
    """
    for name in _LITELLM_LOGGERS:
        logging.getLogger(name).addFilter(_NO_DEBUG)


def union_languages(
    analyze: Callable[[str, str], list[dict[str, Any]]], languages: tuple[str, ...]
) -> Callable[[str, str], list[dict[str, Any]]]:
    """An analyzer that runs `analyze` once per language and merges the spans.

    A request does not say which language it is in, and a name found under
    one language but not the other must not escape: over-detecting is the
    safe side (overlaps resolve to the earlier, longer span).
    """

    def merged(text: str, _language: str) -> list[dict[str, Any]]:
        seen, out = set(), []
        for language in languages:
            for span in analyze(text, language):
                key = (span["start"], span["end"], span["category"])
                if key not in seen:
                    seen.add(key)
                    out.append(span)
        return out

    return merged


class _Ctx:
    """What the reply needs, kept per call."""

    __slots__ = ("restorer",)

    def __init__(self, restorer: Restorer):
        self.restorer = restorer


def _holdback(buffer: str) -> int:
    return len(split_incomplete(buffer)[1])


class RedactTransform:
    """Redact on the way out, restore on the way back.

    Parameters
    ----------
    redactor : Redactor
        Placeholders and the local token store.
    non_text : {"refuse", "pass"}
        What to do with a message part that is not text (an image, audio,
        a file): `refuse` (default) fails the request, since its content
        cannot be redacted; `pass` sends it as it is.
    """

    redacts = True

    def __init__(self, redactor: Redactor, *, non_text: str = "refuse"):
        if non_text not in ("refuse", "pass"):
            raise ValueError("non_text must be 'refuse' or 'pass'")
        self.redactor = redactor
        self.non_text = non_text
        scrub_litellm_logs()

    # -- selection -----------------------------------------------------------------

    def selects(self, ctx: TransformContext) -> bool:
        return ctx.treatment == TREATMENT

    # -- out -----------------------------------------------------------------------

    def forward(self, messages: list[dict], ctx: TransformContext) -> tuple[list[dict], Any]:
        if ctx.treatment != TREATMENT or (ctx.reach and ctx.reach <= LOCAL_ONLY):
            return messages, None
        if self.non_text == "refuse":
            _refuse_non_text(messages)
        principal = current_principal.get()
        rctx = RedactContext(
            principal=principal.principal if principal is not None else "anonymous",
            request_id=ctx.request_id or secrets.token_hex(12),
        )
        mapping: list[str | None] = [None]

        def red(text: str) -> str:
            if not text:
                return text
            out, mapping[0] = self.redactor.redact(text, rctx, mapping[0])
            return out

        def red_json(raw: str) -> str:
            # Tool arguments are JSON: the strings inside are what carry
            # values (a JSON dump escapes `é`, so the raw text would hide it).
            try:
                doc = json.loads(raw)
            except ValueError:
                return red(raw)
            changed = [False]

            def walk(node):
                if isinstance(node, str):
                    new = red(node)
                    changed[0] = changed[0] or new != node
                    return new
                if isinstance(node, list):
                    return [walk(v) for v in node]
                if isinstance(node, dict):
                    return {walk(k): walk(v) for k, v in node.items()}
                return node

            doc = walk(doc)
            return json.dumps(doc, ensure_ascii=False) if changed[0] else raw

        out = map_text(messages, red, red_json)
        if mapping[0] is None:
            return out, None  # no text at all: nothing was minted, nothing to restore
        return out, _Ctx(self.redactor.restorer(mapping[0], rctx))

    # -- back ----------------------------------------------------------------------

    def inverse(self, text: str, state: _Ctx) -> str:
        return state.restorer.text(text)

    def inverse_arguments(self, text: str, state: _Ctx) -> str:
        return state.restorer.json_string(text)

    def restorer(self, state: _Ctx) -> HoldbackRestorer:
        return HoldbackRestorer(lambda t: self.inverse(t, state), _holdback)

    def arguments_restorer(self, state: _Ctx) -> HoldbackRestorer:
        return HoldbackRestorer(lambda t: self.inverse_arguments(t, state), _holdback)


def _refuse_non_text(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type", "text") not in _TEXT_PARTS:
                raise RedactError("message part is not text and cannot be redacted")
