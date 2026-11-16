"""The decision step: recognizers, skip rule, model, sticky floor, failure policy.

Order per request: validate the declared class, read the sticky floor,
run the recognizers, and only when none of them settles the class and
no source is already at the strictest level ask the provider. Every
failure resolves through `on_error` and is recorded on the Decision;
nothing is swallowed.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from routellm.decision import window
from routellm.decision.config import DecisionConfig
from routellm.decision.models import Decision, DecisionError, DecisionRefused, RequestView
from routellm.decision.recognizers import Recognizer
from routellm.decision.sticky import StickyStore, conversation_key

logger = logging.getLogger(__name__)

__all__ = ["DecisionClassifier", "DecisionProvider"]


class DecisionProvider(Protocol):
    """Anything that labels a request: `async decide(view) -> Decision`."""

    def decide(self, view: RequestView) -> Awaitable[Decision]: ...


ProviderFactory = Callable[[DecisionConfig, Any], DecisionProvider]


class DecisionClassifier:
    """Label each request with a data class, an intent and an area."""

    def __init__(
        self,
        config: DecisionConfig,
        recognizers: Sequence[Recognizer] = (),
        provider: DecisionProvider | None = None,
        provider_factory: ProviderFactory | None = None,
        sticky: StickyStore | None = None,
    ):
        self.config = config
        self.recognizers = list(recognizers)
        self._provider = provider
        self._factory = provider_factory
        self.sticky = sticky
        if sticky is None and config.sticky.enabled:
            self.sticky = StickyStore(config.sticky.ttl_s, config.sticky.max_entries)

    @property
    def endpoint(self) -> str:
        """Registry endpoint the model call goes to."""
        return self.config.endpoint

    def bind(self, transport: Any) -> None:
        """Build the provider over `transport`, the model-call seam."""
        if self._factory is None:
            raise RuntimeError("no provider factory to bind a transport to")
        self._provider = self._factory(self.config, transport)

    # -- classification -------------------------------------------------

    async def classify(
        self,
        messages: list[dict[str, Any]],
        *,
        declared_class: str | None = None,
        conversation_id: str | None = None,
        user: str | None = None,
        use_recognizers: bool = True,
    ) -> Decision:
        """Return the Decision for one request.

        Raises
        ------
        DecisionRefused
            A failure occurred and `on_error` is `refuse`.
        """
        cfg = self.config
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            messages = []
        key = conversation_id or (
            conversation_key(messages, user) if self.sticky is not None else None
        )
        if declared_class is not None and declared_class not in cfg.data_classes:
            return self._failed(
                DecisionError("unknown_declared_class", f"{declared_class!r} is not configured")
            )

        chosen = window.select(messages, cfg.input)
        joined = "\n".join(t for t in (window.message_text(m) for m in chosen) if t)
        view = RequestView(
            messages=chosen,
            text=window.truncate(joined, cfg.input),
            full_text=window.full_text(messages),
            declared_class=declared_class,
            conversation_id=key,
        )

        # Declared, the sticky floor and a recognizer hit each name a class
        # without a model; the strictest of them is the floor.
        known = self._known(declared_class, self._floor_rank(key))
        hit = await self._recognize(view) if use_recognizers else None
        if hit is not None:
            known.append(
                (
                    cfg.rank(hit.data_class),
                    Decision(
                        hit.data_class,
                        source="recognizer",
                        detail={"recognizer": hit.name, **hit.detail},
                    ),
                )
            )
        floor = max(known, key=lambda pair: pair[0]) if known else None
        if floor is not None and (hit is not None or floor[0] >= cfg.rank(cfg.most_restrictive)):
            return self._remember(key, floor[1])

        try:
            decision = await self._ask(view)
        except DecisionError as err:
            return self._failed(err)
        if floor is not None and cfg.rank(decision.data_class) < floor[0]:
            decision = Decision(
                floor[1].data_class,
                decision.intent,
                decision.area,
                decision.confidence,
                floor[1].source,
                detail={"model_class": decision.data_class},
            )
        return self._remember(key, decision)

    # -- parts ------------------------------------------------------------

    def _floor_rank(self, key: str | None) -> int | None:
        return self.sticky.floor(key) if self.sticky is not None else None

    def _known(self, declared: str | None, floor: int | None) -> list[tuple[int, Decision]]:
        names = list(self.config.data_classes)
        known: list[tuple[int, Decision]] = []
        if declared is not None:
            known.append((self.config.rank(declared), Decision(declared, source="declared")))
        if floor is not None:
            known.append((floor, Decision(names[floor], source="sticky")))
        return known

    async def _recognize(self, view: RequestView):
        for recognizer in self.recognizers:
            if recognizer.blocking:
                match = await asyncio.to_thread(recognizer.recognize, view.full_text)
            else:
                match = recognizer.recognize(view.full_text)
            if match is not None:
                return match
        return None

    async def _ask(self, view: RequestView) -> Decision:
        if self._provider is None:
            raise DecisionError("transport", "no provider bound")
        cfg = self.config
        try:
            decision = await asyncio.wait_for(
                self._provider.decide(view), timeout=cfg.timeout_ms / 1000
            )
        except TimeoutError as exc:
            raise DecisionError("timeout", f"no answer within {cfg.timeout_ms} ms") from exc
        except DecisionError:
            raise
        except Exception as exc:
            raise DecisionError("transport", f"{type(exc).__name__}: {exc}") from exc
        if decision.data_class not in cfg.selectable:
            raise DecisionError("invalid_answer", f"data_class {decision.data_class!r}")
        if decision.confidence is None or decision.confidence < cfg.min_confidence:
            raise DecisionError(
                "low_confidence", f"{decision.confidence} below {cfg.min_confidence}"
            )
        return decision

    def _failed(self, err: DecisionError) -> Decision:
        logger.warning("decision step failed: %s (on_error=%s)", err, self.config.on_error)
        if self.config.on_error == "refuse":
            raise DecisionRefused(err)
        return Decision(
            self.config.most_restrictive,
            source="error",
            error=err.kind,
            detail={"message": err.message} if err.message else {},
        )

    def _remember(self, key: str | None, decision: Decision) -> Decision:
        # A declarable class is consent only its caller gave: never carried
        # into a later turn that did not declare it.
        if self.sticky is not None and decision.data_class not in self.config.declarable:
            self.sticky.record(key, self.config.rank(decision.data_class))
        return decision
