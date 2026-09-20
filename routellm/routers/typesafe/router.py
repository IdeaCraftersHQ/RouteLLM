"""Router backed by TypeSafe's Jev model via System One.

Asks Jev a single Noul (yes/no) question about the incoming prompt and
uses the returned probability directly as the strong-model win rate.

Import this module through `routellm.routers.routers`, which registers
`JevRouter` in `ROUTER_CLS` and binds the `Router` base it subclasses.
Importing it first, before `routers.py` has executed, is a circular
import.
"""
from __future__ import annotations

import logging

from routellm.routers.routers import Router
from routellm.routers.typesafe import require_typesafe_sdk

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = (
    "Would a frontier-class model answer this prompt materially better "
    "than a small, cheap model?"
)

DEFAULT_CRITERIA = {
    "true": (
        "The prompt needs multi-step reasoning, code beyond boilerplate, "
        "handling of long or ambiguous input, or precise domain knowledge."
    ),
    "false": (
        "The prompt is a greeting, a simple lookup, a short rewrite, or "
        "plain formatting work."
    ),
}

QUESTION_KEY = "strong"


class JevRouter(Router):
    """Route by asking TypeSafe's Jev model one Noul question.

    Jev returns a probability in [0, 1] for a yes answer, which maps
    directly onto the strong-model win rate the controller thresholds on.

    SDK exceptions (``TypeSafeRateLimitError``, ``TypeSafeAPITimeoutError``,
    ``TypeSafeBadRequestError`` when a truncated prompt still exceeds the
    state token budget, ...) propagate to the caller once the SDK has
    exhausted its own retries. There is no neutral 0.5 fallback: the
    controller's resilience layer wraps completion calls only, so a hidden
    default would silently skew threshold calibration instead of surfacing
    the failure.

    Parameters
    ----------
    model : str, optional
        Jev model id. ``None`` falls through to the SDK's own default
        (``TYPESAFE_DEFAULT_MODEL``, else ``jev-latest``).
    timeout : float, optional
        Per-request timeout in seconds. ``None`` uses the SDK default.
    max_retries : int, default 3
        Retry budget handed to the SDK's ``RetryPolicy``.
    max_prompt_chars : int, default 100_000
        Prompts longer than this are truncated before being sent.
    instructions : str, optional
        Override for the default Noul question.
    base_url : str, optional
        Override for the API base URL. ``None`` uses the SDK default.
    transport : httpx2.BaseTransport, optional
        Custom transport; exists so tests can inject a ``MockTransport``.

    Notes
    -----
    No ``api_key`` parameter is exposed: credentials come from the
    environment so that keys never land in ``config.yaml``.
    """

    def __init__(
        self,
        model=None,
        timeout=None,
        max_retries=3,
        max_prompt_chars=100_000,
        instructions=None,
        base_url=None,
        transport=None,
    ):
        typesafe_sdk = require_typesafe_sdk()

        self.max_prompt_chars = max_prompt_chars
        self.instructions = (
            DEFAULT_INSTRUCTIONS if instructions is None else instructions
        )
        self._noul = typesafe_sdk.Noul(
            instructions=self.instructions,
            criteria=DEFAULT_CRITERIA,
        )
        self.client = typesafe_sdk.TypeSafeClient(
            model=model,
            timeout=timeout,
            retry=typesafe_sdk.RetryPolicy(max_retries=max_retries),
            base_url=base_url,
            transport=transport,
        )

    def calculate_strong_win_rate(self, prompt):
        """Ask Jev whether the prompt warrants the strong model.

        Parameters
        ----------
        prompt : str
            Input prompt; truncated to ``max_prompt_chars`` before sending.

        Returns
        -------
        float
            Jev's probability of a yes answer, in [0, 1].
        """
        text = prompt[: self.max_prompt_chars]
        response = self.client.system_one(
            {"prompt": text},
            {QUESTION_KEY: self._noul},
        )
        logger.debug(
            "jev system_one model=%s input_tokens=%s",
            response.model,
            response.usage.input_tokens,
        )
        return response.nouls[QUESTION_KEY].noul

    def close(self):
        """Close the underlying TypeSafe HTTP client."""
        self.client.close()
