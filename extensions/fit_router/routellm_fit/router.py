"""Router backed by a fit advisor trained on scored traces.

Asks the advisor what it thinks of the prompt and uses the returned
confidence directly as the strong-model win rate. The advice's `domain`
and `steering_text` are ignored: routellm needs one number, and the
advisor's confidence is it.

The `Router` base comes from `routellm.routers.base`, not from
`routers.py`, so importing this module pulls in no torch. The class is
not registered here: the `routellm.routers` entry point declared in
this package's pyproject does that at discovery time.
"""

import logging
from typing import Any, Optional

from routellm.routers.base import Router

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:8080"


class FitRouter(Router):
    """Route on a trained fit advisor's confidence.

    Parameters
    ----------
    endpoint : str, optional
        Base URL of a `fit serve` process. Defaults to
        `http://localhost:8080`.
    timeout_ms : int, optional
        Per-request timeout in milliseconds (default 5000).
    advisor : object, optional
        A ready advisor, used instead of building a remote one. It only
        has to implement `generate_advice(context) -> Advice`. Exists
        so a local export, or a test stand-in, can be injected without
        a server.

    Notes
    -----
    Train the advisor with `python -m routellm_fit.train`, then serve
    it with `fit serve` and point `endpoint` at it.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_ms: int = 5000,
        advisor: Any = None,
    ):
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self._timeout_ms = timeout_ms
        self._advisor = advisor

    def _get_advisor(self):
        """Return the advisor, building a remote one on first use.

        fit is imported here rather than at module import, so a config
        that never routes on `fit` does not need fit installed.

        Raises
        ------
        RuntimeError
            If fit is not installed.
        """
        if self._advisor is None:
            try:
                from fit.advisor import RemoteAdvisor
            except ImportError as exc:
                raise RuntimeError(
                    "the fit router needs fit: pip install "
                    f"'routellm-fit' with its dependencies ({exc})"
                ) from exc

            self._advisor = RemoteAdvisor(
                endpoint=self.endpoint, timeout_ms=self._timeout_ms
            )
        return self._advisor

    def calculate_strong_win_rate(self, prompt):
        """Return the advisor's confidence for this prompt.

        Parameters
        ----------
        prompt : str
            The incoming prompt.

        Returns
        -------
        float
            The win rate, clamped to [0, 1]: it is compared against a
            threshold in that range, and an advisor that returns
            something outside it would otherwise route every request
            the same way.

        Raises
        ------
        RuntimeError
            If the advisor cannot be reached or answers with something
            that is not a number. The message names the endpoint, so
            the controller's fallback chain sees an ordinary failure
            with a readable cause.
        """
        advisor = self._get_advisor()

        try:
            advice = advisor.generate_advice({"prompt": prompt})
            confidence = float(advice.confidence)
        except Exception as exc:
            raise RuntimeError(
                f"fit advisor at {self.endpoint} gave no usable "
                f"confidence: {exc}"
            ) from exc

        if confidence < 0.0 or confidence > 1.0:
            logger.warning(
                "fit advisor at %s returned confidence %s outside "
                "[0, 1]; clamping",
                self.endpoint,
                confidence,
            )
        return max(0.0, min(1.0, confidence))
