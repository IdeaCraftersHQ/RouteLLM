"""Router backed by TypeSafe's Jev model via System One.

Asks Jev a single Noul (yes/no) question about the incoming prompt and
uses the returned probability directly as the strong-model win rate.

The `Router` base comes from `routellm.routers.base`, not from
`routers.py`, so importing this module does not pull in torch and does
not form a cycle with the module that registers it in `ROUTER_CLS`.
"""
from __future__ import annotations

import logging

from routellm.prompts import PromptFile, resolve
from routellm.routers.base import Router
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

#: Section of the prompt file this router reads, and the types its keys take.
PROMPT_SECTION = "router"
PROMPT_SCHEMA = {"instructions": str, "criteria": dict}

#: The only keys a Noul's criteria mapping may carry.
_CRITERIA_KEYS = frozenset({"true", "false"})


def _check_criteria(criteria, path):
    """Check prompt-file criteria carry only string "true"/"false" entries.

    The generic loader validates that `criteria` is a mapping; the
    yes/no shape is this router's own requirement, so it is checked
    here rather than in the loader.

    Parameters
    ----------
    criteria : dict
        The `router.criteria` mapping read from the prompt file.
    path : str or os.PathLike
        Prompt file path, named in the error message.

    Raises
    ------
    ValueError
        If a key is outside {"true", "false"} or a value is not a str.
    """
    for key, value in criteria.items():
        if key not in _CRITERIA_KEYS:
            raise ValueError(f"{path}: unknown key {key!r} in 'router.criteria'")
        if not isinstance(value, str):
            raise ValueError(
                f"{path}: router.criteria.{key} must be str, "
                f"got {type(value).__name__}"
            )


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
    criteria : dict, optional
        Override for the default yes/no criteria passed to the Noul.
    prompt_file : str or os.PathLike, optional
        YAML prompt file supplying ``router.instructions`` and
        ``router.criteria``. Most specific wins: explicit kwarg, then
        file value, then built-in default.
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
        criteria=None,
        prompt_file=None,
        base_url=None,
        transport=None,
    ):
        typesafe_sdk = require_typesafe_sdk()

        section = (
            PromptFile.load(prompt_file).section(PROMPT_SECTION, PROMPT_SCHEMA)
            if prompt_file is not None
            else {}
        )
        file_criteria = section.get("criteria")
        if file_criteria is not None:
            _check_criteria(file_criteria, prompt_file)
        self.instructions, instructions_source = resolve(
            instructions, section.get("instructions"), DEFAULT_INSTRUCTIONS
        )
        self.criteria, criteria_source = resolve(
            criteria, file_criteria, dict(DEFAULT_CRITERIA)
        )
        logger.debug(
            "jev router prompt sources instructions=%s criteria=%s",
            instructions_source,
            criteria_source,
        )

        self.max_prompt_chars = max_prompt_chars
        self._noul = typesafe_sdk.Noul(
            instructions=self.instructions,
            criteria=self.criteria,
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
