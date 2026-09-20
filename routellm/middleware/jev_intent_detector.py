"""Intent detection backed by TypeSafe's Jev model.

Classifies a prompt with a single Choice question instead of embeddings
and cosine similarity. Exposes the same two public inference methods as
DomainIntentDetector, so either class fits the `intent_detector` slot of
IntentModelSelector.
"""

import logging
from typing import Any, Dict, List, Optional

from routellm.middleware.intent_model_selector import IntentModelMapping
from routellm.routers.typesafe import require_typesafe_sdk

logger = logging.getLogger(__name__)

#: Label returned when no configured intent fits, or when the model is
#: not confident enough in the one it picked.
GENERAL_INTENT = "general"

#: Description attached to the no-match label in the Choice criteria.
GENERAL_DESCRIPTION = "none of the listed intents fit"

#: Key under which the single classification question is submitted.
_QUESTION_NAME = "intent"


class JevIntentDetector:
    """Intent detector that classifies prompts with TypeSafe System One.

    Each configured intent becomes one criterion of a single Choice
    question, described by its mapping's `description`. A "general"
    criterion covers prompts that match none of them.

    Migrating from DomainIntentDetector: this class has no
    `add_examples`; fold the wording of those examples into each
    mapping's `description` (or into `descriptions`), which is what the
    model reads.
    """

    def __init__(
        self,
        intent_mappings: List[IntentModelMapping],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        confidence_floor: float = 0.5,
        descriptions: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
    ):
        """Initialize the Jev-backed intent detector.

        Parameters
        ----------
        intent_mappings : list[IntentModelMapping]
            Mappings whose `intent` labels become the choice criteria.
        model : str, optional
            TypeSafe model to classify with (default None, which lets
            the SDK pick its default).
        timeout : float, optional
            Per-request timeout in seconds (default None, the SDK
            default).
        confidence_floor : float, optional
            Minimum confidence to accept a detected intent (default
            0.5). Below it, detect_intent falls back to "general".
        descriptions : dict[str, str], optional
            Per-intent description overrides, keyed by intent label
            (default None). Takes precedence over
            `IntentModelMapping.description`.
        transport : httpx2.BaseTransport, optional
            Transport handed to the SDK client (default None). Useful
            for tests.

        Raises
        ------
        ImportError
            If the typesafe-sdk optional dependency is not installed.
        """
        typesafe_sdk = require_typesafe_sdk()

        self.intent_mappings = intent_mappings
        self.model = model
        self.timeout = timeout
        self.confidence_floor = confidence_floor
        self.descriptions = descriptions or {}

        self.criteria = self._build_criteria()
        self._question = typesafe_sdk.Choice(
            instructions=(
                "Classify the user prompt into exactly one intent. "
                "Pick 'general' when none of the other intents fit."
            ),
            criteria=self.criteria,
        )
        self._client = typesafe_sdk.TypeSafeClient(
            model=model,
            timeout=timeout,
            transport=transport,
        )

    def _build_criteria(self) -> Dict[str, Optional[str]]:
        """Build the Choice criteria from the configured mappings.

        Empty descriptions become None so the model relies on the label
        alone rather than on an empty hint.

        Returns
        -------
        dict[str, str | None]
            Criteria keyed by intent label, always including "general".
        """
        criteria: Dict[str, Optional[str]] = {}
        for mapping in self.intent_mappings:
            description = self.descriptions.get(mapping.intent, mapping.description)
            criteria[mapping.intent] = description or None
        criteria[GENERAL_INTENT] = self.descriptions.get(
            GENERAL_INTENT, GENERAL_DESCRIPTION
        )
        return criteria

    def _ask(self, prompt: str):
        """Submit the prompt as one Choice question and return its answer.

        Parameters
        ----------
        prompt : str
            The prompt to classify.

        Returns
        -------
        typesafe_sdk.ChoiceAnswer
            Answer carrying `choice`, `confidence` and `probabilities`.
        """
        response = self._client.system_one(
            {"prompt": prompt},
            {_QUESTION_NAME: self._question},
        )
        return response.choices[_QUESTION_NAME]

    def detect_intent(self, prompt: str) -> str:
        """Detect the intent of a prompt.

        Parameters
        ----------
        prompt : str
            The prompt to classify.

        Returns
        -------
        str
            The detected intent label. Returns "general" when the
            answer's confidence is below `confidence_floor`, or when
            the model picked a label outside the configured criteria.
        """
        answer = self._ask(prompt)
        if answer.confidence < self.confidence_floor:
            logger.debug(
                "Intent %r below confidence floor %.2f (%.2f); using %r",
                answer.choice,
                self.confidence_floor,
                answer.confidence,
                GENERAL_INTENT,
            )
            return GENERAL_INTENT
        if answer.choice not in self.criteria:
            logger.debug("Unknown intent %r; using %r", answer.choice, GENERAL_INTENT)
            return GENERAL_INTENT
        return answer.choice

    def get_intent_confidence(self, prompt: str) -> Dict[str, float]:
        """Get the probability of each configured intent for a prompt.

        Parameters
        ----------
        prompt : str
            The prompt to analyze.

        Returns
        -------
        dict[str, float]
            Probabilities keyed by intent label, restricted to the
            configured intents plus "general". Unlike
            DomainIntentDetector's normalized similarities, these come
            straight from the model and are not re-normalized.
        """
        probabilities = self._ask(prompt).probabilities
        return {
            intent: probability
            for intent, probability in probabilities.items()
            if intent in self.criteria
        }
