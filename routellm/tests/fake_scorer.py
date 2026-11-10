"""A scorer the comparison tests name through a `module:` spec.

Deterministic and offline: the score is the length of the output over
100, so a longer answer scores higher. Every call is recorded in
`CALLS` so a test can see what was judged.
"""

from types import SimpleNamespace

CALLS: list[tuple[str, dict]] = []


class LengthScorer:
    def score(self, output, context):
        CALLS.append((output, dict(context)))
        return SimpleNamespace(score=min(len(output) / 100, 1.0), breakdown={})


def make():
    return LengthScorer()
