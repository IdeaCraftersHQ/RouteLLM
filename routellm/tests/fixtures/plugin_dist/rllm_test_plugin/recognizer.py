"""A recognizer that hits on a configured word."""

from routellm.decision.recognizers import Match


class WordRecognizer:
    blocking = False

    def __init__(self, spec):
        self.name, self.data_class = spec.name, spec.data_class
        self._word = spec.options["word"]

    def recognize(self, text, language=None):
        return (
            Match(self.name, self.data_class, {"word": self._word}) if self._word in text else None
        )


def make_recognizer(spec):
    return WordRecognizer(spec)
