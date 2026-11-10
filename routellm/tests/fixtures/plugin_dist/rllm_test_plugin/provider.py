"""Decision providers: one honest, one answering outside the label sets."""

from routellm.decision.models import Decision


class _Provider:
    def __init__(self, data_class):
        self._data_class = data_class

    async def decide(self, view):
        return Decision(self._data_class, intent="chat", confidence=0.9)


def make_provider(config, transport):
    return _Provider(config.most_restrictive)


def make_bad_provider(config, transport):
    return _Provider("not-a-configured-class")
