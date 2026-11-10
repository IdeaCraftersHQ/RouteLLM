"""A transform that rewrites nothing; `tag` shows the options arrived."""


class TagTransform:
    def __init__(self, tag="seen"):
        self.tag = tag

    def selects(self, ctx):
        return False

    def forward(self, messages, ctx):
        return messages, None

    def inverse(self, text, state):
        return text
