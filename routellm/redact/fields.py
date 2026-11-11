"""What a redact class may send besides message text: every request field, classified.

A request carries more than `messages`. Free text hides in `user`, a
message's `name`, tool and function definitions, a response schema, a
prefill. Each field is in exactly one list; one the lists do not name
is refused, so a field added upstream never leaves the host unseen.

ALLOWED     sent as it is: numbers, flags, stop sequences, headers the
            caller set. `stream_options`, `tool_choice` and
            `function_call` are shape-checked first.
REDACTED    free text: every string in it goes through the redactor.
            Identifiers in it (function, property, `required` names, a
            tool-call id) cannot be swapped without breaking the
            contract with the model, so one the redactor would change
            is refused instead.
REFUSED     known, and not safely redactable.
(other)     refused, unless unset (`None`, `[]`, `{}`).

Nothing in REDACTED comes back in a reply (definitions and prefills are
inputs, not outputs), so only message text and tool-call arguments are
restored; see `routellm.redact.transform`.
"""

from collections.abc import Callable
from typing import Any

from routellm.redact.tokens import RedactError

__all__ = ["ALLOWED", "REDACTED", "REFUSED", "check_known", "redact_fields", "redact_messages"]

Redact = Callable[[str], str]

ALLOWED = frozenset(
    {
        "extra_headers",
        "frequency_penalty",
        "function_call",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "timeout",
        "tool_choice",
        "top_logprobs",
        "top_p",
    }
)
REDACTED = frozenset({"functions", "metadata", "prediction", "response_format", "tools", "user"})
REFUSED = frozenset({"audio", "web_search_options"})
#: Fields the call always carries; the messages are handled by the transform.
_CALL = frozenset({"model", "messages"})

#: Schema keywords whose string values are structure, not text: refused if they hold a value.
_STRUCTURAL = frozenset({"type", "format", "$ref", "$schema", "$id", "$anchor", "pattern"})
#: Schema keywords whose keys are names (and whose values are schemas).
_NAMED = frozenset({"properties", "$defs", "definitions", "patternProperties"})
#: Schema keywords that hold data: keys and strings are all text.
_DATA = frozenset({"default", "examples", "example", "const", "enum"})
_MESSAGE_TEXT = frozenset({"refusal", "reasoning_content"})
_MESSAGE_PLAIN = frozenset({"role", "content", "cache_control"})


def _unset(value: Any) -> bool:
    return value is None or (isinstance(value, list | dict) and not value)


def check_known(kwargs: dict) -> None:
    """Raise when a call carries a field no list names (or a refused one) with a value."""
    for key, value in kwargs.items():
        if key in _CALL or key in ALLOWED or key in REDACTED or _unset(value):
            continue
        if key in REFUSED:
            raise RedactError(f"request field {key!r} cannot be redacted")
        raise RedactError(f"request field {key!r} is not known to the redactor")


def _bad(what: str) -> RedactError:
    return RedactError(f"{what} cannot be redacted")


class _Walker:
    def __init__(self, red: Redact):
        self.red = red

    # -- leaves -------------------------------------------------------------------------

    def text(self, value: Any, what: str) -> str:
        if not isinstance(value, str):
            raise _bad(what)
        return self.red(value)

    def name(self, value: Any, what: str) -> Any:
        """An identifier: kept as it is, or refused when it holds a value."""
        if not isinstance(value, str) or self.red(value) != value:
            raise RedactError(f"{what} carries personal data or is not a string")
        return value

    def data(self, node: Any) -> Any:
        """Free-form data (a default, an example, metadata): every key and string."""
        if isinstance(node, str):
            return self.red(node)
        if isinstance(node, list):
            return [self.data(v) for v in node]
        if isinstance(node, dict):
            return {self.red(k): self.data(v) for k, v in node.items()}
        return node

    # -- schemas ------------------------------------------------------------------------

    def schema(self, node: Any, key: str | None = None) -> Any:
        if isinstance(node, str):
            return self.name(node, f"schema {key}") if key in _STRUCTURAL else self.red(node)
        if isinstance(node, list):
            if key == "required":
                return [self.name(v, "a required name") for v in node]
            if key in _DATA:
                return self.data(node)
            return [self.schema(v, key) for v in node]
        if not isinstance(node, dict):
            return node
        if key in _DATA:
            return self.data(node)
        if key in _NAMED:
            return {self.name(k, "a schema property name"): self.schema(v) for k, v in node.items()}
        return {k: self.schema(v, k) for k, v in node.items()}

    def function(self, fn: Any, what: str) -> dict:
        if not isinstance(fn, dict) or set(fn) - {"name", "description", "parameters", "strict"}:
            raise _bad(what)
        out = dict(fn)
        out["name"] = self.name(fn.get("name"), "a function name")
        if "description" in fn:
            out["description"] = self.text(fn["description"], what)
        if "parameters" in fn:
            out["parameters"] = self.schema(fn["parameters"])
        return out

    def tool(self, tool: Any) -> dict:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise _bad("a tool that is not a function")
        if set(tool) - {"type", "function"}:
            raise _bad("a tool field")
        return {"type": "function", "function": self.function(tool.get("function"), "a tool")}

    # -- fields -------------------------------------------------------------------------

    def response_format(self, value: Any) -> dict:
        if not isinstance(value, dict):
            raise _bad("response_format")
        kind = value.get("type")
        if kind in ("text", "json_object") and set(value) == {"type"}:
            return value
        if kind != "json_schema" or set(value) - {"type", "json_schema"}:
            raise _bad("response_format")
        spec = value.get("json_schema")
        if not isinstance(spec, dict) or set(spec) - {"name", "description", "schema", "strict"}:
            raise _bad("response_format json_schema")
        out = dict(spec)
        out["name"] = self.name(spec.get("name"), "a response schema name")
        if "description" in spec:
            out["description"] = self.text(spec["description"], "response_format")
        if "schema" in spec:
            out["schema"] = self.schema(spec["schema"])
        return {"type": "json_schema", "json_schema": out}

    def prediction(self, value: Any) -> dict:
        if not isinstance(value, dict) or set(value) - {"type", "content"}:
            raise _bad("prediction")
        out = {"type": value.get("type")}
        content = value.get("content")
        if isinstance(content, str):
            out["content"] = self.red(content)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict) or set(part) - {"type", "text"}:
                    raise _bad("a prediction part")
                parts.append({**part, "text": self.text(part.get("text"), "a prediction part")})
            out["content"] = parts
        else:
            raise _bad("prediction")
        return out

    def metadata(self, value: Any) -> dict:
        if not isinstance(value, dict):
            raise _bad("metadata")
        return {self.name(k, "a metadata key"): self.data(v) for k, v in value.items()}

    def choice(self, value: Any, what: str) -> Any:
        """`tool_choice` / `function_call`: a keyword or a named function."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if set(value) == {"name"}:
                fn = value
            elif set(value) == {"type", "function"} and isinstance(value["function"], dict):
                fn = value["function"]
            else:
                raise _bad(what)
            if set(fn) == {"name"}:
                self.name(fn["name"], what)
                return value
        raise _bad(what)

    def stream_options(self, value: Any) -> Any:
        if not isinstance(value, dict) or any(
            k != "include_usage" or not isinstance(v, bool) for k, v in value.items()
        ):
            raise _bad("stream_options")
        return value


def redact_fields(kwargs: dict, red: Redact) -> dict:
    """Return a copy of `kwargs` with every field but `messages` made safe to send.

    Raises
    ------
    RedactError
        A field is unknown, refused, malformed, or an identifier holds
        a value. Nothing is to be sent.
    """
    check_known(kwargs)
    walk = _Walker(red)
    out = dict(kwargs)
    for key, value in kwargs.items():
        if value is None or key in _CALL:
            continue
        if key == "user":
            out[key] = walk.text(value, "user")
        elif key == "tools":
            out[key] = [walk.tool(t) for t in value]
        elif key == "functions":
            out[key] = [walk.function(f, "a function") for f in value]
        elif key == "response_format":
            out[key] = walk.response_format(value)
        elif key == "prediction":
            out[key] = walk.prediction(value)
        elif key == "metadata":
            out[key] = walk.metadata(value)
        elif key in ("tool_choice", "function_call"):
            out[key] = walk.choice(value, key)
        elif key == "stream_options":
            out[key] = walk.stream_options(value)
    return out


def redact_messages(messages: list[dict], red: Redact) -> None:
    """Redact, in place, what a message carries beside `content` and tool arguments.

    `messages` is a copy `map_text` already rewrote. Raises `RedactError`
    on a field it does not know.
    """
    walk = _Walker(red)
    for message in messages:
        if not isinstance(message, dict):
            raise _bad("a message that is not an object")
        for key, value in list(message.items()):
            if key in _MESSAGE_PLAIN or _unset(value):
                continue
            if key == "name":
                message[key] = walk.text(value, "a message name")
            elif key == "tool_call_id":
                walk.name(value, "a tool call id")
            elif key in _MESSAGE_TEXT:
                message[key] = walk.text(value, key)
            elif key == "function_call":
                walk.name(value.get("name") if isinstance(value, dict) else None, "a function name")
                if set(value) - {"name", "arguments"}:
                    raise _bad("function_call")
            elif key == "tool_calls":
                for call in value:
                    _tool_call(walk, call)
            else:
                raise RedactError(f"message field {key!r} is not known to the redactor")


def _tool_call(walk: _Walker, call: Any) -> None:
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict) or set(call) - {"id", "type", "function", "index"}:
        raise _bad("a tool call")
    if set(fn) - {"name", "arguments"}:
        raise _bad("a tool call function")
    if "id" in call:
        walk.name(call["id"], "a tool call id")
    walk.name(fn.get("name"), "a function name")
