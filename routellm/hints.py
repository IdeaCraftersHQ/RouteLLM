"""Operator-facing hints shared by modules that must not import each other.

`jev` is registered by the optional typesafe extension, so both the
server (building the intent detector) and the controller (validating a
tier's router) have to tell an operator the same thing when it is
missing. The controller cannot import the server, so the text lives
here, in a module that imports nothing of its own.
"""

#: How to obtain everything the `jev` name provides.
TYPESAFE_INSTALL_HINT = (
    "`jev` is registered by the typesafe extension. Install it with "
    "`pip install -e extensions/typesafe`."
)
