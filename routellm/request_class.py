"""The data class and operation a caller declares on a request.

Carriage, either of:

* headers `X-Data-Class` and `X-Operation`;
* body `metadata: {data_class, operation}` (an OpenAI-compatible field).

When both name the same field and differ, the request is refused (400).
The declaration is validated against the policy and the caller, attached
to the request context next to the principal, and never forwarded: the
handler drops `metadata` from what it hands the controller, and no
header is copied upstream, so a provider (and the decision classifier,
and any shadow or comparison call) never sees it.

`operation` is free-form `<tool>.<step>`, at most 128 characters. It is
recorded only (audit) and never used for routing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from routellm.auth import Principal
from routellm.policy import Policy, PolicyRefusal

__all__ = [
    "CLASS_HEADER",
    "OPERATION_HEADER",
    "RequestClass",
    "RequestClassRefusal",
    "check_policy_principals",
    "current_request_class",
    "declare",
]

CLASS_HEADER = "X-Data-Class"
OPERATION_HEADER = "X-Operation"
MAX_OPERATION = 128

_CLASS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_OPERATION = re.compile(rf"^[\x21-\x7e]{{1,{MAX_OPERATION}}}$")


@dataclass(frozen=True)
class RequestClass:
    """What a request declared, once validated.

    `data_class` is the policy class name (the policy's `default_class`
    when none was declared), or the declared name when no policy is
    loaded, or None when nothing was declared. `operation` is recorded
    only.
    """

    data_class: str | None = None
    operation: str | None = None


#: The declaration of the request being served, for code with no
#: `Request` in hand. None outside a request.
current_request_class: ContextVar[RequestClass | None] = ContextVar(
    "current_request_class", default=None
)


class RequestClassRefusal(Exception):
    """The declaration is refused; `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _pick(field: str, header: str | None, body: Any, shape: re.Pattern, what: str) -> str | None:
    if body is not None and not isinstance(body, str):
        raise RequestClassRefusal(400, f"metadata.{field} must be a string")
    for source, value in (("header", header), ("metadata", body)):
        if value is not None and not shape.fullmatch(value):
            raise RequestClassRefusal(400, f"{what} in the {source} is malformed")
    if header is not None and body is not None and header != body:
        raise RequestClassRefusal(
            400, f"{what} differs between the header and metadata.{field}; send one"
        )
    return header if header is not None else body


def declare(
    headers: Mapping[str, str],
    metadata: Mapping[str, Any] | None,
    principal: Principal | None,
    policy: Policy | None,
    classifier_classes: Collection[str] = (),
) -> RequestClass:
    """Parse, validate and return a request's declaration.

    Parameters
    ----------
    headers : mapping
        Request headers (case-insensitive lookup expected).
    metadata : mapping, optional
        The body's `metadata` object.
    principal : Principal, optional
        The authenticated caller; None when auth is off.
    policy : Policy, optional
        The loaded policy. With one, only `Policy.principals` decides
        what a caller may send (the keys file's `allowed_classes` is
        unused). Without one, `Principal.allowed_classes` does.
    classifier_classes : collection of str
        The decision classifier's label set. Without a policy a declared
        class outside it is ignored rather than refused.

    Raises
    ------
    RequestClassRefusal
        400 for a malformed or conflicting declaration, an unknown class,
        or a missing class when the policy has no default; 403 for a
        class the caller may not send or a vaulted class.
    """
    meta = metadata or {}
    headers = {k.lower(): v for k, v in headers.items()}
    declared = _pick(
        "data_class",
        headers.get(CLASS_HEADER.lower()),
        meta.get("data_class"),
        _CLASS,
        "data class",
    )
    operation = _pick(
        "operation",
        headers.get(OPERATION_HEADER.lower()),
        meta.get("operation"),
        _OPERATION,
        f"operation (1-{MAX_OPERATION} visible characters)",
    )

    if policy is None:
        if declared is None:
            return RequestClass(None, operation)
        if principal is not None and not principal.allows(declared):
            raise RequestClassRefusal(
                403, f"data class {declared!r} is not allowed for this caller"
            )
        return RequestClass(declared if declared in classifier_classes else None, operation)

    try:
        name, _ = policy.resolve(declared)
    except PolicyRefusal as exc:
        status = 403 if exc.code == "vault" else 400
        raise RequestClassRefusal(status, str(exc)) from None
    if not policy.principal_allows(principal.principal if principal else None, name):
        raise RequestClassRefusal(403, f"data class {name!r} is not allowed for this caller")
    return RequestClass(name, operation)


def check_policy_principals(policy: Policy, resolver: Any) -> None:
    """Startup check of the policy's principals against the keys file.

    With a policy loaded, the policy decides what each principal may
    send. Every principal the policy names must hold a key, or its
    entry could never match; a keys-file `allowed_classes` is unused and
    gets one warning.

    Raises
    ------
    PolicyError
        Naming each policy principal with no key.
    """
    from routellm.policy import PolicyError

    log = logging.getLogger(__name__)
    if resolver is None:
        if policy.principals:
            log.warning(
                "policy %s names principals but `auth:` is off: every request has no "
                "principal and is refused (403); turn auth on",
                policy.source,
            )
        return
    known = resolver.principals() if hasattr(resolver, "principals") else None
    if known is None:
        return
    missing = [p for p in policy.principals if p not in {x.principal for x in known}]
    if missing:
        raise PolicyError(
            f"policy {policy.source}: principals {', '.join(map(repr, missing))} "
            "have no key in the auth keys file"
        )
    if any(x.allowed_classes for x in known):
        log.warning(
            "auth keys file sets allowed_classes but a policy is loaded: the policy's "
            "`principals` decides; allowed_classes is unused"
        )
