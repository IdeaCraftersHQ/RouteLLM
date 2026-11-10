"""The HTTP side of audit: one `Trail` per request, settled once per response.

A handler starts a `Trail` first thing, then returns every response
through `settle`, which writes the event and, with a policy loaded and
the event unwritable, swaps the response for a 503. The event is written
before the response leaves, so a request that cannot be audited is not
answered.
"""

from __future__ import annotations

import logging
from typing import Any

import shortuuid
from fastapi import Request
from fastapi.responses import JSONResponse

from routellm.audit import Auditor, AuditRecord, AuditUnavailable, current_audit

logger = logging.getLogger(__name__)

__all__ = ["Trail", "settle", "unavailable"]


def unavailable(request_id: str) -> JSONResponse:
    """The answer to a request whose audit event could not be written."""
    return JSONResponse(
        {
            "object": "error",
            "message": f"Audit trail unavailable; request refused (request id {request_id})",
            "type": "server_error",
            "request_id": request_id,
        },
        status_code=503,
    )


class Trail:
    """One request's audit record and the auditor that writes it."""

    def __init__(self, auditor: Auditor, record: AuditRecord):
        self.auditor = auditor
        self.record = record

    @classmethod
    def start(
        cls,
        auditor: Auditor | None,
        http_request: Request,
        request_id: str | None,
        route: str,
        prompt: str | None = None,
        streaming: bool = False,
    ) -> Trail | None:
        """Open the trail for a request; None with audit off.

        Sets `current_audit` so the router can add what it decides.
        """
        if auditor is None:
            return None
        principal = getattr(http_request.state, "principal", None)
        record = AuditRecord(
            request_id=request_id or shortuuid.uuid(),
            route=route,
            principal=getattr(principal, "principal", None),
            profile=getattr(principal, "profile", None),
            prompt_hmac=auditor.digest(prompt) if prompt is not None else None,
            streaming=streaming,
        )
        current_audit.set(record)
        return cls(auditor, record)

    def declared(self, request_class: Any) -> None:
        """Record the validated data class and operation."""
        self.record.data_class = request_class.data_class
        self.record.operation = request_class.operation

    def write(self, outcome: str, **fields: Any) -> bool:
        """Write one event; False when a loaded policy requires it and it failed."""
        try:
            self.auditor.emit(self.record, outcome, **fields)
        except AuditUnavailable:
            return False
        return True

    def stream_done(self, res: Any):
        """Callback for the stream's end: writes the `outcome` event.

        The response has started, so a failed write cannot refuse it; it
        is logged and counted by the auditor.
        """

        def done(outcome: str, reason: str | None, tokens_in, tokens_out) -> None:
            self.write(
                outcome,
                phase="outcome",
                status=200,
                reason=reason,
                res=res,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        return done


def settle(
    trail: Trail | None,
    response: Any,
    outcome: str,
    status: int,
    reason: str | None = None,
    **fields: Any,
) -> Any:
    """Write the request's event, then return `response` (or a 503 if it must not go).

    `fields` pass to the event: `phase`, `res`, `tokens_in`, `tokens_out`.
    """
    if trail is None:
        return response
    if trail.write(outcome, status=status, reason=reason, **fields):
        return response
    return unavailable(trail.record.request_id)
