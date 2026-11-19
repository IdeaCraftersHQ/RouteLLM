"""AWS event-stream binary framing: the encoder botocore does not have.

botocore only decodes (`botocore.eventstream.EventStreamBuffer`); a server
speaking `application/vnd.amazon.eventstream` must write the frames. One
message on the wire:

    total length   4 bytes, big endian (the whole message)
    headers length 4 bytes
    prelude CRC    4 bytes, CRC-32 of the first 8 bytes
    headers        name length (1), name, type (1), value
    payload        raw bytes
    message CRC    4 bytes, CRC-32 of everything before it

Only string headers (type 7: 2-byte length, UTF-8 value) are written; they
are the only kind the Bedrock streams use (`:event-type`, `:content-type`,
`:message-type`, `:exception-type`).
"""

from __future__ import annotations

import json
import struct
import zlib

__all__ = ["MEDIA_TYPE", "encode_message", "event", "exception"]

MEDIA_TYPE = "application/vnd.amazon.eventstream"

_STRING = 7
_MAX_HEADER_VALUE = 0xFFFF


def _header(name: str, value: str) -> bytes:
    raw_name, raw_value = name.encode(), value.encode()
    if not 0 < len(raw_name) < 256 or len(raw_value) > _MAX_HEADER_VALUE:
        raise ValueError(f"event-stream header out of range: {name!r}")
    return (
        bytes([len(raw_name)])
        + raw_name
        + bytes([_STRING])
        + struct.pack(">H", len(raw_value))
        + raw_value
    )


def encode_message(headers: dict[str, str], payload: bytes) -> bytes:
    """One framed message: prelude, headers, payload, both CRC-32s."""
    block = b"".join(_header(name, value) for name, value in headers.items())
    prelude = struct.pack(">II", 12 + len(block) + len(payload) + 4, len(block))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    body = prelude + block + payload
    return body + struct.pack(">I", zlib.crc32(body))


def event(event_type: str, payload: dict) -> bytes:
    """A JSON event: `:event-type` names it."""
    headers = {
        ":event-type": event_type,
        ":content-type": "application/json",
        ":message-type": "event",
    }
    return encode_message(headers, json.dumps(payload, separators=(",", ":")).encode())


def exception(exception_type: str, message: str) -> bytes:
    """An in-stream error: clients raise it as `exception_type`."""
    headers = {
        ":exception-type": exception_type,
        ":content-type": "application/json",
        ":message-type": "exception",
    }
    return encode_message(headers, json.dumps({"message": message}).encode())
