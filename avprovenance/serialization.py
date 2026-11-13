"""Small deterministic serialization helpers used by every layer."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, BinaryIO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class HashingSink:
    """Write-only stream that digests and optionally forwards bytes.

    The built-in WAV writer only needs ``write``, ``flush`` and ``tell``. This
    sink lets encoded containers be hashed, or hashed while spooling to disk,
    without ever keeping the full payload in memory.
    """

    def __init__(self, target: BinaryIO | None = None) -> None:
        self._target = target
        self._digest = hashlib.sha256()
        self._size = 0

    def write(self, data: bytes) -> int:
        self._digest.update(data)
        self._size += len(data)
        if self._target is not None:
            self._target.write(data)
        return len(data)

    def flush(self) -> None:
        if self._target is not None:
            self._target.flush()

    def tell(self) -> int:
        return self._size

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()

    @property
    def size_bytes(self) -> int:
        return self._size


def new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))
