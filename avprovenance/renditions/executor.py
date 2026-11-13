"""Execute a normalized plan and convert media failures to domain errors."""

from __future__ import annotations

import os
from typing import Any

from ..errors import RenderFailed, UnsupportedMedia
from ..media import (
    MediaOperationFailure,
    RenderOutcome,
    RenderedMedia,
    apply_media_operations,
    render_media_to_file,
)


def execute_plan(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    try:
        return apply_media_operations(primary_bytes, operations, secondary_bytes)
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc


def execute_plan_to_file(
    primary_path: str | os.PathLike[str],
    operations: list[dict[str, Any]],
    secondary_path: str | os.PathLike[str] | None,
    destination: str | os.PathLike[str],
) -> RenderedMedia:
    try:
        return render_media_to_file(
            primary_path,
            operations,
            secondary_path,
            destination,
        )
    except MediaOperationFailure as exc:
        raise RenderFailed(
            "The rendition could not be produced.",
            details={"reason": exc.code, "detail": exc.message},
        ) from exc
    except UnsupportedMedia as exc:
        raise RenderFailed(
            "The rendition produced unsupported media bytes.",
            details={"reason": exc.code},
        ) from exc
