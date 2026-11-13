"""Route a normalized plan to the matching media implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from ..serialization import HashingSink
from .audio_ops import render_audio, render_audio_files
from .errors import MediaOperationFailure
from .inspector import inspect_media, inspect_media_file
from .video_ops import render_video


@dataclass(frozen=True)
class RenderOutcome:
    data: bytes
    descriptor: MediaDescriptor
    operation_types: list[str]


@dataclass(frozen=True)
class RenderedMedia:
    digest: str
    size_bytes: int
    descriptor: MediaDescriptor
    operation_types: list[str]


def apply_media_operations(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> RenderOutcome:
    primary = inspect_media(primary_bytes)
    secondary = _inspect_secondary(primary, secondary_bytes)

    if primary.kind == "audio":
        output = render_audio(primary_bytes, operations, secondary_bytes)
    elif primary.kind == "video":
        output = render_video(primary_bytes, operations, secondary_bytes)
    else:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    descriptor = inspect_media(output)
    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return RenderOutcome(
        data=output,
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )


def render_media_to_file(
    primary_path: str | os.PathLike[str],
    operations: list[dict[str, Any]],
    secondary_path: str | os.PathLike[str] | None,
    destination: str | os.PathLike[str],
) -> RenderedMedia:
    """Render stored parent blobs into ``destination`` with bounded memory.

    Audio plans stream both parents and the encoded container, so no full
    copy of an input or output blob is held in memory. The descriptor is read
    back from the written file, matching the inspection that ingest performs.
    """
    primary = inspect_media_file(primary_path)
    secondary = (
        inspect_media_file(secondary_path)
        if secondary_path is not None
        else None
    )
    if secondary is not None and secondary.kind != primary.kind:
        raise MediaOperationFailure(
            "MEDIA_KIND_MISMATCH",
            "Primary and secondary media must share a media kind.",
        )
    if primary.kind not in {"audio", "video"}:
        raise UnsupportedMedia("The primary media kind has no renderer.")

    with open(destination, "wb") as handle:
        sink = HashingSink(handle)
        if primary.kind == "audio":
            render_audio_files(
                os.fspath(primary_path),
                operations,
                os.fspath(secondary_path) if secondary_path is not None else None,
                sink,
            )
        else:
            sink.write(
                render_video(
                    _read_file(primary_path),
                    operations,
                    _read_file(secondary_path) if secondary_path is not None else None,
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = inspect_media_file(destination)
    if descriptor.kind != primary.kind:
        raise UnsupportedMedia("The renderer changed the media kind unexpectedly.")
    return RenderedMedia(
        digest=sink.digest,
        size_bytes=sink.size_bytes,
        descriptor=descriptor,
        operation_types=[str(operation["type"]) for operation in operations],
    )


def _inspect_secondary(
    primary: MediaDescriptor,
    secondary_bytes: bytes | None,
) -> MediaDescriptor | None:
    if secondary_bytes is None:
        return None
    secondary = inspect_media(secondary_bytes)
    if secondary.kind != primary.kind:
        raise MediaOperationFailure(
            "MEDIA_KIND_MISMATCH",
            "Primary and secondary media must share a media kind.",
        )
    return secondary


def _read_file(path: str | os.PathLike[str]) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
