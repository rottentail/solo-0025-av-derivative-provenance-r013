"""PCM audio operations with explicit clipping and length rules.

Samples live in packed 16-bit arrays and every operation works one channel at
a time, so each stage needs at most one transformed copy per channel. The
sample arithmetic is unchanged: gain and normalize round and clip each sample
individually, normalize scales against the peak of the whole clip, and mix
clips the weighted sum exactly once.
"""

from __future__ import annotations

import array
import math
from typing import Any

from .errors import MediaOperationFailure
from .wav import (
    AudioClip,
    decode_wav,
    decode_wav_file,
    encode_wav,
    encode_wav_stream,
    replace_channels,
)


PCM_MIN = -32768
PCM_MAX = 32767


def _clip_sample(value: float) -> int:
    if value >= PCM_MAX:
        return PCM_MAX
    if value <= PCM_MIN:
        return PCM_MIN
    return int(round(value))


def _gain_factor(db: float) -> float:
    return math.pow(10.0, float(db) / 20.0)


def _scale_channel_in_place(values: array.array, factor: float) -> None:
    """Round-then-clip one sample at a time, matching the previous arithmetic."""
    clip = _clip_sample
    for index, value in enumerate(values):
        values[index] = clip(value * factor)


def _channel_peak(values: array.array) -> int:
    if not values:
        return 0
    return max(max(values), -min(values))


def trim_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    start = int(round(float(operation["start_seconds"]) * clip.sample_rate))
    duration = int(round(float(operation["duration_seconds"]) * clip.sample_rate))
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "AUDIO_TRIM_OUT_OF_RANGE",
            "The requested audio trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, duration))
    if end <= start:
        raise MediaOperationFailure(
            "AUDIO_TRIM_EMPTY",
            "The requested audio trim produces no sample frames.",
        )
    channels: list[array.array] = []
    for values in clip.channels:
        if start:
            del values[:start]
        trim_length = end - start
        if trim_length < len(values):
            del values[trim_length:]
        channels.append(values)
    return replace_channels(clip, channels)


def gain_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    factor = _gain_factor(float(operation["db"]))
    for values in clip.channels:
        _scale_channel_in_place(values, factor)
    return replace_channels(clip, clip.channels)


def normalize_audio(clip: AudioClip, operation: dict[str, Any]) -> AudioClip:
    peak = max(
        (_channel_peak(channel) for channel in clip.channels),
        default=0,
    )
    if peak == 0:
        return clip
    target = float(operation["target_peak"])
    desired = max(1.0, target * PCM_MAX)
    factor = desired / peak
    for values in clip.channels:
        _scale_channel_in_place(values, factor)
    return replace_channels(clip, clip.channels)


def _mix_channel(
    left: array.array,
    right: array.array,
    frame_count: int,
    primary_weight: float,
    secondary_factor: float,
) -> array.array:
    values = array.array("h")
    append = values.append
    left_count = len(left)
    right_count = len(right)
    shared = min(left_count, right_count, frame_count)
    for frame_index in range(shared):
        combined = (
            left[frame_index] * primary_weight
            + right[frame_index] * secondary_factor
        )
        append(_clip_sample(combined))
    for frame_index in range(shared, frame_count):
        left_value = left[frame_index] if frame_index < left_count else 0
        right_value = right[frame_index] if frame_index < right_count else 0
        combined = left_value * primary_weight + right_value * secondary_factor
        append(_clip_sample(combined))
    return values


def mix_audio(
    primary: AudioClip,
    secondary: AudioClip,
    operation: dict[str, Any],
) -> AudioClip:
    if primary.sample_rate != secondary.sample_rate:
        raise MediaOperationFailure(
            "AUDIO_MIX_RATE_MISMATCH",
            "Mixed audio sources must share a sample rate.",
        )
    if primary.channel_count != secondary.channel_count:
        raise MediaOperationFailure(
            "AUDIO_MIX_CHANNEL_MISMATCH",
            "Mixed audio sources must share a channel count.",
        )
    frame_count = max(primary.frame_count, secondary.frame_count)
    primary_weight = float(operation["primary_weight"])
    secondary_weight = float(operation["secondary_weight"])
    secondary_factor = _gain_factor(float(operation["secondary_gain"])) * secondary_weight

    mixed: list[array.array] = []
    for channel_index in range(primary.channel_count):
        mixed.append(
            _mix_channel(
                primary.channels[channel_index],
                secondary.channels[channel_index],
                frame_count,
                primary_weight,
                secondary_factor,
            )
        )
    return replace_channels(primary, mixed)


def _apply_operations(
    clip: AudioClip,
    operations: list[dict[str, Any]],
    secondary: AudioClip | None,
) -> AudioClip:
    mixed = False
    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                clip = trim_audio(clip, operation)
            elif operation_type == "audio_gain":
                clip = gain_audio(clip, operation)
            elif operation_type == "audio_normalize":
                clip = normalize_audio(clip, operation)
            elif operation_type == "audio_mix":
                if secondary is None:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_SOURCE_MISSING",
                        "audio_mix requires a secondary media asset.",
                    )
                if mixed:
                    raise MediaOperationFailure(
                        "AUDIO_MIX_DUPLICATE",
                        "An audio plan may mix the secondary source only once.",
                    )
                clip = mix_audio(clip, secondary, operation)
                mixed = True
                # The secondary clip is never consulted again; releasing it
                # keeps later normalize and encode steps below budget.
                secondary = None
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_AUDIO_OPERATION",
                    f"Unsupported audio operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "AUDIO_OPERATION_INVALID",
            "The audio operation payload is malformed.",
        ) from exc
    return clip


def render_audio(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    clip = _apply_operations(
        decode_wav(primary_bytes),
        operations,
        decode_wav(secondary_bytes) if secondary_bytes is not None else None,
    )
    return encode_wav(clip)


def render_audio_files(
    primary_path: str,
    operations: list[dict[str, Any]],
    secondary_path: str | None,
    sink: Any,
) -> Any:
    """Decode parent files, apply the plan, and encode the result into ``sink``."""
    clip = _apply_operations(
        decode_wav_file(primary_path),
        operations,
        decode_wav_file(secondary_path) if secondary_path is not None else None,
    )
    return encode_wav_stream(clip, sink)
