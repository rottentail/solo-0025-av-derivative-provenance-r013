"""PCM WAV inspection, chunked decoding, and deterministic encoding.

Audio samples are kept in packed 16-bit arrays and containers are streamed in
bounded chunks. The encoded bytes and the per-sample arithmetic are identical
to the former list-based implementation; only the storage representation and
the number of simultaneously resident copies change.
"""

from __future__ import annotations

import array
import io
import os
import sys
import wave
from dataclasses import dataclass
from typing import Any, Iterable

from ..errors import UnsupportedMedia
from ..models import MediaDescriptor
from ..serialization import HashingSink


SAMPLE_WIDTH = 2
FRAME_CHUNK = 65536


@dataclass(frozen=True)
class AudioClip:
    sample_rate: int
    channels: tuple[array.array, ...]

    @property
    def frame_count(self) -> int:
        return len(self.channels[0]) if self.channels else 0

    @property
    def channel_count(self) -> int:
        return len(self.channels)


def _open_reader(data: bytes) -> wave.Wave_read:
    try:
        reader = wave.open(io.BytesIO(data), "rb")
    except (EOFError, wave.Error) as exc:
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file.",
            details={"reason": str(exc)},
        ) from exc
    return reader


def _open_path_reader(path: str | os.PathLike[str]) -> wave.Wave_read:
    try:
        reader = wave.open(os.fspath(path), "rb")
    except (EOFError, wave.Error, OSError) as exc:
        raise UnsupportedMedia(
            "The audio payload is not a readable RIFF/WAVE file.",
            details={"reason": str(exc)},
        ) from exc
    return reader


def _validate_params(reader: wave.Wave_read) -> None:
    if reader.getcomptype() != "NONE":
        raise UnsupportedMedia(
            "Compressed WAV payloads are not supported.",
            details={"compression": reader.getcomptype()},
        )
    sample_width = reader.getsampwidth()
    if sample_width != SAMPLE_WIDTH:
        raise UnsupportedMedia(
            "Only signed 16-bit PCM WAV is supported.",
            details={"sample_width_bytes": sample_width},
        )
    channels = reader.getnchannels()
    if channels not in (1, 2):
        raise UnsupportedMedia(
            "WAV input must contain one or two channels.",
            details={"channels": channels},
        )
    sample_rate = reader.getframerate()
    if sample_rate < 8000 or sample_rate > 192000:
        raise UnsupportedMedia(
            "WAV sample rate is outside the supported range.",
            details={"sample_rate": sample_rate},
        )
    if reader.getnframes() <= 0:
        raise UnsupportedMedia("WAV input has no sample frames.")


def _descriptor_of(reader: wave.Wave_read) -> MediaDescriptor:
    _validate_params(reader)
    sample_rate = reader.getframerate()
    frame_count = reader.getnframes()
    channels = reader.getnchannels()
    return MediaDescriptor(
        kind="audio",
        mime_type="audio/wav",
        container="wav",
        properties={
            "codec": "pcm_s16le",
            "sample_rate": sample_rate,
            "channels": channels,
            "frame_count": frame_count,
            "duration_seconds": round(frame_count / sample_rate, 9),
        },
    )


def inspect_wav(data: bytes) -> MediaDescriptor:
    reader = _open_reader(data)
    try:
        return _descriptor_of(reader)
    finally:
        reader.close()


def inspect_wav_file(path: str | os.PathLike[str]) -> MediaDescriptor:
    reader = _open_path_reader(path)
    try:
        return _descriptor_of(reader)
    finally:
        reader.close()


def _clip_from_reader(reader: wave.Wave_read) -> AudioClip:
    _validate_params(reader)
    channels = reader.getnchannels()
    sample_rate = reader.getframerate()
    frame_count = reader.getnframes()
    buffers = tuple(array.array("h") for _ in range(channels))
    remaining = frame_count
    while remaining > 0:
        raw = reader.readframes(min(FRAME_CHUNK, remaining))
        if not raw:
            break
        values = array.array("h")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        frames_read = len(values) // channels
        if frames_read == 0:
            break
        aligned = frames_read * channels
        for index, buffer in enumerate(buffers):
            buffer.extend(values[index:aligned:channels])
        remaining -= frames_read
    # A short final chunk is accepted as before: the single-call reader also
    # returned whatever whole frames it could decode.
    return AudioClip(sample_rate=sample_rate, channels=buffers)


def decode_wav(data: bytes) -> AudioClip:
    reader = _open_reader(data)
    try:
        return _clip_from_reader(reader)
    finally:
        reader.close()


def decode_wav_file(path: str | os.PathLike[str]) -> AudioClip:
    reader = _open_path_reader(path)
    try:
        return _clip_from_reader(reader)
    finally:
        reader.close()


def _validate_clip(clip: AudioClip) -> None:
    if not clip.channels:
        raise UnsupportedMedia("Cannot encode an audio clip without channels.")
    if any(len(channel) != clip.frame_count for channel in clip.channels):
        raise UnsupportedMedia("Audio channels do not share one frame count.")


def _interleave_chunk(
    channels: tuple[array.array, ...],
    start: int,
    end: int,
) -> array.array:
    channel_count = len(channels)
    frames = end - start
    interleaved = array.array("h", bytes(frames * channel_count * SAMPLE_WIDTH))
    for channel_index, values in enumerate(channels):
        interleaved[channel_index::channel_count] = values[start:end]
    return interleaved


def _write_frames(
    writer: wave.Wave_write,
    channels: tuple[array.array, ...],
    start: int,
    end: int,
) -> None:
    values = _interleave_chunk(channels, start, end)
    if sys.byteorder != "little":
        values.byteswap()
    writer.writeframesraw(values.tobytes())


def _configure_writer(writer: wave.Wave_write, clip: AudioClip) -> None:
    _validate_clip(clip)
    writer.setnchannels(clip.channel_count)
    writer.setsampwidth(SAMPLE_WIDTH)
    writer.setframerate(clip.sample_rate)
    writer.setnframes(clip.frame_count)


def encode_wav(clip: AudioClip) -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        _configure_writer(writer, clip)
        _write_all_frames(writer, clip)
    return payload.getvalue()


def encode_wav_stream(
    clip: AudioClip,
    sink: HashingSink | Any,
) -> HashingSink | Any:
    """Encode the container straight into ``sink`` in bounded chunks."""
    with wave.open(sink, "wb") as writer:
        _configure_writer(writer, clip)
        _write_all_frames(writer, clip)
    return sink


def _write_all_frames(writer: wave.Wave_write, clip: AudioClip) -> None:
    frame_count = clip.frame_count
    frame_index = 0
    while frame_index < frame_count:
        frame_end = min(frame_count, frame_index + FRAME_CHUNK)
        _write_frames(writer, clip.channels, frame_index, frame_end)
        frame_index = frame_end


def replace_channels(
    clip: AudioClip,
    channels: Iterable[list[int] | array.array],
) -> AudioClip:
    values = tuple(
        channel
        if isinstance(channel, array.array) and channel.typecode == "h"
        else array.array("h", channel)
        for channel in channels
    )
    if not values:
        raise UnsupportedMedia("Audio operations produced no channels.")
    expected = len(values[0])
    if any(len(channel) != expected for channel in values):
        raise UnsupportedMedia("Audio operation produced misaligned channels.")
    return AudioClip(sample_rate=clip.sample_rate, channels=values)
