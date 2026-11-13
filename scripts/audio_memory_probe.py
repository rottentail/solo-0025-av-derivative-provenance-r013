#!/usr/bin/env python3
"""Measure the memory envelope of the WAV mixing and normalize chain.

The script generates one near-limit stereo fixture (24 MiB of PCM), drives the
real HTTP boundary in a child process, and reports the highest resident set of
that process for ingest, rendition, and integrity replay. It also prints the
stored output digest so the same run can be compared against another revision.

Usage:

    python3 scripts/audio_memory_probe.py            # measure current tree
    python3 scripts/audio_memory_probe.py --workdir /tmp/avp-memory

The measurement is per process, so it includes the HTTP server, the request
buffer, the decoded parents, and the encoded result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PRIMARY_FRAMES = 6_290_000
SECONDARY_FRAMES = 5_900_000
SAMPLE_RATE = 48000
QUIET_FRAMES = 4_200_000


def _write_stereo(path: Path, frames: int, *, peak_from: int | None) -> None:
    import array
    import math

    chunk_frames = 65536
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        index = 0
        while index < frames:
            count = min(chunk_frames, frames - index)
            values = array.array("h", bytes(count * 4))
            for offset in range(count):
                position = index + offset
                if peak_from is None:
                    left_amplitude = 8000
                    right_amplitude = 6000
                    left_phase = 330
                    right_phase = 440
                elif position < peak_from:
                    left_amplitude = 200
                    right_amplitude = 140
                    left_phase = 220
                    right_phase = 330
                else:
                    # One loud section at the end proves normalization must see
                    # the whole clip instead of a block-local peak.
                    left_amplitude = 30000
                    right_amplitude = 21000
                    left_phase = 220
                    right_phase = 330
                values[offset * 2] = int(
                    left_amplitude
                    * math.sin(2 * math.pi * left_phase * position / SAMPLE_RATE)
                )
                values[offset * 2 + 1] = int(
                    right_amplitude
                    * math.sin(2 * math.pi * right_phase * position / SAMPLE_RATE)
                )
            writer.writeframesraw(values.tobytes())
            index += count


def _build_fixture(directory: Path) -> tuple[Path, Path]:
    primary = directory / "primary_quiet_then_peak.wav"
    secondary = directory / "secondary_near_limit.wav"
    if not primary.exists():
        _write_stereo(primary, PRIMARY_FRAMES, peak_from=QUIET_FRAMES)
    if not secondary.exists():
        _write_stereo(secondary, SECONDARY_FRAMES, peak_from=None)
    return primary, secondary


def _process_rss_kib(pid: int) -> int:
    highest = 0
    try:
        threads = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return 0
    for entry in threads:
        try:
            with open(f"/proc/{pid}/task/{entry}/statm") as handle:
                highest = max(highest, int(handle.read().split()[1]) * 4)
        except OSError:
            continue
    return highest


class _Sampler:
    def __init__(self, pid: int, interval: float = 0.02) -> None:
        self.pid = pid
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(_process_rss_kib(self.pid))
            time.sleep(self.interval)

    def __enter__(self) -> "_Sampler":
        self.samples.clear()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_kib(self) -> int:
        return max(self.samples, default=0)


def _post(port: int, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    connection.request(
        "POST",
        path,
        body=payload,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
        },
    )
    response = connection.getresponse()
    value = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, value


def _upload(port: int, path: Path, sampler_pid: int) -> tuple[str, int]:
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    with _Sampler(sampler_pid) as sampler:
        status, body = _post(
            port,
            "/v1/assets",
            {"filename": path.name, "data_base64": encoded},
        )
    if status != 201:
        raise AssertionError(f"asset upload failed: {status} {body}")
    return str(body["id"]), sampler.peak_kib


def _server_command(work: Path, port_file: Path) -> list[str]:
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from avprovenance.http_api import build_http_server\n"
        f"server = build_http_server('127.0.0.1', 0, Path({str(work)!r}))\n"
        f"Path({str(port_file)!r}).write_text(str(server.server_address[1]))\n"
        "server.serve_forever(poll_interval=0.02)\n"
    )
    return [sys.executable, "-I", "-c", code]


def measure(workdir: Path) -> dict[str, object]:
    fixture_dir = workdir / "fixtures"
    data_dir = workdir / "data"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()
    primary, secondary = _build_fixture(fixture_dir)

    port_file = workdir / "port.txt"
    port_file.unlink(missing_ok=True)
    process = subprocess.Popen(_server_command(data_dir, port_file))
    try:
        deadline = time.time() + 30
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        if not port_file.exists():
            raise RuntimeError("The probe server did not start.")
        port = int(port_file.read_text().strip())

        primary_id, primary_peak = _upload(port, primary, process.pid)
        secondary_id, secondary_peak = _upload(port, secondary, process.pid)

        operations = [
            {
                "type": "media_trim",
                "start_seconds": 0.05,
                "duration_seconds": 130.0,
            },
            {"type": "audio_gain", "db": -3},
            {
                "type": "audio_mix",
                "secondary_gain": -6,
                "primary_weight": 0.75,
                "secondary_weight": 0.4,
            },
            {"type": "audio_normalize", "target_peak": 0.8},
        ]
        started = time.time()
        with _Sampler(process.pid) as sampler:
            status, rendition = _post(
                port,
                "/v1/renditions",
                {
                    "primary_asset_id": primary_id,
                    "secondary_asset_id": secondary_id,
                    "output_name": "mixed-normalized.wav",
                    "operations": operations,
                },
            )
        rendition_peak = sampler.peak_kib
        rendition_seconds = round(time.time() - started, 2)
        if status != 201:
            raise AssertionError(f"rendition failed: {status} {rendition}")

        started = time.time()
        with _Sampler(process.pid) as sampler:
            status, check = _post(
                port,
                f"/v1/renditions/{rendition['id']}/integrity-checks",
                {},
            )
        verify_peak = sampler.peak_kib
        verify_seconds = round(time.time() - started, 2)
        if status != 201 or check.get("status") != "verified":
            raise AssertionError(f"integrity check failed: {status} {check}")

        primary_asset = rendition["output_asset"]
        return {
            "fixture_bytes": {
                "primary": primary.stat().st_size,
                "secondary": secondary.stat().st_size,
            },
            "ingest_peak_kib": max(primary_peak, secondary_peak),
            "rendition_peak_kib": rendition_peak,
            "verify_peak_kib": verify_peak,
            "rendition_seconds": rendition_seconds,
            "verify_seconds": verify_seconds,
            "output_bytes": primary_asset["size_bytes"],
            "output_sha256": primary_asset["sha256"],
            "integrity_status": check["status"],
        }
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for generated fixtures and probe data.",
    )
    args = parser.parse_args()
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        result = measure(workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="avp-memory-") as temporary:
            result = measure(Path(temporary))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
