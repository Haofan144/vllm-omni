#!/usr/bin/env python3
"""Apply a bounded, step-shaped CUDA memory pressure profile.

This helper is intentionally model agnostic.  It runs in a separate process,
allocates memory in small chunks, and emits JSONL telemetry to stdout.  The
experiment runner owns its lifetime and captures the telemetry as an artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from typing import Any


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"timestamp": datetime.now(UTC).isoformat(), "event": event, **values},
            sort_keys=True,
        ),
        flush=True,
    )


def memory_state(torch, device: int) -> tuple[int, int, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    pressure = 1.0 - free_bytes / total_bytes
    return free_bytes, total_bytes, pressure


def reach_pressure(torch, device: int, chunks: list, target: float, chunk_bytes: int, reserve_bytes: int) -> None:
    """Allocate touched byte tensors until target pressure or reserve is met."""
    while True:
        free_bytes, total_bytes, pressure = memory_state(torch, device)
        if pressure >= target:
            emit("target_reached", target=target, pressure=pressure, chunks=len(chunks))
            return
        request_bytes = min(chunk_bytes, max(0, free_bytes - reserve_bytes))
        if request_bytes < 1024 * 1024:
            emit("reserve_reached", target=target, pressure=pressure, chunks=len(chunks))
            return
        try:
            tensor = torch.empty(request_bytes, dtype=torch.uint8, device=f"cuda:{device}")
            tensor.fill_(1)  # Commit the allocation instead of relying on lazy pages.
            chunks.append(tensor)
        except torch.cuda.OutOfMemoryError as exc:
            emit("allocation_oom", target=target, pressure=pressure, error=str(exc))
            torch.cuda.empty_cache()
            return


def release_to_pressure(torch, device: int, chunks: list, target: float) -> None:
    while chunks:
        _, _, pressure = memory_state(torch, device)
        if pressure <= target:
            break
        chunks.pop()
        torch.cuda.empty_cache()
    _, _, pressure = memory_state(torch, device)
    emit("partial_release", target=target, pressure=pressure, chunks=len(chunks))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--baseline-seconds", type=float, default=15.0)
    parser.add_argument("--high-target", type=float, default=0.84)
    parser.add_argument("--high-seconds", type=float, default=20.0)
    parser.add_argument("--critical-target", type=float, default=0.92)
    parser.add_argument("--critical-seconds", type=float, default=15.0)
    parser.add_argument("--recovery-target", type=float, default=0.78)
    parser.add_argument("--recovery-seconds", type=float, default=20.0)
    parser.add_argument("--post-release-seconds", type=float, default=45.0)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--reserve-mib", type=int, default=1536)
    args = parser.parse_args()
    if not 0 < args.recovery_target < args.high_target < args.critical_target < 1:
        parser.error("targets must satisfy 0 < recovery < high < critical < 1")

    import torch

    torch.cuda.set_device(args.device)
    chunks: list[Any] = []
    _, total, initial = memory_state(torch, args.device)
    emit("started", total_bytes=total, initial_pressure=initial)
    try:
        time.sleep(args.baseline_seconds)
        reach_pressure(
            torch,
            args.device,
            chunks,
            args.high_target,
            args.chunk_mib * 1024**2,
            args.reserve_mib * 1024**2,
        )
        time.sleep(args.high_seconds)
        reach_pressure(
            torch,
            args.device,
            chunks,
            args.critical_target,
            args.chunk_mib * 1024**2,
            args.reserve_mib * 1024**2,
        )
        time.sleep(args.critical_seconds)
        release_to_pressure(torch, args.device, chunks, args.recovery_target)
        time.sleep(args.recovery_seconds)
        chunks.clear()
        torch.cuda.empty_cache()
        _, _, pressure = memory_state(torch, args.device)
        emit("released", pressure=pressure)
        time.sleep(args.post_release_seconds)
    finally:
        chunks.clear()
        torch.cuda.empty_cache()
        _, _, pressure = memory_state(torch, args.device)
        emit("finished", pressure=pressure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
