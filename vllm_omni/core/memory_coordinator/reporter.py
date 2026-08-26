from __future__ import annotations

import time
from collections.abc import Callable

from vllm_omni.core.memory_coordinator.protocol import RankMemoryReport


class RankMemoryReporter:
    """Collect a memory snapshot on the GPU owned by one worker rank."""

    def __init__(
        self,
        *,
        stage_id: int,
        replica_id: int,
        rank: int,
        device_id: int,
        device_memory_provider: Callable[[], tuple[int, int]],
        process_memory_provider: Callable[[], tuple[int, int]],
        node_id: str = "",
        device_uuid: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stage_id = stage_id
        self._replica_id = replica_id
        self._rank = rank
        self._device_id = device_id
        self._device_memory_provider = device_memory_provider
        self._process_memory_provider = process_memory_provider
        self._node_id = node_id
        self._device_uuid = device_uuid
        self._clock = clock

    def report(self) -> RankMemoryReport:
        free_bytes, total_bytes = self._device_memory_provider()
        allocated_bytes, reserved_bytes = self._process_memory_provider()
        return RankMemoryReport(
            stage_id=self._stage_id,
            replica_id=self._replica_id,
            rank=self._rank,
            device_id=self._device_id,
            timestamp_monotonic_s=self._clock(),
            device_total_bytes=total_bytes,
            device_free_bytes=free_bytes,
            process_allocated_bytes=allocated_bytes,
            process_reserved_bytes=reserved_bytes,
            node_id=self._node_id,
            device_uuid=self._device_uuid,
        )
