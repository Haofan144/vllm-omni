from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _pressure(free: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - free / total))


class SafetyState(str, Enum):
    """Fail-safe admission states emitted by the reactive controller."""

    NORMAL = "normal"
    HIGH_PRESSURE = "high_pressure"
    CRITICAL = "critical"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class RankMemoryReport:
    stage_id: int
    replica_id: int
    rank: int
    device_id: int
    timestamp_monotonic_s: float
    device_total_bytes: int
    device_free_bytes: int
    process_allocated_bytes: int
    process_reserved_bytes: int
    node_id: str = ""
    device_uuid: str = ""
    baseline_generation: int = 0
    baseline_device_used_bytes: int | None = None
    baseline_process_allocated_bytes: int | None = None
    baseline_process_reserved_bytes: int | None = None
    sample_started_monotonic_s: float = 0.0
    sample_finished_monotonic_s: float = 0.0

    @property
    def hbm_pressure(self) -> float:
        return _pressure(self.device_free_bytes, self.device_total_bytes)

    def hbm_pressure_with_guard(self, *, guard_bytes: int = 0, guard_ratio: float = 0.0) -> float:
        guard = max(guard_bytes, int(self.device_total_bytes * guard_ratio))
        return _pressure(max(0, self.device_free_bytes - guard), self.device_total_bytes)

    @property
    def baseline_complete(self) -> bool:
        return self.baseline_generation > 0 and self.baseline_device_used_bytes is not None

    @property
    def dynamic_process_reserved_bytes(self) -> int | None:
        if self.baseline_process_reserved_bytes is None:
            return None
        return max(0, self.process_reserved_bytes - self.baseline_process_reserved_bytes)

    @property
    def external_or_unattributed_bytes(self) -> int:
        """Conservative device usage not covered by this rank's torch pool.

        This includes other processes as well as CUDA/NCCL/custom allocations
        that PyTorch's caching allocator cannot attribute. It is telemetry,
        not capacity that may be added back to physical free HBM.
        """
        device_used = max(0, self.device_total_bytes - self.device_free_bytes)
        return max(0, device_used - self.process_reserved_bytes)


@dataclass(frozen=True)
class ReplicaMemoryReport:
    stage_id: int
    replica_id: int
    timestamp_monotonic_s: float
    rank_reports: tuple[RankMemoryReport, ...]
    expected_rank_count: int
    kv_total_blocks: int | None
    kv_free_blocks: int | None
    running_requests: int
    waiting_requests: int
    configured_max_num_seqs: int
    report_generation: int = 0
    trigger_reason: str = "periodic"

    @property
    def complete(self) -> bool:
        return len(self.rank_reports) == self.expected_rank_count

    @property
    def baseline_complete(self) -> bool:
        return self.complete and all(report.baseline_complete for report in self.rank_reports)

    @property
    def hbm_pressure(self) -> float:
        if not self.rank_reports:
            return 1.0
        return max(report.hbm_pressure for report in self.rank_reports)

    def hbm_pressure_with_guard(self, *, guard_bytes: int = 0, guard_ratio: float = 0.0) -> float:
        if not self.rank_reports:
            return 1.0
        return max(
            report.hbm_pressure_with_guard(guard_bytes=guard_bytes, guard_ratio=guard_ratio)
            for report in self.rank_reports
        )

    @property
    def kv_pressure(self) -> float:
        if not self.kv_total_blocks or self.kv_free_blocks is None:
            return 0.0
        return _pressure(self.kv_free_blocks, self.kv_total_blocks)

    @property
    def pressure(self) -> float:
        return max(self.hbm_pressure, self.kv_pressure)


@dataclass(frozen=True)
class StageBudgetDecision:
    stage_id: int
    replica_id: int
    generation: int
    effective_max_num_seqs: int
    pressure: float
    reason: str
    instance_id: str = ""
    based_on_report_generation: int = 0
    safety_state: str = SafetyState.NORMAL.value
    pressure_source: str = "none"
    physical_hbm_pressure: float = 0.0
    kv_pressure: float = 0.0
    report_age_ms: float = 0.0
