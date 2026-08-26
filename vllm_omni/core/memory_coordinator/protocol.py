from __future__ import annotations

from dataclasses import dataclass


def _pressure(free: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - free / total))


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

    @property
    def hbm_pressure(self) -> float:
        return _pressure(self.device_free_bytes, self.device_total_bytes)


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

    @property
    def complete(self) -> bool:
        return len(self.rank_reports) == self.expected_rank_count

    @property
    def hbm_pressure(self) -> float:
        if not self.rank_reports:
            return 1.0
        return max(report.hbm_pressure for report in self.rank_reports)

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
