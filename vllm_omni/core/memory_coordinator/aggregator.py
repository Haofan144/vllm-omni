from __future__ import annotations

import time
from collections.abc import Sequence

from vllm_omni.core.memory_coordinator.protocol import RankMemoryReport, ReplicaMemoryReport


class ReplicaMemoryAggregator:
    """Validate and combine rank reports using the most pressured GPU."""

    def __init__(self, *, stage_id: int, replica_id: int, expected_rank_count: int) -> None:
        if expected_rank_count < 1:
            raise ValueError("expected_rank_count must be positive")
        self.stage_id = stage_id
        self.replica_id = replica_id
        self.expected_rank_count = expected_rank_count

    def aggregate(
        self,
        rank_reports: Sequence[RankMemoryReport],
        *,
        kv_total_blocks: int | None,
        kv_free_blocks: int | None,
        running_requests: int,
        waiting_requests: int,
        configured_max_num_seqs: int,
    ) -> ReplicaMemoryReport:
        by_rank: dict[int, RankMemoryReport] = {}
        for report in rank_reports:
            if report.stage_id != self.stage_id or report.replica_id != self.replica_id:
                continue
            if report.rank < 0:
                continue
            previous = by_rank.get(report.rank)
            if previous is None or report.timestamp_monotonic_s > previous.timestamp_monotonic_s:
                by_rank[report.rank] = report

        reports = tuple(by_rank[rank] for rank in sorted(by_rank))
        timestamp = min((report.timestamp_monotonic_s for report in reports), default=time.monotonic())
        return ReplicaMemoryReport(
            stage_id=self.stage_id,
            replica_id=self.replica_id,
            timestamp_monotonic_s=timestamp,
            rank_reports=reports,
            expected_rank_count=self.expected_rank_count,
            kv_total_blocks=kv_total_blocks,
            kv_free_blocks=kv_free_blocks,
            running_requests=running_requests,
            waiting_requests=waiting_requests,
            configured_max_num_seqs=configured_max_num_seqs,
        )
