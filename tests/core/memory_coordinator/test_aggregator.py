# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.core.memory_coordinator import DynamicHBMConfig, RankMemoryReport, ReplicaMemoryAggregator
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rank(rank: int, free: int, total: int = 1000, *, timestamp: float = 1.0) -> RankMemoryReport:
    return RankMemoryReport(
        stage_id=1,
        replica_id=2,
        rank=rank,
        device_id=rank,
        timestamp_monotonic_s=timestamp,
        device_total_bytes=total,
        device_free_bytes=free,
        process_allocated_bytes=0,
        process_reserved_bytes=0,
    )


def test_aggregator_uses_most_pressured_rank() -> None:
    aggregator = ReplicaMemoryAggregator(stage_id=1, replica_id=2, expected_rank_count=3)

    report = aggregator.aggregate(
        [_rank(0, 500), _rank(1, 40), _rank(2, 600)],
        kv_total_blocks=100,
        kv_free_blocks=80,
        running_requests=3,
        waiting_requests=5,
        configured_max_num_seqs=16,
    )

    assert report.complete
    assert report.hbm_pressure == pytest.approx(0.96)
    assert report.pressure == pytest.approx(0.96)


def test_aggregator_deduplicates_rank_using_newest_report() -> None:
    aggregator = ReplicaMemoryAggregator(stage_id=1, replica_id=2, expected_rank_count=1)

    report = aggregator.aggregate(
        [_rank(0, 100, timestamp=1.0), _rank(0, 700, timestamp=2.0)],
        kv_total_blocks=None,
        kv_free_blocks=None,
        running_requests=0,
        waiting_requests=0,
        configured_max_num_seqs=8,
    )

    assert len(report.rank_reports) == 1
    assert report.rank_reports[0].device_free_bytes == 700


def test_aggregator_marks_missing_rank_incomplete() -> None:
    aggregator = ReplicaMemoryAggregator(stage_id=1, replica_id=2, expected_rank_count=2)

    report = aggregator.aggregate(
        [_rank(0, 500)],
        kv_total_blocks=100,
        kv_free_blocks=50,
        running_requests=0,
        waiting_requests=0,
        configured_max_num_seqs=8,
    )

    assert not report.complete


def test_engine_core_collects_all_replica_ranks(monkeypatch) -> None:
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    scheduler = MagicMock()
    scheduler._dynamic_hbm_config = DynamicHBMConfig(enabled=True, sample_interval_ms=1)
    scheduler._configured_max_num_seqs = 16
    scheduler.running = [object(), object()]
    scheduler.waiting = [object()] * 3
    scheduler.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 20
    scheduler.kv_cache_config.num_blocks = 100
    engine.scheduler = scheduler
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_id=1),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
    )
    engine.model_executor = MagicMock()
    future = MagicMock()
    future.done.return_value = True
    future.result.return_value = [
        _rank(0, 500).__dict__,
        _rank(1, 50).__dict__,
    ]
    engine.model_executor.collective_rpc.return_value = future
    monkeypatch.setenv("VLLM_OMNI_REPLICA_ID", "2")

    engine._maybe_start_dynamic_hbm_report()
    engine._maybe_finish_dynamic_hbm_report()

    report = scheduler.update_replica_memory_report.call_args.args[0]
    assert report.complete
    assert report.expected_rank_count == 2
    assert report.running_requests == 2
    assert report.waiting_requests == 3
    assert report.hbm_pressure == pytest.approx(0.95)
    engine.model_executor.collective_rpc.assert_called_once_with(
        "report_rank_memory",
        timeout=1.5,
        non_block=True,
    )
