# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import (
    BudgetAllocator,
    DynamicHBMConfig,
    RankMemoryReport,
    ReplicaMemoryReport,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _report(
    *,
    hbm_pressures: tuple[float, ...] = (0.5,),
    kv_pressure: float = 0.5,
    cap: int = 16,
    expected_ranks: int | None = None,
) -> ReplicaMemoryReport:
    total_bytes = 1000
    total_blocks = 100
    ranks = tuple(
        RankMemoryReport(
            stage_id=0,
            replica_id=3,
            rank=rank,
            device_id=rank,
            timestamp_monotonic_s=1.0,
            device_total_bytes=total_bytes,
            device_free_bytes=round(total_bytes * (1 - pressure)),
            process_allocated_bytes=100,
            process_reserved_bytes=120,
        )
        for rank, pressure in enumerate(hbm_pressures)
    )
    return ReplicaMemoryReport(
        stage_id=0,
        replica_id=3,
        timestamp_monotonic_s=1.0,
        rank_reports=ranks,
        expected_rank_count=len(ranks) if expected_ranks is None else expected_ranks,
        kv_total_blocks=total_blocks,
        kv_free_blocks=round(total_blocks * (1 - kv_pressure)),
        running_requests=0,
        waiting_requests=0,
        configured_max_num_seqs=cap,
    )


def test_most_pressured_rank_applies_multiplicative_decrease() -> None:
    allocator = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)

    decision = allocator.allocate(_report(hbm_pressures=(0.4, 0.91, 0.5)))

    assert decision.replica_id == 3
    assert decision.effective_max_num_seqs == 8
    assert decision.reason == "high_pressure"


def test_critical_kv_pressure_immediately_uses_minimum() -> None:
    config = DynamicHBMConfig(enabled=True, min_num_seqs=2)
    allocator = BudgetAllocator(config, 16)

    decision = allocator.allocate(_report(kv_pressure=0.96))

    assert decision.effective_max_num_seqs == 2
    assert decision.reason == "critical_pressure"


def test_additive_increase_requires_stable_low_pressure() -> None:
    config = DynamicHBMConfig(enabled=True, scale_up_stable_samples=3)
    allocator = BudgetAllocator(config, 8)
    allocator.allocate(_report(hbm_pressures=(0.91,), cap=16))
    assert allocator.current_cap == 4

    first = allocator.allocate(_report(cap=16))
    second = allocator.allocate(_report(cap=16))
    third = allocator.allocate(_report(cap=16))

    assert first.effective_max_num_seqs == 4
    assert second.effective_max_num_seqs == 4
    assert third.effective_max_num_seqs == 5
    assert third.reason == "stable_headroom"


def test_incomplete_rank_reports_hold_then_decrease() -> None:
    config = DynamicHBMConfig(enabled=True, missing_report_grace_samples=1)
    allocator = BudgetAllocator(config, 16)
    incomplete = _report(hbm_pressures=(0.2,), expected_ranks=2)

    first = allocator.allocate(incomplete)
    second = allocator.allocate(incomplete)

    assert first.effective_max_num_seqs == 16
    assert first.reason == "incomplete_rank_reports_hold"
    assert second.effective_max_num_seqs == 8
    assert second.reason == "incomplete_rank_reports_decrease"


def test_hysteresis_holds_current_cap() -> None:
    allocator = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)

    decision = allocator.allocate(_report(hbm_pressures=(0.80,)))

    assert decision.effective_max_num_seqs == 16
    assert decision.reason == "within_hysteresis"


@pytest.mark.parametrize(
    "value",
    [
        {"low_watermark": 0.9, "high_watermark": 0.8},
        {"scale_down_ratio": 1.0},
        {"sample_interval_ms": 0},
        {"report_timeout_ms": 100, "sample_interval_ms": 500},
        {"missing_report_grace_samples": -1},
        {"min_num_seqs": 0},
    ],
)
def test_dynamic_hbm_config_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="dynamic_hbm"):
        DynamicHBMConfig.from_value(value)
