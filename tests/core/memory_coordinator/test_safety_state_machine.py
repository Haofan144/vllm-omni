# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import (
    BudgetAllocator,
    DynamicHBMConfig,
    RankMemoryReport,
    ReplicaMemoryReport,
    SafetyState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _report(
    hbm: float,
    kv: float = 0.0,
    *,
    cap: int = 16,
    complete: bool = True,
) -> ReplicaMemoryReport:
    rank = RankMemoryReport(
        stage_id=0,
        replica_id=0,
        rank=0,
        device_id=0,
        timestamp_monotonic_s=1.0,
        device_total_bytes=1000,
        device_free_bytes=round(1000 * (1 - hbm)),
        process_allocated_bytes=100,
        process_reserved_bytes=120,
    )
    return ReplicaMemoryReport(
        stage_id=0,
        replica_id=0,
        timestamp_monotonic_s=1.0,
        rank_reports=(rank,),
        expected_rank_count=1 if complete else 2,
        kv_total_blocks=100,
        kv_free_blocks=round(100 * (1 - kv)),
        running_requests=0,
        waiting_requests=0,
        configured_max_num_seqs=cap,
    )


def test_high_pressure_multiplicatively_decreases() -> None:
    controller = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)
    assert controller.allocate(_report(0.92)).effective_max_num_seqs == 8
    decision = controller.allocate(_report(0.92))
    assert decision.effective_max_num_seqs == 4
    assert decision.safety_state == SafetyState.HIGH_PRESSURE.value


@pytest.mark.parametrize(
    ("hbm", "kv", "source"),
    [(0.96, 0.2, "physical_hbm"), (0.2, 0.96, "kv"), (0.96, 0.96, "physical_hbm+kv")],
)
def test_critical_pressure_stops_admission(hbm, kv, source) -> None:
    controller = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)
    decision = controller.allocate(_report(hbm, kv))
    assert decision.effective_max_num_seqs == 0
    assert decision.safety_state == SafetyState.CRITICAL.value
    assert decision.pressure_source == source


def test_critical_cap_does_not_expand_in_high_or_hysteresis() -> None:
    controller = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)
    controller.allocate(_report(0.96))
    high = controller.allocate(_report(0.92))
    hysteresis = controller.allocate(_report(0.85))
    assert high.effective_max_num_seqs == 0
    assert hysteresis.effective_max_num_seqs == 0
    assert hysteresis.safety_state == SafetyState.RECOVERING.value


def test_recovery_requires_complete_and_stable_windows() -> None:
    config = DynamicHBMConfig(
        enabled=True,
        recovery_complete_samples=2,
        scale_up_stable_samples=2,
    )
    controller = BudgetAllocator(config, 16)
    controller.allocate(_report(0.96))
    decisions = [controller.allocate(_report(0.5)) for _ in range(4)]
    assert [d.effective_max_num_seqs for d in decisions] == [0, 0, 0, 1]
    assert decisions[-1].safety_state == SafetyState.RECOVERING.value


def test_incomplete_reports_hold_then_decrease_without_expansion() -> None:
    config = DynamicHBMConfig(enabled=True, missing_report_grace_samples=1)
    controller = BudgetAllocator(config, 16)
    first = controller.allocate(_report(0.2, complete=False))
    second = controller.allocate(_report(0.2, complete=False))
    third = controller.allocate(_report(0.2, complete=False))
    assert first.effective_max_num_seqs == 16
    assert first.safety_state == SafetyState.INCOMPLETE.value
    assert second.effective_max_num_seqs == 8
    assert third.effective_max_num_seqs == 4
    assert third.safety_state == SafetyState.STALE.value


def test_guard_can_cross_critical_watermark() -> None:
    config = DynamicHBMConfig(enabled=True, guard_bytes=100)
    controller = BudgetAllocator(config, 16)
    decision = controller.allocate(_report(0.85))
    assert decision.physical_hbm_pressure == pytest.approx(0.95)
    assert decision.effective_max_num_seqs == 0


def test_configured_cap_is_always_an_upper_bound() -> None:
    controller = BudgetAllocator(DynamicHBMConfig(enabled=True), 16)
    decision = controller.allocate(_report(0.5, cap=4))
    assert decision.effective_max_num_seqs == 4
