# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pytest

from vllm_omni.core.memory_coordinator import (
    DynamicHBMConfig,
    RankMemoryReport,
    ReplicaMemoryReport,
    SafetyState,
    StageBudgetDecision,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rank(**overrides) -> RankMemoryReport:
    values = {
        "stage_id": 0,
        "replica_id": 1,
        "rank": 0,
        "device_id": 0,
        "timestamp_monotonic_s": 1.0,
        "device_total_bytes": 1000,
        "device_free_bytes": 200,
        "process_allocated_bytes": 250,
        "process_reserved_bytes": 400,
    }
    values.update(overrides)
    return RankMemoryReport(**values)


def test_dynamic_hbm_m1_defaults_are_fail_closed() -> None:
    config = DynamicHBMConfig()
    assert not config.enabled
    assert config.critical_admission_cap == 0
    assert config.disconnect_admission_cap == 0
    assert config.fail_closed_on_disconnect
    assert config.recovery_complete_samples == 3


@pytest.mark.parametrize(
    "value",
    [
        {"critical_admission_cap": -1},
        {"disconnect_admission_cap": -1},
        {"min_num_seqs": 1, "critical_admission_cap": 2},
        {"min_num_seqs": 1, "disconnect_admission_cap": 2},
        {"recovery_complete_samples": 0},
        {"guard_bytes": -1},
        {"guard_ratio": -0.01},
        {"guard_ratio": 1.0},
        {"immediate_sample_min_interval_ms": 0},
    ],
)
def test_dynamic_hbm_rejects_invalid_m1_config(value) -> None:
    with pytest.raises(ValueError, match="dynamic_hbm"):
        DynamicHBMConfig.from_value(value)


def test_rank_report_separates_raw_guarded_and_unattributed_memory() -> None:
    rank = _rank(
        baseline_generation=1,
        baseline_device_used_bytes=700,
        baseline_process_allocated_bytes=200,
        baseline_process_reserved_bytes=300,
    )
    assert rank.hbm_pressure == pytest.approx(0.8)
    assert rank.hbm_pressure_with_guard(guard_bytes=100) == pytest.approx(0.9)
    assert rank.dynamic_process_reserved_bytes == 100
    assert rank.external_or_unattributed_bytes == 400
    assert rank.baseline_complete


def test_rank_report_clamps_accounting_deltas() -> None:
    rank = _rank(
        device_free_bytes=800,
        process_reserved_bytes=300,
        baseline_generation=1,
        baseline_device_used_bytes=400,
        baseline_process_reserved_bytes=500,
    )
    assert rank.dynamic_process_reserved_bytes == 0
    assert rank.external_or_unattributed_bytes == 0


def test_replica_report_keeps_physical_and_kv_pressure_separate() -> None:
    report = ReplicaMemoryReport(
        stage_id=0,
        replica_id=1,
        timestamp_monotonic_s=1.0,
        rank_reports=(_rank(device_free_bytes=400), _rank(rank=1, device_free_bytes=100)),
        expected_rank_count=2,
        kv_total_blocks=100,
        kv_free_blocks=30,
        running_requests=2,
        waiting_requests=3,
        configured_max_num_seqs=16,
    )
    assert report.complete
    assert report.hbm_pressure == pytest.approx(0.9)
    assert report.kv_pressure == pytest.approx(0.7)
    assert report.pressure == pytest.approx(0.9)
    assert not report.baseline_complete


def test_empty_rank_report_is_fail_safe() -> None:
    report = ReplicaMemoryReport(0, 1, 1.0, (), 1, None, None, 0, 0, 8)
    assert not report.complete
    assert report.hbm_pressure == 1.0
    assert report.kv_pressure == 0.0


def test_budget_decision_new_fields_round_trip() -> None:
    decision = StageBudgetDecision(
        stage_id=0,
        replica_id=1,
        generation=2,
        effective_max_num_seqs=0,
        pressure=0.97,
        reason="critical_pressure",
        safety_state=SafetyState.CRITICAL.value,
        pressure_source="physical_hbm",
        physical_hbm_pressure=0.97,
        kv_pressure=0.4,
        report_age_ms=3.5,
    )
    restored = StageBudgetDecision(**asdict(decision))
    assert restored == decision
