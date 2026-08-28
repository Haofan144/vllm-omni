# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.core.memory_coordinator import DynamicHBMConfig
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _engine(*, enabled=True, has_kv=True) -> StageEngineCoreProc:
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    config = DynamicHBMConfig(
        enabled=enabled,
        sample_interval_ms=500,
        report_timeout_ms=1500,
        immediate_sample_min_interval_ms=100,
    )
    scheduler = MagicMock()
    scheduler._dynamic_hbm_config = config
    scheduler._configured_max_num_seqs = 16
    scheduler._safety_max_num_seqs = 16
    scheduler._effective_max_num_seqs = 16
    scheduler.running = [object(), object()]
    scheduler.waiting = [object()] * 3
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[object()] if has_kv else [],
        num_blocks=100,
    )
    scheduler.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 20
    engine.scheduler = scheduler
    engine.model_executor = MagicMock()
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(stage_id=0),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
    )
    return engine


def test_baseline_capture_is_gated_by_config() -> None:
    disabled = _engine(enabled=False)
    disabled._initialize_dynamic_hbm_monitoring()
    disabled.model_executor.collective_rpc.assert_not_called()

    enabled = _engine(enabled=True)
    enabled.model_executor.collective_rpc.return_value = [{"generation": 1}]
    enabled._initialize_dynamic_hbm_monitoring()
    enabled.model_executor.collective_rpc.assert_called_once_with(
        "capture_rank_memory_baseline",
        timeout=1.5,
    )


def test_baseline_failure_does_not_escape() -> None:
    engine = _engine()
    engine.model_executor.collective_rpc.side_effect = RuntimeError("baseline failed")
    engine._initialize_dynamic_hbm_monitoring()


def test_immediate_sample_is_rate_limited(monkeypatch) -> None:
    engine = _engine()
    future = MagicMock()
    engine.model_executor.collective_rpc.return_value = future
    times = iter([0.0, 0.05, 0.10])
    monkeypatch.setattr("vllm_omni.engine.stage_engine_core_proc.time.monotonic", lambda: next(times))

    engine._maybe_start_dynamic_hbm_report()
    engine._dynamic_hbm_report_future = None
    engine.request_dynamic_hbm_sample("kv_exhausted")
    engine._maybe_start_dynamic_hbm_report()
    assert engine.model_executor.collective_rpc.call_count == 1
    engine._maybe_start_dynamic_hbm_report()
    assert engine.model_executor.collective_rpc.call_count == 2
    assert engine._dynamic_hbm_active_trigger_reason == "kv_exhausted"


def test_pending_future_prevents_duplicate_collective_rpc() -> None:
    engine = _engine()
    engine._dynamic_hbm_report_future = MagicMock()
    engine.request_dynamic_hbm_sample("manual")
    engine._maybe_start_dynamic_hbm_report()
    engine.model_executor.collective_rpc.assert_not_called()


def test_local_kv_exhaustion_applies_guard_and_requests_sample() -> None:
    engine = _engine()
    engine.scheduler.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 0
    engine.request_dynamic_hbm_sample = MagicMock()
    engine._maybe_apply_local_kv_safety()
    engine.scheduler.apply_dynamic_hbm_local_kv_guard.assert_called_once_with()
    engine.request_dynamic_hbm_sample.assert_called_once_with("kv_exhausted")


def test_model_without_kv_cache_skips_local_kv_guard() -> None:
    engine = _engine(has_kv=False)
    engine._maybe_apply_local_kv_safety()
    engine.scheduler.kv_cache_manager.block_pool.get_num_free_blocks.assert_not_called()


def test_report_send_failure_applies_disconnect_guard(monkeypatch) -> None:
    engine = _engine()
    client = MagicMock()
    client.send_memory_report.side_effect = RuntimeError("send failed")
    engine._dynamic_hbm_coord_client = client
    future = MagicMock()
    future.done.return_value = True
    future.result.return_value = [
        {
            "stage_id": 0,
            "replica_id": 0,
            "rank": 0,
            "device_id": 0,
            "timestamp_monotonic_s": 1.0,
            "device_total_bytes": 1000,
            "device_free_bytes": 500,
            "process_allocated_bytes": 100,
            "process_reserved_bytes": 120,
        }
    ]
    engine._dynamic_hbm_report_future = future
    engine._dynamic_hbm_report_started_s = 1.0
    monkeypatch.setattr("vllm_omni.engine.stage_engine_core_proc.time.monotonic", lambda: 1.0)
    monkeypatch.setenv("VLLM_OMNI_REPLICA_ID", "0")
    engine._maybe_finish_dynamic_hbm_report()
    engine.scheduler.apply_dynamic_hbm_disconnect_guard.assert_called_once_with()


def test_poll_failure_applies_disconnect_guard() -> None:
    engine = _engine()
    client = MagicMock()
    client.poll_budget_decisions.side_effect = RuntimeError("poll failed")
    engine._dynamic_hbm_coord_client = client
    engine._apply_dynamic_hbm_decisions()
    engine.scheduler.apply_dynamic_hbm_disconnect_guard.assert_called_once_with()


def test_valid_decision_is_applied_and_acknowledged(monkeypatch) -> None:
    engine = _engine()
    client = MagicMock()
    client._instance_id = "instance"
    client._stage_id = 0
    client._replica_id = 0
    client.poll_budget_decisions.return_value = [
        SimpleNamespace(
            instance_id="instance",
            stage_id=0,
            replica_id=0,
            decision_generation=4,
            effective_max_num_seqs=0,
            pressure=0.97,
            reason="critical_pressure",
            based_on_report_generation=3,
            safety_state="critical",
            pressure_source="physical_hbm",
        )
    ]
    engine.scheduler.apply_stage_budget_decision.return_value = True
    engine.scheduler._safety_max_num_seqs = 0
    engine.scheduler._effective_max_num_seqs = 0
    engine._dynamic_hbm_coord_client = client
    monkeypatch.setattr("vllm_omni.engine.stage_engine_core_proc.time.monotonic", lambda: 10.0)

    engine._apply_dynamic_hbm_decisions()

    engine.scheduler.apply_stage_budget_decision.assert_called_once()
    client.send_budget_applied.assert_called_once_with(
        decision_generation=4,
        applied_safety_cap=0,
        effective_cap=0,
        occupied_slots=2,
        applied_monotonic_s=10.0,
    )


def test_stale_or_wrong_identity_decision_is_not_acknowledged() -> None:
    engine = _engine()
    client = MagicMock()
    client._instance_id = "new"
    client._stage_id = 0
    client._replica_id = 0
    client.poll_budget_decisions.return_value = [
        SimpleNamespace(instance_id="old", stage_id=0, replica_id=0),
    ]
    engine._dynamic_hbm_coord_client = client
    engine._apply_dynamic_hbm_decisions()
    engine.scheduler.apply_stage_budget_decision.assert_not_called()
    client.send_budget_applied.assert_not_called()
