# SPDX-License-Identifier: Apache-2.0

"""CPU-only control-chain integration for centralized dynamic-HBM safety."""

from types import SimpleNamespace

import pytest

from vllm_omni.core.memory_coordinator import DynamicHBMConfig, SafetyState
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ReplicaScheduler(OmniSchedulerMixin):
    def __init__(self, running: int):
        self.max_num_running_reqs = 16
        self.max_num_scheduled_tokens = 4096
        self.running = [object()] * running
        self.waiting = [object()] * 8
        self.num_waiting_for_streaming_input = 0
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(stage_id=0, dynamic_hbm={"enabled": True})
        )
        self._init_dynamic_hbm_scheduling_state()


def _apply(scheduler, generation, cap, state, pressure, report_generation):
    return scheduler.apply_stage_budget_decision(
        generation=generation,
        effective_max_num_seqs=cap,
        pressure=pressure,
        reason=state.value,
        based_on_report_generation=report_generation,
        safety_state=state.value,
        pressure_source="shared_physical_hbm",
    )


def test_shared_pressure_shrinks_both_replicas_without_evicting_running() -> None:
    first = _ReplicaScheduler(running=6)
    second = _ReplicaScheduler(running=2)

    for scheduler in (first, second):
        assert _apply(scheduler, 1, 8, SafetyState.HIGH_PRESSURE, 0.92, 1)
        assert _apply(scheduler, 2, 0, SafetyState.CRITICAL, 0.97, 2)

    assert first._effective_max_num_seqs == 0
    assert second._effective_max_num_seqs == 0
    assert len(first.running) == 6
    assert len(second.running) == 2
    assert first._dynamic_max_num_running_reqs() == 6
    assert second._dynamic_max_num_running_reqs() == 2
    assert not first._dynamic_hbm_allows_new_admission()
    assert not second._dynamic_hbm_allows_new_admission()


def test_cap_recovery_releases_only_available_admission_slots() -> None:
    scheduler = _ReplicaScheduler(running=0)
    _apply(scheduler, 1, 0, SafetyState.CRITICAL, 0.97, 1)
    assert not scheduler._dynamic_hbm_allows_new_admission()

    _apply(scheduler, 2, 1, SafetyState.RECOVERING, 0.5, 2)
    assert scheduler._dynamic_hbm_allows_new_admission()
    scheduler.running.append(scheduler.waiting.pop())
    assert not scheduler._dynamic_hbm_allows_new_admission()

    _apply(scheduler, 3, 2, SafetyState.RECOVERING, 0.5, 3)
    assert scheduler._dynamic_hbm_allows_new_admission()
    scheduler.running.append(scheduler.waiting.pop())
    assert not scheduler._dynamic_hbm_allows_new_admission()


def test_disconnect_then_fresh_central_decision_reconciles_state() -> None:
    scheduler = _ReplicaScheduler(running=3)
    assert scheduler.apply_dynamic_hbm_disconnect_guard()
    assert scheduler._dynamic_hbm_safety_state is SafetyState.DISCONNECTED
    assert scheduler._effective_max_num_seqs == 0
    assert len(scheduler.running) == 3

    assert _apply(scheduler, 1, 4, SafetyState.RECOVERING, 0.5, 1)
    assert scheduler._dynamic_hbm_safety_state is SafetyState.RECOVERING
    assert scheduler._effective_max_num_seqs == 4
    assert scheduler._dynamic_hbm_allows_new_admission()


def test_local_kv_guard_is_reconciled_by_newer_central_report() -> None:
    scheduler = _ReplicaScheduler(running=1)
    assert scheduler.apply_dynamic_hbm_local_kv_guard()
    assert scheduler._last_budget_generation == 0
    assert scheduler._effective_max_num_seqs == 0

    assert _apply(scheduler, 1, 2, SafetyState.RECOVERING, 0.5, 1)
    assert scheduler._effective_max_num_seqs == 2
    assert scheduler._dynamic_hbm_allows_new_admission()
