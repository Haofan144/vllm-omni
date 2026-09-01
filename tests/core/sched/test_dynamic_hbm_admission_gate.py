# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.core.memory_coordinator import DynamicHBMConfig, SafetyState
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    def __init__(self, *, cap: int = 16, tokens: int = 4096, running: int = 0, config=None):
        self.max_num_running_reqs = cap
        self.max_num_scheduled_tokens = tokens
        self.running = [object()] * running
        self.waiting = []
        self.num_waiting_for_streaming_input = 0
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=0,
                dynamic_hbm=config or {"enabled": True},
                async_chunk=False,
            )
        )
        self._init_dynamic_hbm_scheduling_state()


def _apply(scheduler: _Scheduler, *, generation: int, cap: int, state: SafetyState, report=1) -> bool:
    pressure = 0.96 if state is SafetyState.CRITICAL else 0.92
    return scheduler.apply_stage_budget_decision(
        generation=generation,
        based_on_report_generation=report,
        effective_max_num_seqs=cap,
        pressure=pressure,
        reason=state.value,
        safety_state=state.value,
        pressure_source="physical_hbm",
    )


def test_scheduler_initializes_independent_safety_cap() -> None:
    scheduler = _Scheduler()
    assert scheduler._configured_max_num_seqs == 16
    assert scheduler._safety_max_num_seqs == 16
    assert scheduler._effective_max_num_seqs == 16
    assert scheduler._dynamic_hbm_safety_state is SafetyState.NORMAL


def test_high_decision_updates_only_safety_budget() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=8, state=SafetyState.HIGH_PRESSURE)
    assert scheduler._configured_max_num_seqs == 16
    assert scheduler._safety_max_num_seqs == 8
    assert scheduler._effective_max_num_seqs == 8
    assert scheduler._effective_max_num_scheduled_tokens == 2048


def test_critical_state_clamps_inconsistent_wire_cap_to_zero() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=4, state=SafetyState.CRITICAL)
    assert scheduler._safety_max_num_seqs == 0
    assert scheduler._effective_max_num_seqs == 0
    assert scheduler._effective_max_num_scheduled_tokens == 1
    assert not scheduler._dynamic_hbm_allows_new_admission()


def test_running_requests_are_retained_in_execution_budget() -> None:
    scheduler = _Scheduler(running=6)
    _apply(scheduler, generation=1, cap=0, state=SafetyState.CRITICAL)
    assert scheduler._effective_max_num_seqs == 0
    assert scheduler._dynamic_max_num_running_reqs() == 6
    assert scheduler._effective_max_num_scheduled_tokens == 1536
    assert len(scheduler.running) == 6


def test_admission_resumes_only_below_effective_cap() -> None:
    scheduler = _Scheduler(running=2)
    _apply(scheduler, generation=1, cap=2, state=SafetyState.HIGH_PRESSURE)
    assert not scheduler._dynamic_hbm_allows_new_admission()
    scheduler.running.pop()
    assert scheduler._dynamic_hbm_allows_new_admission()


def test_stale_decision_and_unknown_state_are_rejected() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=2, cap=8, state=SafetyState.HIGH_PRESSURE, report=5)
    assert not _apply(scheduler, generation=1, cap=16, state=SafetyState.NORMAL, report=6)
    assert not scheduler.apply_stage_budget_decision(
        generation=3,
        based_on_report_generation=6,
        effective_max_num_seqs=16,
        pressure=0.5,
        reason="invalid",
        safety_state="not-a-state",
    )
    assert scheduler._effective_max_num_seqs == 8


def test_older_report_cannot_expand_cap() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=4, state=SafetyState.HIGH_PRESSURE, report=5)
    assert not _apply(scheduler, generation=2, cap=16, state=SafetyState.NORMAL, report=4)
    assert scheduler._effective_max_num_seqs == 4


def test_local_kv_guard_does_not_consume_central_generation() -> None:
    scheduler = _Scheduler()
    assert scheduler.apply_dynamic_hbm_local_kv_guard()
    assert scheduler._last_budget_generation == 0
    assert scheduler._dynamic_hbm_safety_state is SafetyState.CRITICAL
    assert scheduler._effective_max_num_seqs == 0
    assert _apply(scheduler, generation=1, cap=1, state=SafetyState.RECOVERING)
    assert scheduler._effective_max_num_seqs == 1


@pytest.mark.parametrize("fail_closed", [True, False])
def test_disconnect_guard_respects_policy(fail_closed) -> None:
    scheduler = _Scheduler(config={"enabled": True, "fail_closed_on_disconnect": fail_closed})
    applied = scheduler.apply_dynamic_hbm_disconnect_guard()
    assert applied is fail_closed
    expected = 0 if fail_closed else 16
    assert scheduler._effective_max_num_seqs == expected
    assert scheduler._last_budget_generation == 0


def test_token_scaling_can_be_disabled() -> None:
    scheduler = _Scheduler(config={"enabled": True, "scale_token_budget": False})
    _apply(scheduler, generation=1, cap=4, state=SafetyState.HIGH_PRESSURE)
    assert scheduler._effective_max_num_scheduled_tokens == 4096


class _BlockPool:
    def __init__(self, free_blocks: int):
        self.free_blocks = free_blocks

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


class _WaitingRequest:
    request_id = "request-0"
    num_prompt_tokens = 64
    max_tokens = 32
    num_computed_tokens = 0
    num_output_tokens = 0


def _resource_scheduler(mode: str, free_blocks: int) -> _Scheduler:
    scheduler = _Scheduler(
        config={"enabled": True, "resource_admission_mode": mode},
    )
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=_BlockPool(free_blocks))
    scheduler.waiting = [_WaitingRequest()]
    return scheduler


def test_resource_admission_defaults_to_shadow() -> None:
    scheduler = _resource_scheduler("shadow", free_blocks=2)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.shadow_would_defer
    assert scheduler._resource_admission_counts["shadow_would_defer"] == 1


def test_resource_admission_can_be_explicitly_enforced() -> None:
    scheduler = _resource_scheduler("enforce", free_blocks=2)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert not decision.allowed
    assert decision.reason.value == "kv_peak_risk"


def test_resource_admission_off_is_noop() -> None:
    scheduler = _resource_scheduler("off", free_blocks=0)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.reason.value == "disabled"
