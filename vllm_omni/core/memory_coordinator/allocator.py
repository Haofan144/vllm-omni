from __future__ import annotations

import math

from vllm_omni.core.memory_coordinator.config import DynamicHBMConfig
from vllm_omni.core.memory_coordinator.protocol import (
    ReplicaMemoryReport,
    SafetyState,
    StageBudgetDecision,
)


class BudgetAllocator:
    """Watermark allocator with additive increase and multiplicative decrease.

    High pressure applies multiplicative decrease immediately. Low pressure
    applies additive increase only after a configurable stable sample count.
    """

    def __init__(self, config: DynamicHBMConfig, initial_max_num_seqs: int) -> None:
        if initial_max_num_seqs < config.min_num_seqs:
            raise ValueError("configured max_num_seqs must be >= dynamic_hbm.min_num_seqs")
        self.config = config
        self._cap = initial_max_num_seqs
        self._generation = 0
        self._low_pressure_samples = 0
        self._missing_report_samples = 0
        self._recovery_complete_samples = 0
        self._state = SafetyState.NORMAL

    @property
    def current_cap(self) -> int:
        return self._cap

    @property
    def state(self) -> SafetyState:
        return self._state

    def allocate(
        self,
        report: ReplicaMemoryReport,
        *,
        pressure_override: float | None = None,
    ) -> StageBudgetDecision:
        physical_pressure = report.hbm_pressure_with_guard(
            guard_bytes=self.config.guard_bytes,
            guard_ratio=self.config.guard_ratio,
        )
        kv_pressure = report.kv_pressure
        pressure = (
            max(physical_pressure, kv_pressure)
            if pressure_override is None
            else min(1.0, max(0.0, pressure_override))
        )
        if physical_pressure > kv_pressure:
            pressure_source = "physical_hbm"
        elif kv_pressure > physical_pressure:
            pressure_source = "kv"
        else:
            pressure_source = "physical_hbm+kv"
        configured_cap = report.configured_max_num_seqs
        current = min(self._cap, configured_cap)

        if not report.complete:
            self._missing_report_samples += 1
            self._low_pressure_samples = 0
            if self._missing_report_samples <= self.config.missing_report_grace_samples:
                cap = current
                reason = "incomplete_rank_reports_hold"
                state = SafetyState.INCOMPLETE
            else:
                cap = max(
                    self.config.critical_admission_cap,
                    math.floor(current * self.config.scale_down_ratio),
                )
                reason = "incomplete_rank_reports_decrease"
                state = SafetyState.STALE
        elif pressure >= self.config.critical_watermark:
            self._missing_report_samples = 0
            cap = self.config.critical_admission_cap
            reason = "critical_pressure"
            self._low_pressure_samples = 0
            self._recovery_complete_samples = 0
            state = SafetyState.CRITICAL
        elif pressure >= self.config.high_watermark:
            self._missing_report_samples = 0
            if current < self.config.min_num_seqs:
                cap = current
                reason = "high_pressure_hold_critical_cap"
            else:
                cap = max(
                    self.config.min_num_seqs,
                    math.floor(current * self.config.scale_down_ratio),
                )
                reason = "high_pressure"
            self._low_pressure_samples = 0
            self._recovery_complete_samples = 0
            state = SafetyState.HIGH_PRESSURE
        elif pressure <= self.config.low_watermark:
            self._missing_report_samples = 0
            if self._state in {
                SafetyState.CRITICAL,
                SafetyState.HIGH_PRESSURE,
                SafetyState.INCOMPLETE,
                SafetyState.STALE,
                SafetyState.DISCONNECTED,
                SafetyState.RECOVERING,
            } and self._recovery_complete_samples < self.config.recovery_complete_samples:
                self._recovery_complete_samples += 1
                self._low_pressure_samples = 0
                cap = current
                reason = "recovering_complete_reports"
                state = SafetyState.RECOVERING
            else:
                self._low_pressure_samples += 1
                state = SafetyState.RECOVERING if current < configured_cap else SafetyState.NORMAL
                cap = current
                reason = "awaiting_stable_headroom"
            if self._low_pressure_samples >= self.config.scale_up_stable_samples:
                cap = min(configured_cap, current + self.config.scale_up_step)
                reason = "stable_headroom" if cap != current else "at_configured_limit"
                self._low_pressure_samples = 0
                state = SafetyState.RECOVERING if cap < configured_cap else SafetyState.NORMAL
        else:
            self._missing_report_samples = 0
            cap = current
            self._low_pressure_samples = 0
            self._recovery_complete_samples = 0
            if self._state in {
                SafetyState.CRITICAL,
                SafetyState.HIGH_PRESSURE,
                SafetyState.INCOMPLETE,
                SafetyState.STALE,
                SafetyState.RECOVERING,
            } and current < self.config.min_num_seqs:
                reason = "within_hysteresis_hold_critical_cap"
                state = SafetyState.RECOVERING
            else:
                reason = "within_hysteresis"
                state = SafetyState.NORMAL

        floor = (
            self.config.critical_admission_cap
            if state
            in {
                SafetyState.CRITICAL,
                SafetyState.HIGH_PRESSURE,
                SafetyState.INCOMPLETE,
                SafetyState.STALE,
                SafetyState.RECOVERING,
            }
            else self.config.min_num_seqs
        )
        cap = min(configured_cap, max(floor, cap))
        if cap != self._cap:
            self._generation += 1
        self._cap = cap
        self._state = state
        return StageBudgetDecision(
            stage_id=report.stage_id,
            replica_id=report.replica_id,
            generation=self._generation,
            effective_max_num_seqs=cap,
            pressure=pressure,
            reason=reason,
            safety_state=state.value,
            pressure_source=pressure_source if report.complete else "missing_rank_report",
            physical_hbm_pressure=physical_pressure,
            kv_pressure=kv_pressure,
        )
