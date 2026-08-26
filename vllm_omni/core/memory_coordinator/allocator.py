from __future__ import annotations

import math

from vllm_omni.core.memory_coordinator.config import DynamicHBMConfig
from vllm_omni.core.memory_coordinator.protocol import ReplicaMemoryReport, StageBudgetDecision


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

    @property
    def current_cap(self) -> int:
        return self._cap

    def allocate(
        self,
        report: ReplicaMemoryReport,
        *,
        pressure_override: float | None = None,
    ) -> StageBudgetDecision:
        pressure = report.pressure if pressure_override is None else min(1.0, max(0.0, pressure_override))
        configured_cap = report.configured_max_num_seqs
        current = min(self._cap, configured_cap)

        if not report.complete:
            self._missing_report_samples += 1
            self._low_pressure_samples = 0
            if self._missing_report_samples <= self.config.missing_report_grace_samples:
                cap = current
                reason = "incomplete_rank_reports_hold"
            else:
                cap = max(
                    self.config.min_num_seqs,
                    math.floor(current * self.config.scale_down_ratio),
                )
                reason = "incomplete_rank_reports_decrease"
        elif pressure >= self.config.critical_watermark:
            self._missing_report_samples = 0
            cap = self.config.min_num_seqs
            reason = "critical_pressure"
            self._low_pressure_samples = 0
        elif pressure >= self.config.high_watermark:
            self._missing_report_samples = 0
            cap = max(
                self.config.min_num_seqs,
                math.floor(current * self.config.scale_down_ratio),
            )
            reason = "high_pressure"
            self._low_pressure_samples = 0
        elif pressure <= self.config.low_watermark:
            self._missing_report_samples = 0
            self._low_pressure_samples += 1
            if self._low_pressure_samples >= self.config.scale_up_stable_samples:
                cap = min(configured_cap, current + self.config.scale_up_step)
                reason = "stable_headroom" if cap != current else "at_configured_limit"
                self._low_pressure_samples = 0
            else:
                cap = current
                reason = "awaiting_stable_headroom"
        else:
            self._missing_report_samples = 0
            cap = current
            reason = "within_hysteresis"
            self._low_pressure_samples = 0

        cap = min(configured_cap, max(self.config.min_num_seqs, cap))
        if cap != self._cap:
            self._generation += 1
        self._cap = cap
        return StageBudgetDecision(
            stage_id=report.stage_id,
            replica_id=report.replica_id,
            generation=self._generation,
            effective_max_num_seqs=cap,
            pressure=pressure,
            reason=reason,
        )
