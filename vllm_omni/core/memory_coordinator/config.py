from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DynamicHBMConfig:
    """Process-local dynamic HBM admission settings."""

    enabled: bool = False
    min_num_seqs: int = 1
    sample_interval_ms: int = 500
    report_timeout_ms: int = 1500
    missing_report_grace_samples: int = 1
    low_watermark: float = 0.75
    high_watermark: float = 0.90
    critical_watermark: float = 0.95
    scale_up_step: int = 1
    scale_down_ratio: float = 0.5
    scale_up_stable_samples: int = 5
    scale_token_budget: bool = True
    # ``min_num_seqs`` is the normal operating floor. Under critical HBM
    # pressure it must be possible to stop *new* admission completely while
    # allowing already-running requests to drain.
    critical_admission_cap: int = 0
    disconnect_admission_cap: int = 0
    recovery_complete_samples: int = 3
    guard_bytes: int = 0
    guard_ratio: float = 0.0
    immediate_sample_min_interval_ms: int = 100
    fail_closed_on_disconnect: bool = True
    # M2 resource estimator rollout. ``shadow`` records counterfactual
    # resource decisions without changing scheduling; ``enforce`` applies the
    # local AR KV decision. Global commitment remains a later milestone.
    resource_admission_mode: str = "shadow"
    resource_profile_path: str | None = None
    resource_profile_device_type: str | None = None
    resource_target_coverage: float = 0.95
    resource_profile_min_samples: int = 20
    resource_observation_path: str | None = None
    resource_observation_flush_size: int = 1

    def __post_init__(self) -> None:
        if self.min_num_seqs < 1:
            raise ValueError("dynamic_hbm.min_num_seqs must be at least 1")
        if self.sample_interval_ms < 1:
            raise ValueError("dynamic_hbm.sample_interval_ms must be positive")
        if self.report_timeout_ms < self.sample_interval_ms:
            raise ValueError("dynamic_hbm.report_timeout_ms must be >= sample_interval_ms")
        if self.missing_report_grace_samples < 0:
            raise ValueError("dynamic_hbm.missing_report_grace_samples must be non-negative")
        if not 0 < self.low_watermark < self.high_watermark < self.critical_watermark < 1:
            raise ValueError("dynamic_hbm watermarks must satisfy 0 < low < high < critical < 1")
        if self.scale_up_step < 1:
            raise ValueError("dynamic_hbm.scale_up_step must be at least 1")
        if not 0 < self.scale_down_ratio < 1:
            raise ValueError("dynamic_hbm.scale_down_ratio must be between 0 and 1")
        if self.scale_up_stable_samples < 1:
            raise ValueError("dynamic_hbm.scale_up_stable_samples must be at least 1")
        if self.critical_admission_cap < 0:
            raise ValueError("dynamic_hbm.critical_admission_cap must be non-negative")
        if self.critical_admission_cap > self.min_num_seqs:
            raise ValueError("dynamic_hbm.critical_admission_cap must not exceed min_num_seqs")
        if self.disconnect_admission_cap < 0:
            raise ValueError("dynamic_hbm.disconnect_admission_cap must be non-negative")
        if self.disconnect_admission_cap > self.min_num_seqs:
            raise ValueError("dynamic_hbm.disconnect_admission_cap must not exceed min_num_seqs")
        if self.recovery_complete_samples < 1:
            raise ValueError("dynamic_hbm.recovery_complete_samples must be at least 1")
        if self.guard_bytes < 0:
            raise ValueError("dynamic_hbm.guard_bytes must be non-negative")
        if not 0 <= self.guard_ratio < 1:
            raise ValueError("dynamic_hbm.guard_ratio must be in [0, 1)")
        if self.immediate_sample_min_interval_ms < 1:
            raise ValueError("dynamic_hbm.immediate_sample_min_interval_ms must be positive")
        if self.resource_admission_mode not in {"off", "shadow", "enforce"}:
            raise ValueError(
                "dynamic_hbm.resource_admission_mode must be off, shadow, or enforce"
            )
        if not 0.0 < self.resource_target_coverage <= 1.0:
            raise ValueError(
                "dynamic_hbm.resource_target_coverage must be in (0, 1]"
            )
        if self.resource_profile_min_samples < 1:
            raise ValueError(
                "dynamic_hbm.resource_profile_min_samples must be positive"
            )
        if self.resource_profile_path and not self.resource_profile_device_type:
            raise ValueError(
                "dynamic_hbm.resource_profile_device_type is required with "
                "resource_profile_path"
            )
        if self.resource_observation_flush_size < 1:
            raise ValueError(
                "dynamic_hbm.resource_observation_flush_size must be positive"
            )

    @classmethod
    def from_value(cls, value: DynamicHBMConfig | dict[str, Any] | None) -> DynamicHBMConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("dynamic_hbm must be a mapping or null")
        return cls(**value)
