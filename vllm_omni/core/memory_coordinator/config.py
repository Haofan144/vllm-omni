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

    @classmethod
    def from_value(cls, value: DynamicHBMConfig | dict[str, Any] | None) -> DynamicHBMConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("dynamic_hbm must be a mapping or null")
        return cls(**value)
