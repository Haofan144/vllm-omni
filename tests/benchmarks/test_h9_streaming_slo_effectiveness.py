from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.hbm_admission.hypothesis_validation.H9_streaming_slo_effectiveness import (
    analyze_calibration_cell,
    paired_bootstrap_ci,
    parse_slo_case,
)


def _case(*, continuity: float, completed: int = 100) -> dict:
    return {
        "status": "completed",
        "arm": {"name": "C_dynamic_off_pressure"},
        "repeat": 1,
        "benchmark": {
            "completed": completed,
            "failed": 0,
            "audio_continuity_ok_rate": continuity,
            "mean_audio_underrun_s": 0.4,
        },
        "scheduler": {"oom_mentions": 0, "traceback_mentions": 0},
        "pressure": {
            "allocation_oom_count": 0,
            "target_reached_count": 2,
            "targets_reached": [0.84, 0.92],
        },
    }


def test_parse_slo_case_uses_request_continuity_rate() -> None:
    row = parse_slo_case(_case(continuity=0.65), intended_requests=100)
    assert row["continuity_violation_rate"] == 0.35
    assert row["all_requests_completed"]
    assert row["pressure_targets_reached"]


def test_calibration_selects_midrange_slo_violation(tmp_path: Path) -> None:
    (tmp_path / "cases" / "repeat_01_C").mkdir(parents=True)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"arguments": {"num_prompts": 100}})
    )
    (tmp_path / "cases" / "repeat_01_C" / "status.json").write_text(
        json.dumps(_case(continuity=0.55))
    )
    report = analyze_calibration_cell(tmp_path, low=0.30, high=0.80)
    assert report["valid"]
    assert report["qualifies"]
    assert report["continuity_violation_rate"] == pytest.approx(0.45)


def test_paired_bootstrap_ci_is_deterministic_and_positive() -> None:
    first = paired_bootstrap_ci([0.2, 0.3, 0.4], samples=1000)
    second = paired_bootstrap_ci([0.2, 0.3, 0.4], samples=1000)
    assert first == second
    assert first[0] is not None and first[0] > 0
