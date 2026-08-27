# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_runner():
    path = Path(__file__).parents[2] / "benchmarks/hbm_admission/run_milestone0_characterization.py"
    spec = importlib.util.spec_from_file_location("milestone0_characterization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reaction_delay_uses_pressure_and_controller_timestamps() -> None:
    runner = _load_runner()
    case = {
        "pressure": {
            "events": [
                {
                    "event": "pressure_ramp_started",
                    "phase": "high",
                    "target": 0.84,
                    "timestamp": "2026-08-26T12:00:00.100+00:00",
                },
                {"event": "partial_release", "timestamp": "2026-08-26T12:01:00+00:00"},
            ]
        },
        "scheduler": {
            "cap_changes": [
                {"old_cap": 32, "new_cap": 16, "pressure": 0.84, "log_second_of_day": 43200.35},
                {"old_cap": 16, "new_cap": 17, "log_second_of_day": 43260.5},
            ]
        },
    }

    result = runner.reaction_delays(case)

    assert result["downscale_reaction_ms"] == pytest.approx(250)
    assert result["recovery_reaction_ms"] == pytest.approx(500)


def test_summarize_reuses_existing_matrix_artifacts(tmp_path: Path) -> None:
    runner = _load_runner()
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "model: org/model\n"
        "deploy_config: deploy.yaml\n"
        "repeats: 3\n"
        "metric_map: {throughput: metrics.rps}\n"
        "workloads: [{name: low}, {name: medium}, {name: high}]\n"
        "pressure_profiles: [{name: gradual}, {name: spike}]\n"
    )
    cell_dir = tmp_path / "cell"
    case_dir = cell_dir / "cases" / "repeat_01_A"
    case_dir.mkdir(parents=True)
    (tmp_path / "matrix_plan.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "name": "low__gradual__default",
                        "workload": "low",
                        "pressure_profile": "gradual",
                        "controller": "default",
                        "output_dir": str(cell_dir),
                    }
                ]
            }
        )
    )
    (case_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "repeat": 1,
                "arm": {"name": "A_dynamic_off_no_pressure", "dynamic": False, "pressure": False},
                "acceptance": {"passed": True},
                "benchmark": {"metrics": {"rps": 12.5}},
                "scheduler": {"oom_mentions": 0, "cap_change_count": 0},
                "gpu": {"memory_used_mib": {"peak": 1000}},
                "pressure": {},
            }
        )
    )

    report = runner.summarize(spec, tmp_path)

    assert report["counts"]["observed_runs"] == 1
    assert report["coverage"]["ready_for_paper_plots"] is False
    assert "metric_throughput" in (tmp_path / "runs.csv").read_text()
    assert (tmp_path / "MILESTONE0_SUMMARY.md").exists()


def test_coverage_rejects_terminal_but_unaccepted_runs() -> None:
    runner = _load_runner()
    spec = {
        "repeats": 3,
        "workloads": [{"name": "low"}, {"name": "medium"}, {"name": "high"}],
        "pressure_profiles": [{"name": "gradual"}, {"name": "spike"}],
    }
    rows = [
        {
            "arm": arm,
            "status": "acceptance_failed" if arm.startswith("D_") else "completed",
            "accepted": not arm.startswith("D_"),
            "dynamic": arm.startswith(("B_", "D_")),
        }
        for arm in (
            "A_dynamic_off_no_pressure",
            "B_dynamic_on_no_pressure",
            "C_dynamic_off_pressure",
            "D_dynamic_on_pressure",
            "E_fixed_cap_8",
        )
    ]

    result = runner.coverage(spec, rows)

    assert result["checks"]["all_cases_terminal"]
    assert not result["checks"]["all_observed_runs_completed_and_accepted"]
    assert not result["checks"]["all_dynamic_runs_accepted"]
    assert not result["ready_for_paper_plots"]


def test_median_rows_excludes_unaccepted_runs() -> None:
    runner = _load_runner()
    base = {
        "cell": "low",
        "workload": "low",
        "pressure_profile": "gradual",
        "controller": "default",
        "arm": "D_dynamic_on_pressure",
        "repeat": 1,
        "dynamic": True,
        "external_pressure": True,
        "fixed_cap": None,
        "metric_throughput": 10.0,
    }

    assert runner.median_rows([base | {"status": "acceptance_failed", "accepted": False}]) == []
