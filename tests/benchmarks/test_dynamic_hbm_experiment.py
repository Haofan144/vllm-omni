# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_runner():
    path = Path(__file__).parents[2] / "benchmarks/hbm_admission/run_dynamic_hbm_experiment.py"
    spec = importlib.util.spec_from_file_location("dynamic_hbm_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_matrix_runner():
    path = Path(__file__).parents[2] / "benchmarks/hbm_admission/run_dynamic_hbm_matrix.py"
    spec = importlib.util.spec_from_file_location("dynamic_hbm_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_centralized_cap_events_by_stage(tmp_path: Path) -> None:
    runner = _load_runner()
    log = tmp_path / "server.log"
    log.write_text(
        "INFO [StageRuntime] Local dynamic-HBM coordinator started at tcp://127.0.0.1:26000\n"
        "INFO [HBMCoordinator] stage=0 replica=0 first central memory report sent ranks=1 pressure=0.4000\n"
        "INFO [HBMCoordinator] stage=1 replica=0 first central memory report sent ranks=1 pressure=0.4000\n"
        "INFO [HBMCoordinator] registered memory stream stage=0 replica=0 devices=[('node', 'gpu')]\n"
        "INFO [HBMCoordinator] registered memory stream stage=1 replica=0 devices=[('node', 'gpu')]\n"
        "INFO [HBMCoordinator] central decision stage=0 replica=0 cap=16 "
        "pressure=0.8400 reason=shared_device_high_pressure generation=2\n"
        "INFO [HBMCoordinator] central decision stage=1 replica=0 cap=16 "
        "pressure=0.8400 reason=shared_device_high_pressure generation=2\n"
        "INFO 08-13 12:00:01.250 [HBMCoordinator] stage=0 replica=0 cap=32->16 "
        "pressure=0.8400 reason=shared_device_high_pressure generation=2 mode=centralized\n"
        "INFO 08-13 12:00:01.300 [HBMCoordinator] stage=1 replica=0 cap=32->16 "
        "pressure=0.8400 reason=shared_device_high_pressure generation=2 mode=centralized\n"
    )

    parsed = runner.parse_server_events(log)

    assert parsed["cap_change_count"] == 2
    assert parsed["local_coordinator_start_count"] == 1
    assert parsed["first_report_stage_ids"] == [0, 1]
    assert parsed["registered_stream_stage_ids"] == [0, 1]
    assert parsed["per_stage"]["0"]["shared_device_changes"] == 1
    assert parsed["per_stage"]["0"]["shared_device_decisions"] == 1
    assert parsed["per_stage"]["1"]["minimum_observed_cap"] == 16
    assert parsed["cap_changes"][0]["log_second_of_day"] == pytest.approx(43201.25)


def test_generic_metric_mapping_and_analysis() -> None:
    runner = _load_runner()
    cases = []
    for arm, throughput, latency in (
        ("A_dynamic_off_no_pressure", 100.0, 10.0),
        ("B_dynamic_on_no_pressure", 97.0, 10.5),
    ):
        cases.append(
            {
                "status": "completed",
                "arm": {"name": arm},
                "benchmark": {"custom": {"rps": throughput, "p99_ms": latency}},
                "scheduler": {},
            }
        )
    analysis = runner.analyze_summary(
        {"cases": json.loads(json.dumps(cases))},
        {"throughput": "custom.rps", "p99_latency_ms": "custom.p99_ms"},
    )

    assert analysis["comparisons"]["monitoring_throughput_ratio_B_over_A"] == pytest.approx(0.97)
    assert analysis["comparisons"]["monitoring_p99_latency_ratio_B_over_A"] == pytest.approx(1.05)


def test_acceptance_failure_still_contributes_runtime_metrics() -> None:
    runner = _load_runner()
    analysis = runner.analyze_summary(
        {
            "cases": [
                {
                    "status": "acceptance_failed",
                    "arm": {"name": "D_dynamic_on_pressure"},
                    "benchmark": {"request_throughput": 5.0},
                    "scheduler": {"per_stage": {"0": {"shared_device_decisions": 0}}},
                    "acceptance": {"passed": False},
                }
            ]
        },
        {"throughput": "request_throughput"},
    )

    assert analysis["normalized_metrics"]["D_dynamic_on_pressure"]["throughput"]["median"] == 5.0
    assert analysis["safety"]["dynamic_pressure_completed_runs"] == 1
    assert analysis["safety"]["dynamic_pressure_accepted_runs"] == 0


def test_pressure_parser_and_dynamic_acceptance(tmp_path: Path) -> None:
    runner = _load_runner()
    pressure_log = tmp_path / "pressure.jsonl"
    pressure_log.write_text(
        '{"event":"target_reached","target":0.84,"pressure":0.841}\n'
        '{"event":"target_reached","target":0.92,"pressure":0.921}\n'
    )
    pressure = runner.parse_pressure_events(pressure_log)
    scheduler = {
        "first_report_stage_ids": [0, 1],
        "registered_stream_stage_ids": [0, 1],
        "minimum_observed_cap": 2,
        "oom_mentions": 0,
        "per_stage": {"0": {"shared_device_decisions": 1}, "1": {"shared_device_decisions": 1}},
    }

    acceptance = runner.evaluate_case_acceptance(
        runner.Arm("D_dynamic_on_pressure", True, True),
        scheduler
        | {
            "local_coordinator_start_count": 1,
            "traceback_mentions": 0,
        },
        pressure,
        {0, 1},
        2,
        True,
    )

    assert pressure["peak_reported_pressure"] == pytest.approx(0.921)
    assert acceptance["passed"]


def test_acceptance_rejects_failed_execution_and_wrong_coordinator_lifecycle() -> None:
    runner = _load_runner()

    failed_static = runner.evaluate_case_acceptance(
        runner.Arm("A_dynamic_off_no_pressure", False, False),
        {
            "local_coordinator_start_count": 1,
            "first_reports": [],
            "registered_streams": [],
        },
        {},
        {0, 1},
        2,
        False,
    )

    assert not failed_static["passed"]
    assert not failed_static["checks"]["case_execution_succeeded"]
    assert not failed_static["checks"]["local_coordinator_start_matches_arm"]


def test_generic_matrix_expands_workloads_pressure_and_controllers(tmp_path: Path) -> None:
    matrix = _load_matrix_runner()
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text("stages: []\n")
    spec = tmp_path / "experiment.yaml"
    spec.write_text(
        "model: org/model\n"
        "deploy_config: deploy.yaml\n"
        "request_profile: custom\n"
        "benchmark_command: [python, load.py, --url, 'http://{host}:{port}', --out, '{result_json}']\n"
        "metric_map: {throughput: metrics.rps}\n"
        "workloads:\n"
        "  - {name: low, concurrency: 2, num_prompts: 10}\n"
        "  - {name: high, concurrency: 8, num_prompts: 20}\n"
        "pressure_profiles:\n"
        "  - {name: step, options: {pressure_chunk_mib: 64}}\n"
        "  - {name: spike, options: {pressure_chunk_mib: 512}}\n"
        "controllers:\n"
        "  - {name: fast, options: {sample_interval_ms: 100}}\n"
    )

    cells = matrix.build_cells(spec, tmp_path / "results", python="python-test")

    assert len(cells) == 4
    assert {cell.concurrency for cell in cells} == {2, 8}
    spike = next(cell for cell in cells if cell.pressure_profile == "spike")
    assert "--benchmark-command-json" in spike.command
    assert spike.command[spike.command.index("--pressure-chunk-mib") + 1] == "512"
    assert spike.command[spike.command.index("--sample-interval-ms") + 1] == "100"
    assert str(deploy.resolve()) in spike.command


def test_generic_matrix_rejects_unknown_runner_option(tmp_path: Path) -> None:
    matrix = _load_matrix_runner()
    (tmp_path / "deploy.yaml").write_text("stages: []\n")
    spec = tmp_path / "experiment.yaml"
    spec.write_text("model: org/model\ndeploy_config: deploy.yaml\nrunner_options: {misspelled_watermark: 0.9}\n")

    with pytest.raises(ValueError, match="unknown runner options"):
        matrix.build_cells(spec, tmp_path / "results")
