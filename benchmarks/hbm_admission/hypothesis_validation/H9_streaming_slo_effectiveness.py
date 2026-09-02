#!/usr/bin/env python3
"""H9-SLO: test whether Dynamic HBM protects streaming-audio continuity.

The experiment deliberately does *not* claim that admission control improves
end-to-end latency.  It tests whether pausing new admissions under physical
HBM pressure reduces the fraction of completed requests that violate the
benchmark's audio-continuity SLO.  Queueing/E2E latency and throughput are
reported as the cost of that protection.

Workflow::

    baseline   A-only, no-pressure runs; verifies that the chosen workload is
               healthy before pressure is introduced.
    calibrate  C-only pressure sweep; selects cells with a 30%-80% continuity
               violation rate without server or sidecar OOM.
    run        counterbalanced paired C/D repeats at one frozen cell.
    analyze    re-analyze an existing formal output directory.

``audio_continuity_ok_rate`` is produced by the vLLM-Omni benchmark from
per-request audio underrun measurements.  Its current continuity threshold is
100 ms.  The threshold must be frozen in the benchmark implementation/config
before baseline/calibration/formal data are collected; this script never
chooses a threshold after looking at formal C/D results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_RUNNER = HERE.parent / "run_dynamic_hbm_experiment.py"
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_long"
# Must match vllm_omni.benchmarks.patch.patch._AUDIO_CONTINUITY_THRESHOLD_ENV.
# Read once here purely to label reports with the threshold actually in effect
# for the benchmark client subprocess; this script does not set the variable
# itself (it must be frozen in the environment before baseline data are
# collected, per the module docstring).
_CONTINUITY_THRESHOLD_ENV = "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S"
_CONTINUITY_DEFAULT_THRESHOLD_S = 0.1


def _active_continuity_threshold_s() -> float:
    raw = os.environ.get(_CONTINUITY_THRESHOLD_ENV)
    if raw is None:
        return _CONTINUITY_DEFAULT_THRESHOLD_S
    try:
        return float(raw)
    except ValueError:
        return _CONTINUITY_DEFAULT_THRESHOLD_S


sys.path.insert(0, str(HERE.parent))
from run_dynamic_hbm_effectiveness import parse_case  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric(benchmark: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(benchmark.get(name))
        if value is not None:
            return value
    return None


def load_statuses(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((output_dir / "cases").glob("*/status.json"))
    ]


def parse_slo_case(case: dict[str, Any], *, intended_requests: int) -> dict[str, Any]:
    benchmark = case.get("benchmark") or {}
    scheduler = case.get("scheduler") or {}
    pressure = case.get("pressure") or {}
    arm = str((case.get("arm") or {}).get("name"))
    continuity_ok = _metric(benchmark, "audio_continuity_ok_rate")
    completed = int(benchmark.get("completed") or 0)
    failed = int(benchmark.get("failed") or 0)
    valid_continuity = continuity_ok is not None and completed > 0
    mechanism: dict[str, Any] = {}
    if arm == "D_dynamic_on_pressure":
        metadata = case.get("_analysis_metadata") or {}
        mechanism = parse_case(
            case,
            float(metadata.get("critical_watermark", 0.90)),
            int(metadata.get("critical_admission_cap", 0)),
        )
    return {
        "arm": arm,
        "repeat": int(case.get("repeat") or 0),
        "status": case.get("status"),
        "completed": completed,
        "failed": failed,
        "all_requests_completed": completed == intended_requests and failed == 0,
        "continuity_metric_available": valid_continuity,
        "continuity_ok_rate": continuity_ok,
        "continuity_violation_rate": 1.0 - continuity_ok if valid_continuity else None,
        "mean_audio_underrun_s": _metric(benchmark, "mean_audio_underrun_s"),
        "p99_audio_underrun_s": _metric(benchmark, "p99_audio_underrun_s"),
        "median_audio_underrun_s": _metric(benchmark, "median_audio_underrun_s"),
        "p99_audio_ttfp_ms": _metric(benchmark, "p99_audio_ttfp_ms"),
        "p99_e2el_ms": _metric(benchmark, "p99_e2el_ms"),
        "request_throughput": _metric(benchmark, "request_throughput"),
        "oom_mentions": int(scheduler.get("oom_mentions") or 0),
        "traceback_mentions": int(scheduler.get("traceback_mentions") or 0),
        "sidecar_allocation_ooms": int(pressure.get("allocation_oom_count") or 0),
        "pressure_targets_reached": int(pressure.get("target_reached_count") or 0) >= 2,
        "critical_cap_reached": bool(mechanism.get("critical_cap_reached")),
        "paused_with_waiting": bool(mechanism.get("paused_with_waiting")),
        "resumed": bool(mechanism.get("resumed")),
        "shared_device_all_stages": bool(mechanism.get("shared_device_all_stages")),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(row.get(key)) for row in rows) if rows else 0.0


def paired_bootstrap_ci(
    differences: list[float], *, confidence: float = 0.95, samples: int = 20_000, seed: int = 20260831
) -> tuple[float | None, float | None]:
    if not differences:
        return None, None
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        statistics.fmean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[max(0, int(tail * samples))]
    high = estimates[min(samples - 1, int((1.0 - tail) * samples) - 1)]
    return low, high


def analyze_formal(
    output_dir: Path,
    *,
    min_pairs: int = 20,
    min_absolute_improvement: float = 0.20,
    max_violation_ratio: float = 0.50,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing base-runner metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    arguments = metadata.get("arguments") or {}
    intended = int(arguments.get("num_prompts") or metadata.get("num_prompts") or 0)
    statuses = load_statuses(output_dir)
    for case in statuses:
        case["_analysis_metadata"] = {
            "critical_watermark": arguments.get("critical_watermark", 0.90),
            "critical_admission_cap": arguments.get("critical_admission_cap", 0),
        }
    rows = [parse_slo_case(case, intended_requests=intended) for case in statuses]
    by_arm = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in ("C_dynamic_off_pressure", "D_dynamic_on_pressure")
    }
    c_rows, d_rows = by_arm.values()
    c_by_repeat = {row["repeat"]: row for row in c_rows}
    d_by_repeat = {row["repeat"]: row for row in d_rows}
    pairs: list[dict[str, Any]] = []
    for repeat in sorted(c_by_repeat.keys() & d_by_repeat.keys()):
        c, d = c_by_repeat[repeat], d_by_repeat[repeat]
        if c["continuity_violation_rate"] is None or d["continuity_violation_rate"] is None:
            continue
        pairs.append(
            {
                "repeat": repeat,
                "c_violation_rate": c["continuity_violation_rate"],
                "d_violation_rate": d["continuity_violation_rate"],
                "absolute_improvement": c["continuity_violation_rate"] - d["continuity_violation_rate"],
            }
        )
    c_violation = _mean(row["continuity_violation_rate"] for row in c_rows)
    d_violation = _mean(row["continuity_violation_rate"] for row in d_rows)
    absolute = c_violation - d_violation if c_violation is not None and d_violation is not None else None
    ratio = d_violation / c_violation if c_violation else None
    differences = [float(pair["absolute_improvement"]) for pair in pairs]
    ci_low, ci_high = paired_bootstrap_ci(differences, samples=bootstrap_samples)
    all_rows = c_rows + d_rows
    gates = {
        "at_least_min_complete_pairs": len(pairs) >= min_pairs,
        "all_runs_have_continuity_metric": bool(all_rows)
        and all(row["continuity_metric_available"] for row in all_rows),
        "all_requests_complete_without_failures": bool(all_rows)
        and all(row["all_requests_completed"] for row in all_rows),
        "no_server_oom_or_traceback": all(
            row["oom_mentions"] == 0 and row["traceback_mentions"] == 0 for row in all_rows
        ),
        "pressure_targets_reached": bool(all_rows)
        and all(row["pressure_targets_reached"] for row in all_rows),
        "sidecar_never_oomed": sum(row["sidecar_allocation_ooms"] for row in all_rows) == 0,
        "absolute_violation_improvement": absolute is not None
        and absolute >= min_absolute_improvement,
        "relative_violation_ratio": ratio is not None and ratio <= max_violation_ratio,
        "paired_bootstrap_ci_excludes_zero": ci_low is not None and ci_low > 0.0,
        "dynamic_critical_cap_rate_at_least_90pct": _rate(d_rows, "critical_cap_reached") >= 0.90,
        "dynamic_pause_with_waiting_rate_at_least_90pct": _rate(d_rows, "paused_with_waiting") >= 0.90,
        "dynamic_resume_rate_at_least_90pct": _rate(d_rows, "resumed") >= 0.90,
        "shared_device_decision_rate_at_least_90pct": _rate(d_rows, "shared_device_all_stages") >= 0.90,
    }
    report = {
        "schema_version": 1,
        "hypothesis": "H9-SLO",
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "MATCHES EXPECTATION" if all(gates.values()) else "NOT PROVEN",
        "slo_definition": {
            "primary": "per-request audio continuity violation",
            "source_metric": "1 - audio_continuity_ok_rate",
            "benchmark_continuity_threshold_s": _active_continuity_threshold_s(),
        },
        "expectation": {
            "min_complete_pairs": min_pairs,
            "min_absolute_improvement": min_absolute_improvement,
            "max_dynamic_over_uncontrolled_violation_ratio": max_violation_ratio,
            "mechanism_evidence_rate": 0.90,
        },
        "gates": gates,
        "metrics": {
            "complete_pairs": len(pairs),
            "uncontrolled_violation_rate": c_violation,
            "dynamic_violation_rate": d_violation,
            "absolute_violation_improvement": absolute,
            "dynamic_over_uncontrolled_violation_ratio": ratio,
            "paired_improvement_ci95": [ci_low, ci_high],
            "uncontrolled_median_mean_underrun_s": _median(c_rows, "mean_audio_underrun_s"),
            "dynamic_median_mean_underrun_s": _median(d_rows, "mean_audio_underrun_s"),
            "uncontrolled_median_p99_e2el_ms": _median(c_rows, "p99_e2el_ms"),
            "dynamic_median_p99_e2el_ms": _median(d_rows, "p99_e2el_ms"),
            "uncontrolled_median_throughput": _median(c_rows, "request_throughput"),
            "dynamic_median_throughput": _median(d_rows, "request_throughput"),
            "dynamic_critical_cap_rate": _rate(d_rows, "critical_cap_reached"),
            "dynamic_pause_rate": _rate(d_rows, "paused_with_waiting"),
            "dynamic_resume_rate": _rate(d_rows, "resumed"),
            "dynamic_shared_device_rate": _rate(d_rows, "shared_device_all_stages"),
        },
        "pairs": pairs,
        "runs": rows,
    }
    write_json(output_dir / "H9_SLO_result.json", report)
    write_markdown(output_dir / "H9_SLO_result.md", report)
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# H9-SLO: streaming-audio continuity protection",
        "",
        f"**Verdict:** {report['verdict']}",
        "",
        "The primary metric is the fraction of completed requests that violate the benchmark's 100 ms audio-continuity SLO. E2E latency and throughput are reported as protection cost, not required to improve.",
        "",
        "| Metric | C: dynamic off | D: dynamic on |",
        "|---|---:|---:|",
        f"| Continuity violation rate | {metrics['uncontrolled_violation_rate']} | {metrics['dynamic_violation_rate']} |",
        f"| Median run mean underrun (s) | {metrics['uncontrolled_median_mean_underrun_s']} | {metrics['dynamic_median_mean_underrun_s']} |",
        f"| Median P99 E2E (ms) | {metrics['uncontrolled_median_p99_e2el_ms']} | {metrics['dynamic_median_p99_e2el_ms']} |",
        f"| Median throughput (req/s) | {metrics['uncontrolled_median_throughput']} | {metrics['dynamic_median_throughput']} |",
        "",
        f"Absolute violation improvement: `{metrics['absolute_violation_improvement']}`",
        f"Paired bootstrap 95% CI: `{metrics['paired_improvement_ci95']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["gates"].items())
    lines.extend(["", "Raw paired and per-run evidence is in `H9_SLO_result.json`.", ""])
    path.write_text("\n".join(lines))


def common_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    arms: list[str],
    repeats: int,
    concurrency: int,
    critical_target: float,
) -> list[str]:
    command = [
        sys.executable,
        str(BASE_RUNNER),
        "--output-dir",
        str(output_dir),
        "--model",
        str(args.model),
        "--deploy-config",
        str(args.deploy_config),
        "--dataset-path",
        str(args.dataset_path),
        "--device",
        str(args.device),
        "--port",
        str(args.port),
        "--arms",
        *arms,
        "--repeats",
        str(repeats),
        "--concurrency",
        str(concurrency),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--min-num-seqs",
        "2",
        "--critical-admission-cap",
        "0",
        "--disconnect-admission-cap",
        "0",
        "--guard-mib",
        str(args.guard_mib),
        "--sample-interval-ms",
        "500",
        "--report-timeout-ms",
        "1500",
        "--low-watermark",
        "0.72",
        "--high-watermark",
        "0.82",
        "--critical-watermark",
        "0.90",
        "--scale-down-ratio",
        "0.5",
        "--scale-up-step",
        "2",
        "--scale-up-stable-samples",
        "6",
        "--recovery-complete-samples",
        "3",
        "--pressure-trigger-mode",
        "first-response",
        "--pressure-baseline-seconds",
        "8",
        "--pressure-high-target",
        "0.84",
        "--pressure-high-seconds",
        "10",
        "--pressure-critical-target",
        str(critical_target),
        "--pressure-critical-seconds",
        str(args.pressure_critical_seconds),
        "--pressure-recovery-target",
        "0.76",
        "--pressure-recovery-seconds",
        "15",
        "--pressure-post-release-seconds",
        "30",
        "--pressure-chunk-mib",
        "64",
        "--pressure-reserve-mib",
        str(args.pressure_reserve_mib),
        "--startup-timeout",
        str(args.startup_timeout),
        "--benchmark-timeout",
        str(args.benchmark_timeout),
    ]
    for extra in args.server_extra_arg:
        command.append(f"--server-extra-arg={extra}")
    return command


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        return subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True
        ).returncode


def analyze_baseline(output_dir: Path, *, max_violation_rate: float) -> dict[str, Any]:
    metadata = json.loads((output_dir / "metadata.json").read_text())
    intended = int(metadata["arguments"]["num_prompts"])
    rows = [parse_slo_case(case, intended_requests=intended) for case in load_statuses(output_dir)]
    violation = _mean(row["continuity_violation_rate"] for row in rows)
    gates = {
        "runs_present": bool(rows),
        "continuity_metric_available": bool(rows) and all(row["continuity_metric_available"] for row in rows),
        "all_requests_completed": bool(rows) and all(row["all_requests_completed"] for row in rows),
        "healthy_baseline_violation_rate": violation is not None and violation <= max_violation_rate,
    }
    report = {
        "schema_version": 1,
        "hypothesis": "H9-SLO-baseline",
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "continuity_threshold_s": _active_continuity_threshold_s(),
        "violation_rate": violation,
        "max_allowed_violation_rate": max_violation_rate,
        "gates": gates,
        "runs": rows,
    }
    write_json(output_dir / "H9_SLO_baseline.json", report)
    return report


def analyze_calibration_cell(output_dir: Path, *, low: float, high: float) -> dict[str, Any]:
    metadata = json.loads((output_dir / "metadata.json").read_text())
    intended = int(metadata["arguments"]["num_prompts"])
    rows = [parse_slo_case(case, intended_requests=intended) for case in load_statuses(output_dir)]
    violation = _mean(row["continuity_violation_rate"] for row in rows)
    valid = (
        bool(rows)
        and all(row["continuity_metric_available"] and row["all_requests_completed"] for row in rows)
        and all(row["oom_mentions"] == 0 and row["sidecar_allocation_ooms"] == 0 for row in rows)
        and all(row["pressure_targets_reached"] for row in rows)
    )
    return {
        "attempts": len(rows),
        "continuity_violation_rate": violation,
        "valid": valid,
        "qualifies": valid and violation is not None and low <= violation <= high,
        "distance_from_midpoint": abs(violation - (low + high) / 2.0) if violation is not None else None,
        "runs": rows,
    }


def baseline(args: argparse.Namespace) -> int:
    command = common_command(
        args,
        args.output_dir,
        arms=["A_dynamic_off_no_pressure"],
        repeats=args.repeats,
        concurrency=args.concurrency,
        critical_target=0.90,
    )
    runner_exit = run_command(command, args.output_dir / "orchestrator.log")
    report = analyze_baseline(args.output_dir, max_violation_rate=args.max_baseline_violation_rate)
    report["runner_exit_code"] = runner_exit
    write_json(args.output_dir / "H9_SLO_baseline.json", report)
    return 0 if report["verdict"] == "PASS" else 2


def calibrate(args: argparse.Namespace) -> int:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "hypothesis": "H9-SLO-calibration",
        "selection_rule": "valid C-only cell nearest the midpoint of the preregistered violation band",
        "violation_band": [args.min_calibration_violation, args.max_calibration_violation],
        "cells": [],
    }
    for concurrency in args.concurrencies:
        for critical in args.critical_targets:
            cell_dir = args.output_dir / f"c{concurrency}_p{critical:.3f}"
            command = common_command(
                args,
                cell_dir,
                arms=["C_dynamic_off_pressure"],
                repeats=args.repeats,
                concurrency=concurrency,
                critical_target=critical,
            )
            runner_exit = run_command(command, cell_dir / "orchestrator.log")
            result = analyze_calibration_cell(
                cell_dir,
                low=args.min_calibration_violation,
                high=args.max_calibration_violation,
            )
            result.update(
                concurrency=concurrency,
                critical_target=critical,
                runner_exit_code=runner_exit,
                output_dir=str(cell_dir),
            )
            manifest["cells"].append(result)
            write_json(args.output_dir / "H9_SLO_calibration.json", manifest)
    candidates = [cell for cell in manifest["cells"] if cell["qualifies"]]
    candidates.sort(key=lambda cell: (cell["distance_from_midpoint"], cell["concurrency"], cell["critical_target"]))
    manifest["recommended_cell"] = candidates[0] if candidates else None
    manifest["verdict"] = "PASS" if candidates else "NO QUALIFYING CELL"
    write_json(args.output_dir / "H9_SLO_calibration.json", manifest)
    return 0 if candidates else 2


def formal(args: argparse.Namespace) -> int:
    command = common_command(
        args,
        args.output_dir,
        arms=["C_dynamic_off_pressure", "D_dynamic_on_pressure"],
        repeats=args.repeats,
        concurrency=args.concurrency,
        critical_target=args.critical_target,
    )
    runner_exit = run_command(command, args.output_dir / "orchestrator.log")
    report = analyze_formal(
        args.output_dir,
        min_pairs=args.min_pairs,
        min_absolute_improvement=args.min_absolute_improvement,
        max_violation_ratio=args.max_violation_ratio,
        bootstrap_samples=args.bootstrap_samples,
    )
    report["base_runner_exit_code"] = runner_exit
    write_json(args.output_dir / "H9_SLO_result.json", report)
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(item: argparse.ArgumentParser) -> None:
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        item.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
        item.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
        item.add_argument("--device", type=int, default=0)
        item.add_argument("--port", type=int, default=8000)
        item.add_argument("--num-prompts", type=int, default=160)
        item.add_argument("--num-warmups", type=int, default=4)
        item.add_argument("--max-num-seqs", type=int, default=48)
        item.add_argument("--guard-mib", type=int, default=0)
        item.add_argument("--pressure-reserve-mib", type=int, default=512)
        item.add_argument("--pressure-critical-seconds", type=float, default=45.0)
        item.add_argument("--server-extra-arg", action="append", default=[])
        item.add_argument("--startup-timeout", type=int, default=1800)
        item.add_argument("--benchmark-timeout", type=int, default=3600)

    baseline_parser = sub.add_parser("baseline")
    shared(baseline_parser)
    baseline_parser.add_argument("--concurrency", type=int, default=32)
    baseline_parser.add_argument("--repeats", type=int, default=5)
    baseline_parser.add_argument("--max-baseline-violation-rate", type=float, default=0.10)

    calibration = sub.add_parser("calibrate")
    shared(calibration)
    calibration.add_argument("--concurrencies", type=int, nargs="+", default=[16, 24, 32, 48])
    calibration.add_argument("--critical-targets", type=float, nargs="+", default=[0.88, 0.90, 0.92, 0.94])
    calibration.add_argument("--repeats", type=int, default=3)
    calibration.add_argument("--min-calibration-violation", type=float, default=0.30)
    calibration.add_argument("--max-calibration-violation", type=float, default=0.80)

    run_parser = sub.add_parser("run")
    shared(run_parser)
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--critical-target", type=float, required=True)
    run_parser.add_argument("--repeats", type=int, default=20)
    run_parser.add_argument("--min-pairs", type=int, default=20)
    run_parser.add_argument("--min-absolute-improvement", type=float, default=0.20)
    run_parser.add_argument("--max-violation-ratio", type=float, default=0.50)
    run_parser.add_argument("--bootstrap-samples", type=int, default=20_000)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--min-pairs", type=int, default=20)
    analysis.add_argument("--min-absolute-improvement", type=float, default=0.20)
    analysis.add_argument("--max-violation-ratio", type=float, default=0.50)
    analysis.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "baseline":
        return baseline(args)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "run":
        return formal(args)
    report = analyze_formal(
        args.output_dir,
        min_pairs=args.min_pairs,
        min_absolute_improvement=args.min_absolute_improvement,
        max_violation_ratio=args.max_violation_ratio,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}, indent=2))
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
