#!/usr/bin/env python3
"""Calibrate, run, and analyze the dynamic-HBM effectiveness experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BASE_RUNNER = Path(__file__).with_name("run_dynamic_hbm_experiment.py")
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_smoke"

CAP_RE = re.compile(
    r"\[HBMCoordinator\] stage=(\d+) replica=(\d+) cap=(\d+)->(\d+) "
    r"pressure=([0-9.]+) reason=(\S+)"
)
PAUSE_RE = re.compile(
    r"\[HBMAdmission\] paused .*waiting=(\d+) running=(\d+) .*effective_cap=(\d+)"
)
RESUME_RE = re.compile(r"\[HBMAdmission\] resumed .*waiting=(\d+) running=(\d+)")
LOG_TIME_RE = re.compile(
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:[,.](?P<fraction>\d+))?"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def seconds_of_day(line: str) -> float | None:
    match = LOG_TIME_RE.search(line)
    if match is None:
        return None
    return (
        int(match.group("hour")) * 3600
        + int(match.group("minute")) * 60
        + int(match.group("second"))
        + float(f"0.{match.group('fraction') or '0'}")
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def gpu_samples(path: Path) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    if not path.exists():
        return samples
    for row in csv.reader(path.open(newline="")):
        if len(row) < 4:
            continue
        try:
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
            free_mib, total_mib = float(row[2]), float(row[3])
        except ValueError:
            continue
        samples.append((stamp.timestamp() % 86400, 1.0 - free_mib / total_mib))
    return samples


def nearest_pressure(samples: list[tuple[float, float]], event_s: float) -> float | None:
    if not samples:
        return None
    return min(samples, key=lambda item: abs(item[0] - event_s))[1]


def first_crossing(samples: list[tuple[float, float]], threshold: float) -> float | None:
    return next((stamp for stamp, pressure in samples if pressure >= threshold), None)


def parse_case(case: dict[str, Any], critical_watermark: float, critical_cap: int) -> dict[str, Any]:
    artifacts = case.get("artifacts", {})
    server_log = Path(artifacts.get("server_log", ""))
    gpu_csv = Path(artifacts.get("gpu_csv", ""))
    lines = server_log.read_text(errors="replace").splitlines() if server_log.is_file() else []
    samples = gpu_samples(gpu_csv)
    crossing = first_crossing(samples, critical_watermark)
    caps: list[dict[str, Any]] = []
    pauses: list[dict[str, Any]] = []
    resumes: list[dict[str, Any]] = []
    for line in lines:
        stamp = seconds_of_day(line)
        if match := CAP_RE.search(line):
            observed = float(match.group(5))
            nvml = nearest_pressure(samples, stamp) if stamp is not None else None
            caps.append(
                {
                    "stage_id": int(match.group(1)),
                    "old_cap": int(match.group(3)),
                    "new_cap": int(match.group(4)),
                    "controller_pressure": observed,
                    "nvml_pressure": nvml,
                    "pressure_abs_error": abs(observed - nvml) if nvml is not None else None,
                    "time_s": stamp,
                    "reason": match.group(6),
                }
            )
        if match := PAUSE_RE.search(line):
            pauses.append({"time_s": stamp, "waiting": int(match.group(1)), "running": int(match.group(2))})
        if match := RESUME_RE.search(line):
            resumes.append({"time_s": stamp, "waiting": int(match.group(1)), "running": int(match.group(2))})
    critical_events = [item for item in caps if item["new_cap"] == critical_cap]
    first_critical = min(
        (item["time_s"] for item in critical_events if item["time_s"] is not None), default=None
    )
    latency = None
    if crossing is not None and first_critical is not None:
        latency = first_critical - crossing
        if latency < -43200:
            latency += 86400
        elif latency > 43200:
            latency -= 86400
    scheduler = case.get("scheduler", {})
    pressure = case.get("pressure", {})
    benchmark = case.get("benchmark", {})
    server_failure = case.get("status") not in {"completed", "acceptance_failed"}
    oom = scheduler.get("oom_mentions", 0) > 0
    return {
        "arm": case.get("arm", {}).get("name"),
        "repeat": case.get("repeat"),
        "status": case.get("status"),
        "oom_or_server_failure": bool(oom or server_failure),
        "oom_mentions": scheduler.get("oom_mentions", 0),
        "sidecar_allocation_ooms": pressure.get("allocation_oom_count", 0),
        "request_throughput": benchmark.get("request_throughput"),
        "completed_requests": benchmark.get("completed"),
        "critical_crossing_to_cap_s": latency,
        "cap_events": caps,
        "critical_cap_reached": bool(critical_events),
        "paused_with_waiting": any(item["waiting"] > 0 for item in pauses),
        "resumed": bool(resumes),
        "pause_events": pauses,
        "resume_events": resumes,
        "shared_device_all_stages": bool(scheduler.get("per_stage"))
        and all(value.get("shared_device_decisions", 0) > 0 for value in scheduler["per_stage"].values()),
    }


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return statistics.fmean(bool(row.get(key)) for row in rows) if rows else None


def analyze(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing base-runner metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    arguments = metadata["arguments"]
    cases = [json.loads(path.read_text()) for path in sorted((output_dir / "cases").glob("*/status.json"))]
    rows = [parse_case(case, arguments["critical_watermark"], arguments["critical_admission_cap"]) for case in cases]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    c_rows = by_arm.get("C_dynamic_off_pressure", [])
    d_rows = by_arm.get("D_dynamic_on_pressure", [])
    e_name = next((name for name in by_arm if name.startswith("E_fixed_cap_")), None)
    e_rows = by_arm.get(e_name, []) if e_name else []
    c_failure = rate(c_rows, "oom_or_server_failure")
    d_failure = rate(d_rows, "oom_or_server_failure")
    latencies = [float(row["critical_crossing_to_cap_s"]) for row in d_rows if isinstance(row.get("critical_crossing_to_cap_s"), (int, float)) and row["critical_crossing_to_cap_s"] >= 0]
    errors = [
        float(event["pressure_abs_error"])
        for row in d_rows
        for event in row["cap_events"]
        if isinstance(event.get("pressure_abs_error"), (int, float))
    ]
    def throughput(items: list[dict[str, Any]]) -> float | None:
        values = [float(row["request_throughput"]) for row in items if isinstance(row.get("request_throughput"), (int, float))]
        return statistics.median(values) if values else None
    d_tp, e_tp = throughput(d_rows), throughput(e_rows)
    gates: dict[str, bool] = {
        "at_least_20_attempts_per_arm": bool(c_rows and d_rows and e_rows) and min(map(len, (c_rows, d_rows, e_rows))) >= 20,
        "uncontrolled_failure_rate_at_least_30pct": c_failure is not None and c_failure >= 0.30,
        "dynamic_failure_rate_at_most_20pct_of_uncontrolled": c_failure is not None and c_failure > 0 and d_failure is not None and d_failure <= 0.20 * c_failure,
        "dynamic_pause_evidence_at_least_90pct": (rate(d_rows, "paused_with_waiting") or 0) >= 0.90,
        "dynamic_resume_evidence_at_least_90pct": (rate(d_rows, "resumed") or 0) >= 0.90,
        "shared_device_decisions_at_least_90pct": (rate(d_rows, "shared_device_all_stages") or 0) >= 0.90,
        "critical_cap_reached_at_least_90pct": (rate(d_rows, "critical_cap_reached") or 0) >= 0.90,
        "p95_reaction_latency_at_most_1_5s": percentile(latencies, 0.95) is not None and percentile(latencies, 0.95) <= 1.5,
        "p95_pressure_error_at_most_0_03": percentile(errors, 0.95) is not None and percentile(errors, 0.95) <= 0.03,
        "sidecar_never_oomed": sum(row["sidecar_allocation_ooms"] for row in rows) == 0,
        "dynamic_throughput_exceeds_fixed_cap": d_tp is not None and e_tp is not None and d_tp > e_tp,
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "result": "passed" if all(gates.values()) else "not_proven",
        "gates": gates,
        "metrics": {
            "uncontrolled_failure_rate": c_failure,
            "dynamic_failure_rate": d_failure,
            "dynamic_pause_rate": rate(d_rows, "paused_with_waiting"),
            "dynamic_resume_rate": rate(d_rows, "resumed"),
            "dynamic_shared_decision_rate": rate(d_rows, "shared_device_all_stages"),
            "reaction_latency_p95_s": percentile(latencies, 0.95),
            "pressure_abs_error_p95": percentile(errors, 0.95),
            "dynamic_median_throughput": d_tp,
            "fixed_cap_median_throughput": e_tp,
        },
        "attempt_counts": {name: len(items) for name, items in by_arm.items()},
        "runs": rows,
    }
    write_json(output_dir / "effectiveness_report.json", report)
    write_summary(output_dir / "EFFECTIVENESS_SUMMARY.md", report)
    return report


def write_summary(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# Dynamic HBM effectiveness summary", "",
        f"Overall result: **{report['result']}**", "", "## Metrics", "",
        f"- Uncontrolled failure rate: `{metrics['uncontrolled_failure_rate']}`",
        f"- Dynamic failure rate: `{metrics['dynamic_failure_rate']}`",
        f"- Dynamic pause/resume rate: `{metrics['dynamic_pause_rate']}` / `{metrics['dynamic_resume_rate']}`",
        f"- Shared-device decision rate: `{metrics['dynamic_shared_decision_rate']}`",
        f"- P95 critical crossing to cap: `{metrics['reaction_latency_p95_s']}` seconds",
        f"- P95 controller/NVML pressure error: `{metrics['pressure_abs_error_p95']}`",
        f"- Dynamic/fixed median throughput: `{metrics['dynamic_median_throughput']}` / `{metrics['fixed_cap_median_throughput']}`",
        "", "## Acceptance gates", "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["gates"].items())
    lines.extend(["", "Raw per-run evidence is in `effectiveness_report.json` and `cases/`."])
    path.write_text("\n".join(lines) + "\n")


def common_command(args: argparse.Namespace, output_dir: Path, *, repeats: int, concurrency: int, critical: float) -> list[str]:
    command = [
        sys.executable, str(BASE_RUNNER), "--output-dir", str(output_dir),
        "--model", str(args.model), "--deploy-config", str(args.deploy_config),
        "--dataset-path", str(args.dataset_path), "--device", str(args.device),
        "--port", str(args.port), "--repeats", str(repeats),
        "--concurrency", str(concurrency), "--num-prompts", str(args.num_prompts),
        "--num-warmups", str(args.num_warmups), "--max-num-seqs", str(args.max_num_seqs),
        "--min-num-seqs", "2", "--critical-admission-cap", "0",
        "--disconnect-admission-cap", "0", "--guard-mib", "0",
        "--sample-interval-ms", "500", "--report-timeout-ms", "1500",
        "--low-watermark", "0.72", "--high-watermark", "0.82",
        "--critical-watermark", "0.90", "--scale-down-ratio", "0.5",
        "--scale-up-step", "2", "--scale-up-stable-samples", "6",
        "--recovery-complete-samples", "3", "--pressure-trigger-mode", "first-response",
        "--pressure-baseline-seconds", "8", "--pressure-high-target", "0.84",
        "--pressure-high-seconds", "10", "--pressure-critical-target", str(critical),
        "--pressure-critical-seconds", str(args.pressure_critical_seconds),
        "--pressure-recovery-target", "0.76",
        "--pressure-recovery-seconds", "15", "--pressure-post-release-seconds", "30",
        "--pressure-chunk-mib", "64", "--pressure-reserve-mib", str(args.pressure_reserve_mib),
        "--startup-timeout", str(args.startup_timeout), "--benchmark-timeout", str(args.benchmark_timeout),
    ]
    for extra in args.server_extra_arg:
        # Use the --opt=value form: argparse rejects a flag-like value
        # (e.g. --stage-overrides) supplied as a separate token.
        command.append(f"--server-extra-arg={extra}")
    return command


def run_allowing_expected_failures(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as output:
        completed = subprocess.run(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, text=True)
    return completed.returncode


def calibrate(args: argparse.Namespace) -> int:
    manifest: dict[str, Any] = {"schema_version": 1, "cells": []}
    for concurrency in args.concurrencies:
        for critical in args.critical_targets:
            cell = args.output_dir / f"c{concurrency}_p{critical:.3f}"
            command = common_command(args, cell, repeats=args.repeats, concurrency=concurrency, critical=critical)
            command.extend(("--arms", "C_dynamic_off_pressure"))
            exit_code = run_allowing_expected_failures(command, cell / "orchestrator.log")
            report = analyze_calibration(cell)
            manifest["cells"].append({"concurrency": concurrency, "critical_target": critical, "runner_exit_code": exit_code, **report})
            write_json(args.output_dir / "calibration_report.json", manifest)
    candidates = [cell for cell in manifest["cells"] if 0.30 <= cell["failure_rate"] <= 0.80 and cell["sidecar_ooms"] == 0]
    manifest["recommended_cells"] = candidates
    write_json(args.output_dir / "calibration_report.json", manifest)
    return 0 if candidates else 2


def analyze_calibration(output_dir: Path) -> dict[str, Any]:
    statuses = [json.loads(path.read_text()) for path in sorted((output_dir / "cases").glob("*/status.json"))]
    failed = sum(case.get("status") not in {"completed", "acceptance_failed"} or case.get("scheduler", {}).get("oom_mentions", 0) > 0 for case in statuses)
    return {
        "attempts": len(statuses),
        "failures": failed,
        "failure_rate": failed / len(statuses) if statuses else 0.0,
        "sidecar_ooms": sum(case.get("pressure", {}).get("allocation_oom_count", 0) for case in statuses),
    }


def formal(args: argparse.Namespace) -> int:
    command = common_command(args, args.output_dir, repeats=args.repeats, concurrency=args.concurrency, critical=args.critical_target)
    command.extend(("--arms", "C_dynamic_off_pressure", "D_dynamic_on_pressure", "--include-fixed-cap", str(args.fixed_cap)))
    runner_exit = run_allowing_expected_failures(command, args.output_dir / "orchestrator.log")
    report = analyze(args.output_dir)
    report["base_runner_exit_code"] = runner_exit
    write_json(args.output_dir / "effectiveness_report.json", report)
    return 0 if report["result"] == "passed" else 2


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
        item.add_argument("--num-prompts", type=int, default=240)
        item.add_argument("--num-warmups", type=int, default=4)
        item.add_argument("--max-num-seqs", type=int, default=32)
        item.add_argument("--pressure-reserve-mib", type=int, default=512)
        item.add_argument("--pressure-critical-seconds", type=float, default=15.0)
        item.add_argument("--server-extra-arg", action="append", default=[])
        item.add_argument("--startup-timeout", type=int, default=1800)
        item.add_argument("--benchmark-timeout", type=int, default=3600)
    calibration = sub.add_parser("calibrate")
    shared(calibration)
    calibration.add_argument("--critical-targets", type=float, nargs="+", default=[0.93, 0.95, 0.97])
    calibration.add_argument("--concurrencies", type=int, nargs="+", default=[16, 24, 32])
    calibration.add_argument("--repeats", type=int, default=3)
    formal_parser = sub.add_parser("formal")
    shared(formal_parser)
    formal_parser.add_argument("--critical-target", type=float, required=True)
    formal_parser.add_argument("--concurrency", type=int, required=True)
    formal_parser.add_argument("--repeats", type=int, default=20)
    formal_parser.add_argument("--fixed-cap", type=int, default=4)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate":
        return calibrate(args)
    if args.command == "formal":
        return formal(args)
    report = analyze(args.output_dir)
    print(json.dumps({"result": report["result"], "gates": report["gates"]}, indent=2))
    return 0 if report["result"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
