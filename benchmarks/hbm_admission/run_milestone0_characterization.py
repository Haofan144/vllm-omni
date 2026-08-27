#!/usr/bin/env python3
"""Run and summarize the Milestone-0 HBM characterization study.

This experiment-layer wrapper reuses ``run_dynamic_hbm_matrix.py`` and its
existing artifacts. It does not import or modify the serving hot path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
MATRIX_RUNNER = Path(__file__).with_name("run_dynamic_hbm_matrix.py")
CORE_ARMS = {
    "A_dynamic_off_no_pressure",
    "B_dynamic_on_no_pressure",
    "C_dynamic_off_pressure",
    "D_dynamic_on_pressure",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_probe(command: list[str]) -> dict[str, Any]:
    """Run a bounded, read-only environment probe without failing the study."""
    try:
        completed = subprocess.run(command, cwd=REPO, check=False, capture_output=True, text=True, timeout=15)
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def build_manifest(spec_path: Path, argv: list[str]) -> dict[str, Any]:
    spec_bytes = spec_path.read_bytes()
    packages: dict[str, str | None] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("vllm", "vllm-omni", "torch", "ray", "pyyaml"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
    except ImportError:
        pass
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "command": argv,
        "spec": {
            "source": str(spec_path),
            "sha256": hashlib.sha256(spec_bytes).hexdigest(),
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "packages": packages,
        "git": {
            "revision": run_probe(["git", "rev-parse", "HEAD"]),
            "status": run_probe(["git", "status", "--short"]),
            "remotes": run_probe(["git", "remote", "-v"]),
        },
        "gpu": run_probe(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("experiment spec must be a YAML mapping")
    return payload


def resolve_metric(payload: dict[str, Any], dotted_path: str) -> float | None:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def second_of_day(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1e6


def elapsed_seconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    elapsed = end - start
    if elapsed < -12 * 3600:  # A run may cross UTC midnight.
        elapsed += 24 * 3600
    return elapsed if elapsed >= 0 else None


def _milliseconds(value: float | None) -> float | None:
    return value * 1000 if value is not None else None


def reaction_delays(case: dict[str, Any]) -> dict[str, float | None]:
    pressure_events = case.get("pressure", {}).get("events", [])
    cap_changes = case.get("scheduler", {}).get("cap_changes", [])
    release_event = next(
        (event for event in pressure_events if event.get("event") in {"partial_release", "released"}),
        None,
    )
    first_down = next(
        (change for change in cap_changes if change.get("new_cap", 0) < change.get("old_cap", 0)),
        None,
    )
    first_up = next(
        (change for change in cap_changes if change.get("new_cap", 0) > change.get("old_cap", 0)),
        None,
    )
    # Attribute a downscale to the latest pressure ramp whose target had been
    # crossed. Using the first/high ramp would overstate reaction time when the
    # controller intentionally waits for the later critical ramp.
    down_pressure = first_down.get("pressure") if first_down else None
    eligible_ramps = [
        event
        for event in pressure_events
        if event.get("event") == "pressure_ramp_started"
        and isinstance(event.get("target"), (int, float))
        and isinstance(down_pressure, (int, float))
        and event["target"] <= down_pressure + 1e-3
    ]
    high_event = eligible_ramps[-1] if eligible_ramps else None
    if high_event is None:  # Compatibility with artifacts created before Milestone 0.
        reached = [event for event in pressure_events if event.get("event") == "target_reached"]
        high_event = reached[-1] if reached else None
    return {
        "downscale_reaction_ms": _milliseconds(
            elapsed_seconds(
                second_of_day(high_event.get("timestamp")) if high_event else None,
                first_down.get("log_second_of_day") if first_down else None,
            )
        ),
        "recovery_reaction_ms": _milliseconds(
            elapsed_seconds(
                second_of_day(release_event.get("timestamp")) if release_event else None,
                first_up.get("log_second_of_day") if first_up else None,
            )
        ),
    }


def case_row(cell: dict[str, Any], case: dict[str, Any], metric_map: dict[str, str]) -> dict[str, Any]:
    scheduler = case.get("scheduler", {})
    gpu = case.get("gpu", {})
    pressure = case.get("pressure", {})
    arm = case.get("arm", {})
    row: dict[str, Any] = {
        "cell": cell["name"],
        "workload": cell["workload"],
        "pressure_profile": cell["pressure_profile"],
        "controller": cell["controller"],
        "arm": arm.get("name"),
        "repeat": case.get("repeat"),
        "status": case.get("status"),
        "accepted": case.get("acceptance", {}).get("passed"),
        "dynamic": arm.get("dynamic"),
        "external_pressure": arm.get("pressure"),
        "fixed_cap": arm.get("fixed_cap"),
        "oom_mentions": scheduler.get("oom_mentions"),
        "preemption_mentions": scheduler.get("preemption_mentions"),
        "cap_change_count": scheduler.get("cap_change_count"),
        "minimum_observed_cap": scheduler.get("minimum_observed_cap"),
        "local_coordinator_start_count": scheduler.get("local_coordinator_start_count"),
        "gpu_peak_memory_mib": gpu.get("memory_used_mib", {}).get("peak"),
        "gpu_mean_memory_mib": gpu.get("memory_used_mib", {}).get("mean"),
        "gpu_peak_pressure": gpu.get("pressure", {}).get("peak"),
        "gpu_mean_utilization_percent": gpu.get("gpu_utilization_percent", {}).get("mean"),
        "sidecar_peak_pressure": pressure.get("peak_reported_pressure"),
        **reaction_delays(case),
    }
    benchmark = case.get("benchmark", {})
    for name, dotted_path in metric_map.items():
        row[f"metric_{name}"] = resolve_metric(benchmark, dotted_path)
    return row


def median_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed" and row.get("accepted") is True:
            groups[(str(row["cell"]), str(row["arm"]))].append(row)
    aggregated: list[dict[str, Any]] = []
    identity = {"cell", "workload", "pressure_profile", "controller", "arm"}
    skipped = {"repeat", "status", "accepted", "dynamic", "external_pressure"}
    for _, selected in sorted(groups.items()):
        result = {key: selected[0][key] for key in identity}
        result["runs"] = len(selected)
        result["accepted_runs"] = sum(row.get("accepted") is True for row in selected)
        for key in selected[0]:
            if key in identity or key in skipped:
                continue
            values = [row.get(key) for row in selected]
            numeric = [
                float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric:
                result[f"median_{key}"] = statistics.median(numeric)
        aggregated.append(result)
    return aggregated


def ratio(numerator: Any, denominator: Any) -> float | None:
    return float(numerator) / float(denominator) if isinstance(numerator, (int, float)) and denominator else None


def build_comparisons(aggregates: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregates:
        by_cell[str(row["cell"])][str(row["arm"])] = row
    comparisons: list[dict[str, Any]] = []
    for cell, arms in sorted(by_cell.items()):
        item: dict[str, Any] = {"cell": cell}
        pairs = {
            "monitoring_B_over_A": ("B_dynamic_on_no_pressure", "A_dynamic_off_no_pressure"),
            "pressure_no_control_C_over_A": ("C_dynamic_off_pressure", "A_dynamic_off_no_pressure"),
            "reactive_D_over_C": ("D_dynamic_on_pressure", "C_dynamic_off_pressure"),
        }
        fixed_arm = next((name for name in arms if name.startswith("E_fixed_cap_")), None)
        if fixed_arm:
            pairs["reactive_D_over_fixed_E"] = ("D_dynamic_on_pressure", fixed_arm)
        for label, (numerator_arm, denominator_arm) in pairs.items():
            for metric in metric_names:
                key = f"median_metric_{metric}"
                item[f"{label}__{metric}"] = ratio(
                    arms.get(numerator_arm, {}).get(key), arms.get(denominator_arm, {}).get(key)
                )
        item["no_control_pressure_oom_mentions"] = arms.get("C_dynamic_off_pressure", {}).get("median_oom_mentions")
        item["reactive_pressure_oom_mentions"] = arms.get("D_dynamic_on_pressure", {}).get("median_oom_mentions")
        comparisons.append(item)
    return comparisons


def coverage(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {str(row.get("arm")) for row in rows}
    checks = {
        "all_ABCD_arms_present": CORE_ARMS <= arms,
        "fixed_cap_baseline_present": any(arm.startswith("E_fixed_cap_") for arm in arms),
        "at_least_three_load_levels": len(spec.get("workloads", [])) >= 3,
        "at_least_two_pressure_shapes": len(spec.get("pressure_profiles", [])) >= 2,
        "at_least_three_repetitions": int(spec.get("repeats", 3)) >= 3,
        "all_cases_terminal": bool(rows)
        and all(row.get("status") in {"completed", "acceptance_failed", "failed"} for row in rows),
        "all_observed_runs_completed_and_accepted": bool(rows)
        and all(row.get("status") == "completed" and row.get("accepted") is True for row in rows),
        "all_dynamic_runs_accepted": bool(rows)
        and all(
            row.get("status") == "completed" and row.get("accepted") is True
            for row in rows
            if row.get("dynamic") is True
        ),
    }
    return {
        "checks": checks,
        "ready_for_paper_plots": all(checks.values()),
        "warnings": [name for name, passed in checks.items() if not passed],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_summary(report: dict[str, Any]) -> str:
    coverage_result = report["coverage"]
    lines = [
        "# Milestone 0 characterization summary",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Paper-plot readiness: **{'yes' if coverage_result['ready_for_paper_plots'] else 'no'}**",
        "",
        "## Coverage",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'MISSING'}: `{name}`" for name, passed in coverage_result["checks"].items())
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `manifest.json`: source revision, dirty state, software, and GPU inventory.",
            "- `runs.csv`: one row per repetition and arm.",
            "- `aggregates.csv`: median values for each matrix cell and arm.",
            "- `comparisons.csv`: A/B/C/D/E ratios used by baseline plots.",
            "",
            "## Known observability gaps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["observability_gaps"])
    return "\n".join(lines) + "\n"


def summarize(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    plan_path = output_dir / "matrix_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"missing matrix plan: {plan_path}")
    cells = json.loads(plan_path.read_text()).get("cells", [])
    metric_map = spec.get("metric_map") or {
        "throughput": "request_throughput",
        "p99_latency_ms": "p99_e2el_ms",
        "p99_first_output_ms": "p99_audio_ttfp_ms",
    }
    rows: list[dict[str, Any]] = []
    missing_status_files: list[str] = []
    for cell in cells:
        status_paths = sorted((Path(cell["output_dir"]) / "cases").glob("*/status.json"))
        if not status_paths:
            missing_status_files.append(cell["name"])
        for status_path in status_paths:
            rows.append(case_row(cell, json.loads(status_path.read_text()), metric_map))
    aggregates = median_rows(rows)
    comparisons = build_comparisons(aggregates, list(metric_map))
    coverage_result = coverage(spec, rows)
    expected_arm_count = len(spec.get("arms", CORE_ARMS)) + int(
        spec.get("runner_options", {}).get("include_fixed_cap") is not None
    )
    expected_runs = len(cells) * int(spec.get("repeats", 3)) * expected_arm_count
    coverage_result["checks"]["all_planned_cells_observed"] = not missing_status_files
    coverage_result["checks"]["all_planned_runs_observed"] = len(rows) >= expected_runs
    coverage_result["ready_for_paper_plots"] = all(coverage_result["checks"].values())
    coverage_result["warnings"] = [name for name, passed in coverage_result["checks"].items() if not passed]
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "spec": str(spec_path),
        "coverage": coverage_result,
        "counts": {
            "planned_cells": len(cells),
            "expected_runs": expected_runs,
            "observed_runs": len(rows),
            "accepted_runs": sum(row.get("status") == "completed" and row.get("accepted") is True for row in rows),
            "invalid_runs": sum(row.get("status") != "completed" or row.get("accepted") is not True for row in rows),
            "missing_cells": missing_status_files,
        },
        "comparisons": comparisons,
        "observability_gaps": [
            "GPU telemetry is device-wide; it cannot attribute HBM use to an individual pipeline stage.",
            "Logs expose physical pressure and scheduler caps, but not per-request logical reservation demand.",
            "Reaction delay uses wall-clock timestamps from two processes; clock synchronization must be stated.",
            "Burst and mixed-size claims require corresponding workload generators/datasets in the matrix spec.",
        ],
    }
    write_json(output_dir / "milestone0_report.json", report)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "aggregates.csv", aggregates)
    write_csv(output_dir / "comparisons.csv", comparisons)
    (output_dir / "MILESTONE0_SUMMARY.md").write_text(render_summary(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="freeze metadata and expand the matrix only")
    parser.add_argument("--summarize-only", action="store_true", help="regenerate tables from existing artifacts")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        write_json(output_dir / "manifest.json", build_manifest(spec_path, sys.argv))
        (output_dir / "experiment_spec.yaml").write_text(spec_path.read_text())
        command = [
            sys.executable,
            str(MATRIX_RUNNER),
            "--spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.continue_on_error:
            command.append("--continue-on-error")
        completed = subprocess.run(command, cwd=REPO, check=False)
        if args.dry_run:
            return completed.returncode
    report = summarize(spec_path, output_dir)
    print(f"Milestone 0 report: {output_dir / 'milestone0_report.json'}")
    if not args.summarize_only and completed.returncode:
        return completed.returncode
    return 0 if report["coverage"]["ready_for_paper_plots"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
