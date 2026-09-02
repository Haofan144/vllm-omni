#!/usr/bin/env python3
"""H9 - Dynamic HBM reduces real OOMs/failures or raises pressure success.

Run calibration first; it finds a pressure/concurrency cell where the dynamic-
off C arm fails in 30%-80% of attempts without the pressure sidecar itself
OOMing.  Freeze that cell, then run at least 20 fresh C/D repetitions.  H9 is
supported when D has either a materially lower OOM/server-failure rate or a
materially higher request success rate, and D's pause/cap/resume mechanism is
visible in at least 90% of attempts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EFFECT_RUNNER = HERE.parent / "run_dynamic_hbm_effectiveness.py"
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_smoke"

sys.path.insert(0, str(HERE.parent))
from run_dynamic_hbm_effectiveness import parse_case  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def analyze(
    output_dir: Path,
    *,
    min_attempts: int = 20,
    min_absolute_improvement: float = 0.20,
    max_failure_rate_ratio: float = 0.20,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing experiment metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    arguments = metadata["arguments"]
    statuses = [json.loads(path.read_text()) for path in sorted((output_dir / "cases").glob("*/status.json"))]
    rows = [parse_case(case, float(arguments["critical_watermark"]),
                       int(arguments["critical_admission_cap"])) for case in statuses]
    by_arm = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in ("C_dynamic_off_pressure", "D_dynamic_on_pressure")
    }
    intended = int(arguments["num_prompts"])

    def failure_rate(items: list[dict[str, Any]]) -> float | None:
        return statistics.fmean(bool(row["oom_or_server_failure"]) for row in items) if items else None

    def request_success_rate(items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        return sum(min(int(row.get("completed_requests") or 0), intended) for row in items) / (len(items) * intended)

    c, d = by_arm["C_dynamic_off_pressure"], by_arm["D_dynamic_on_pressure"]
    c_fail, d_fail = failure_rate(c), failure_rate(d)
    c_success, d_success = request_success_rate(c), request_success_rate(d)
    failure_improvement = c_fail - d_fail if c_fail is not None and d_fail is not None else None
    success_improvement = d_success - c_success if c_success is not None and d_success is not None else None
    failure_ratio = d_fail / c_fail if c_fail else None
    failure_claim = (
        failure_improvement is not None and failure_improvement >= min_absolute_improvement
        and failure_ratio is not None and failure_ratio <= max_failure_rate_ratio
    )
    success_claim = success_improvement is not None and success_improvement >= min_absolute_improvement

    def rate(key: str) -> float:
        return statistics.fmean(bool(row.get(key)) for row in d) if d else 0.0

    sidecar_ooms = sum(int(row["sidecar_allocation_ooms"]) for row in rows)
    gates = {
        "at_least_min_attempts_each_arm": min(len(c), len(d)) >= min_attempts,
        "calibrated_uncontrolled_failure_rate_30_to_80pct": c_fail is not None and 0.30 <= c_fail <= 0.80,
        "oom_failure_or_request_success_materially_improved": failure_claim or success_claim,
        "dynamic_critical_cap_reached_at_least_90pct": rate("critical_cap_reached") >= 0.90,
        "dynamic_paused_with_waiting_at_least_90pct": rate("paused_with_waiting") >= 0.90,
        "dynamic_resumed_at_least_90pct": rate("resumed") >= 0.90,
        "shared_device_decisions_at_least_90pct": rate("shared_device_all_stages") >= 0.90,
        "pressure_sidecar_never_oomed": sidecar_ooms == 0,
    }
    report = {
        "schema_version": 1,
        "hypothesis": "H9",
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "MATCHES EXPECTATION" if all(gates.values()) else "NOT PROVEN",
        "expectation": {
            "min_attempts_each_arm": min_attempts,
            "calibration_failure_rate_range": [0.30, 0.80],
            "min_absolute_improvement": min_absolute_improvement,
            "max_dynamic_over_uncontrolled_failure_ratio": max_failure_rate_ratio,
            "mechanism_evidence_rate": 0.90,
        },
        "gates": gates,
        "metrics": {
            "attempts_C": len(c), "attempts_D": len(d),
            "uncontrolled_failure_rate": c_fail, "dynamic_failure_rate": d_fail,
            "absolute_failure_rate_improvement": failure_improvement,
            "dynamic_over_uncontrolled_failure_ratio": failure_ratio,
            "uncontrolled_request_success_rate": c_success,
            "dynamic_request_success_rate": d_success,
            "absolute_request_success_improvement": success_improvement,
            "failure_claim_passed": failure_claim, "success_claim_passed": success_claim,
            "dynamic_critical_cap_rate": rate("critical_cap_reached"),
            "dynamic_pause_rate": rate("paused_with_waiting"),
            "dynamic_resume_rate": rate("resumed"),
            "dynamic_shared_decision_rate": rate("shared_device_all_stages"),
            "sidecar_allocation_ooms": sidecar_ooms,
        },
        "runs": rows,
    }
    _write_json(output_dir / "H9_result.json", report)
    _write_markdown(output_dir / "H9_result.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    m = report["metrics"]
    lines = [
        "# H9: OOM / high-pressure success effectiveness", "",
        f"**Verdict:** {report['verdict']}", "",
        "| Metric | C: dynamic off | D: dynamic on | Improvement |", "|---|---:|---:|---:|",
        f"| OOM or server-failure rate | {m['uncontrolled_failure_rate']} | {m['dynamic_failure_rate']} | {m['absolute_failure_rate_improvement']} |",
        f"| Request success rate | {m['uncontrolled_request_success_rate']} | {m['dynamic_request_success_rate']} | {m['absolute_request_success_improvement']} |",
        "", "## Gates", "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["gates"].items())
    path.write_text("\n".join(lines) + "\n")


def _shared_cli(args: argparse.Namespace) -> list[str]:
    cli = [
        "--output-dir", str(args.output_dir), "--model", str(args.model),
        "--deploy-config", str(args.deploy_config), "--dataset-path", str(args.dataset_path),
        "--device", str(args.device), "--port", str(args.port),
        "--num-prompts", str(args.num_prompts), "--num-warmups", str(args.num_warmups),
        "--max-num-seqs", str(args.max_num_seqs), "--pressure-reserve-mib", str(args.pressure_reserve_mib),
        "--pressure-critical-seconds", str(args.pressure_critical_seconds),
        "--startup-timeout", str(args.startup_timeout), "--benchmark-timeout", str(args.benchmark_timeout),
    ]
    for extra in args.server_extra_arg:
        # --opt=value form: argparse rejects a flag-like value as a
        # separate token (e.g. --stage-overrides).
        cli.append(f"--server-extra-arg={extra}")
    return cli


def run_calibration(args: argparse.Namespace) -> int:
    command = [sys.executable, str(EFFECT_RUNNER), "calibrate", *_shared_cli(args),
               "--critical-targets", *map(str, args.critical_targets),
               "--concurrencies", *map(str, args.concurrencies), "--repeats", str(args.repeats)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "H9_calibration.log").open("w") as log:
        return subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def run_formal(args: argparse.Namespace) -> int:
    command = [sys.executable, str(EFFECT_RUNNER), "formal", *_shared_cli(args),
               "--critical-target", str(args.critical_target), "--concurrency", str(args.concurrency),
               "--repeats", str(args.repeats), "--fixed-cap", str(args.fixed_cap)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "H9_orchestrator.log").open("w") as log:
        runner = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    report = analyze(args.output_dir, min_attempts=args.min_attempts,
                     min_absolute_improvement=args.min_absolute_improvement,
                     max_failure_rate_ratio=args.max_failure_rate_ratio)
    report["underlying_runner_exit_code"] = runner.returncode
    _write_json(args.output_dir / "H9_result.json", report)
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibrate")
    formal = sub.add_parser("run")
    analysis = sub.add_parser("analyze")
    for item in (calibration, formal):
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        item.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
        item.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
        item.add_argument("--device", type=int, default=0); item.add_argument("--port", type=int, default=8000)
        item.add_argument("--num-prompts", type=int, default=240); item.add_argument("--num-warmups", type=int, default=4)
        item.add_argument("--max-num-seqs", type=int, default=32); item.add_argument("--pressure-reserve-mib", type=int, default=512)
        item.add_argument("--pressure-critical-seconds", type=float, default=15.0)
        item.add_argument("--server-extra-arg", action="append", default=[])
        item.add_argument("--startup-timeout", type=int, default=1800); item.add_argument("--benchmark-timeout", type=int, default=3600)
    calibration.add_argument("--critical-targets", type=float, nargs="+", default=[0.93, 0.95, 0.97])
    calibration.add_argument("--concurrencies", type=int, nargs="+", default=[16, 24, 32])
    calibration.add_argument("--repeats", type=int, default=3)
    formal.add_argument("--critical-target", type=float, required=True)
    formal.add_argument("--concurrency", type=int, required=True)
    formal.add_argument("--repeats", type=int, default=20); formal.add_argument("--fixed-cap", type=int, default=4)
    for item in (formal, analysis):
        if item is analysis:
            item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--min-attempts", type=int, default=20)
        item.add_argument("--min-absolute-improvement", type=float, default=0.20)
        item.add_argument("--max-failure-rate-ratio", type=float, default=0.20)
    return root


def main() -> int:
    args = parser().parse_args(); args.output_dir = args.output_dir.resolve()
    if args.command == "calibrate": return run_calibration(args)
    if args.command == "run": return run_formal(args)
    report = analyze(args.output_dir, min_attempts=args.min_attempts,
                     min_absolute_improvement=args.min_absolute_improvement,
                     max_failure_rate_ratio=args.max_failure_rate_ratio)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}, indent=2))
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
