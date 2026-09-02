#!/usr/bin/env python3
"""Real-GPU B0-vs-enforce comparison for the Milestone-2 AR resource estimator.

Unlike run_dynamic_hbm_experiment.py (which compares the Milestone-1 HBM
pressure/watermark controller across arms A-E), this script holds the
Milestone-1 controller fixed (``dynamic_hbm.enabled: true`` in every arm, no
external memory pressure) and varies exactly one thing:
``dynamic_hbm.resource_admission_mode``.

Arms
----
B0  resource_admission_mode: off      -- plain slot-count admission (baseline)
B4  resource_admission_mode: enforce  -- profile + P95 quantile KV-block gate,
                                         with a pre-trained resource_profile_path
                                         (produced by analyze_ar_shadow_trace.py)

The online calibrator (EWMA) is unconditionally live once observations start
flowing in enforce mode -- there is no config toggle to disable it today, so
B4 here already includes whatever correction the calibrator learns online;
this script does not attempt a clean B4-vs-B5 (EWMA on/off) split.

The talker's KV pool is intentionally shrunk (--stage0-gpu-mem-util, well
below the qwen3_tts.yaml default of 0.3) and concurrency pushed above what
the shrunk pool can hold for every in-flight request's worst-case (max_tokens)
reservation, so that B0's blindness to per-request KV cost and B4's
resource-aware gate can actually produce different behavior. Reuses server /
benchmark-client plumbing from run_dynamic_hbm_experiment.py; does not reuse
its Arm/CORE_ARMS Milestone-1 pressure axis or its acceptance-gate logic
(both are about the watermark controller, not this admission-mode axis).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dynamic_hbm_experiment as base  # noqa: E402

REPO = base.REPO


@dataclass(frozen=True)
class AdmissionArm:
    name: str
    resource_admission_mode: str  # "off" | "enforce"
    use_profile: bool


ARMS = (
    AdmissionArm("B0_slot_count_only", "off", use_profile=False),
    AdmissionArm("B4_profile_quantile_enforce", "enforce", use_profile=True),
)


def prepare_config(args: argparse.Namespace, arm: AdmissionArm, case_dir: Path) -> Path:
    config = yaml.safe_load(args.deploy_config.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("stages"), list):
        raise ValueError(f"deploy config has no stages list: {args.deploy_config}")
    config = deepcopy(config)
    found = False
    for stage in config["stages"]:
        if int(stage["stage_id"]) != args.ar_stage_id:
            continue
        found = True
        stage["max_num_seqs"] = args.max_num_seqs
        stage["gpu_memory_utilization"] = args.stage0_gpu_mem_util
        dynamic_hbm: dict[str, Any] = {
            "enabled": True,
            "min_num_seqs": 1,
            "critical_admission_cap": 0,
            "disconnect_admission_cap": 0,
            "resource_admission_mode": arm.resource_admission_mode,
            "resource_target_coverage": args.coverage,
            "resource_profile_min_samples": args.profile_min_samples,
        }
        if arm.use_profile:
            dynamic_hbm.update(
                {
                    "resource_profile_path": str(args.profile_path),
                    "resource_profile_device_type": args.device_type,
                }
            )
        if args.resource_observation:
            dynamic_hbm.update(
                {
                    "resource_observation_path": str(
                        case_dir
                        / "resource_observations.stage-{stage_id}.replica-{replica_id}.pid-{pid}.jsonl"
                    ),
                    "resource_observation_flush_size": 1,
                }
            )
        stage["dynamic_hbm"] = dynamic_hbm
    if not found:
        raise ValueError(f"--ar-stage-id {args.ar_stage_id} absent from deploy config")
    path = case_dir / "deploy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_arm(args: argparse.Namespace, arm: AdmissionArm) -> dict[str, Any]:
    case_dir = args.output_dir / "cases" / arm.name
    case_dir.mkdir(parents=True, exist_ok=True)
    deploy = prepare_config(args, arm, case_dir)
    server_log = case_dir / "server.log"
    gpu_csv = case_dir / "gpu.csv"
    client_log = case_dir / "client.log"
    result_json = case_dir / "benchmark.json"
    status: dict[str, Any] = {
        "status": "running",
        "arm": asdict(arm),
        "started_at": base.utc_now(),
        "artifacts": {"deploy": str(deploy), "server_log": str(server_log), "gpu_csv": str(gpu_csv)},
    }
    base.write_json(case_dir / "status.json", status)
    server = monitor = client = None
    server_file = monitor_file = None
    try:
        base.wait_gpu_release(args.device, args.release_threshold_mib)
        server, server_file = base.start_server(args, deploy, server_log)
        monitor, monitor_file = base.start_gpu_monitor(args.device, gpu_csv)
        command = base.benchmark_command(args, result_json)
        status["benchmark_command"] = command
        with client_log.open("w") as output:
            client = subprocess.Popen(
                command,
                cwd=REPO,
                env={**os.environ, "PYTHONPATH": str(REPO)},
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                client.wait(timeout=args.benchmark_timeout)
            except subprocess.TimeoutExpired:
                base.stop_group(client, timeout=5)
                raise
        status["benchmark_exit_code"] = client.returncode
        status["status"] = "completed" if client.returncode == 0 and result_json.exists() else "failed"
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        base.stop_group(client, timeout=5)
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
        if monitor_file is not None:
            monitor_file.close()
        base.stop_group(server)
        if server_file is not None:
            server_file.close()
    status["finished_at"] = base.utc_now()
    status["gpu"] = base.parse_gpu_csv(gpu_csv) if gpu_csv.exists() else {}
    # A fingerprint mismatch between --profile-path and the live server's
    # (model_id, device_type, dtype, tp_size, block_size, execution_mode)
    # degrades ARProfileStore.read_jsonl to a silent hard-fallback (logged as
    # a WARNING, not an error) -- surface it here so a broken B4 arm cannot be
    # mistaken for a working one.
    status["profile_load_warning"] = (
        "Unable to load AR resource profile" in server_log.read_text(errors="replace")
        if server_log.exists()
        else None
    )
    if result_json.exists():
        status["benchmark"] = json.loads(result_json.read_text())
    status["artifacts"].update(client_log=str(client_log), benchmark_json=str(result_json))
    base.write_json(case_dir / "status.json", status)
    return status


def summarize(statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, status in statuses.items():
        benchmark = status.get("benchmark", {})
        summary[name] = {
            "status": status["status"],
            "profile_load_warning": status.get("profile_load_warning"),
            "completed": benchmark.get("completed"),
            "failed": benchmark.get("failed"),
            "request_throughput": benchmark.get("request_throughput"),
            "p99_e2el_ms": benchmark.get("p99_e2el_ms"),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=base.DEFAULT_DEPLOY)
    parser.add_argument("--ar-stage-id", type=int, default=0)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--device-type", required=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--profile-min-samples", type=int, default=20)
    parser.add_argument("--stage0-gpu-mem-util", type=float, default=0.11)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--dataset-path", type=Path, default=REPO / "benchmarks/build_dataset/seed_tts_long")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--extra-body", default='{"voice":"Vivian","language":"English","task_type":"CustomVoice"}')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=150)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--request-profile", default="qwen3-tts")
    parser.add_argument("--benchmark-command-json")
    parser.add_argument("--server-command-json")
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--resource-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safetensors-load-strategy", default="prefetch")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=3600)
    parser.add_argument("--release-threshold-mib", type=int, default=1024)
    parser.add_argument("--arms", nargs="+", choices=[arm.name for arm in ARMS])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [arm for arm in ARMS if args.arms is None or arm.name in args.arms]
    statuses: dict[str, dict[str, Any]] = {}
    for arm in selected:
        print(f"=== running {arm.name} ===", flush=True)
        statuses[arm.name] = run_arm(args, arm)
    summary = summarize(statuses)
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
