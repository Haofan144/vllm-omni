#!/usr/bin/env python3
"""Run a repeatable dynamic-HBM experiment and retain all artifacts.

The default profile tests Qwen3-TTS 0.6B.  Other models can use a built-in
request profile or provide ``--benchmark-command-json`` as a JSON argv array.
The placeholders {host}, {port}, {model}, {concurrency}, {num_prompts},
{result_json}, and {repo} are expanded without invoking a shell.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DYNAMIC_EVENT_RE = re.compile(
    r"\[HBMCoordinator\] stage=(\d+) replica=(\d+) cap=(\d+)->(\d+) pressure=([0-9.]+) reason=(\S+)"
)
FIRST_REPORT_RE = re.compile(
    r"\[HBMCoordinator\] stage=(\d+) replica=(\d+) first central memory report sent "
    r"ranks=(\d+) pressure=([0-9.]+)"
)
REGISTERED_STREAM_RE = re.compile(
    r"\[HBMCoordinator\] registered memory stream stage=(\d+) replica=(\d+) devices=(.+)$"
)
LOCAL_COORDINATOR_STARTED_RE = re.compile(r"\[StageRuntime\] Local dynamic-HBM coordinator started at (\S+)")
CENTRAL_DECISION_RE = re.compile(
    r"\[HBMCoordinator\] central decision stage=(\d+) replica=(\d+) cap=(\d+) "
    r"pressure=([0-9.]+) reason=(\S+) generation=(\d+)"
)
LOG_TIME_RE = re.compile(r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?:[,.](?P<fraction>\d+))?")


@dataclass(frozen=True)
class Arm:
    name: str
    dynamic: bool
    pressure: bool
    fixed_cap: int | None = None


CORE_ARMS = (
    Arm("A_dynamic_off_no_pressure", False, False),
    Arm("B_dynamic_on_no_pressure", True, False),
    Arm("C_dynamic_off_pressure", False, True),
    Arm("D_dynamic_on_pressure", True, True),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stop_group(proc: subprocess.Popen | None, timeout: int = 45) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def wait_health(proc: subprocess.Popen, host: str, port: int, timeout: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"server exited with {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy; see {log_path}")


def wait_gpu_release(device: int, threshold_mib: int, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if float(output.strip().splitlines()[0]) <= threshold_mib:
            return
        time.sleep(2)
    raise TimeoutError("GPU memory did not return below release threshold")


def prepare_config(args: argparse.Namespace, arm: Arm, case_dir: Path) -> Path:
    config = yaml.safe_load(args.deploy_config.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("stages"), list):
        raise ValueError(f"deploy config has no stages list: {args.deploy_config}")
    config = deepcopy(config)
    selected = set(args.dynamic_stage_ids) if args.dynamic_stage_ids else None
    found: set[int] = set()
    for stage in config["stages"]:
        stage_id = int(stage["stage_id"])
        if selected is not None and stage_id not in selected:
            continue
        found.add(stage_id)
        effective_cap = arm.fixed_cap or args.max_num_seqs
        stage["max_num_seqs"] = effective_cap
        stage["dynamic_hbm"] = {
            "enabled": arm.dynamic,
            "min_num_seqs": min(args.min_num_seqs, effective_cap),
            "critical_admission_cap": min(args.critical_admission_cap, effective_cap),
            "disconnect_admission_cap": min(args.disconnect_admission_cap, effective_cap),
            "recovery_complete_samples": args.recovery_complete_samples,
            "guard_bytes": args.guard_mib * 1024**2,
            "guard_ratio": args.guard_ratio,
            "immediate_sample_min_interval_ms": args.immediate_sample_min_interval_ms,
            "fail_closed_on_disconnect": args.fail_closed_on_disconnect,
            "sample_interval_ms": args.sample_interval_ms,
            "report_timeout_ms": args.report_timeout_ms,
            "missing_report_grace_samples": args.missing_report_grace_samples,
            "low_watermark": args.low_watermark,
            "high_watermark": args.high_watermark,
            "critical_watermark": args.critical_watermark,
            "scale_down_ratio": args.scale_down_ratio,
            "scale_up_step": args.scale_up_step,
            "scale_up_stable_samples": args.scale_up_stable_samples,
        }
    if selected is not None and found != selected:
        raise ValueError(f"dynamic stage IDs absent from deploy config: {sorted(selected - found)}")
    path = case_dir / "deploy.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def start_server(args: argparse.Namespace, deploy: Path, log_path: Path) -> tuple[subprocess.Popen, Any]:
    values = {
        "python": sys.executable,
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "deploy": str(deploy),
        "repo": str(REPO),
    }
    if args.server_command_json:
        raw = json.loads(args.server_command_json)
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("server-command-json must be a JSON array of strings")
        command = [item.format(**values) for item in raw]
    else:
        command = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            args.model,
            "--omni",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--deploy-config",
            str(deploy),
            "--log-stats",
            "--safetensors-load-strategy",
            args.safetensors_load_strategy,
            "--init-timeout",
            str(args.startup_timeout),
            "--stage-init-timeout",
            str(args.startup_timeout),
        ]
    command.extend(args.server_extra_arg)
    log_file = log_path.open("w")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    proc = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        wait_health(proc, args.host, args.port, args.startup_timeout, log_path)
    except Exception:
        stop_group(proc)
        log_file.close()
        raise
    return proc, log_file


def start_gpu_monitor(device: int, path: Path) -> tuple[subprocess.Popen, Any]:
    output = path.open("w")
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=timestamp,memory.used,memory.free,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        "200",
    ]
    return subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True), output


def benchmark_command(args: argparse.Namespace, result_json: Path) -> list[str]:
    values = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "result_json": str(result_json),
        "repo": str(REPO),
    }
    if args.benchmark_command_json:
        raw = json.loads(args.benchmark_command_json)
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("benchmark-command-json must be a JSON array of strings")
        return [item.format(**values) for item in raw]
    if args.request_profile != "qwen3-tts":
        raise ValueError("non-Qwen profiles require --benchmark-command-json")
    vllm_cli = Path(sys.executable).parent / "vllm"
    return [
        str(vllm_cli) if vllm_cli.exists() else "vllm",
        "bench",
        "serve",
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--backend",
        "openai-audio-speech",
        "--endpoint",
        "/v1/audio/speech",
        "--dataset-name",
        "seed-tts-text",
        "--dataset-path",
        str(args.dataset_path),
        "--seed-tts-locale",
        args.locale,
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--extra-body",
        args.extra_body,
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        args.request_rate,
        "--percentile-metrics",
        "ttft,e2el,audio_rtf,audio_ttfp,audio_duration,audio_underrun",
        "--save-result",
        "--result-dir",
        str(result_json.parent),
        "--result-filename",
        result_json.name,
    ]


def start_pressure(args: argparse.Namespace, path: Path) -> tuple[subprocess.Popen, Any]:
    output = path.open("w")
    command = [
        sys.executable,
        str(Path(__file__).with_name("memory_pressure.py")),
        "--device",
        "0",  # CUDA_VISIBLE_DEVICES makes the selected physical GPU local device zero.
        "--baseline-seconds",
        str(args.pressure_baseline_seconds),
        "--high-target",
        str(args.pressure_high_target),
        "--high-seconds",
        str(args.pressure_high_seconds),
        "--critical-target",
        str(args.pressure_critical_target),
        "--critical-seconds",
        str(args.pressure_critical_seconds),
        "--recovery-target",
        str(args.pressure_recovery_target),
        "--recovery-seconds",
        str(args.pressure_recovery_seconds),
        "--post-release-seconds",
        str(args.pressure_post_release_seconds),
        "--chunk-mib",
        str(args.pressure_chunk_mib),
        "--reserve-mib",
        str(args.pressure_reserve_mib),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    return (
        subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        ),
        output,
    )


def wait_for_benchmark_response(
    proc: subprocess.Popen,
    server_log: Path,
    start_offset: int,
    timeout: int,
) -> None:
    """Wait until the benchmark has completed its first HTTP warmup request."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"benchmark exited with {proc.returncode} before its first HTTP response")
        with server_log.open("rb") as handle:
            handle.seek(start_offset)
            if b'"POST ' in handle.read():
                return
        time.sleep(0.5)
    raise TimeoutError("benchmark did not produce an HTTP response before pressure trigger timeout")


def parse_gpu_csv(path: Path) -> dict[str, Any]:
    used: list[float] = []
    pressure: list[float] = []
    utilization: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            try:
                used_mib, free_mib, total_mib = map(lambda value: float(value.strip()), row[1:4])
                used.append(used_mib)
                pressure.append(1.0 - free_mib / total_mib)
                utilization.append(float(row[4].strip()))
            except ValueError:
                continue
    if not used:
        return {"samples": 0}
    return {
        "samples": len(used),
        "memory_used_mib": {"mean": statistics.fmean(used), "peak": max(used)},
        "pressure": {"mean": statistics.fmean(pressure), "peak": max(pressure)},
        "gpu_utilization_percent": {"mean": statistics.fmean(utilization), "peak": max(utilization)},
    }


def parse_server_events(path: Path) -> dict[str, Any]:
    contents = path.read_text(errors="replace")
    changes = []
    central_decisions = []
    first_reports = []
    registered_streams = []
    local_coordinator_addresses = LOCAL_COORDINATOR_STARTED_RE.findall(contents)
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if match := CENTRAL_DECISION_RE.search(line):
            central_decisions.append(
                {
                    "stage_id": int(match.group(1)),
                    "replica_id": int(match.group(2)),
                    "cap": int(match.group(3)),
                    "pressure": float(match.group(4)),
                    "reason": match.group(5),
                    "generation": int(match.group(6)),
                    "line_number": line_number,
                }
            )
        if match := FIRST_REPORT_RE.search(line):
            first_reports.append(
                {
                    "stage_id": int(match.group(1)),
                    "replica_id": int(match.group(2)),
                    "rank_count": int(match.group(3)),
                    "pressure": float(match.group(4)),
                    "line_number": line_number,
                }
            )
        if match := REGISTERED_STREAM_RE.search(line):
            registered_streams.append(
                {
                    "stage_id": int(match.group(1)),
                    "replica_id": int(match.group(2)),
                    "devices": match.group(3),
                    "line_number": line_number,
                }
            )
        match = DYNAMIC_EVENT_RE.search(line)
        if match is None:
            continue
        time_match = LOG_TIME_RE.search(line)
        log_second = None
        if time_match:
            log_second = (
                int(time_match.group("hour")) * 3600
                + int(time_match.group("minute")) * 60
                + int(time_match.group("second"))
                + float(f"0.{time_match.group('fraction') or '0'}")
            )
        changes.append(
            {
                "stage_id": int(match.group(1)),
                "replica_id": int(match.group(2)),
                "old_cap": int(match.group(3)),
                "new_cap": int(match.group(4)),
                "pressure": float(match.group(5)),
                "reason": match.group(6),
                "line_number": line_number,
                "log_second_of_day": log_second,
            }
        )
    lowered = contents.lower()
    return {
        "local_coordinator_start_count": len(local_coordinator_addresses),
        "local_coordinator_addresses": local_coordinator_addresses,
        "first_reports": first_reports,
        "first_report_stage_ids": sorted({item["stage_id"] for item in first_reports}),
        "registered_streams": registered_streams,
        "registered_stream_stage_ids": sorted({item["stage_id"] for item in registered_streams}),
        "central_decisions": central_decisions,
        "cap_changes": changes,
        "cap_change_count": len(changes),
        "minimum_observed_cap": min((item["new_cap"] for item in changes), default=None),
        "reasons": {
            reason: sum(item["reason"] == reason for item in changes)
            for reason in sorted({item["reason"] for item in changes})
        },
        "oom_mentions": lowered.count("out of memory") + lowered.count("cuda oom"),
        "preemption_mentions": lowered.count("preempt"),
        "traceback_mentions": lowered.count("traceback"),
        "per_stage": {
            str(stage_id): {
                "cap_change_count": sum(item["stage_id"] == stage_id for item in changes),
                "minimum_observed_cap": min(
                    (item["new_cap"] for item in changes if item["stage_id"] == stage_id),
                    default=None,
                ),
                "shared_device_changes": sum(
                    item["stage_id"] == stage_id and item["reason"].startswith("shared_device_") for item in changes
                ),
                "shared_device_decisions": sum(
                    item["stage_id"] == stage_id and item["reason"].startswith("shared_device_")
                    for item in central_decisions
                ),
            }
            for stage_id in sorted(
                {item["stage_id"] for item in changes} | {item["stage_id"] for item in central_decisions}
            )
        },
    }


def parse_pressure_events(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    reached = [event for event in events if event.get("event") == "target_reached"]
    pressures = [float(event["pressure"]) for event in events if isinstance(event.get("pressure"), (int, float))]
    return {
        "events": events,
        "target_reached_count": len(reached),
        "targets_reached": [float(event["target"]) for event in reached],
        "peak_reported_pressure": max(pressures, default=None),
        "allocation_oom_count": sum(event.get("event") == "allocation_oom" for event in events),
    }


def evaluate_case_acceptance(
    arm: Arm,
    scheduler: dict[str, Any],
    pressure: dict[str, Any],
    expected_stage_ids: set[int],
    expected_pressure_cap: int,
    execution_succeeded: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "case_execution_succeeded": execution_succeeded,
        "local_coordinator_start_matches_arm": scheduler.get("local_coordinator_start_count", 0) == int(arm.dynamic),
    }
    if arm.dynamic:
        checks["all_stages_sent_memory_reports"] = (
            set(scheduler.get("first_report_stage_ids", [])) == expected_stage_ids
        )
        checks["all_stages_registered_centrally"] = (
            set(scheduler.get("registered_stream_stage_ids", [])) == expected_stage_ids
        )
    else:
        checks["no_central_memory_streams"] = not scheduler.get("first_reports") and not scheduler.get(
            "registered_streams"
        )
    if arm.pressure:
        checks["pressure_high_and_critical_targets_reached"] = pressure.get("target_reached_count", 0) >= 2
    if arm.dynamic and arm.pressure:
        per_stage = scheduler.get("per_stage", {})
        checks["all_stages_received_shared_device_decisions"] = all(
            per_stage.get(str(stage_id), {}).get("shared_device_decisions", 0) > 0 for stage_id in expected_stage_ids
        )
        checks["critical_admission_cap_reached"] = (
            scheduler.get("minimum_observed_cap") == expected_pressure_cap
        )
        checks["no_server_oom"] = scheduler.get("oom_mentions", 0) == 0
        checks["no_server_traceback"] = scheduler.get("traceback_mentions", 0) == 0
    return {"passed": all(checks.values()), "checks": checks}


def resolve_metric(payload: dict[str, Any], dotted_path: str) -> float | None:
    value: Any = payload
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return float(value) if isinstance(value, (int, float)) else None


def analyze_summary(summary: dict[str, Any], metric_map: dict[str, str]) -> dict[str, Any]:
    """Build model-independent comparisons from normalized metric paths."""
    completed = [case for case in summary["cases"] if case.get("status") in {"completed", "acceptance_failed"}]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for case in completed:
        by_arm.setdefault(case["arm"]["name"], []).append(case)
    normalized: dict[str, Any] = {}
    for arm, cases in by_arm.items():
        normalized[arm] = {}
        for name, path in metric_map.items():
            values = [resolve_metric(case.get("benchmark", {}), path) for case in cases]
            numeric = [value for value in values if value is not None]
            if numeric:
                normalized[arm][name] = {
                    "median": statistics.median(numeric),
                    "mean": statistics.fmean(numeric),
                    "min": min(numeric),
                    "max": max(numeric),
                }

    def ratio(numerator_arm: str, denominator_arm: str, metric: str) -> float | None:
        numerator = normalized.get(numerator_arm, {}).get(metric, {}).get("median")
        denominator = normalized.get(denominator_arm, {}).get(metric, {}).get("median")
        return numerator / denominator if numerator is not None and denominator else None

    dynamic_pressure_cases = by_arm.get("D_dynamic_on_pressure", [])
    stage_ids = {
        stage_id for case in dynamic_pressure_cases for stage_id in case.get("scheduler", {}).get("per_stage", {})
    }
    shared_stage_ids = {
        stage_id
        for case in dynamic_pressure_cases
        for stage_id, stage in case.get("scheduler", {}).get("per_stage", {}).items()
        if stage.get("shared_device_decisions", 0) > 0
    }
    return {
        "normalized_metrics": normalized,
        "comparisons": {
            "monitoring_throughput_ratio_B_over_A": ratio(
                "B_dynamic_on_no_pressure", "A_dynamic_off_no_pressure", "throughput"
            ),
            "monitoring_p99_latency_ratio_B_over_A": ratio(
                "B_dynamic_on_no_pressure", "A_dynamic_off_no_pressure", "p99_latency_ms"
            ),
            "dynamic_vs_fixed_throughput_ratio_D_over_E": next(
                (
                    ratio("D_dynamic_on_pressure", arm, "throughput")
                    for arm in normalized
                    if arm.startswith("E_fixed_cap_")
                ),
                None,
            ),
            "dynamic_vs_static_pressure_throughput_ratio_D_over_C": ratio(
                "D_dynamic_on_pressure", "C_dynamic_off_pressure", "throughput"
            ),
            "dynamic_vs_static_pressure_p99_latency_ratio_D_over_C": ratio(
                "D_dynamic_on_pressure", "C_dynamic_off_pressure", "p99_latency_ms"
            ),
        },
        "safety": {
            "dynamic_pressure_completed_runs": len(dynamic_pressure_cases),
            "dynamic_pressure_accepted_runs": sum(
                bool(case.get("acceptance", {}).get("passed")) for case in dynamic_pressure_cases
            ),
            "dynamic_pressure_oom_mentions": sum(
                case.get("scheduler", {}).get("oom_mentions", 0) for case in dynamic_pressure_cases
            ),
            "stages_with_cap_events": sorted(stage_ids),
            "stages_with_shared_device_events": sorted(shared_stage_ids),
            "all_observed_stages_received_shared_decisions": bool(stage_ids) and stage_ids == shared_stage_ids,
        },
    }


def run_case(args: argparse.Namespace, arm: Arm, repeat: int) -> dict[str, Any]:
    case_dir = args.output_dir / "cases" / f"repeat_{repeat:02d}_{arm.name}"
    status_path = case_dir / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        return json.loads(status_path.read_text())
    case_dir.mkdir(parents=True, exist_ok=True)
    deploy = prepare_config(args, arm, case_dir)
    deploy_payload = yaml.safe_load(deploy.read_text())
    expected_stage_ids = {
        int(stage["stage_id"])
        for stage in deploy_payload["stages"]
        if args.dynamic_stage_ids is None or int(stage["stage_id"]) in args.dynamic_stage_ids
    }
    server_log, gpu_csv = case_dir / "server.log", case_dir / "gpu.csv"
    pressure_log, client_log = case_dir / "pressure.jsonl", case_dir / "client.log"
    result_json = case_dir / "benchmark.json"
    status: dict[str, Any] = {
        "status": "running",
        "arm": asdict(arm),
        "repeat": repeat,
        "started_at": utc_now(),
        "artifacts": {"deploy": str(deploy), "server_log": str(server_log), "gpu_csv": str(gpu_csv)},
    }
    write_json(status_path, status)
    server = monitor = pressure = client = None
    server_file = monitor_file = pressure_file = None
    try:
        wait_gpu_release(args.device, args.release_threshold_mib)
        server, server_file = start_server(args, deploy, server_log)
        monitor, monitor_file = start_gpu_monitor(args.device, gpu_csv)
        command = benchmark_command(args, result_json)
        status["benchmark_command"] = command
        with client_log.open("w") as output:
            trigger_offset = server_log.stat().st_size
            client = subprocess.Popen(
                command,
                cwd=REPO,
                env={**os.environ, "PYTHONPATH": str(REPO)},
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if arm.pressure:
                if args.pressure_trigger_mode == "first-response":
                    wait_for_benchmark_response(
                        client,
                        server_log,
                        trigger_offset,
                        args.pressure_trigger_timeout,
                    )
                pressure, pressure_file = start_pressure(args, pressure_log)
            try:
                client.wait(timeout=args.benchmark_timeout)
            except subprocess.TimeoutExpired:
                stop_group(client, timeout=5)
                raise
        status["benchmark_exit_code"] = client.returncode
        if pressure is not None:
            try:
                pressure.wait(timeout=args.pressure_wait_timeout)
            except subprocess.TimeoutExpired:
                stop_group(pressure, timeout=5)
        status["status"] = "completed" if client.returncode == 0 and result_json.exists() else "failed"
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop_group(client, timeout=5)
        stop_group(pressure, timeout=5)
        if pressure_file is not None:
            pressure_file.close()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
        if monitor_file is not None:
            monitor_file.close()
        stop_group(server)
        if server_file is not None:
            server_file.close()
    status["finished_at"] = utc_now()
    status["scheduler"] = parse_server_events(server_log) if server_log.exists() else {}
    status["pressure"] = parse_pressure_events(pressure_log) if pressure_log.exists() else {}
    status["gpu"] = parse_gpu_csv(gpu_csv) if gpu_csv.exists() else {}
    if result_json.exists():
        status["benchmark"] = json.loads(result_json.read_text())
    status["artifacts"].update(
        pressure_log=str(pressure_log) if arm.pressure else None,
        client_log=str(client_log),
        benchmark_json=str(result_json),
    )
    status["acceptance"] = evaluate_case_acceptance(
        arm,
        status["scheduler"],
        status["pressure"],
        expected_stage_ids,
        min(args.critical_admission_cap, arm.fixed_cap or args.max_num_seqs),
        status["status"] == "completed",
    )
    if status["status"] == "completed" and not status["acceptance"]["passed"]:
        status["status"] = "acceptance_failed"
    write_json(status_path, status)
    return status


def aggregate(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    cases = [json.loads(path.read_text()) for path in sorted((output_dir / "cases").glob("*/status.json"))]
    arms: dict[str, Any] = {}
    metric_names = ("request_throughput", "p99_e2el_ms", "p99_audio_ttfp_ms", "median_audio_rtf")
    for arm in {case["arm"]["name"] for case in cases}:
        selected = [
            case
            for case in cases
            if case["arm"]["name"] == arm and case["status"] in {"completed", "acceptance_failed"}
        ]
        item: dict[str, Any] = {
            "completed_runs": len(selected),
            "accepted_runs": sum(bool(case.get("acceptance", {}).get("passed")) for case in selected),
        }
        for metric in metric_names:
            values = [case.get("benchmark", {}).get(metric) for case in selected]
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            if numeric:
                item[metric] = {"mean": statistics.fmean(numeric), "min": min(numeric), "max": max(numeric)}
        item["oom_mentions"] = sum(case.get("scheduler", {}).get("oom_mentions", 0) for case in selected)
        item["cap_change_count"] = sum(case.get("scheduler", {}).get("cap_change_count", 0) for case in selected)
        minima = [case.get("scheduler", {}).get("minimum_observed_cap") for case in selected]
        minima = [value for value in minima if isinstance(value, int)]
        item["minimum_observed_cap"] = min(minima) if minima else None
        arms[arm] = item
    summary = {"metadata": metadata, "arms": arms, "cases": cases}
    write_json(output_dir / "summary.json", summary)
    metric_map = metadata.get("metric_map") or {
        "throughput": "request_throughput",
        "p99_latency_ms": "p99_e2el_ms",
        "p99_first_output_ms": "p99_audio_ttfp_ms",
    }
    write_json(output_dir / "analysis.json", analyze_summary(summary, metric_map))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--request-profile", default="qwen3-tts")
    parser.add_argument("--benchmark-command-json", help="JSON argv template for any model/endpoint")
    parser.add_argument("--server-command-json", help="JSON argv template for a custom server command")
    parser.add_argument(
        "--metric-map-json",
        help='JSON map from normalized names to result paths, e.g. {"throughput":"metrics.rps"}',
    )
    parser.add_argument("--dataset-path", type=Path, default=REPO / "benchmarks/build_dataset/seed_tts_smoke")
    parser.add_argument("--locale", default="en")
    parser.add_argument("--extra-body", default='{"voice":"Vivian","language":"English","task_type":"CustomVoice"}')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--dynamic-stage-ids", type=int, nargs="+")
    parser.add_argument("--include-fixed-cap", type=int, metavar="CAP")
    parser.add_argument("--min-num-seqs", type=int, default=2)
    parser.add_argument("--critical-admission-cap", type=int, default=0)
    parser.add_argument("--disconnect-admission-cap", type=int, default=0)
    parser.add_argument("--recovery-complete-samples", type=int, default=3)
    parser.add_argument("--guard-mib", type=int, default=0)
    parser.add_argument("--guard-ratio", type=float, default=0.0)
    parser.add_argument("--immediate-sample-min-interval-ms", type=int, default=100)
    parser.add_argument(
        "--fail-closed-on-disconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    parser.add_argument("--report-timeout-ms", type=int, default=1500)
    parser.add_argument("--missing-report-grace-samples", type=int, default=1)
    parser.add_argument("--low-watermark", type=float, default=0.72)
    parser.add_argument("--high-watermark", type=float, default=0.82)
    parser.add_argument("--critical-watermark", type=float, default=0.90)
    parser.add_argument("--scale-down-ratio", type=float, default=0.5)
    parser.add_argument("--scale-up-step", type=int, default=2)
    parser.add_argument("--scale-up-stable-samples", type=int, default=6)
    parser.add_argument("--pressure-baseline-seconds", type=float, default=15)
    parser.add_argument("--pressure-high-target", type=float, default=0.84)
    parser.add_argument("--pressure-high-seconds", type=float, default=20)
    parser.add_argument("--pressure-critical-target", type=float, default=0.92)
    parser.add_argument("--pressure-critical-seconds", type=float, default=15)
    parser.add_argument("--pressure-recovery-target", type=float, default=0.78)
    parser.add_argument("--pressure-recovery-seconds", type=float, default=20)
    parser.add_argument("--pressure-post-release-seconds", type=float, default=45)
    parser.add_argument(
        "--pressure-trigger-mode",
        choices=("immediate", "first-response"),
        default="immediate",
        help="start the sidecar immediately or after the first benchmark HTTP response",
    )
    parser.add_argument("--pressure-trigger-timeout", type=int, default=600)
    parser.add_argument("--pressure-chunk-mib", type=int, default=128)
    parser.add_argument("--pressure-reserve-mib", type=int, default=1536)
    parser.add_argument("--pressure-wait-timeout", type=int, default=180)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=3600)
    parser.add_argument("--release-threshold-mib", type=int, default=1024)
    parser.add_argument("--safetensors-load-strategy", default="prefetch")
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--arms", nargs="+", choices=[arm.name for arm in CORE_ARMS])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.low_watermark < args.high_watermark < args.critical_watermark < 1:
        parser.error("watermarks must satisfy 0 < low < high < critical < 1")
    if args.min_num_seqs > args.max_num_seqs:
        parser.error("min-num-seqs must not exceed max-num-seqs")
    if not 0 <= args.critical_admission_cap <= args.min_num_seqs:
        parser.error("critical-admission-cap must be in [0, min-num-seqs]")
    if not 0 <= args.disconnect_admission_cap <= args.min_num_seqs:
        parser.error("disconnect-admission-cap must be in [0, min-num-seqs]")
    if args.recovery_complete_samples < 1:
        parser.error("recovery-complete-samples must be at least 1")
    if args.guard_mib < 0:
        parser.error("guard-mib must be non-negative")
    if not 0 <= args.guard_ratio < 1:
        parser.error("guard-ratio must be in [0, 1)")
    if args.immediate_sample_min_interval_ms < 1:
        parser.error("immediate-sample-min-interval-ms must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    if args.summarize_only:
        aggregate(args.output_dir, json.loads(metadata_path.read_text()))
        return 0
    selected = [arm for arm in CORE_ARMS if args.arms is None or arm.name in args.arms]
    if args.include_fixed_cap:
        selected.append(Arm(f"E_fixed_cap_{args.include_fixed_cap}", False, True, args.include_fixed_cap))
    gpu = subprocess.check_output(
        ["nvidia-smi", f"--id={args.device}", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    metadata = {
        "created_at": utc_now(),
        "model": args.model,
        "deploy_config": str(args.deploy_config.resolve()),
        "gpu": gpu,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "arms": [asdict(arm) for arm in selected],
        "metric_map": json.loads(args.metric_map_json) if args.metric_map_json else None,
        "arguments": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "deploy_config": str(args.deploy_config),
            "dataset_path": str(args.dataset_path),
        },
    }
    write_json(metadata_path, metadata)
    # Counterbalance adjacent repeats by reversing arm order.
    failed = False
    for repeat in range(1, args.repeats + 1):
        order = selected if repeat % 2 else list(reversed(selected))
        for arm in order:
            result = run_case(args, arm, repeat)
            failed = failed or result.get("status") != "completed"
            aggregate(args.output_dir, metadata)
    aggregate(args.output_dir, metadata)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
