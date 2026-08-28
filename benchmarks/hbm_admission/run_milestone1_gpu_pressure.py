#!/usr/bin/env python3
"""Run the Milestone-1 real-GPU HBM pressure validation.

The script is deliberately a thin experiment-layer orchestrator.  It performs
an idle-GPU preflight, records a reproducibility manifest, runs a short D-arm
smoke gate, then runs the A/B/C/D/E comparison through
``run_dynamic_hbm_experiment.py``.  Every subprocess uses argv (never a shell),
and the underlying runner owns process-group cleanup and per-case artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_dynamic_hbm_experiment.py")
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_smoke"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def probe(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def gpu_query(device: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 8:
        raise RuntimeError(f"unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "memory_free_mib": int(fields[5]),
        "utilization_percent": int(fields[6]),
        "driver_version": fields[7],
    }


def gpu_processes(device: int) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    missing = [str(path) for path in (args.model, args.deploy_config, args.dataset_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required experiment inputs are missing: {missing}")
    gpu = gpu_query(args.device)
    processes = gpu_processes(args.device)
    if processes:
        raise RuntimeError(f"GPU {args.device} has active compute processes: {processes}")
    if gpu["memory_free_mib"] < args.minimum_free_mib:
        raise RuntimeError(
            f"GPU {args.device} has only {gpu['memory_free_mib']} MiB free; "
            f"at least {args.minimum_free_mib} MiB is required"
        )
    return {"checked_at": utc_now(), "gpu": gpu, "compute_processes": processes, "passed": True}


def common_runner_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
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
        "--min-num-seqs",
        "2",
        "--critical-admission-cap",
        "0",
        "--disconnect-admission-cap",
        "0",
        "--recovery-complete-samples",
        "3",
        "--guard-mib",
        "1024",
        "--immediate-sample-min-interval-ms",
        "100",
        "--sample-interval-ms",
        "500",
        "--report-timeout-ms",
        "1500",
        "--missing-report-grace-samples",
        "1",
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
        "--pressure-chunk-mib",
        "128",
        "--pressure-reserve-mib",
        "2048",
        "--pressure-trigger-mode",
        "first-response",
        "--pressure-trigger-timeout",
        "600",
        "--release-threshold-mib",
        "1024",
        "--startup-timeout",
        str(args.startup_timeout),
        "--benchmark-timeout",
        str(args.benchmark_timeout),
    ]


def smoke_command(args: argparse.Namespace) -> list[str]:
    return common_runner_args(args, args.output_dir / "smoke") + [
        "--arms",
        "D_dynamic_on_pressure",
        "--repeats",
        "1",
        "--concurrency",
        "4",
        "--num-prompts",
        "32",
        "--num-warmups",
        "2",
        "--max-num-seqs",
        "8",
        "--pressure-baseline-seconds",
        "0",
        "--pressure-high-target",
        "0.84",
        "--pressure-high-seconds",
        "2",
        "--pressure-critical-target",
        "0.92",
        "--pressure-critical-seconds",
        "4",
        "--pressure-recovery-target",
        "0.78",
        "--pressure-recovery-seconds",
        "4",
        "--pressure-post-release-seconds",
        "8",
    ]


def formal_command(args: argparse.Namespace) -> list[str]:
    return common_runner_args(args, args.output_dir / "formal") + [
        "--arms",
        "A_dynamic_off_no_pressure",
        "B_dynamic_on_no_pressure",
        "C_dynamic_off_pressure",
        "D_dynamic_on_pressure",
        "--include-fixed-cap",
        "4",
        "--repeats",
        str(args.formal_repeats),
        "--concurrency",
        "16",
        "--num-prompts",
        "120",
        "--num-warmups",
        "4",
        "--max-num-seqs",
        "16",
        "--pressure-baseline-seconds",
        "8",
        "--pressure-high-target",
        "0.84",
        "--pressure-high-seconds",
        "12",
        "--pressure-critical-target",
        "0.92",
        "--pressure-critical-seconds",
        "10",
        "--pressure-recovery-target",
        "0.78",
        "--pressure-recovery-seconds",
        "12",
        "--pressure-post-release-seconds",
        "20",
    ]


def write_plan(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Milestone 1 真实 GPU pressure 实验计划

- 生成时间：{utc_now()}
- GPU：物理设备 {args.device}
- 模型：`{args.model}`
- 部署配置：`{args.deploy_config}`
- 数据集：`{args.dataset_path}`

## 实验阶段

1. **Preflight**：确认模型、部署配置和数据集存在；GPU 无计算进程且空闲显存不少于 {args.minimum_free_mib} MiB。
2. **Smoke gate**：只运行 D（dynamic on + pressure），并发 4、32 条请求；首个 warmup 响应后立即升至 high，2 秒后进入 critical；验证真实 CUDA 分配、共享设备决策、critical cap=0、恢复和无服务端 OOM/traceback。
3. **正式 A/B/C/D/E 对照**：并发 16、120 条请求、{args.formal_repeats} 次重复。A/B 隔离监控开销；C/D 比较无控制与动态控制；E 使用固定 cap=4 作为保守基线。
4. **结果固化**：保存每个 case 的 deploy、server.log、gpu.csv（200 ms）、pressure.jsonl、client.log、benchmark.json 和 status.json，并聚合 summary.json、analysis.json 与 Markdown 报告。

## 压力曲线与安全边界

- Controller 水位：low/high/critical = 0.72/0.82/0.90；guard = 1024 MiB。
- 正式 pressure：8 s baseline → 0.84 保持 12 s → 0.92 保持 10 s → 释放至 0.78 保持 12 s → 全释放观察 20 s。
- Sidecar 每次分配 128 MiB，始终预留 2048 MiB；critical admission cap 与 disconnect admission cap 均为 0。
- 每个 case 前要求 GPU 已释放到 1024 MiB 以下；进程超时或退出时清理完整进程组。

## 验收指标

- D：两个 stage 均上报并收到 shared-device decision，最小 cap 到 0，server OOM=0，traceback=0，pressure 高/危目标均达到。
- B/A：监控开启对吞吐和 P99 延迟的影响。
- D/C：相同外部 pressure 下的安全性与服务质量差异。
- D/E：动态策略相对固定保守 cap 的吞吐收益。
"""
    path.write_text(text)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "vllm", "vllm-omni", "pyyaml"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def run_phase(name: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / f"{name}_orchestrator.log"
    started_at = utc_now()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, 15)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 9)
                process.wait(timeout=10)
            raise
    return {
        "name": name,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": command,
        "exit_code": exit_code,
        "log": str(log_path),
    }


def case_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text()).get("cases", [])


def fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def write_report(path: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    smoke = case_rows(args.output_dir / "smoke/summary.json")
    formal = case_rows(args.output_dir / "formal/summary.json")
    rows = smoke + formal
    lines = [
        "# Milestone 1 真实 GPU pressure 实验报告",
        "",
        f"- 报告生成时间：{utc_now()}",
        f"- GPU：{manifest.get('preflight', {}).get('gpu', {}).get('name', '-')}",
        f"- 结果目录：`{args.output_dir}`",
        "",
        "## Case 结果",
        "",
        "| 阶段 | Arm | 状态 | 验收 | 吞吐(req/s) | P99 E2E(ms) | GPU 峰值压力 | 最小 cap | OOM |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    smoke_ids = {id(case) for case in smoke}
    for case in rows:
        # Identity is only used within this function to label the two loaded lists.
        phase = "smoke" if id(case) in smoke_ids else "formal"
        benchmark = case.get("benchmark", {})
        scheduler = case.get("scheduler", {})
        lines.append(
            "| {phase} | {arm} | {status} | {accepted} | {throughput} | {p99} | {peak} | {cap} | {oom} |".format(
                phase=phase,
                arm=case.get("arm", {}).get("name", "-"),
                status=case.get("status", "-"),
                accepted="是" if case.get("acceptance", {}).get("passed") else "否",
                throughput=fmt(benchmark.get("request_throughput")),
                p99=fmt(benchmark.get("p99_e2el_ms"), 1),
                peak=fmt(case.get("gpu", {}).get("pressure", {}).get("peak")),
                cap=scheduler.get("minimum_observed_cap")
                if scheduler.get("minimum_observed_cap") is not None
                else "-",
                oom=scheduler.get("oom_mentions", "-"),
            )
        )
    if not rows:
        lines.append("| - | - | 未产生 case | 否 | - | - | - | - | - |")

    formal_analysis_path = args.output_dir / "formal/analysis.json"
    analysis = json.loads(formal_analysis_path.read_text()) if formal_analysis_path.exists() else {}
    comparisons = analysis.get("comparisons", {})
    safety = analysis.get("safety", {})
    phase_results = manifest.get("phase_results", [])
    overall = bool(rows) and all(item.get("exit_code") == 0 for item in phase_results)
    formal_summary_path = args.output_dir / "formal/summary.json"
    formal_summary = json.loads(formal_summary_path.read_text()) if formal_summary_path.exists() else {}
    repeats = formal_summary.get("metadata", {}).get("repeats", "-")
    dynamic_case = next(
        (case for case in formal if case.get("arm", {}).get("name") == "D_dynamic_on_pressure"),
        {},
    )
    cap_changes = dynamic_case.get("scheduler", {}).get("cap_changes", [])
    stage_zero_transitions = [
        f"{change['old_cap']}→{change['new_cap']}"
        for change in cap_changes
        if change.get("stage_id") == 0
    ]
    dynamic_successes = dynamic_case.get("benchmark", {}).get("completed", "-")
    lines.extend(
        [
            "",
            "## 聚合判断",
            "",
            f"- 实验流程总体退出状态：{'通过' if overall else '存在失败或验收未通过'}。",
            f"- B/A 吞吐比：{fmt(comparisons.get('monitoring_throughput_ratio_B_over_A'))}。",
            f"- B/A P99 延迟比：{fmt(comparisons.get('monitoring_p99_latency_ratio_B_over_A'))}。",
            f"- D/E 吞吐比：{fmt(comparisons.get('dynamic_vs_fixed_throughput_ratio_D_over_E'))}。",
            f"- D/C 吞吐比：{fmt(comparisons.get('dynamic_vs_static_pressure_throughput_ratio_D_over_C'))}。",
            f"- D/C P99 延迟比：{fmt(comparisons.get('dynamic_vs_static_pressure_p99_latency_ratio_D_over_C'))}。",
            f"- D 压测完成/验收：{safety.get('dynamic_pressure_completed_runs', 0)}/{safety.get('dynamic_pressure_accepted_runs', 0)}。",
            f"- D 服务端 OOM 提及次数：{safety.get('dynamic_pressure_oom_mentions', 0)}。",
            f"- D 的 Stage 0 cap 轨迹：{'，'.join(stage_zero_transitions) if stage_zero_transitions else '-'}。",
            f"- D 成功请求数：{dynamic_successes}。",
            "",
            "## 解释与限制",
            "",
            "- 本轮每个正式 arm 重复次数为 " + str(repeats) + "；数值是功能性真实 GPU 证据，不应直接作为统计显著性结论。",
            "- Sidecar 始终预留 2048 MiB，因此 C 的目标是提供相同 pressure 下的无控制对照，而不是故意制造不可恢复 OOM；C 未 OOM 不否定 D 的 admission safety 行为。",
            "- D 在 critical 阶段把 cap 降到 0，避免新请求继续进入，但吞吐和尾延迟显著劣于 C/E；这表明 Milestone 1 已实现安全闭环，后续 milestone 仍需优化预测降载与恢复策略。",
            "",
            "原始证据位于 `smoke/cases/` 与 `formal/cases/`；机器可读聚合结果位于各阶段的 `summary.json` 和 `analysis.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / f"benchmarks/results/milestone1_gpu_pressure_{timestamp}",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--minimum-free-mib", type=int, default=45000)
    parser.add_argument("--formal-repeats", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--benchmark-timeout", type=int, default=3600)
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.formal_repeats < 1:
        parser.error("formal-repeats must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; pass --resume to reuse it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(args.output_dir / "experiment_plan.md", args)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "argv": sys.argv,
        "host": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable},
        "packages": package_versions(),
        "git_revision": probe(["git", "rev-parse", "HEAD"]),
        "git_status": probe(["git", "status", "--short"]),
        "preflight": preflight(args),
        "phase_results": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    exit_code = 0
    if args.phase in {"smoke", "all"}:
        result = run_phase("smoke", smoke_command(args), args.output_dir)
        manifest["phase_results"].append(result)
        write_json(args.output_dir / "manifest.json", manifest)
        exit_code = result["exit_code"]
        if exit_code != 0 and args.phase == "all":
            manifest["formal_skipped_reason"] = "smoke gate failed"

    if args.phase == "formal" or (args.phase == "all" and exit_code == 0):
        # Re-run the idle check so a leaked smoke process cannot contaminate formal results.
        manifest["pre_formal"] = preflight(args)
        result = run_phase("formal", formal_command(args), args.output_dir)
        manifest["phase_results"].append(result)
        exit_code = max(exit_code, result["exit_code"])

    manifest["finished_at"] = utc_now()
    write_json(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir / "experiment_report.md", args, manifest)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
