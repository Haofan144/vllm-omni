#!/usr/bin/env python3
"""Fast H9 danger-cell probe.

Answers one question as cheaply as possible: *does the C arm (dynamic HBM
monitoring OFF) actually crash under Scheme A* — a shrunken stage-0 KV pool plus
a long-audio workload and a wide critical-pressure window?

It is a thin wrapper around ``run_dynamic_hbm_experiment.py``: it builds the
base-runner argv for ONE case, shells out, then reads back the
``cases/repeat_01_<arm>/status.json`` and ``server.log`` the base runner writes
and classifies the outcome. It does not re-implement server launch, the pressure
sidecar, or artifact parsing.

Default run: one ``C_dynamic_off_pressure`` case at the most aggressive single
point (concurrency 48, ``gmu0=0.11``, 45 s critical window) — ~10-13 min on an
A6000. With ``--bisect`` it will, if that first probe does not crash, halve
``gmu0`` toward a smaller pool (max 3 steps); if the first probe DOES crash it
instead runs one D-arm confirmation at the same ``gmu0``.

Outputs under ``benchmarks/results/H9_probe_<UTC stamp>/``:
  * ``run_<nn>_<arm>_gmu<g>/``  — the base runner's own output dir (full evidence)
  * ``probe_result.json``       — one record per probe step
  * ``probe_result.md``         — table + one-line verdict + recommended next cmd
  * ``probe.log``               — stdout/stderr of each base-runner invocation
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_RUNNER = REPO / "benchmarks/hbm_admission/run_dynamic_hbm_experiment.py"
# Prefer the repo venv interpreter for the base-runner subprocess (it has vllm
# installed); fall back to whatever is running this script.
_VENV_PY = REPO / ".venv/bin/python"
PYEXE = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_long"

KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
KV_GIB_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
MAX_CONC_RE = re.compile(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x")
DESIRED_GMU_RE = re.compile(r"Desired GPU memory utilization is \(([\d.]+),")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stage_overrides(gmu0: float, gmu1: float, hbm_limit_gb: float | None = None) -> str:
    """Stage-0 KV-pool shrink override.

    ``hbm_limit_gb`` (when set) is a HARD KV byte cap independent of the
    boot-time gpu_memory_utilization profiling floor; it is paired with
    ``hbm_admission_guard: false`` so the static deferral/preempt guard does not
    protect the C arm (it auto-enables whenever hbm_limit_gb is set).
    """
    stage0: dict[str, Any] = {"gpu_memory_utilization": gmu0}
    if hbm_limit_gb is not None:
        stage0["hbm_limit_gb"] = hbm_limit_gb
        stage0["hbm_admission_guard"] = False
    return json.dumps(
        {"0": stage0, "1": {"gpu_memory_utilization": gmu1}},
        separators=(",", ":"),
    )


def base_runner_argv(args: argparse.Namespace, *, arm: str, gmu0: float, gmu1: float, out_dir: Path,
                     hbm_limit_gb: float | None = None) -> list[str]:
    """Assemble the single-case argv for run_dynamic_hbm_experiment.py."""
    return [
        PYEXE, str(BASE_RUNNER),
        "--output-dir", str(out_dir),
        "--model", str(args.model),
        "--deploy-config", str(args.deploy_config),
        "--dataset-path", str(args.dataset_path),
        "--device", str(args.device),
        "--port", str(args.port),
        "--arms", arm,
        "--repeats", "1",
        "--concurrency", str(args.concurrency),
        "--max-num-seqs", str(args.concurrency),
        "--num-prompts", str(args.num_prompts),
        "--num-warmups", "4",
        # dynamic_hbm block knobs (match the effectiveness runner's defaults so a
        # later formal run is comparable). Harmless for the C arm (enabled=False).
        "--min-num-seqs", "2",
        "--critical-admission-cap", "0",
        "--disconnect-admission-cap", "0",
        "--guard-mib", "0",
        "--low-watermark", "0.72",
        "--high-watermark", "0.82",
        "--critical-watermark", "0.90",
        "--scale-down-ratio", "0.5",
        "--scale-up-step", "2",
        "--scale-up-stable-samples", "6",
        "--recovery-complete-samples", "3",
        # pressure profile
        "--pressure-trigger-mode", "first-response",
        "--pressure-baseline-seconds", "8",
        "--pressure-high-target", "0.84",
        "--pressure-high-seconds", "10",
        "--pressure-critical-target", str(args.pressure_critical_target),
        "--pressure-critical-seconds", str(args.pressure_critical_seconds),
        "--pressure-recovery-target", "0.76",
        "--pressure-recovery-seconds", "15",
        "--pressure-post-release-seconds", "20",
        "--pressure-chunk-mib", "64",
        "--pressure-reserve-mib", str(args.pressure_reserve_mib),
        "--startup-timeout", str(args.startup_timeout),
        "--benchmark-timeout", str(args.benchmark_timeout),
        # shrink the talker KV pool; `=` form because argparse rejects a
        # flag-like value (--stage-overrides) supplied as a separate token.
        "--server-extra-arg=--stage-overrides",
        f"--server-extra-arg={stage_overrides(gmu0, gmu1, hbm_limit_gb)}",
    ]


def _server_log_facts(server_log: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "kv_cache_tokens": None,
        "kv_cache_gib": None,
        "max_concurrency_x": None,
        "effective_gmu0": None,
        "oom_lines": 0,
        "cuda_oom_lines": 0,
        "preempt_lines": 0,
        "kv_full_lines": 0,
    }
    if not server_log.is_file():
        return facts
    text = server_log.read_text(errors="replace")
    if m := KV_TOKENS_RE.search(text):
        facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := KV_GIB_RE.search(text):
        facts["kv_cache_gib"] = float(m.group(1))
    if m := MAX_CONC_RE.search(text):
        facts["max_concurrency_x"] = float(m.group(1))
    if m := DESIRED_GMU_RE.search(text):
        facts["effective_gmu0"] = float(m.group(1))
    low = text.lower()
    facts["oom_lines"] = low.count("out of memory")
    facts["cuda_oom_lines"] = low.count("cuda out of memory")
    facts["preempt_lines"] = low.count("preempt")
    facts["kv_full_lines"] = low.count("no available kv") + low.count("cannot allocate") + low.count(
        "kv cache is full"
    )
    # A boot-time KV-cache-config failure ("No available memory for the cache
    # blocks") means the server never started — NOT a runtime overflow under
    # load. It gives no C-vs-D signal because the D arm would fail identically.
    facts["boot_kv_config_failure"] = (
        "no available memory for the cache blocks" in low
        or "_check_enough_kv_cache_memory" in text
    )
    return facts


def _read_case(out_dir: Path, arm: str) -> dict[str, Any]:
    """Load status.json + server.log facts for the single case the runner wrote."""
    case_dir = out_dir / "cases" / f"repeat_01_{arm}"
    status_path = case_dir / "status.json"
    if not status_path.is_file():
        return {"status": "missing", "error": f"no status.json under {case_dir}"}
    status = json.loads(status_path.read_text())
    scheduler = status.get("scheduler", {})
    pressure = status.get("pressure", {})
    benchmark = status.get("benchmark", {})
    gpu = status.get("gpu", {})
    server_log = Path(status.get("artifacts", {}).get("server_log", case_dir / "server.log"))
    facts = _server_log_facts(server_log)

    oom_mentions = int(scheduler.get("oom_mentions", 0))
    case_status = status.get("status")
    boot_fail = bool(facts.get("boot_kv_config_failure"))
    # "crashed" mirrors run_dynamic_hbm_effectiveness.parse_case, but a
    # boot-time KV-config failure is excluded — it is a mis-sized config, not a
    # runtime danger point.
    crashed = (case_status not in {"completed", "acceptance_failed"} or oom_mentions > 0) and not boot_fail

    return {
        "case_dir": str(case_dir),
        "server_log": str(server_log),
        "status": case_status,
        "crashed": bool(crashed),
        "oom_mentions": oom_mentions,
        "preemption_mentions": int(scheduler.get("preemption_mentions", 0)),
        "traceback_mentions": int(scheduler.get("traceback_mentions", 0)),
        "minimum_observed_cap": scheduler.get("minimum_observed_cap"),
        "cap_change_count": scheduler.get("cap_change_count"),
        "completed": benchmark.get("completed"),
        "failed": benchmark.get("failed"),
        "request_throughput": benchmark.get("request_throughput"),
        "p99_e2el_ms": benchmark.get("p99_e2el_ms"),
        "allocation_oom_count": int(pressure.get("allocation_oom_count", 0)),
        "peak_reported_pressure": pressure.get("peak_reported_pressure"),
        "targets_reached": pressure.get("targets_reached"),
        "gpu_pressure_peak": (gpu.get("pressure") or {}).get("peak"),
        "gpu_mem_used_peak_mib": (gpu.get("memory_used_mib") or {}).get("peak"),
        "acceptance_passed": (status.get("acceptance") or {}).get("passed"),
        "boot_kv_config_failure": boot_fail,
        **facts,
    }


def classify(rec: dict[str, Any], arm: str, gmu0: float) -> tuple[str, str]:
    """Return (classification, recommended-next-command)."""
    if rec.get("status") == "missing":
        return "run-error", "inspect probe.log — the base runner did not produce a status.json"

    if rec.get("boot_kv_config_failure"):
        return (
            "boot-failure-too-small",
            f"gmu0={gmu0:.3f} is below the boot-time KV-config floor — the server never "
            f"started ('No available memory for the cache blocks'), so there is no C-vs-D "
            f"signal. gpu_memory_utilization alone cannot go this low. Switch to a hard KV "
            f"byte cap with the static guard disabled:\n"
            f"    python {HERE}/probe_h9_danger.py --hbm-limit-gb 0.8 "
            f"--concurrency {rec.get('_concurrency', 56)} --bisect",
        )

    if rec["allocation_oom_count"] > 0:
        return (
            "sidecar-oom-invalid",
            f"raise --pressure-reserve-mib to 512 (or --gmu0 {gmu0 + 0.02:.2f}) and re-probe — "
            "a sidecar OOM does not demonstrate admission-control effectiveness",
        )

    if arm == "D_dynamic_on_pressure":
        paused = rec["preemption_mentions"] >= 0  # placeholder; real check below on server log
        # D-arm success = survived AND showed the pause/resume mechanism.
        log_text = ""
        try:
            log_text = Path(rec["server_log"]).read_text(errors="replace")
        except OSError:
            pass
        has_pause = "[HBMAdmission] paused" in log_text
        has_resume = "[HBMAdmission] resumed" in log_text
        if not rec["crashed"] and has_pause and has_resume:
            return (
                "D-survives-with-mechanism",
                f"freeze --gmu0 {gmu0:.2f}; run the 20-repeat formal:\n"
                f"    python {HERE}/H9_oom_success_effectiveness.py run "
                f"--output-dir benchmarks/results/H9_formal_v2 "
                f"--dataset-path {DEFAULT_DATASET} --critical-target {rec.get('_critical_target', 0.90)} "
                f"--concurrency {rec.get('_concurrency', 48)} --max-num-seqs {rec.get('_concurrency', 48)} "
                f"--pressure-critical-seconds {rec.get('_critical_seconds', 45)} "
                f"--pressure-reserve-mib {rec.get('_reserve_mib', 256)} --num-prompts 320 --repeats 20 "
                f"--fixed-cap 4 --server-extra-arg=--stage-overrides "
                f"'--server-extra-arg={stage_overrides(gmu0, rec.get('_gmu1', 0.10))}'",
            )
        if rec["crashed"]:
            return (
                "D-also-crashes",
                f"pool too small for the controller to keep min_num_seqs alive — "
                f"raise --gmu0 to {gmu0 + 0.02:.2f} and re-probe C then D",
            )
        return (
            "D-inconclusive",
            "D survived but pause/resume markers not both found — inspect server.log",
        )

    # C arm
    if rec["crashed"]:
        return (
            "danger-point-found",
            f"freeze --gmu0 {gmu0:.2f}; confirm with a 3-repeat C-only run, then a D run:\n"
            f"    python {HERE}/probe_h9_danger.py --gmu0 {gmu0:.2f} --repeats 3\n"
            f"    python {HERE}/probe_h9_danger.py --gmu0 {gmu0:.2f} --arm D_dynamic_on_pressure",
        )

    mcx = rec.get("max_concurrency_x")
    kv_gib = rec.get("kv_cache_gib")
    if (mcx is not None and mcx <= 2.5) or rec["preemption_mentions"] > 0:
        return (
            "on-the-edge",
            f"C survived but KV headroom is thin (max_concurrency={mcx}x, "
            f"preemptions={rec['preemption_mentions']}) — drop --gmu0 to {max(gmu0 - 0.02, 0.05):.2f} and re-probe",
        )
    return (
        "pool-still-too-big",
        f"C survived comfortably (max_concurrency={mcx}x, KV pool={kv_gib} GiB) — "
        f"drop --gmu0 to {max(gmu0 - 0.03, 0.05):.2f} (or --concurrency 56) and re-probe",
    )


def run_one(args: argparse.Namespace, *, arm: str, gmu0: float, gmu1: float, step: int,
            root: Path, log_fh, hbm_limit_gb: float | None = None) -> dict[str, Any]:
    tag = f"hbm{hbm_limit_gb:.2f}" if hbm_limit_gb is not None else f"gmu{gmu0:.3f}"
    out_dir = root / f"run_{step:02d}_{arm}_{tag}"
    argv = base_runner_argv(args, arm=arm, gmu0=gmu0, gmu1=gmu1, out_dir=out_dir, hbm_limit_gb=hbm_limit_gb)
    log_fh.write(f"\n{'=' * 80}\nstep {step}  arm={arm}  gmu0={gmu0}  gmu1={gmu1}  hbm_limit_gb={hbm_limit_gb}\n")
    log_fh.write("CMD: " + " ".join(argv) + "\n")
    log_fh.flush()
    t0 = time.monotonic()
    proc = subprocess.run(argv, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    duration = time.monotonic() - t0
    rec = _read_case(out_dir, arm)
    rec.update(
        step=step, arm=arm, gmu0=gmu0, gmu1=gmu1, hbm_limit_gb=hbm_limit_gb,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts, pressure_critical_target=args.pressure_critical_target,
        pressure_critical_seconds=args.pressure_critical_seconds,
        pressure_reserve_mib=args.pressure_reserve_mib,
        base_runner_exit_code=proc.returncode, duration_s=round(duration, 1),
        out_dir=str(out_dir),
        # echoes used by classify() to build the next command
        _critical_target=args.pressure_critical_target, _concurrency=args.concurrency,
        _critical_seconds=args.pressure_critical_seconds, _reserve_mib=args.pressure_reserve_mib,
        _gmu1=gmu1,
    )
    cls, nxt = classify(rec, arm, gmu0)
    rec["classification"] = cls
    rec["recommended_next"] = nxt
    return rec


def write_report(root: Path, records: list[dict[str, Any]]) -> None:
    write_json(root / "probe_result.json", records)
    lines: list[str] = ["# H9 danger-cell probe", ""]
    last = records[-1]
    knob = (f"hbm_limit_gb={last['hbm_limit_gb']}" if last.get("hbm_limit_gb") is not None
            else f"gmu0={last['gmu0']}")
    lines += [
        f"**Verdict:** `{last['classification']}` (step {last['step']}, arm {last['arm']}, "
        f"{knob}, concurrency={last['concurrency']})",
        "",
        f"**Recommended next:**",
        "```",
        last["recommended_next"],
        "```",
        "",
        "| step | arm | gmu0 | hbm_gb | conc | crashed | boot_fail | classification | KV tok | KV GiB | maxconc | oom | preempt | sidecar_oom | done/fail | thrpt | peak_press | status | dur s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['step']} | {r['arm'].split('_')[0]} | {r['gmu0']} | {r.get('hbm_limit_gb')} | {r['concurrency']} | "
            f"{'YES' if r.get('crashed') else 'no'} | {'YES' if r.get('boot_kv_config_failure') else 'no'} | "
            f"{r.get('classification', '-')} | "
            f"{r.get('kv_cache_tokens')} | {r.get('kv_cache_gib')} | {r.get('max_concurrency_x')} | "
            f"{r.get('oom_mentions')} | {r.get('preemption_mentions')} | {r.get('allocation_oom_count')} | "
            f"{r.get('completed')}/{r.get('failed')} | "
            f"{round(r['request_throughput'], 3) if isinstance(r.get('request_throughput'), (int, float)) else r.get('request_throughput')} | "
            f"{round(r['peak_reported_pressure'], 4) if isinstance(r.get('peak_reported_pressure'), (int, float)) else r.get('peak_reported_pressure')} | "
            f"{r.get('status')} | {r.get('duration_s')} |"
        )
    lines += ["", f"Full per-step evidence: `{root}/run_*/cases/`.", ""]
    (root / "probe_result.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmu0", type=float, default=0.11, help="stage-0 (talker) gpu_memory_utilization override")
    p.add_argument("--gmu1", type=float, default=0.10, help="stage-1 (code2wav) gpu_memory_utilization override")
    p.add_argument("--hbm-limit-gb", type=float, default=None,
                   help="stage-0 HARD KV byte cap (GiB) + hbm_admission_guard:false. Use when "
                        "gpu_memory_utilization alone hits the boot-time KV-config floor. When set, "
                        "--gmu0 is still applied (keep it modest, e.g. 0.20, so the server boots) "
                        "and the hard cap is what actually bites. --bisect halves this value.")
    p.add_argument("--concurrency", type=int, default=48, help="request concurrency == max_num_seqs")
    p.add_argument("--num-prompts", type=int, default=160)
    p.add_argument("--arm", default="C_dynamic_off_pressure",
                   choices=["C_dynamic_off_pressure", "D_dynamic_on_pressure"])
    p.add_argument("--repeats", type=int, default=1,
                   help="repeats for the base runner (1 = probe; 3 = quick confirm)")
    p.add_argument("--pressure-critical-target", type=float, default=0.90)
    p.add_argument("--pressure-critical-seconds", type=float, default=45.0)
    p.add_argument("--pressure-reserve-mib", type=int, default=256)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--benchmark-timeout", type=int, default=2400)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--bisect", action="store_true",
                   help="if the first C probe does not crash, halve gmu0 toward a smaller pool "
                        "(max 3 steps); if it DOES crash, run one D-arm confirmation instead")
    p.add_argument("-n", "--print-cmd", action="store_true",
                   help="print the base-runner argv for the first step and exit (no launch)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.repeats != 1:
        # The base runner writes cases/repeat_01.., ..repeat_NN..; _read_case only
        # inspects repeat_01. A >1 repeats run still produces repeat_01, so the
        # probe classification reflects the first repeat; the full evidence for
        # all repeats is under the run dir.
        pass

    root = args.output_dir or (REPO / f"benchmarks/results/H9_probe_{utc_stamp()}")
    root.mkdir(parents=True, exist_ok=True)

    if args.print_cmd:
        argv = base_runner_argv(args, arm=args.arm, gmu0=args.gmu0, gmu1=args.gmu1,
                                out_dir=root / "run_00_print", hbm_limit_gb=args.hbm_limit_gb)
        print(" ".join(argv))
        return 0

    for tool in ("model", "deploy_config", "dataset_path"):
        path = getattr(args, tool)
        if not Path(path).exists():
            print(f"ERROR: --{tool.replace('_', '-')} not found: {path}", file=sys.stderr)
            return 2
    if not (Path(args.dataset_path) / "en" / "meta.lst").is_file():
        print(f"ERROR: dataset has no en/meta.lst: {args.dataset_path}", file=sys.stderr)
        return 2

    hbm = args.hbm_limit_gb  # None => gpu_memory_utilization mode; float => hard-cap mode

    def _step(step: int, *, arm: str, gmu0: float, hbm_limit_gb: float | None) -> dict[str, Any]:
        rec = run_one(args, arm=arm, gmu0=gmu0, gmu1=args.gmu1, step=step, root=root,
                      log_fh=log_fh, hbm_limit_gb=hbm_limit_gb)
        records.append(rec)
        write_report(root, records)
        knob = f"hbm_limit_gb={hbm_limit_gb}" if hbm_limit_gb is not None else f"gmu0={gmu0}"
        print(f"[step {step}] {arm.split('_')[0]} {knob} -> crashed={rec['crashed']} "
              f"boot_fail={rec.get('boot_kv_config_failure')} classification={rec['classification']}  "
              f"({rec['duration_s']}s)")
        return rec

    records: list[dict[str, Any]] = []
    log_path = root / "probe.log"
    with log_path.open("w") as log_fh:
        # step 0 — the primary probe
        rec = _step(0, arm=args.arm, gmu0=args.gmu0, hbm_limit_gb=hbm)

        if args.bisect and args.arm == "C_dynamic_off_pressure":
            crashed_ok = rec["crashed"] and rec["allocation_oom_count"] == 0

            if crashed_ok:
                # C already crashes at this point — confirm the D arm survives it.
                _step(1, arm="D_dynamic_on_pressure", gmu0=args.gmu0, hbm_limit_gb=hbm)

            elif hbm is not None:
                # hard-cap mode: halve hbm_limit_gb toward a smaller pool.
                val = hbm
                for step in range(1, 4):
                    val = round(val / 2, 3)
                    rec = _step(step, arm="C_dynamic_off_pressure", gmu0=args.gmu0, hbm_limit_gb=val)
                    if rec["crashed"] and rec["allocation_oom_count"] == 0:
                        _step(step + 1, arm="D_dynamic_on_pressure", gmu0=args.gmu0, hbm_limit_gb=val)
                        break
                    if rec.get("boot_kv_config_failure") or val <= 0.1:
                        break

            else:
                # gpu_memory_utilization mode: step gmu0 down. Stop on a crash OR
                # on a boot-config failure (floor hit — switch to hard-cap mode).
                gmu = args.gmu0
                for step in range(1, 4):
                    gmu = round(gmu - 0.025, 3)
                    rec = _step(step, arm="C_dynamic_off_pressure", gmu0=gmu, hbm_limit_gb=None)
                    if rec["crashed"] and rec["allocation_oom_count"] == 0:
                        _step(step + 1, arm="D_dynamic_on_pressure", gmu0=gmu, hbm_limit_gb=None)
                        break
                    if rec.get("boot_kv_config_failure") or gmu <= 0.04:
                        break

    write_report(root, records)
    print(f"\nreport: {root}/probe_result.md")
    print(f"json:   {root}/probe_result.json")
    print(f"verdict: {records[-1]['classification']}")
    print(f"next:\n{records[-1]['recommended_next']}")
    # exit 0 only if BOTH: a C step crashed under load (not a boot failure, no
    # sidecar OOM) AND — if a D step ran — D survived with the pause/resume
    # mechanism visible.
    c_danger = any(
        r["arm"] == "C_dynamic_off_pressure" and r.get("crashed")
        and not r.get("boot_kv_config_failure") and r.get("allocation_oom_count", 0) == 0
        for r in records
    )
    d_steps = [r for r in records if r["arm"] == "D_dynamic_on_pressure"]
    d_ok = (not d_steps) or any(
        r["classification"] == "D-survives-with-mechanism" for r in d_steps
    )
    return 0 if (c_danger and d_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
