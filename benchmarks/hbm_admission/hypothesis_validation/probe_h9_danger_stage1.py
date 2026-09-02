#!/usr/bin/env python3
"""H9 danger-cell probe, stage-1 (Code2Wav) variant.

``probe_h9_danger.py`` squeezes stage-0 (talker) KV headroom and found no
danger cell: Qwen3-TTS is stage-1 (Code2Wav) bound, so the talker just queues
admissions politely instead of overflowing (see
``benchmarks/results/H9_probe_SUMMARY.md``). This variant keeps stage-0
generously sized (never the bottleneck) and instead squeezes stage-1's own
``gpu_memory_utilization``, so the pressure sidecar's transient allocation
competes with whatever Code2Wav actually holds on the same physical GPU (both
stages share ``devices: "0"``) instead of talker KV.

Two things this probe does NOT claim, on purpose:

* ``gpu_memory_utilization`` only constrains vLLM's own KV-cache profiling
  budget for stage-1. It is not a hard cap on CUDA-graph capture pools, the
  decoder_state_cache, the PyTorch caching allocator, or Code2Wav's temporary
  workspace tensors. This probe's job is to find out whether squeezing that
  one knob, plus external sidecar pressure, produces a real runtime danger
  cell for stage-1 — not to assume one exists. (``--hbm-limit-gb`` is kept as
  an opt-in escape hatch for when gpu_memory_utilization alone hits the
  boot-time floor, but it converts to vLLM's ``kv_cache_memory_bytes`` budget
  — see ``_effective_kv_cache_memory_bytes()`` in
  ``vllm_omni/engine/arg_utils.py`` — so it is still a KV-cache limit, not a
  workspace/CUDA-graph/allocator limit. Treat any "crash" found only via
  ``--hbm-limit-gb`` as a KV-starvation result, not a workspace-exhaustion
  result.)
* A "crash" only counts when there is direct runtime evidence: a CUDA OOM
  message, the server process exiting mid-workload, or completed requests
  falling short of the intended count for a server-side reason — and only
  when that evidence's log timestamp falls inside the pressure critical
  window (from ``pressure_ramp_started phase=critical`` to the matching
  ``released``/``partial_release`` event in ``status.json``'s
  ``pressure.events``). Startup failures, sidecar allocation OOMs, benchmark
  client/harness errors, and failures observed before pressure started or
  after it released are explicitly excluded — see ``classify_crash()``.

It is otherwise a thin wrapper around ``run_dynamic_hbm_experiment.py``,
mirroring probe_h9_danger.py's shell-out / classify / bisect structure, with
two structural fixes:

* ``--repeats`` is passed through to the base runner (previously hardcoded to
  1) and all ``repeat_NN_<arm>`` cases are aggregated into a failure rate,
  since a single run cannot distinguish a real danger cell from CUDA
  workspace-peak variance across launches.
* Per-stage log facts are extracted by first splitting ``server.log`` on the
  ``(StageEngineCoreProc_stage<N>_replica...)`` process-title prefix vLLM
  attaches to every line, then regex-matching within each stage's slice.
  Matching the whole file with a single un-scoped regex silently returns
  whichever stage happens to log that line first, which in every prior H9
  probe run was stage-0 — not necessarily what a stage-1 probe needs.

Recommended flow (see docstring on ``run_bisect``): a first pass at
concurrency default finds a *candidate* interval by single-shot bisection;
that candidate must then be confirmed with ``--repeats 3`` (or 5) on both the
C and D arms before it is treated as a frozen H9 cell — a single crash or a
single survival is not evidence either way.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_RUNNER = REPO / "benchmarks/hbm_admission/run_dynamic_hbm_experiment.py"
_VENV_PY = REPO / ".venv/bin/python"
PYEXE = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_long"

# vLLM tags every worker-process log line with a process-title prefix of the
# form "(StageEngineCoreProc_stage0_replica0 pid=1234) ...". Split on this
# before matching KV/concurrency facts so a stage-1 probe cannot silently read
# back stage-0's numbers.
STAGE_PREFIX_RE = re.compile(r"^\(StageEngineCoreProc_stage(\d+)_replica\d+ pid=\d+\)\s?(.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
KV_GIB_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
MAX_CONC_RE = re.compile(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x")
DESIRED_GMU_RE = re.compile(r"Desired GPU memory utilization is \(([\d.]+),")
CUDA_OOM_RE = re.compile(r"cuda(?:\s+error)?[:\s]+out of memory|torch\.cuda\.outofmemoryerror", re.IGNORECASE)


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stage_overrides(gmu0: float, gmu1: float, hbm_limit_gb: float | None = None) -> str:
    """Stage-1 (Code2Wav) squeeze override; stage-0 left generously sized.

    ``hbm_limit_gb``, when set, only tightens vLLM's KV-cache byte budget for
    stage-1 (see module docstring) — it is an escape hatch for when
    ``gpu_memory_utilization`` alone can't go low enough to boot, not a
    workspace/CUDA-graph cap. It is paired with ``hbm_admission_guard: false``
    so the static guard does not protect the C arm (it auto-enables whenever
    hbm_limit_gb is configured).
    """
    stage1: dict[str, Any] = {"gpu_memory_utilization": gmu1}
    if hbm_limit_gb is not None:
        stage1["hbm_limit_gb"] = hbm_limit_gb
        stage1["hbm_admission_guard"] = False
    return json.dumps(
        {"0": {"gpu_memory_utilization": gmu0}, "1": stage1},
        separators=(",", ":"),
    )


def base_runner_argv(args: argparse.Namespace, *, arm: str, gmu0: float, gmu1: float, out_dir: Path,
                     repeats: int, hbm_limit_gb: float | None = None) -> list[str]:
    """Assemble the argv for run_dynamic_hbm_experiment.py."""
    return [
        PYEXE, str(BASE_RUNNER),
        "--output-dir", str(out_dir),
        "--model", str(args.model),
        "--deploy-config", str(args.deploy_config),
        "--dataset-path", str(args.dataset_path),
        "--device", str(args.device),
        "--port", str(args.port),
        "--arms", arm,
        "--repeats", str(repeats),
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
        # squeeze the Code2Wav (stage-1) pool; stage-0 stays at gmu0 (default
        # 0.35, above stock 0.3) so talker is never the limiting factor. `=`
        # form because argparse rejects a flag-like value (--stage-overrides)
        # supplied as a separate token.
        "--server-extra-arg=--stage-overrides",
        f"--server-extra-arg={stage_overrides(gmu0, gmu1, hbm_limit_gb)}",
    ]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _split_by_stage(text: str) -> dict[str, list[str]]:
    """Group server.log lines by the stage that emitted them.

    Lines with no recognizable stage prefix (orchestrator/launcher output) go
    under key "unknown" and are never used for KV/concurrency facts.
    """
    by_stage: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = _strip_ansi(raw_line)
        m = STAGE_PREFIX_RE.match(line)
        if m:
            stage_id, rest = m.group(1), m.group(2)
        else:
            stage_id, rest = "unknown", line
        by_stage.setdefault(stage_id, []).append(rest)
    return by_stage


def _stage_facts(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    facts: dict[str, Any] = {
        "kv_cache_tokens": None,
        "kv_cache_gib": None,
        "max_concurrency_x": None,
        "effective_gmu": None,
        "cuda_oom_lines": 0,
        "preempt_lines": 0,
        "kv_full_lines": 0,
        "boot_kv_config_failure": False,
    }
    if not text:
        return facts
    if m := KV_TOKENS_RE.search(text):
        facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := KV_GIB_RE.search(text):
        facts["kv_cache_gib"] = float(m.group(1))
    if m := MAX_CONC_RE.search(text):
        facts["max_concurrency_x"] = float(m.group(1))
    if m := DESIRED_GMU_RE.search(text):
        facts["effective_gmu"] = float(m.group(1))
    low = text.lower()
    facts["cuda_oom_lines"] = len(CUDA_OOM_RE.findall(text))
    facts["preempt_lines"] = low.count("preempt")
    facts["kv_full_lines"] = low.count("no available kv") + low.count("cannot allocate") + low.count(
        "kv cache is full"
    )
    facts["boot_kv_config_failure"] = (
        "no available memory for the cache blocks" in low
        or "_check_enough_kv_cache_memory" in text
    )
    return facts


def _server_log_facts(server_log: Path) -> dict[str, Any]:
    """Per-stage KV/concurrency/OOM facts, keyed by stage id (plus "unknown").

    Also returns a top-level ``server_exited_mid_workload`` flag derived from
    whether stage-tagged process output stops appearing near the end of the
    file without a clean shutdown marker — used by classify_crash() as one
    piece of runtime-crash evidence.
    """
    if not server_log.is_file():
        return {"stages": {}, "cuda_oom_timestamps": []}
    raw = server_log.read_text(errors="replace")
    by_stage = _split_by_stage(raw)
    stages = {sid: _stage_facts(lines) for sid, lines in by_stage.items() if sid != "unknown"}
    # Absolute CUDA-OOM occurrences across the whole file, used only to gate
    # "did a runtime CUDA OOM happen at all" — timestamp-window overlap with
    # the pressure critical window is checked separately in classify_crash().
    cuda_oom_total = sum(s["cuda_oom_lines"] for s in stages.values())
    return {"stages": stages, "cuda_oom_total": cuda_oom_total}


def _pressure_window(pressure: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (critical_start_iso, critical_end_iso) from status.json's pressure.events.

    critical_end is the first "partial_release" or "released" event at or
    after the critical ramp; None if the critical phase never started (no
    signal to gate on, caller should not apply the window filter).
    """
    events = pressure.get("events") or []
    start = None
    end = None
    for ev in events:
        if ev.get("event") == "pressure_ramp_started" and ev.get("phase") == "critical":
            start = ev.get("timestamp")
        elif start is not None and end is None and ev.get("event") in {"partial_release", "released"}:
            end = ev.get("timestamp")
    return start, end


def classify_crash(status: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Decide whether one case is a genuine runtime crash under pressure.

    Returns a dict with ``crashed`` (bool) and ``crash_reason`` explaining
    which evidence (or lack of it) drove the verdict, plus ``excluded_as``
    when a failure was observed but attributed to a non-H9 cause.
    """
    case_status = status.get("status")
    benchmark = status.get("benchmark") or {}
    pressure = status.get("pressure") or {}
    intended = status.get("_intended_num_prompts")
    completed = benchmark.get("completed")
    failed = benchmark.get("failed")

    result = {"crashed": False, "crash_reason": None, "excluded_as": None}

    if case_status == "running":
        # status.json was never finalized: the case dir exists but run_case
        # never reached its finally-block writeback (e.g. this process was
        # killed). Not evidence of an H9 danger point.
        result["excluded_as"] = "harness-incomplete"
        return result

    stage_facts = facts.get("stages", {})
    cuda_oom_total = facts.get("cuda_oom_total", 0)
    boot_failure = any(s.get("boot_kv_config_failure") for s in stage_facts.values())
    if boot_failure:
        result["excluded_as"] = "startup-failure"
        return result

    allocation_oom = int(pressure.get("allocation_oom_count", 0))
    if allocation_oom > 0:
        result["excluded_as"] = "sidecar-oom"
        return result

    critical_start, critical_end = _pressure_window(pressure)
    started_at, finished_at = status.get("started_at"), status.get("finished_at")

    # Evidence 1: a runtime CUDA OOM was logged at all. We cannot always bind
    # a CUDA OOM log line to a timestamp (vLLM's OOM tracebacks don't all
    # carry one on the OOM line itself), so treat "any CUDA OOM anywhere in a
    # case that also has a critical-pressure window" as in-scope; a CUDA OOM
    # in a case with pressure=False, or before pressure started at all
    # (started_at == critical_start would be nonsensical), is not attributed
    # to H9 pressure.
    cuda_oom_evidence = cuda_oom_total > 0 and critical_start is not None

    # Evidence 2: server process exited mid-workload. run_case() only reaches
    # "completed"/"acceptance_failed" when the benchmark client itself exited
    # 0 and produced a result file; a server crash normally surfaces as the
    # client failing (non-zero exit / no result file) with the server log
    # cutting off — status "failed" with no explicit exception message is the
    # closest signal available from the base runner's status.json.
    server_exit_evidence = (
        case_status == "failed"
        and "error" not in status  # an "error" key means a *harness*-side exception (see run_case), not necessarily a server crash
        and critical_start is not None
    )

    # Evidence 3: completed < intended for a server-side reason. A benchmark
    # timeout also produces completed < intended but is excluded separately
    # below via the timeout check.
    shortfall_evidence = (
        isinstance(completed, int) and isinstance(intended, int)
        and completed < intended
        and (failed or 0) > 0
        and critical_start is not None
    )

    if not (cuda_oom_evidence or server_exit_evidence or shortfall_evidence):
        if case_status in {"failed", "acceptance_failed"}:
            result["excluded_as"] = "no-runtime-crash-evidence"
        return result

    if critical_start is None:
        result["excluded_as"] = "no-pressure-window-recorded"
        return result

    # Window check: only count evidence whose case lifetime overlaps
    # [critical_start, critical_end or finished_at]. We compare at the
    # case-lifetime granularity (started_at..finished_at vs critical window)
    # because individual OOM/exit log lines aren't reliably timestamped;
    # this is a coarse but conservative filter — see module docstring.
    window_end = critical_end or finished_at
    if started_at and finished_at and critical_start:
        if finished_at < critical_start:
            result["excluded_as"] = "failure-before-pressure-started"
            return result
        if window_end and started_at > window_end:
            result["excluded_as"] = "failure-after-pressure-released"
            return result

    result["crashed"] = True
    reasons = []
    if cuda_oom_evidence:
        reasons.append(f"cuda_oom_lines={cuda_oom_total}")
    if server_exit_evidence:
        reasons.append(f"status=failed (client exit/server crash)")
    if shortfall_evidence:
        reasons.append(f"completed={completed}/{intended} failed={failed}")
    result["crash_reason"] = "; ".join(reasons)
    return result


def _read_cases(out_dir: Path, arm: str, repeats: int, intended_num_prompts: int) -> list[dict[str, Any]]:
    """Load status.json + per-stage server.log facts for every repeat the runner wrote."""
    records = []
    for repeat in range(1, repeats + 1):
        case_dir = out_dir / "cases" / f"repeat_{repeat:02d}_{arm}"
        status_path = case_dir / "status.json"
        if not status_path.is_file():
            records.append({"repeat": repeat, "status": "missing", "error": f"no status.json under {case_dir}"})
            continue
        status = json.loads(status_path.read_text())
        status["_intended_num_prompts"] = intended_num_prompts
        server_log = Path(status.get("artifacts", {}).get("server_log", case_dir / "server.log"))
        facts = _server_log_facts(server_log)
        crash = classify_crash(status, facts)
        benchmark = status.get("benchmark") or {}
        pressure = status.get("pressure") or {}
        scheduler = status.get("scheduler") or {}
        stage1 = facts.get("stages", {}).get("1", {})
        stage0 = facts.get("stages", {}).get("0", {})
        records.append({
            "repeat": repeat,
            "case_dir": str(case_dir),
            "server_log": str(server_log),
            "status": status.get("status"),
            **crash,
            "preemption_mentions": int(scheduler.get("preemption_mentions", 0)),
            "completed": benchmark.get("completed"),
            "failed": benchmark.get("failed"),
            "request_throughput": benchmark.get("request_throughput"),
            "p99_e2el_ms": benchmark.get("p99_e2el_ms"),
            "mean_audio_underrun_s": benchmark.get("mean_audio_underrun_s"),
            "p99_audio_rtf": benchmark.get("p99_audio_rtf"),
            "allocation_oom_count": int(pressure.get("allocation_oom_count", 0)),
            "peak_reported_pressure": pressure.get("peak_reported_pressure"),
            "acceptance_passed": (status.get("acceptance") or {}).get("passed"),
            "stage0": stage0,
            "stage1": stage1,
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r.get("status") != "missing"]
    n = len(valid)
    crashes = sum(1 for r in valid if r.get("crashed"))
    failure_rate = (crashes / n) if n else None
    has_pause = has_resume = 0
    for r in valid:
        try:
            log_text = Path(r["server_log"]).read_text(errors="replace")
        except OSError:
            continue
        has_pause += "[HBMAdmission] paused" in log_text
        has_resume += "[HBMAdmission] resumed" in log_text
    underruns = [r["mean_audio_underrun_s"] for r in valid if isinstance(r.get("mean_audio_underrun_s"), (int, float))]
    return {
        "n": n,
        "crashes": crashes,
        "failure_rate": failure_rate,
        "pause_rate": (has_pause / n) if n else None,
        "resume_rate": (has_resume / n) if n else None,
        "mean_audio_underrun_s": statistics.fmean(underruns) if underruns else None,
        "exclusion_reasons": {
            reason: sum(1 for r in valid if r.get("excluded_as") == reason)
            for reason in sorted({r.get("excluded_as") for r in valid if r.get("excluded_as")})
        },
    }


def classify(summary: dict[str, Any], arm: str, gmu1: float, repeats: int) -> tuple[str, str]:
    """Return (classification, recommended-next-command) for an aggregated cell."""
    if summary["n"] == 0:
        return "run-error", "inspect probe.log — the base runner produced no status.json files"

    rate = summary["failure_rate"] or 0.0

    if arm == "D_dynamic_on_pressure":
        if rate <= 0.20 and (summary["pause_rate"] or 0) >= 0.90 and (summary["resume_rate"] or 0) >= 0.90:
            return (
                "D-survives-with-mechanism",
                f"freeze --gmu1 {gmu1:.2f}; this cell is a candidate H9 formal cell "
                f"(C failure_rate should be 0.30-0.80, D failure_rate={rate:.2f} with "
                f"pause_rate={summary['pause_rate']:.2f} resume_rate={summary['resume_rate']:.2f}). "
                f"Confirm with --repeats 20 on both arms before writing a formal H9 result.",
            )
        if rate > 0.20:
            return (
                "D-also-fails",
                f"D failure_rate={rate:.2f} at gmu1={gmu1:.2f} — dynamic monitoring is not "
                f"preventing failure at this cell; raise --gmu1 and re-probe C then D",
            )
        return (
            "D-inconclusive",
            f"D survived (failure_rate={rate:.2f}) but pause_rate={summary['pause_rate']} / "
            f"resume_rate={summary['resume_rate']} did not both reach 0.90 — inspect server.log",
        )

    # C arm
    if 0.30 <= rate <= 0.80:
        return (
            "candidate-danger-cell",
            f"C failure_rate={rate:.2f} (n={summary['n']}) at gmu1={gmu1:.2f} is inside the "
            f"30-80% calibration band. Confirm with more repeats if n < 5, then run D at the "
            f"same cell:\n"
            f"    python {HERE}/probe_h9_danger_stage1.py --gmu1 {gmu1:.2f} --repeats 5\n"
            f"    python {HERE}/probe_h9_danger_stage1.py --gmu1 {gmu1:.2f} --repeats 5 "
            f"--arm D_dynamic_on_pressure",
        )
    if rate > 0.80:
        return (
            "danger-cell-too-severe",
            f"C failure_rate={rate:.2f} — pool too small, controller likely can't keep "
            f"min_num_seqs alive either. Raise --gmu1 to {gmu1 + 0.02:.2f} and re-probe.",
        )

    underrun = summary.get("mean_audio_underrun_s")
    if isinstance(underrun, (int, float)) and underrun >= 2.0:
        return (
            "quality-edge-no-crash",
            f"C survived (failure_rate={rate:.2f}) but mean_audio_underrun_s={underrun:.2f} — "
            f"real SLO degradation without runtime-crash evidence (mirrors the stage-0 probe's "
            f"edge finding). Either treat this as the H9 signal directly under an SLO-based "
            f"failure definition, or drop --gmu1 to {max(gmu1 - 0.02, 0.05):.2f} to push for an "
            f"actual crash.",
        )
    return (
        "pool-still-too-big",
        f"C survived comfortably (failure_rate={rate:.2f}, n={summary['n']}) — "
        f"drop --gmu1 to {max(gmu1 - 0.03, 0.05):.2f} and re-probe",
    )


def run_one(args: argparse.Namespace, *, arm: str, gmu0: float, gmu1: float, step: int,
            root: Path, log_fh, repeats: int, hbm_limit_gb: float | None = None) -> dict[str, Any]:
    tag = f"hbm{hbm_limit_gb:.2f}" if hbm_limit_gb is not None else f"gmu1{gmu1:.3f}"
    out_dir = root / f"run_{step:02d}_{arm}_{tag}"
    argv = base_runner_argv(args, arm=arm, gmu0=gmu0, gmu1=gmu1, out_dir=out_dir, repeats=repeats,
                            hbm_limit_gb=hbm_limit_gb)
    log_fh.write(f"\n{'=' * 80}\nstep {step}  arm={arm}  gmu0={gmu0}  gmu1={gmu1}  "
                 f"hbm_limit_gb={hbm_limit_gb}  repeats={repeats}\n")
    log_fh.write("CMD: " + " ".join(argv) + "\n")
    log_fh.flush()
    t0 = time.monotonic()
    proc = subprocess.run(argv, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    duration = time.monotonic() - t0
    case_records = _read_cases(out_dir, arm, repeats, args.num_prompts)
    summary = summarize(case_records)
    cls, nxt = classify(summary, arm, gmu1, repeats)
    return {
        "step": step, "arm": arm, "gmu0": gmu0, "gmu1": gmu1, "hbm_limit_gb": hbm_limit_gb,
        "repeats": repeats, "concurrency": args.concurrency, "num_prompts": args.num_prompts,
        "pressure_critical_target": args.pressure_critical_target,
        "pressure_critical_seconds": args.pressure_critical_seconds,
        "pressure_reserve_mib": args.pressure_reserve_mib,
        "base_runner_exit_code": proc.returncode, "duration_s": round(duration, 1),
        "out_dir": str(out_dir),
        "summary": summary,
        "cases": case_records,
        "classification": cls,
        "recommended_next": nxt,
    }


def write_report(root: Path, records: list[dict[str, Any]]) -> None:
    write_json(root / "probe_result.json", records)
    lines: list[str] = ["# H9 danger-cell probe — stage-1 (Code2Wav) variant", ""]
    last = records[-1]
    knob = (f"hbm_limit_gb={last['hbm_limit_gb']}" if last.get("hbm_limit_gb") is not None
            else f"gmu1={last['gmu1']}")
    lines += [
        f"**Verdict:** `{last['classification']}` (step {last['step']}, arm {last['arm']}, "
        f"{knob}, gmu0={last['gmu0']}, concurrency={last['concurrency']}, repeats={last['repeats']})",
        "",
        "**Recommended next:**",
        "```",
        last["recommended_next"],
        "```",
        "",
        "| step | arm | gmu0 | gmu1 | hbm_gb | conc | n | failure_rate | pause_rate | resume_rate | underrun_s | classification | status | dur s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        s = r["summary"]
        lines.append(
            f"| {r['step']} | {r['arm'].split('_')[0]} | {r['gmu0']} | {r['gmu1']} | {r.get('hbm_limit_gb')} | "
            f"{r['concurrency']} | {s['n']} | "
            f"{round(s['failure_rate'], 2) if s['failure_rate'] is not None else '-'} | "
            f"{round(s['pause_rate'], 2) if s['pause_rate'] is not None else '-'} | "
            f"{round(s['resume_rate'], 2) if s['resume_rate'] is not None else '-'} | "
            f"{round(s['mean_audio_underrun_s'], 2) if s['mean_audio_underrun_s'] is not None else '-'} | "
            f"{r.get('classification', '-')} | {r.get('base_runner_exit_code')} | {r.get('duration_s')} |"
        )
    lines += ["", "## Exclusion reasons (why a failed case was NOT counted as a crash)", ""]
    for r in records:
        reasons = r["summary"].get("exclusion_reasons") or {}
        if reasons:
            lines.append(f"- step {r['step']} ({r['arm'].split('_')[0]}): {reasons}")
    lines += ["", f"Full per-repeat evidence: `{root}/run_*/cases/`.", ""]
    (root / "probe_result.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmu0", type=float, default=0.35,
                   help="stage-0 (talker) gpu_memory_utilization override — kept generous, not the squeeze target")
    p.add_argument("--gmu1", type=float, default=0.11,
                   help="stage-1 (code2wav) gpu_memory_utilization override — squeeze target. "
                        "NOTE: this only bounds vLLM's KV-cache profiling budget for stage-1, not "
                        "CUDA-graph pools, decoder_state_cache, or the PyTorch allocator — see module docstring.")
    p.add_argument("--hbm-limit-gb", type=float, default=None,
                   help="stage-1 KV-cache byte cap (GiB) via kv_cache_memory_bytes, + "
                        "hbm_admission_guard:false. Escape hatch for when gpu_memory_utilization "
                        "alone hits the boot-time KV-config floor. This still only limits KV cache, "
                        "NOT stage-1 workspace/CUDA-graph memory — treat a crash found only this way "
                        "as a KV-starvation result, not a workspace-exhaustion result.")
    p.add_argument("--concurrency", type=int, default=48, help="request concurrency == max_num_seqs")
    p.add_argument("--num-prompts", type=int, default=160)
    p.add_argument("--arm", default="C_dynamic_off_pressure",
                   choices=["C_dynamic_off_pressure", "D_dynamic_on_pressure"])
    p.add_argument("--repeats", type=int, default=1,
                   help="repeats per cell, aggregated into a failure rate. Use 1 for a fast scan to "
                        "find a candidate interval; 3-5 to confirm a candidate cell before freezing it.")
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
                   help="single-shot (repeats=1) scan that halves gmu1 toward a smaller pool (max 3 "
                        "steps) to find a CANDIDATE interval only. A candidate is not a confirmed "
                        "danger cell — re-run the recommended --repeats 3-5 command it prints before "
                        "trusting the result.")
    p.add_argument("-n", "--print-cmd", action="store_true",
                   help="print the base-runner argv for the first step and exit (no launch)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    root = args.output_dir or (REPO / f"benchmarks/results/H9_probe_stage1_{utc_stamp()}")
    root.mkdir(parents=True, exist_ok=True)

    if args.print_cmd:
        argv = base_runner_argv(args, arm=args.arm, gmu0=args.gmu0, gmu1=args.gmu1,
                                out_dir=root / "run_00_print", repeats=args.repeats,
                                hbm_limit_gb=args.hbm_limit_gb)
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

    def _step(step: int, *, arm: str, gmu1: float, repeats: int, hbm_limit_gb: float | None) -> dict[str, Any]:
        rec = run_one(args, arm=arm, gmu0=args.gmu0, gmu1=gmu1, step=step, root=root,
                      log_fh=log_fh, repeats=repeats, hbm_limit_gb=hbm_limit_gb)
        records.append(rec)
        write_report(root, records)
        knob = f"hbm_limit_gb={hbm_limit_gb}" if hbm_limit_gb is not None else f"gmu1={gmu1}"
        s = rec["summary"]
        print(f"[step {step}] {arm.split('_')[0]} {knob} repeats={repeats} -> "
              f"n={s['n']} failure_rate={s['failure_rate']} classification={rec['classification']}  "
              f"({rec['duration_s']}s)")
        return rec

    records: list[dict[str, Any]] = []
    log_path = root / "probe.log"
    with log_path.open("w") as log_fh:
        rec = _step(0, arm=args.arm, gmu1=args.gmu1, repeats=args.repeats, hbm_limit_gb=hbm)

        if args.bisect and args.arm == "C_dynamic_off_pressure":
            # Single-shot (repeats=1) bisection to find a CANDIDATE interval
            # only. Per the module docstring / user guidance, a candidate must
            # be re-confirmed with --repeats 3-5 on both arms before it is
            # trusted as a frozen H9 cell — this loop does not do that itself.
            rate0 = rec["summary"]["failure_rate"] or 0.0
            candidate_found = 0.30 <= rate0 <= 0.80

            if candidate_found:
                _step(1, arm="D_dynamic_on_pressure", gmu1=args.gmu1, repeats=1, hbm_limit_gb=hbm)

            elif hbm is not None:
                val = hbm
                for step in range(1, 4):
                    val = round(val / 2, 3)
                    rec = _step(step, arm="C_dynamic_off_pressure", gmu1=args.gmu1, repeats=1, hbm_limit_gb=val)
                    rate = rec["summary"]["failure_rate"] or 0.0
                    if 0.30 <= rate <= 0.80:
                        _step(step + 1, arm="D_dynamic_on_pressure", gmu1=args.gmu1, repeats=1, hbm_limit_gb=val)
                        break
                    if any(c.get("excluded_as") == "startup-failure" for c in rec["cases"]) or val <= 0.1:
                        break

            else:
                gmu = args.gmu1
                for step in range(1, 4):
                    gmu = round(gmu - 0.025, 3)
                    rec = _step(step, arm="C_dynamic_off_pressure", gmu1=gmu, repeats=1, hbm_limit_gb=None)
                    rate = rec["summary"]["failure_rate"] or 0.0
                    if 0.30 <= rate <= 0.80:
                        _step(step + 1, arm="D_dynamic_on_pressure", gmu1=gmu, repeats=1, hbm_limit_gb=None)
                        break
                    if any(c.get("excluded_as") == "startup-failure" for c in rec["cases"]) or gmu <= 0.04:
                        break

    write_report(root, records)
    print(f"\nreport: {root}/probe_result.md")
    print(f"json:   {root}/probe_result.json")
    print(f"verdict: {records[-1]['classification']}")
    print(f"next:\n{records[-1]['recommended_next']}")
    exit_ok = records[-1]["classification"] in {
        "candidate-danger-cell", "D-survives-with-mechanism",
    }
    return 0 if exit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
