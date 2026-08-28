#!/usr/bin/env python3
"""H8 - Dynamic HBM has small throughput/latency overhead without pressure.

This is an end-to-end, real-server A/B experiment.  Each repeat starts a fresh
server for A (dynamic HBM disabled) and B (enabled), with the same model,
dataset, concurrency and request count.  Neither arm runs the pressure sidecar.

The analyzer is deliberately paired by repeat.  Its default acceptance gates
are: at least 10 complete A/B pairs, every B run remains below high_watermark,
no B run lowers the cap, the B monitoring path emits reports, and the upper
endpoint of a deterministic paired-bootstrap 95% interval is <=5% throughput
loss and <=10% P99 end-to-end latency inflation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_RUNNER = HERE.parent / "run_dynamic_hbm_experiment.py"
DEFAULT_MODEL = REPO.parent / "huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_DEPLOY = REPO / "vllm_omni/deploy/qwen3_tts.yaml"
DEFAULT_DATASET = REPO / "benchmarks/build_dataset/seed_tts_smoke"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] + fraction * (ordered[hi] - ordered[lo])


def _bootstrap_upper(values: list[float], *, samples: int = 10_000) -> float | None:
    """Deterministic paired-bootstrap upper 95% bound for the median."""
    if not values:
        return None
    rng = random.Random(8)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return _percentile(medians, 0.95)


def _load_cases(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((output_dir / "cases").glob("*/status.json"))
    ]


def analyze(
    output_dir: Path,
    *,
    min_pairs: int = 10,
    max_throughput_loss: float = 0.05,
    max_p99_inflation: float = 0.10,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing experiment metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    high = float(metadata["arguments"]["high_watermark"])
    cases = _load_cases(output_dir)
    indexed = {(case.get("repeat"), case.get("arm", {}).get("name")): case for case in cases}
    repeats = sorted({key[0] for key in indexed if key[0] is not None})
    pairs: list[dict[str, Any]] = []
    incomplete: list[int] = []
    for repeat in repeats:
        a = indexed.get((repeat, "A_dynamic_off_no_pressure"))
        b = indexed.get((repeat, "B_dynamic_on_no_pressure"))
        if not a or not b or a.get("status") != "completed" or b.get("status") != "completed":
            incomplete.append(int(repeat))
            continue
        a_bench, b_bench = a.get("benchmark", {}), b.get("benchmark", {})
        a_tp, b_tp = a_bench.get("request_throughput"), b_bench.get("request_throughput")
        a_p99, b_p99 = a_bench.get("p99_e2el_ms"), b_bench.get("p99_e2el_ms")
        if not all(isinstance(v, (int, float)) and v > 0 for v in (a_tp, b_tp, a_p99, b_p99)):
            incomplete.append(int(repeat))
            continue
        scheduler = b.get("scheduler", {})
        pairs.append({
            "repeat": repeat,
            "throughput_loss": 1.0 - float(b_tp) / float(a_tp),
            "p99_latency_inflation": float(b_p99) / float(a_p99) - 1.0,
            "a_throughput": a_tp,
            "b_throughput": b_tp,
            "a_p99_e2el_ms": a_p99,
            "b_p99_e2el_ms": b_p99,
            "b_peak_pressure": b.get("gpu", {}).get("pressure", {}).get("peak"),
            "b_cap_change_count": scheduler.get("cap_change_count", 0),
            "b_report_stage_ids": scheduler.get("first_report_stage_ids", []),
        })
    throughput_losses = [row["throughput_loss"] for row in pairs]
    latency_inflations = [row["p99_latency_inflation"] for row in pairs]
    tp_upper = _bootstrap_upper(throughput_losses)
    latency_upper = _bootstrap_upper(latency_inflations)
    no_pressure = all(
        isinstance(row["b_peak_pressure"], (int, float)) and row["b_peak_pressure"] < high
        for row in pairs
    )
    reports_present = all(bool(row["b_report_stage_ids"]) for row in pairs)
    no_cap_reduction = all(row["b_cap_change_count"] == 0 for row in pairs)
    gates = {
        "at_least_min_complete_pairs": len(pairs) >= min_pairs,
        "all_attempted_pairs_complete": not incomplete and len(pairs) == len(repeats),
        "dynamic_arm_stayed_below_high_watermark": bool(pairs) and no_pressure,
        "dynamic_arm_emitted_memory_reports": bool(pairs) and reports_present,
        "dynamic_arm_never_reduced_cap": bool(pairs) and no_cap_reduction,
        "throughput_loss_bootstrap_upper_within_limit": tp_upper is not None and tp_upper <= max_throughput_loss,
        "p99_latency_inflation_bootstrap_upper_within_limit": latency_upper is not None and latency_upper <= max_p99_inflation,
    }
    report = {
        "schema_version": 1,
        "hypothesis": "H8",
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "MATCHES EXPECTATION" if all(gates.values()) else "NOT PROVEN",
        "expectation": {
            "min_complete_pairs": min_pairs,
            "max_throughput_loss": max_throughput_loss,
            "max_p99_latency_inflation": max_p99_inflation,
            "bootstrap_confidence": 0.95,
        },
        "gates": gates,
        "metrics": {
            "complete_pairs": len(pairs),
            "incomplete_repeats": incomplete,
            "median_throughput_loss": statistics.median(throughput_losses) if pairs else None,
            "throughput_loss_bootstrap_upper_95": tp_upper,
            "median_p99_latency_inflation": statistics.median(latency_inflations) if pairs else None,
            "p99_latency_inflation_bootstrap_upper_95": latency_upper,
        },
        "pairs": pairs,
    }
    _write_json(output_dir / "H8_result.json", report)
    _write_markdown(output_dir / "H8_result.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics, limits = report["metrics"], report["expectation"]
    lines = [
        "# H8: No-pressure monitoring overhead", "",
        f"**Verdict:** {report['verdict']}", "",
        f"Complete paired repeats: {metrics['complete_pairs']} (required {limits['min_complete_pairs']}).", "",
        "| Metric | Median | Bootstrap upper 95% | Limit |", "|---|---:|---:|---:|",
        f"| Throughput loss | {metrics['median_throughput_loss']} | {metrics['throughput_loss_bootstrap_upper_95']} | {limits['max_throughput_loss']} |",
        f"| P99 latency inflation | {metrics['median_p99_latency_inflation']} | {metrics['p99_latency_inflation_bootstrap_upper_95']} | {limits['max_p99_latency_inflation']} |",
        "", "## Gates", "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["gates"].items())
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    command = [
        sys.executable, str(BASE_RUNNER), "--output-dir", str(args.output_dir),
        "--model", str(args.model), "--deploy-config", str(args.deploy_config),
        "--dataset-path", str(args.dataset_path), "--device", str(args.device),
        "--port", str(args.port), "--arms", "A_dynamic_off_no_pressure", "B_dynamic_on_no_pressure",
        "--repeats", str(args.repeats), "--concurrency", str(args.concurrency),
        "--num-prompts", str(args.num_prompts), "--num-warmups", str(args.num_warmups),
        "--max-num-seqs", str(args.max_num_seqs), "--sample-interval-ms", "500",
        "--report-timeout-ms", "1500", "--low-watermark", "0.72",
        "--high-watermark", "0.82", "--critical-watermark", "0.90",
        "--startup-timeout", str(args.startup_timeout), "--benchmark-timeout", str(args.benchmark_timeout),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "H8_orchestrator.log").open("w") as log:
        runner = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    report = analyze(args.output_dir, min_pairs=args.min_pairs,
                     max_throughput_loss=args.max_throughput_loss,
                     max_p99_inflation=args.max_p99_inflation)
    report["base_runner_exit_code"] = runner.returncode
    _write_json(args.output_dir / "H8_result.json", report)
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    execute.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    execute.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    execute.add_argument("--device", type=int, default=0)
    execute.add_argument("--port", type=int, default=8000)
    execute.add_argument("--repeats", type=int, default=10)
    execute.add_argument("--concurrency", type=int, default=16)
    execute.add_argument("--num-prompts", type=int, default=240)
    execute.add_argument("--num-warmups", type=int, default=4)
    execute.add_argument("--max-num-seqs", type=int, default=16)
    execute.add_argument("--startup-timeout", type=int, default=1800)
    execute.add_argument("--benchmark-timeout", type=int, default=3600)
    for item in (execute, sub.add_parser("analyze")):
        if item is not execute:
            item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--min-pairs", type=int, default=10)
        item.add_argument("--max-throughput-loss", type=float, default=0.05)
        item.add_argument("--max-p99-inflation", type=float, default=0.10)
    return root


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.command == "run":
        return run(args)
    report = analyze(args.output_dir, min_pairs=args.min_pairs,
                     max_throughput_loss=args.max_throughput_loss,
                     max_p99_inflation=args.max_p99_inflation)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}, indent=2))
    return 0 if report["verdict"] == "MATCHES EXPECTATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
