#!/usr/bin/env python3
"""Build and evaluate an AR output-length profile from real shadow traces."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Local environments may have the vLLM 0.28 module move while this checkout
# still targets 0.27. Keep this analysis CLI usable under the same drift as the
# hypothesis harness; production packages with matched versions skip the shim.
_MISSING = "vllm.entrypoints.serve.utils.error_response"
if _MISSING not in sys.modules:
    try:
        sys.modules[_MISSING] = importlib.import_module(
            "vllm.entrypoints.serve.exception_handling.error_response"
        )
    except Exception:
        sys.modules[_MISSING] = SimpleNamespace(
            create_error_response=lambda *args, **kwargs: None
        )

from vllm_omni.core.memory_coordinator import (  # noqa: E402
    ARProfileStore,
    ARRequestResourceContext,
    ARResourceEstimator,
    ProfileFingerprint,
    build_ar_output_profiles,
    read_observations_jsonl,
)


def _estimate(observation, *, output50=None, output_quantile=None) -> int:
    return ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=observation.prompt_tokens,
            max_tokens=observation.requested_max_tokens,
            block_size=observation.block_size,
            allocated_kv_blocks=observation.baseline_allocated_kv_blocks,
            expected_output_tokens=output50,
            quantile_output_tokens=output_quantile,
            workload_class=observation.workload_class,
            target_coverage=1.0 if output_quantile is None else 0.95,
        )
    ).logical.quantile_peak_kv_blocks


def analyze(args: argparse.Namespace) -> dict:
    observations = []
    for path in args.observations:
        observations.extend(read_observations_jsonl(path))
    observations.sort(key=lambda item: item.finished_monotonic_s)

    grouped: dict[str, list] = {}
    for observation in observations:
        grouped.setdefault(observation.workload_class, []).append(observation)

    training = []
    holdout = []
    skipped_classes = {}
    for workload_class, rows in sorted(grouped.items()):
        if len(rows) < args.min_samples:
            skipped_classes[workload_class] = len(rows)
            continue
        split = max(1, min(len(rows) - 1, int(len(rows) * args.train_fraction)))
        training.extend(rows[:split])
        holdout.extend(rows[split:])

    if not training or not holdout:
        raise ValueError(
            "trace has no evaluable train/holdout classes; collect more observations "
            "or lower --min-samples"
        )

    fingerprint = ProfileFingerprint(
        model_id=args.model_id,
        device_type=args.device_type,
        dtype=args.dtype,
        tp_size=args.tp_size,
        block_size=args.block_size,
        execution_mode=args.execution_mode,
    )
    profiles = build_ar_output_profiles(
        training,
        fingerprint=fingerprint,
        min_samples=max(1, int(args.min_samples * args.train_fraction)),
    )
    store = ARProfileStore(profiles)
    store.write_jsonl(args.profile_output)

    rows = []
    per_class: dict[str, list[dict]] = {}
    for observation in holdout:
        profile = store.get(observation.workload_class)
        if profile is None:
            continue
        actual = observation.observed_peak_incremental_kv_blocks
        worst = _estimate(observation)
        profiled = _estimate(
            observation,
            output50=profile.p50_output_tokens,
            output_quantile=profile.output_tokens_at(args.coverage),
        )
        row = {
            "class": observation.workload_class,
            "actual": actual,
            "worst": worst,
            "profiled": profiled,
        }
        rows.append(row)
        per_class.setdefault(observation.workload_class, []).append(row)

    def metrics(items: list[dict]) -> dict:
        count = len(items)
        return {
            "samples": count,
            "worst_coverage": sum(row["worst"] >= row["actual"] for row in items)
            / count,
            "profile_coverage": sum(
                row["profiled"] >= row["actual"] for row in items
            )
            / count,
            "worst_underprediction_rate": sum(
                row["worst"] < row["actual"] for row in items
            )
            / count,
            "profile_underprediction_rate": sum(
                row["profiled"] < row["actual"] for row in items
            )
            / count,
            "worst_mean_excess_blocks": sum(
                max(0, row["worst"] - row["actual"]) for row in items
            )
            / count,
            "profile_mean_excess_blocks": sum(
                max(0, row["profiled"] - row["actual"]) for row in items
            )
            / count,
        }

    aggregate = metrics(rows)
    worst_excess = aggregate["worst_mean_excess_blocks"]
    profile_excess = aggregate["profile_mean_excess_blocks"]
    aggregate["excess_block_reduction"] = (
        1.0 - profile_excess / worst_excess if worst_excess > 0 else 0.0
    )
    report = {
        "schema_version": 1,
        "source_files": [str(path) for path in args.observations],
        "total_observations": len(observations),
        "training_observations": len(training),
        "holdout_observations": len(rows),
        "train_fraction": args.train_fraction,
        "target_coverage": args.coverage,
        "skipped_classes": skipped_classes,
        "fingerprint": fingerprint.__dict__,
        "profiles": [
            {
                "workload_class": profile.workload_class,
                "sample_count": profile.sample_count,
                "p50": profile.p50_output_tokens,
                "p95": profile.p95_output_tokens,
                "p99": profile.p99_output_tokens,
            }
            for profile in profiles
        ],
        "aggregate": aggregate,
        "per_class": {
            workload_class: metrics(items)
            for workload_class, items in sorted(per_class.items())
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device-type", required=True)
    parser.add_argument("--dtype", default="torch.float16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--execution-mode", default="async")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if not 0.0 < args.coverage <= 1.0:
        parser.error("--coverage must be in (0, 1]")
    if args.min_samples < 2:
        parser.error("--min-samples must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    report = analyze(args)
    aggregate = report["aggregate"]
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
