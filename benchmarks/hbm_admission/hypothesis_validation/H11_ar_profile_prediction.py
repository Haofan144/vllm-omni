#!/usr/bin/env python3
"""H11 - AR workload-class quantiles reduce worst-case reservation loss.

This GPU-free experiment validates the profile pipeline, not production model
accuracy. Deterministic train/holdout distributions stand in for observations
that a live server will later export. It compares max_tokens worst-case,
workload-class P95, and oracle KV-block demand on the same holdout requests.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

from vllm_omni.core.memory_coordinator import (  # noqa: E402
    ARProfileStore,
    ARRequestResourceContext,
    ARResourceEstimator,
    ARWorkloadClassifier,
    ProfileFingerprint,
    ResourceObservationCollector,
    build_ar_output_profiles,
)

BLOCK_SIZE = 16
FINGERPRINT = ProfileFingerprint(
    model_id="synthetic-ar",
    device_type="gpu-free",
    dtype="float16",
    tp_size=1,
    block_size=BLOCK_SIZE,
)


def _observations(prompt: int, max_tokens: int, outputs: list[int]):
    classifier = ARWorkloadClassifier()
    workload_class = classifier.classify(
        prompt_tokens=prompt,
        max_tokens=max_tokens,
    )
    completed = []
    for index, actual_output in enumerate(outputs):
        estimate = ARResourceEstimator(BLOCK_SIZE).estimate(
            num_prompt_tokens=prompt,
            max_tokens=max_tokens,
        )
        collector = ResourceObservationCollector()
        request_id = f"train-{prompt}-{index}"
        collector.begin(
            request_id,
            estimate,
            prompt_tokens=prompt,
            requested_max_tokens=max_tokens,
            block_size=BLOCK_SIZE,
        )
        collector.observe(
            request_id,
            allocated_kv_blocks=math.ceil((prompt + actual_output) / BLOCK_SIZE),
            output_tokens=actual_output,
        )
        observation = collector.finish(request_id)
        assert observation is not None
        # Training traces are reclassified from request features; they must not
        # inherit the legacy estimator's generic "unknown" class.
        completed.append(
            observation.__class__(
                **{**observation.__dict__, "workload_class": workload_class}
            )
        )
    return completed


def _evaluate(prompt: int, max_tokens: int, outputs: list[int], store: ARProfileStore):
    classifier = ARWorkloadClassifier()
    workload_class = classifier.classify(prompt_tokens=prompt, max_tokens=max_tokens)
    profile = store.get(workload_class)
    assert profile is not None
    rows = []
    for actual_output in outputs:
        actual = math.ceil((prompt + actual_output) / BLOCK_SIZE)
        worst = ARResourceEstimator(BLOCK_SIZE).estimate(
            num_prompt_tokens=prompt, max_tokens=max_tokens
        ).kv_blocks
        profiled = ARResourceEstimator().estimate(
            ARRequestResourceContext(
                num_prompt_tokens=prompt,
                max_tokens=max_tokens,
                block_size=BLOCK_SIZE,
                expected_output_tokens=profile.p50_output_tokens,
                quantile_output_tokens=profile.p95_output_tokens,
                workload_class=workload_class,
                target_coverage=0.95,
                profile_version=profile.profile_version,
                sample_count=profile.sample_count,
            )
        ).kv_blocks
        rows.append((actual, worst, profiled))
    return rows


def main() -> None:
    # Two stable workload classes. Five percent of samples form a long tail.
    train = _observations(64, 256, [24] * 150 + [48] * 40 + [96] * 10)
    train += _observations(512, 1024, [96] * 150 + [192] * 40 + [448] * 10)
    profiles = build_ar_output_profiles(
        train,
        fingerprint=FINGERPRINT,
        min_samples=20,
    )
    store = ARProfileStore(profiles)

    with tempfile.TemporaryDirectory() as directory:
        profile_path = Path(directory) / "profiles.jsonl"
        store.write_jsonl(profile_path)
        loaded = ARProfileStore.read_jsonl(profile_path)
        loaded.require_fingerprint(FINGERPRINT)

    holdout = _evaluate(64, 256, [24] * 75 + [48] * 20 + [96] * 5, loaded)
    holdout += _evaluate(512, 1024, [96] * 75 + [192] * 20 + [448] * 5, loaded)
    count = len(holdout)
    worst_over = sum(worst - actual for actual, worst, _ in holdout) / count
    profile_over = sum(profile - actual for actual, _, profile in holdout) / count
    profile_coverage = sum(profile >= actual for actual, _, profile in holdout) / count
    worst_coverage = sum(worst >= actual for actual, worst, _ in holdout) / count

    observations = {
        "training_samples": len(train),
        "holdout_samples": count,
        "workload_classes": len(profiles),
        "profiles": [
            {
                "class": profile.workload_class,
                "samples": profile.sample_count,
                "p50": profile.p50_output_tokens,
                "p95": profile.p95_output_tokens,
                "p99": profile.p99_output_tokens,
            }
            for profile in profiles
        ],
        "worst_case_coverage": worst_coverage,
        "profile_p95_coverage": profile_coverage,
        "worst_case_mean_excess_blocks": worst_over,
        "profile_p95_mean_excess_blocks": profile_over,
        "excess_block_reduction": 1.0 - profile_over / worst_over,
    }
    checks = [
        h.Check(
            "profile JSONL round-trip preserves both workload classes",
            all(loaded.get(profile.workload_class) == profile for profile in profiles),
            f"classes={len(profiles)}",
        ),
        h.Check(
            "P95 profile covers at least 95% of deterministic holdout peaks",
            profile_coverage >= 0.95,
            f"coverage={profile_coverage:.3f}",
        ),
        h.Check(
            "P95 profile reduces excess KV blocks versus max_tokens worst-case",
            profile_over < worst_over,
            f"profile={profile_over:.2f}, worst={worst_over:.2f}",
        ),
        h.Check(
            "P95 profile cuts mean excess reservation by at least 50%",
            observations["excess_block_reduction"] >= 0.50,
            f"reduction={observations['excess_block_reduction']:.3f}",
        ),
    ]
    h.write_result(
        "H11",
        "AR workload-class P95 profile reduces worst-case reservation loss",
        "A versioned workload-class quantile profile preserves >=95% holdout "
        "peak coverage while reserving fewer KV blocks than max_tokens.",
        "All JSONL, coverage, and utilization-loss checks pass.",
        observations,
        checks,
        "This establishes correctness of the observation -> profile -> estimator "
        "pipeline on controlled distributions. It is not evidence of production "
        "accuracy; the same analysis must be rerun on live-server traces.",
    )
    print(f"H11 verdict: {'MATCHES' if all(check.passed for check in checks) else 'DEVIATION'}")
    for check in checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name} -- {check.detail}")


if __name__ == "__main__":
    main()
