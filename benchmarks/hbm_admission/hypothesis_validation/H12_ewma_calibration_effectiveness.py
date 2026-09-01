#!/usr/bin/env python3
"""H12 - online EWMA calibration corrects a systematically-underestimating
profile without needing a new profile deploy.

Hypothesis
----------
``OnlineCalibrator`` (Milestone 2 online correction) folds each completed
request's predicted-vs-observed KV peak into an EWMA ratio, isolated per
``(backend, workload_class, profile_version)``. When a workload-class profile
has learned too small an output-length quantile (e.g. the traffic mix shifted
after the profile was captured), repeated underprediction should raise the
correction factor above 1.0 and this should, without any config change, raise
the applied admission demand above what the *uncorrected* quantile estimate
alone would have reported.

Expectation
-----------
1. Before any observations, the calibrator reports a neutral 1.0 correction
   (cold start).
2. After several requests whose real KV usage exceeds the profiled quantile
   by a stable margin, ``snapshot().correction`` rises above 1.0.
3. Driving ``OmniSchedulerMixin`` end to end (real
   ``_dynamic_hbm_resource_admission_decision`` +
   ``_finish_resource_observation``) with such requests raises the applied
   ``required`` KV-block figure — the calibrated admission demand — above the
   raw ``quantile_peak_kv_blocks`` the estimator alone would have reported.
4. A workload class with no observations at all is unaffected by another
   class's correction (isolation).
5. Fallback (no-profile) requests do not get folded into the correction, so
   an always-fallback workload never learns a spurious correction.

Method
------
Drive the REAL ``OmniSchedulerMixin`` through the extended
``SchedulerHarness`` (as in H10/H11): admit/observe/finish a series of
requests carrying a fixed ``ARProfileStore`` quantile that undershoots their
actual KV usage, then read back the calibrator snapshot and the next
resource-admission decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

from vllm_omni.core.memory_coordinator import (  # noqa: E402
    AROutputLengthProfile,
    ARProfileStore,
    ProfileFingerprint,
)

BLOCK_SIZE = 16
FINGERPRINT = ProfileFingerprint(
    model_id="synthetic-ar",
    device_type="gpu-free",
    dtype="float16",
    tp_size=1,
    block_size=BLOCK_SIZE,
)
# prompt=64 -> "ar:p128:o..." bucket; max_tokens=64 -> "o64" bucket.
WORKLOAD_CLASS = "ar:p128:o64:s0"
# Profile quantile (32 output tokens) undershoots the real usage (96 tokens)
# injected below by a stable 3x margin - the "profile went stale" scenario.
UNDERSIZED_PROFILE = ARProfileStore(
    [
        AROutputLengthProfile(
            fingerprint=FINGERPRINT,
            workload_class=WORKLOAD_CLASS,
            sample_count=100,
            p50_output_tokens=16,
            p95_output_tokens=32,
            p99_output_tokens=32,
        )
    ]
)


def _make_harness(*, profile_store: ARProfileStore | None) -> h.SchedulerHarness:
    sch = h.SchedulerHarness(
        cap=16,
        running=0,
        block_size=BLOCK_SIZE,
        free_kv_blocks=10_000,
        config={"enabled": True, "resource_admission_mode": "shadow"},
        track_allocated_blocks=True,
    )
    if profile_store is not None:
        sch._s._dynamic_hbm_ar_profile_store = profile_store
        sch._s._dynamic_hbm_ar_profile_loaded = True
    return sch


# ceil((64 uncached prompt + 96 real output) / 16) = 10 real KV blocks -
# 3x the undersized profile's ceil((64 + 32) / 16) = 6 block quantile peak.
REAL_ALLOCATED_BLOCKS = 10


def _drive_one_underprediction(sch: h.SchedulerHarness, index: int) -> None:
    """Admit (shadow) a fresh (not-yet-generated) request, then observe+finish
    it once its real KV usage exceeds the profiled quantile peak by a stable
    margin - the "profile went stale" scenario the calibrator is meant to
    correct for. ``num_output_tokens`` must stay 0 at admission time: the
    estimator floors expected/quantile output at tokens already generated so
    far, so a nonzero value here would leak the "future" observed usage into
    the "current" admission-time prediction and defeat the scenario."""
    request_id = f"req-{index}"
    request = h.FakeWaitingRequest(
        num_prompt_tokens=64,
        max_tokens=64,
        request_id=request_id,
        num_output_tokens=0,
    )
    sch.set_waiting([request])
    sch.resource_admission_decision()  # begins the shadow observation
    sch.set_allocated_blocks(request_id, REAL_ALLOCATED_BLOCKS)
    request.num_output_tokens = 96  # now report the request's real completion
    sch.finish_resource_observation(request)


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. cold start is neutral ------------------------------------- #
    sch1 = _make_harness(profile_store=UNDERSIZED_PROFILE)
    cold = sch1.calibrator.snapshot(
        backend="ar", workload_class=WORKLOAD_CLASS, profile_version="ar-output-v1"
    )
    obs["cold_start"] = {"correction": cold.correction, "sample_count": cold.sample_count}
    checks.append(h.Check(
        "cold start: correction is neutral (1.0) before any observation",
        cold.correction == 1.0 and cold.sample_count == 0,
        f"{obs['cold_start']}",
    ))

    # ---- 2. repeated underprediction raises correction above 1.0 ------ #
    sch2 = _make_harness(profile_store=UNDERSIZED_PROFILE)
    baseline_decision = sch2.resource_admission_decision()
    baseline_required = baseline_decision.required
    for i in range(20):
        _drive_one_underprediction(sch2, i)
    warmed = sch2.calibrator.snapshot(
        backend="ar", workload_class=WORKLOAD_CLASS, profile_version="ar-output-v1"
    )
    obs["after_20_underpredictions"] = {
        "correction": warmed.correction,
        "sample_count": warmed.sample_count,
        "baseline_required_blocks": baseline_required,
    }
    checks.append(h.Check(
        "20 systematic underpredictions raise correction above 1.0",
        warmed.correction > 1.0,
        f"{obs['after_20_underpredictions']}",
    ))
    checks.append(h.Check(
        "calibrator accumulates one sample per finished observation",
        warmed.sample_count == 20,
        f"{obs['after_20_underpredictions']}",
    ))

    # ---- 3. the calibrated decision demands more than the raw quantile - #
    next_request = h.FakeWaitingRequest(num_prompt_tokens=64, max_tokens=64, request_id="next")
    sch2.set_waiting([next_request])
    calibrated_decision = sch2.resource_admission_decision()
    raw_quantile = calibrated_decision.estimate.logical.quantile_peak_kv_blocks
    obs["calibrated_vs_raw"] = {
        "raw_quantile_peak_kv_blocks": raw_quantile,
        "calibrated_required_blocks": calibrated_decision.required,
        "correction_applied": calibrated_decision.correction,
    }
    checks.append(h.Check(
        "calibrated admission demand exceeds the raw (uncorrected) quantile peak",
        calibrated_decision.required > raw_quantile,
        f"{obs['calibrated_vs_raw']}",
    ))

    # ---- 4. an unrelated workload class is unaffected (isolation) ----- #
    other_snapshot = sch2.calibrator.snapshot(
        backend="ar", workload_class="ar:p2048:o1024:s0", profile_version="ar-output-v1"
    )
    obs["unrelated_workload_class"] = {"correction": other_snapshot.correction}
    checks.append(h.Check(
        "an unrelated workload class keeps a neutral correction",
        other_snapshot.correction == 1.0,
        f"{obs['unrelated_workload_class']}",
    ))

    # ---- 5. fallback (no-profile) requests are not folded in ---------- #
    sch3 = _make_harness(profile_store=None)  # no profile -> every estimate falls back
    for i in range(20):
        _drive_one_underprediction(sch3, i)
    fallback_snapshot = sch3.calibrator.snapshot(
        backend="ar", workload_class=WORKLOAD_CLASS, profile_version=None
    )
    obs["fallback_only_traffic"] = {
        "correction": fallback_snapshot.correction,
        "sample_count": fallback_snapshot.sample_count,
    }
    checks.append(h.Check(
        "an always-fallback workload class never learns a correction",
        fallback_snapshot.correction == 1.0 and fallback_snapshot.sample_count == 0,
        f"{obs['fallback_only_traffic']}",
    ))

    verdict = "MATCHES" if all(c.passed for c in checks) else "DEVIATES"
    print(f"H12 verdict: {verdict}")
    for check in checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name} -- {check.detail}")

    h.write_result(
        "H12",
        "Online EWMA calibration corrects a systematically-underestimating profile",
        "A workload class whose profile quantile undershoots real KV usage by a "
        "stable margin should, after enough completed requests, have its "
        "calibrator correction rise above 1.0 and raise the applied admission "
        "demand above the raw quantile estimate — without redeploying a profile.",
        "Cold start is neutral; repeated underprediction raises correction above "
        "1.0 and the calibrated admission demand exceeds the raw quantile; "
        "unrelated workload classes and always-fallback traffic are unaffected.",
        obs,
        checks,
        "This establishes that OnlineCalibrator closes the gap M2b left open: a "
        "static profile that drifts stale is corrected online instead of always "
        "requiring a new profile deploy, while remaining inert for traffic the "
        "estimator has no confident basis to correct (unknown/fallback classes).",
    )


if __name__ == "__main__":
    main()
