# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import (
    ARRequestResourceContext,
    ARResourceEstimator,
    OnlineCalibrator,
    ResourceObservationCollector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _profiled_estimate(*, workload_class: str, profile_version: str):
    """A quantile estimate with a profile behind it (no fallback_reason) —
    the only kind of estimate the calibrator is meant to learn from."""
    return ARResourceEstimator(16).estimate(
        ARRequestResourceContext(
            num_prompt_tokens=64,
            max_tokens=256,
            block_size=16,
            expected_output_tokens=64,
            quantile_output_tokens=128,
            workload_class=workload_class,
            profile_version=profile_version,
            sample_count=100,
        )
    )


def _observation(*, estimate, observed_allocated_blocks, workload_class="ar:p128:o64:s0", profile_version="v1"):
    collector = ResourceObservationCollector()
    collector.begin("r", estimate)
    collector.observe("r", allocated_kv_blocks=observed_allocated_blocks, output_tokens=1)
    observation = collector.finish("r")
    assert observation is not None
    return observation.__class__(
        **{**observation.__dict__, "workload_class": workload_class, "profile_version": profile_version}
    )


def test_cold_start_returns_neutral_correction() -> None:
    calibrator = OnlineCalibrator(min_samples=5)
    snapshot = calibrator.snapshot(backend="ar", workload_class="ar:p128:o64:s0", profile_version="v1")
    assert snapshot.correction == 1.0
    assert snapshot.sample_count == 0
    assert not snapshot.stale


def test_underprediction_raises_correction_above_one() -> None:
    calibrator = OnlineCalibrator(alpha=0.5, min_samples=1)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    observation = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 2,
    )
    for _ in range(10):
        calibrator.update(observation)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=observation.workload_class, profile_version="v1"
    )
    assert snapshot.correction > 1.0
    assert snapshot.sample_count == 10


def test_overprediction_lowers_correction_below_one() -> None:
    calibrator = OnlineCalibrator(alpha=0.5, min_samples=1)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    observation = _observation(
        estimate=estimate,
        observed_allocated_blocks=max(1, estimate.logical.quantile_peak_kv_blocks // 4),
    )
    for _ in range(10):
        calibrator.update(observation)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=observation.workload_class, profile_version="v1"
    )
    assert snapshot.correction < 1.0


def test_correction_is_clamped() -> None:
    calibrator = OnlineCalibrator(alpha=1.0, min_correction=0.5, max_correction=2.0, min_samples=1)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    huge_underprediction = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 100,
    )
    calibrator.update(huge_underprediction)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=huge_underprediction.workload_class, profile_version="v1"
    )
    assert snapshot.correction == 2.0


def test_below_min_samples_stays_neutral() -> None:
    calibrator = OnlineCalibrator(alpha=1.0, min_samples=5)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    observation = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 3,
    )
    for _ in range(4):
        calibrator.update(observation)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=observation.workload_class, profile_version="v1"
    )
    assert snapshot.correction == 1.0
    assert snapshot.sample_count == 4


def test_stale_state_falls_back_to_neutral() -> None:
    calibrator = OnlineCalibrator(alpha=1.0, min_samples=1, stale_after_s=0.01)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    observation = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 3,
    )
    calibrator.update(observation)
    import time

    time.sleep(0.02)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=observation.workload_class, profile_version="v1"
    )
    assert snapshot.correction == 1.0
    assert snapshot.stale


def test_profile_version_change_starts_fresh_state() -> None:
    calibrator = OnlineCalibrator(alpha=1.0, min_samples=1)
    estimate = _profiled_estimate(workload_class="ar:p128:o64:s0", profile_version="v1")
    observation_v1 = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 3,
        profile_version="v1",
    )
    calibrator.update(observation_v1)
    assert calibrator.snapshot(
        backend="ar", workload_class=observation_v1.workload_class, profile_version="v1"
    ).correction > 1.0
    # A different profile_version must not inherit v1's learned correction.
    fresh = calibrator.snapshot(
        backend="ar", workload_class=observation_v1.workload_class, profile_version="v2"
    )
    assert fresh.correction == 1.0
    assert fresh.sample_count == 0


def test_fallback_observations_are_not_folded_in() -> None:
    calibrator = OnlineCalibrator(alpha=1.0, min_samples=1)
    # No output-length profile => fallback_reason is set on the estimate/observation.
    estimate = ARResourceEstimator(16).estimate(num_prompt_tokens=64, max_tokens=256)
    observation = _observation(
        estimate=estimate,
        observed_allocated_blocks=estimate.logical.quantile_peak_kv_blocks * 5,
    )
    assert observation.fallback_reason == "output_length_profile_unavailable"
    calibrator.update(observation)
    snapshot = calibrator.snapshot(
        backend="ar", workload_class=observation.workload_class, profile_version="v1"
    )
    assert snapshot.correction == 1.0
    assert snapshot.sample_count == 0


def test_invalid_constructor_args_are_rejected() -> None:
    with pytest.raises(ValueError):
        OnlineCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        OnlineCalibrator(min_correction=1.5, max_correction=2.0)
    with pytest.raises(ValueError):
        OnlineCalibrator(min_samples=0)
    with pytest.raises(ValueError):
        OnlineCalibrator(stale_after_s=0)
