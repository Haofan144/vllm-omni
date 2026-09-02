# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator.code2wav_resource_estimator import (
    Code2WavEnvelopeProfile,
    Code2WavObservation,
    Code2WavProfileStore,
    Code2WavRequestContext,
    Code2WavResourceEstimator,
    Code2WavWorkloadClassifier,
    build_code2wav_envelope_profiles,
)
from vllm_omni.core.memory_coordinator.resource_estimator import ResourceDimension
from vllm_omni.core.memory_coordinator.resource_profile import ProfileFingerprint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FINGERPRINT = ProfileFingerprint(
    model_id="qwen3-tts", device_type="A100", dtype="torch.bfloat16", tp_size=1, block_size=16
)


def test_context_rejects_frame_count_above_batch_max() -> None:
    with pytest.raises(ValueError, match="cannot exceed batch_max_frame_count"):
        Code2WavRequestContext(frame_count=100, batch_size=2, batch_max_frame_count=50)


def test_context_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        Code2WavRequestContext(frame_count=10, batch_size=0, batch_max_frame_count=10)


def test_classifier_buckets_are_orderable_and_conservative() -> None:
    classifier = Code2WavWorkloadClassifier()
    assert classifier.bucket_key(batch_size=1, frame_count=25) == (1, 25)
    assert classifier.bucket_key(batch_size=3, frame_count=30) == (4, 50)
    # Anything past the largest declared boundary still buckets to *something*
    # orderable (never crashes, never silently caps at the largest bucket).
    huge_batch, huge_frames = classifier.bucket_key(batch_size=10_000, frame_count=10_000)
    assert huge_batch > classifier.BATCH_SIZE_BUCKETS[-1]
    assert huge_frames > classifier.FRAME_COUNT_BUCKETS[-1]


def test_classify_label_includes_persistent_state_flag() -> None:
    classifier = Code2WavWorkloadClassifier()
    streaming = classifier.classify(batch_size=4, frame_count=25, persistent_state_active=True)
    non_streaming = classifier.classify(batch_size=4, frame_count=25, persistent_state_active=False)
    assert streaming != non_streaming


def test_estimate_without_profile_falls_back_to_hard_placeholder() -> None:
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(
        frame_count=25, batch_size=4, batch_max_frame_count=25, persistent_state_active=True
    )
    estimate = estimator.estimate(context)
    assert estimate.provenance.fallback_reason == "unprofiled_placeholder_constant"
    assert estimate.provenance.profile_version is None
    assert estimate.physical.quantile_transient_peak_bytes == estimate.physical.hard_transient_peak_bytes
    assert estimate.physical.persistent_bytes > 0
    # KV blocks are structurally not applicable to this backend.
    assert estimate.logical.immediate_kv_blocks == 0
    assert ResourceDimension.KV_BLOCKS not in estimate.physical.available_dimensions
    assert ResourceDimension.TRANSIENT_BYTES in estimate.physical.available_dimensions
    assert ResourceDimension.PERSISTENT_BYTES in estimate.physical.available_dimensions


def test_estimate_without_persistent_state_has_zero_persistent_bytes() -> None:
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(
        frame_count=25, batch_size=4, batch_max_frame_count=25, persistent_state_active=False
    )
    estimate = estimator.estimate(context)
    assert estimate.physical.persistent_bytes == 0


def test_estimate_uses_ceiling_matched_profile_when_available() -> None:
    profile = Code2WavEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=8,
        frame_count_bucket=50,
        sample_count=30,
        peak_transient_bytes=1_000_000,
        persistent_bytes_per_request=500_000,
    )
    store = Code2WavProfileStore([profile])
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(
        frame_count=25, batch_size=3, batch_max_frame_count=30, persistent_state_active=True
    )
    estimate = estimator.estimate(context, store)
    assert estimate.provenance.fallback_reason is None
    assert estimate.provenance.profile_version == "code2wav-envelope-v1"
    assert estimate.provenance.sample_count == 30
    assert estimate.physical.quantile_transient_peak_bytes == 1_000_000
    assert estimate.physical.persistent_bytes == 500_000


def test_profile_ceiling_never_rounds_down() -> None:
    # Only a bucket smaller than the request exists -- it must not be used;
    # the estimator must fall back to the conservative hard bound instead.
    profile = Code2WavEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=1,
        frame_count_bucket=25,
        sample_count=10,
        peak_transient_bytes=1,
        persistent_bytes_per_request=1,
    )
    store = Code2WavProfileStore([profile])
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(frame_count=100, batch_size=8, batch_max_frame_count=100)
    estimate = estimator.estimate(context, store)
    assert estimate.provenance.fallback_reason == "unprofiled_placeholder_constant"
    assert estimate.physical.quantile_transient_peak_bytes > 1


def test_profile_estimate_is_capped_at_hard_bound() -> None:
    # A profile whose observed peak somehow exceeds this request's own hard
    # analytical bound must never be trusted past that bound (mirrors
    # evaluate_ar_kv_admission's calibration safety cap).
    profile = Code2WavEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=1,
        frame_count_bucket=25,
        sample_count=10,
        peak_transient_bytes=10**15,
        persistent_bytes_per_request=0,
    )
    store = Code2WavProfileStore([profile])
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(frame_count=1, batch_size=1, batch_max_frame_count=1)
    estimate = estimator.estimate(context, store)
    assert estimate.physical.quantile_transient_peak_bytes == estimate.physical.hard_transient_peak_bytes
    assert estimate.physical.quantile_transient_peak_bytes < 10**15


def test_profile_store_returns_none_with_no_profiles() -> None:
    store = Code2WavProfileStore()
    assert store.get_ceiling(batch_size_bucket=1, frame_count_bucket=25) is None


def test_observation_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="batch fields are invalid"):
        Code2WavObservation(
            batch_size=0, batch_max_frame_count=10, persistent_state_active=False, peak_transient_bytes=1
        )
    with pytest.raises(ValueError, match="non-negative"):
        Code2WavObservation(
            batch_size=1, batch_max_frame_count=10, persistent_state_active=False, peak_transient_bytes=-1
        )


def test_build_envelope_profiles_takes_max_peak_per_bucket() -> None:
    observations = [
        Code2WavObservation(
            batch_size=3, batch_max_frame_count=20, persistent_state_active=False, peak_transient_bytes=500
        ),
        Code2WavObservation(
            batch_size=4, batch_max_frame_count=24, persistent_state_active=False, peak_transient_bytes=900
        ),
        Code2WavObservation(
            batch_size=3, batch_max_frame_count=10, persistent_state_active=False, peak_transient_bytes=100
        ),
    ]
    profiles = build_code2wav_envelope_profiles(observations, fingerprint=_FINGERPRINT, min_samples=1)
    # All three observations bucket to (batch<=4, frame<=25) -- the profile's
    # peak must be the MAX observed peak in that bucket, not a mean, since an
    # empirical peak is a lower bound on the true worst case.
    assert len(profiles) == 1
    assert profiles[0].batch_size_bucket == 4
    assert profiles[0].frame_count_bucket == 25
    assert profiles[0].peak_transient_bytes == 900
    assert profiles[0].sample_count == 3


def test_build_envelope_profiles_respects_min_samples() -> None:
    observations = [
        Code2WavObservation(
            batch_size=1, batch_max_frame_count=10, persistent_state_active=False, peak_transient_bytes=100
        )
    ]
    profiles = build_code2wav_envelope_profiles(observations, fingerprint=_FINGERPRINT, min_samples=2)
    assert profiles == []


def test_built_profile_feeds_estimator_and_replaces_placeholder() -> None:
    observations = [
        Code2WavObservation(
            batch_size=4, batch_max_frame_count=25, persistent_state_active=False, peak_transient_bytes=777
        )
        for _ in range(5)
    ]
    profiles = build_code2wav_envelope_profiles(observations, fingerprint=_FINGERPRINT, min_samples=1)
    store = Code2WavProfileStore(profiles)
    estimator = Code2WavResourceEstimator()
    context = Code2WavRequestContext(frame_count=20, batch_size=3, batch_max_frame_count=25)
    estimate = estimator.estimate(context, store)
    assert estimate.provenance.fallback_reason is None
    assert estimate.physical.quantile_transient_peak_bytes == 777
