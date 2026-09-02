# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator.diffusion_resource_estimator import (
    DiffusionBatchContext,
    DiffusionPhase,
    DiffusionPhaseEnvelopeProfile,
    DiffusionProfileStore,
    DiffusionResourceEstimator,
    DiffusionWorkloadClassifier,
)
from vllm_omni.core.memory_coordinator.resource_estimator import ResourceDimension
from vllm_omni.core.memory_coordinator.resource_profile import ProfileFingerprint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FINGERPRINT = ProfileFingerprint(
    model_id="wan2.2", device_type="A100", dtype="torch.bfloat16", tp_size=1, block_size=16
)


def test_context_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        DiffusionBatchContext(batch_size=0, latent_height=32, latent_width=32)


def test_context_rejects_non_positive_latent_dimensions() -> None:
    with pytest.raises(ValueError, match="latent dimensions must be positive"):
        DiffusionBatchContext(batch_size=1, latent_height=0, latent_width=32)


def test_context_rejects_non_positive_num_frames() -> None:
    with pytest.raises(ValueError, match="num_frames must be at least 1"):
        DiffusionBatchContext(batch_size=1, latent_height=32, latent_width=32, num_frames=0)


def test_context_latent_pixel_count_multiplies_frames() -> None:
    context = DiffusionBatchContext(batch_size=1, latent_height=8, latent_width=8, num_frames=16)
    assert context.latent_pixel_count == 8 * 8 * 16


def test_classifier_buckets_are_orderable_and_conservative() -> None:
    classifier = DiffusionWorkloadClassifier()
    assert classifier.bucket_key(batch_size=1, latent_pixel_count=32 * 32, num_frames=1) == (
        1,
        64 * 64,
        1,
    )
    # Anything past the largest declared boundary still buckets to *something*
    # orderable (never crashes, never silently caps at the largest bucket).
    huge_batch, huge_pixels, huge_frames = classifier.bucket_key(
        batch_size=10_000, latent_pixel_count=10**9, num_frames=10_000
    )
    assert huge_batch > classifier.BATCH_SIZE_BUCKETS[-1]
    assert huge_pixels > classifier.LATENT_PIXEL_BUCKETS[-1]
    assert huge_frames > classifier.FRAME_COUNT_BUCKETS[-1]


def test_classify_label_reflects_cfg_and_vae_mode() -> None:
    classifier = DiffusionWorkloadClassifier()
    a = classifier.classify(
        batch_size=2, latent_pixel_count=64 * 64, num_frames=1, cfg_parallel_size=1, vae_mode="tile"
    )
    b = classifier.classify(
        batch_size=2, latent_pixel_count=64 * 64, num_frames=1, cfg_parallel_size=2, vae_mode="tile"
    )
    c = classifier.classify(
        batch_size=2, latent_pixel_count=64 * 64, num_frames=1, cfg_parallel_size=1, vae_mode="spatial_shard_height"
    )
    assert a != b
    assert a != c


def test_estimate_without_profile_falls_back_to_hard_placeholder() -> None:
    estimator = DiffusionResourceEstimator()
    context = DiffusionBatchContext(batch_size=4, latent_height=32, latent_width=32)
    estimate = estimator.estimate_batch(context)
    assert estimate.provenance.fallback_reason == "unprofiled_placeholder_constant"
    assert estimate.provenance.profile_version is None
    assert estimate.physical.quantile_transient_peak_bytes == estimate.physical.hard_transient_peak_bytes
    # KV blocks are structurally not applicable to this backend.
    assert estimate.logical.immediate_kv_blocks == 0
    assert ResourceDimension.KV_BLOCKS not in estimate.physical.available_dimensions
    assert ResourceDimension.TRANSIENT_BYTES in estimate.physical.available_dimensions
    assert ResourceDimension.PERSISTENT_BYTES in estimate.physical.available_dimensions


def test_estimate_alias_matches_estimate_batch() -> None:
    estimator = DiffusionResourceEstimator()
    context = DiffusionBatchContext(batch_size=2, latent_height=16, latent_width=16)
    assert estimator.estimate(context) == estimator.estimate_batch(context)


def test_hard_bound_scales_with_batch_size_and_frames() -> None:
    estimator = DiffusionResourceEstimator()
    small = estimator.estimate_batch(DiffusionBatchContext(batch_size=1, latent_height=32, latent_width=32))
    bigger_batch = estimator.estimate_batch(
        DiffusionBatchContext(batch_size=2, latent_height=32, latent_width=32)
    )
    more_frames = estimator.estimate_batch(
        DiffusionBatchContext(batch_size=1, latent_height=32, latent_width=32, num_frames=16)
    )
    assert bigger_batch.physical.hard_transient_peak_bytes > small.physical.hard_transient_peak_bytes
    assert more_frames.physical.hard_transient_peak_bytes > small.physical.hard_transient_peak_bytes


def test_hard_bound_scales_with_cfg_parallel_size() -> None:
    estimator = DiffusionResourceEstimator()
    base = estimator.estimate_batch(DiffusionBatchContext(batch_size=1, latent_height=32, latent_width=32))
    cfg2 = estimator.estimate_batch(
        DiffusionBatchContext(batch_size=1, latent_height=32, latent_width=32, cfg_parallel_size=2)
    )
    assert cfg2.physical.hard_transient_peak_bytes == 2 * base.physical.hard_transient_peak_bytes


def test_envelope_peak_is_max_across_phases_not_sum() -> None:
    profile = DiffusionPhaseEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=8,
        latent_pixel_bucket=64 * 64,
        frame_count_bucket=1,
        sample_count=10,
        phase_peak_bytes={
            DiffusionPhase.CONDITION_ENCODING: 100,
            DiffusionPhase.DENOISING: 900,
            DiffusionPhase.VAE_DECODE: 300,
        },
    )
    assert profile.envelope_peak_bytes == 900
    assert profile.envelope_peak_bytes < sum(profile.phase_peak_bytes.values())


def test_profile_rejects_empty_phase_measurements() -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        DiffusionPhaseEnvelopeProfile(
            fingerprint=_FINGERPRINT,
            batch_size_bucket=1,
            latent_pixel_bucket=1,
            frame_count_bucket=1,
            sample_count=1,
            phase_peak_bytes={},
        )


def test_profile_rejects_negative_phase_bytes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DiffusionPhaseEnvelopeProfile(
            fingerprint=_FINGERPRINT,
            batch_size_bucket=1,
            latent_pixel_bucket=1,
            frame_count_bucket=1,
            sample_count=1,
            phase_peak_bytes={DiffusionPhase.DENOISING: -1},
        )


def test_estimate_uses_ceiling_matched_profile_when_available() -> None:
    profile = DiffusionPhaseEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=8,
        latent_pixel_bucket=128 * 128,
        frame_count_bucket=1,
        sample_count=30,
        phase_peak_bytes={DiffusionPhase.DENOISING: 1_000_000},
        persistent_bytes_per_request=500,
    )
    store = DiffusionProfileStore([profile])
    estimator = DiffusionResourceEstimator()
    context = DiffusionBatchContext(batch_size=3, latent_height=100, latent_width=100)
    estimate = estimator.estimate_batch(context, store)
    assert estimate.provenance.fallback_reason is None
    assert estimate.provenance.profile_version == "diffusion-envelope-v1"
    assert estimate.provenance.sample_count == 30
    assert estimate.physical.quantile_transient_peak_bytes == 1_000_000
    assert estimate.physical.persistent_bytes == 500


def test_profile_ceiling_never_rounds_down() -> None:
    # Only a bucket smaller than the request exists -- it must not be used;
    # the estimator must fall back to the conservative hard bound instead.
    profile = DiffusionPhaseEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=1,
        latent_pixel_bucket=64 * 64,
        frame_count_bucket=1,
        sample_count=10,
        phase_peak_bytes={DiffusionPhase.DENOISING: 1},
    )
    store = DiffusionProfileStore([profile])
    estimator = DiffusionResourceEstimator()
    context = DiffusionBatchContext(batch_size=8, latent_height=512, latent_width=512)
    estimate = estimator.estimate_batch(context, store)
    assert estimate.provenance.fallback_reason == "unprofiled_placeholder_constant"
    assert estimate.physical.quantile_transient_peak_bytes > 1


def test_profile_estimate_is_capped_at_hard_bound() -> None:
    # A profile whose observed peak somehow exceeds this request's own hard
    # analytical bound must never be trusted past that bound (mirrors
    # evaluate_ar_kv_admission's/Code2Wav's min(calibrated, hard) safety cap).
    profile = DiffusionPhaseEnvelopeProfile(
        fingerprint=_FINGERPRINT,
        batch_size_bucket=1,
        latent_pixel_bucket=64 * 64,
        frame_count_bucket=1,
        sample_count=10,
        phase_peak_bytes={DiffusionPhase.DENOISING: 10**18},
    )
    store = DiffusionProfileStore([profile])
    estimator = DiffusionResourceEstimator()
    context = DiffusionBatchContext(batch_size=1, latent_height=1, latent_width=1)
    estimate = estimator.estimate_batch(context, store)
    assert estimate.physical.quantile_transient_peak_bytes == estimate.physical.hard_transient_peak_bytes
    assert estimate.physical.quantile_transient_peak_bytes < 10**18


def test_profile_store_returns_none_with_no_profiles() -> None:
    store = DiffusionProfileStore()
    assert store.get_ceiling(batch_size_bucket=1, latent_pixel_bucket=1, frame_count_bucket=1) is None


def test_marginal_estimate_delta_is_non_negative_and_grows_batch() -> None:
    estimator = DiffusionResourceEstimator()
    existing_batch = DiffusionBatchContext(batch_size=2, latent_height=32, latent_width=32)
    marginal = estimator.estimate_marginal(
        request_latent_pixel_count=32 * 32,
        request_num_frames=1,
        candidate_batch=existing_batch,
    )
    assert marginal.delta_transient_bytes >= 0
    assert marginal.with_request.physical.quantile_transient_peak_bytes >= (
        marginal.without_request.physical.quantile_transient_peak_bytes
    )


def test_marginal_estimate_grows_when_new_request_has_larger_latent() -> None:
    estimator = DiffusionResourceEstimator()
    existing_batch = DiffusionBatchContext(batch_size=1, latent_height=16, latent_width=16)
    small_addition = estimator.estimate_marginal(
        request_latent_pixel_count=16 * 16,
        request_num_frames=1,
        candidate_batch=existing_batch,
    )
    large_addition = estimator.estimate_marginal(
        request_latent_pixel_count=128 * 128,
        request_num_frames=1,
        candidate_batch=existing_batch,
    )
    # Adding a request with a much larger latent shape than the existing
    # batch must cost strictly more than adding a same-shape request, since
    # every batch member gets padded to the batch's largest member.
    assert large_addition.delta_transient_bytes > small_addition.delta_transient_bytes


def test_denoising_steps_is_not_a_classifier_dimension() -> None:
    """M2 design doc S9.1: denoising step count must not be assumed to
    affect peak HBM without evidence, so it is deliberately absent from both
    the workload-class key and DiffusionBatchContext entirely."""
    import inspect

    from vllm_omni.core.memory_coordinator.diffusion_resource_estimator import (
        DiffusionBatchContext as _Ctx,
    )

    fields = inspect.signature(_Ctx).parameters
    assert not any("step" in name for name in fields)
