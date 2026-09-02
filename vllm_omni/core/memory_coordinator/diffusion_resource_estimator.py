# SPDX-License-Identifier: Apache-2.0
"""Diffusion resource model (Milestone 2d).

A ``StageExecutionType.DIFFUSION`` stage has no per-token KV-block accounting
at all (see the M2 design doc section 9): its natural unit of scheduling is a
*batch* of same-stage requests run through a fixed number of denoising steps,
not a per-request autoregressive loop. Peak HBM for that batch is a function
of the batch's shape -- resolution/latent size, frame count (video vs. image),
batch size, CFG mode, parallelism, VAE mode -- not of any single request in
isolation. This mirrors, and deliberately reuses, the same batch-envelope
pattern already built for Code2Wav
(``code2wav_resource_estimator.Code2WavProfileStore.get_ceiling``): profile a
discrete grid of workload buckets and, for any unprofiled combination, round
UP to the nearest profiled bucket on every dimension rather than linearly
extrapolating (M2 design doc S9.2 explicitly forbids linear extrapolation
here, since larger latents / more frames do not cost proportionally more in
practice once attention and VAE tiling behavior are accounted for).

Two things make Diffusion's estimator shape different from Code2Wav's,
though, and are why this is a separate module rather than a thin subclass:

* The design doc's primary Diffusion interface (S9.2) is not
  ``estimate(context)`` for one request but ``estimate_batch(batch_context)``
  for the whole candidate batch, plus ``estimate_marginal(request, batch)``
  for the incremental cost of adding one more request to an existing batch.
  Diffusion stages batch multiple independent generation requests into one
  forward pass (unlike Code2Wav, which pads one already-decided batch); the
  marginal-cost question ("does admitting this one additional request still
  fit?") is the one the scheduler actually needs to answer.
* Diffusion's peak is not one working-set number but the envelope (max) of
  several sequential *phases* -- text/condition encoding, denoising, VAE
  encode (image-to-video/editing), VAE decode, post-processing (S9.3) -- each
  with its own peak, not their sum: the phases run sequentially within one
  request's pipeline and do not hold their peaks concurrently.

``denoising_steps`` is deliberately NOT a bucket dimension (S9.1: "Denoising
steps 主要影响执行时间, 应通过实验确定其是否影响 peak HBM; 不能默认线性影响
内存") -- it affects wall-clock, not the peak activation/workspace size,
because every step reuses the same latent buffers. If real profiling later
shows step count does perturb peak HBM for some backend, add it as an
explicit bucket dimension then rather than assuming it a priori.

Placeholder cost constants
---------------------------
Exactly like ``code2wav_resource_estimator.py``, the per-phase byte-cost
constants below are NOT measured. They exist so the batch-envelope
ceiling-lookup structure, marginal-cost estimate, and phase-aware fallback
hierarchy are complete and testable now; every estimate this module produces
carries a ``fallback_reason`` of ``"unprofiled_placeholder_constant"`` until
real ``DiffusionPhaseEnvelopeProfile`` entries (built from actual GPU traces)
replace them. Do not use any absolute byte number from this module to size a
real deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vllm_omni.core.memory_coordinator.resource_estimator import (
    EstimateProvenance,
    LogicalResourceDemand,
    PhysicalResourceDemand,
    RequestResourceEstimate,
    ResourceDimension,
)
from vllm_omni.core.memory_coordinator.resource_profile import ProfileFingerprint

_UNPROFILED_FALLBACK_REASON = "unprofiled_placeholder_constant"


class DiffusionPhase(StrEnum):
    """Sequential pipeline phases whose peaks form an envelope, not a sum."""

    CONDITION_ENCODING = "condition_encoding"
    DENOISING = "denoising"
    VAE_ENCODE = "vae_encode"
    VAE_DECODE = "vae_decode"
    POSTPROCESSING = "postprocessing"


# PLACEHOLDER: not measured. A deliberately round, conservative-leaning guess
# for the per-latent-pixel, per-batch-slot transient (activation/workspace)
# cost of one phase's forward pass, in bytes. See module docstring. Denoising
# is weighted highest since it is the phase that holds attention activations
# across the full latent sequence; VAE encode/decode operate on tiled
# spatial windows in most backends and so cost less per latent pixel.
_PLACEHOLDER_BYTES_PER_LATENT_PIXEL_PER_BATCH_SLOT: dict[DiffusionPhase, int] = {
    DiffusionPhase.CONDITION_ENCODING: 256,
    DiffusionPhase.DENOISING: 4096,
    DiffusionPhase.VAE_ENCODE: 1024,
    DiffusionPhase.VAE_DECODE: 1024,
    DiffusionPhase.POSTPROCESSING: 64,
}
# PLACEHOLDER: not measured. Persistent per-request state (e.g. cached
# condition embeddings held across a multi-request CFG-parallel batch) is
# assumed negligible for a first version -- diffusion requests are typically
# stateless across calls, unlike Code2Wav's streaming overlap buffers.
_PLACEHOLDER_PERSISTENT_BYTES_PER_REQUEST = 0


@dataclass(frozen=True)
class DiffusionBatchContext:
    """Estimator input describing one candidate batch's shape.

    ``latent_height``/``latent_width`` are the denoising-loop latent grid
    (already downsampled by the VAE's spatial compression factor), not pixel
    resolution -- this is the shape that actually drives attention and
    activation memory. ``num_frames`` is 1 for image generation/editing and
    the frame count for video.
    """

    batch_size: int
    latent_height: int
    latent_width: int
    num_frames: int = 1
    cfg_parallel_size: int = 1
    workload_class: str = "unknown"
    target_coverage: float = 0.95
    profile_version: str | None = None
    sample_count: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("DiffusionBatchContext batch_size must be at least 1")
        if self.latent_height < 1 or self.latent_width < 1:
            raise ValueError("DiffusionBatchContext latent dimensions must be positive")
        if self.num_frames < 1:
            raise ValueError("DiffusionBatchContext num_frames must be at least 1")
        if self.cfg_parallel_size < 1:
            raise ValueError("DiffusionBatchContext cfg_parallel_size must be at least 1")

    @property
    def latent_pixel_count(self) -> int:
        """Per-frame latent pixel count times frame count -- the natural unit
        the placeholder per-phase costs above are expressed in."""
        return self.latent_height * self.latent_width * self.num_frames


class DiffusionWorkloadClassifier:
    """Buckets on batch/latent shape, not prompt/output tokens -- a Diffusion
    request has no AR-style prompt/max_tokens notion (M2 design doc S9.1:
    model/resolution/frame-count/batch-size/dtype/CFG-mode/parallelism/
    VAE-mode). ``denoising_steps`` is intentionally excluded; see module
    docstring."""

    BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32)
    LATENT_PIXEL_BUCKETS = (
        64 * 64,
        128 * 128,
        256 * 256,
        512 * 512,
        1024 * 1024,
        2048 * 2048,
    )
    FRAME_COUNT_BUCKETS = (1, 16, 49, 81, 121, 161)

    @staticmethod
    def _bucket(value: int, boundaries: tuple[int, ...]) -> int:
        for boundary in boundaries:
            if value <= boundary:
                return boundary
        return boundaries[-1] * 1_000_000  # effectively "overflow", still orderable

    def bucket_key(
        self, *, batch_size: int, latent_pixel_count: int, num_frames: int
    ) -> tuple[int, int, int]:
        """The numeric (batch_bucket, latent_pixel_bucket, frame_bucket)
        triple, for the profile store's ceiling search -- string
        workload_class keys aren't orderable, so the store needs the
        underlying numeric buckets, exactly as Code2WavWorkloadClassifier
        does for (batch, frame)."""
        return (
            self._bucket(batch_size, self.BATCH_SIZE_BUCKETS),
            self._bucket(latent_pixel_count, self.LATENT_PIXEL_BUCKETS),
            self._bucket(num_frames, self.FRAME_COUNT_BUCKETS),
        )

    def classify(
        self,
        *,
        batch_size: int,
        latent_pixel_count: int,
        num_frames: int,
        cfg_parallel_size: int = 1,
        vae_mode: str | None = None,
        **_backend_specific: Any,
    ) -> str:
        if batch_size < 1 or latent_pixel_count < 1 or num_frames < 1:
            raise ValueError("workload classifier inputs are invalid")
        batch_bucket, pixel_bucket, frame_bucket = self.bucket_key(
            batch_size=batch_size,
            latent_pixel_count=latent_pixel_count,
            num_frames=num_frames,
        )
        resolved_vae_mode = vae_mode or "default"
        return (
            f"diffusion:b{batch_bucket}:px{pixel_bucket}:f{frame_bucket}"
            f":cfg{cfg_parallel_size}:vae{resolved_vae_mode}"
        )


@dataclass(frozen=True)
class DiffusionPhaseEnvelopeProfile:
    """One profiled (batch_size_bucket, latent_pixel_bucket, frame_bucket)
    entry's per-phase peaks.

    ``phase_peak_bytes`` holds one measured transient peak per
    ``DiffusionPhase`` that was actually exercised for this bucket (a
    text-to-image pipeline, say, never runs ``VAE_ENCODE``); a phase absent
    from the mapping is unmeasured for this bucket, not zero-cost.
    ``persistent_bytes_per_request`` mirrors Code2Wav's field for API
    symmetry, even though it defaults to (and today is expected to stay)
    negligible for Diffusion -- see module docstring.
    """

    fingerprint: ProfileFingerprint
    batch_size_bucket: int
    latent_pixel_bucket: int
    frame_count_bucket: int
    sample_count: int
    phase_peak_bytes: dict[DiffusionPhase, int]
    persistent_bytes_per_request: int = 0
    profile_version: str = "diffusion-envelope-v1"

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("profile sample_count must be positive")
        if self.batch_size_bucket < 1 or self.latent_pixel_bucket < 1 or self.frame_count_bucket < 1:
            raise ValueError("profile bucket boundaries must be positive")
        if self.persistent_bytes_per_request < 0:
            raise ValueError("profile persistent_bytes_per_request must be non-negative")
        if not self.phase_peak_bytes:
            raise ValueError("profile must measure at least one phase")
        if any(value < 0 for value in self.phase_peak_bytes.values()):
            raise ValueError("profile phase peak bytes must be non-negative")

    @property
    def envelope_peak_bytes(self) -> int:
        """The batch's overall transient peak: the MAX across phases, not
        their sum (M2 design doc S9.3 -- phases run sequentially and do not
        hold peaks concurrently)."""
        return max(self.phase_peak_bytes.values())


class DiffusionProfileStore:
    """Bucket-ceiling lookup, identical in spirit to
    ``Code2WavProfileStore.get_ceiling`` but over a 3-tuple (batch, latent
    pixels, frames) key: an unprofiled combination rounds UP to the smallest
    profiled bucket that is >= it on every dimension (never down, never
    linearly interpolated/extrapolated). Returns ``None`` when no profiled
    bucket covers the request at all -- callers must fall back to the hard
    analytical bound, not assume zero cost.
    """

    def __init__(self, profiles: list[DiffusionPhaseEnvelopeProfile] | None = None) -> None:
        self._profiles: dict[tuple[int, int, int], DiffusionPhaseEnvelopeProfile] = {
            (profile.batch_size_bucket, profile.latent_pixel_bucket, profile.frame_count_bucket): profile
            for profile in (profiles or [])
        }

    def get_ceiling(
        self,
        *,
        batch_size_bucket: int,
        latent_pixel_bucket: int,
        frame_count_bucket: int,
    ) -> DiffusionPhaseEnvelopeProfile | None:
        candidates = [
            profile
            for profile in self._profiles.values()
            if profile.batch_size_bucket >= batch_size_bucket
            and profile.latent_pixel_bucket >= latent_pixel_bucket
            and profile.frame_count_bucket >= frame_count_bucket
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda p: (p.batch_size_bucket, p.latent_pixel_bucket, p.frame_count_bucket),
        )


@dataclass(frozen=True)
class MarginalResourceEstimate:
    """The incremental cost of adding one more request to an existing
    candidate batch (M2 design doc S9.2: ``ΔPeak(r,B) = Envelope(B∪{r}) -
    Envelope(B)``). ``without_request``/``with_request`` are the full-batch
    estimates before and after; ``delta_transient_bytes`` is their
    difference, floored at zero since a same-shape-or-smaller batch can never
    legitimately cost less by adding a request."""

    without_request: RequestResourceEstimate
    with_request: RequestResourceEstimate
    delta_transient_bytes: int


class DiffusionResourceEstimator:
    """Phase-aware batch-envelope estimator for a Diffusion stage.

    Deliberately does not produce any ``LogicalResourceDemand`` (KV blocks
    are structurally not applicable -- Diffusion has no autoregressive
    token loop); ``available_dimensions`` marks only the physical dimensions
    this estimator actually models. Mirrors
    ``Code2WavResourceEstimator``'s shape but the primary entry point is
    ``estimate_batch`` (the design doc's ``estimate_batch(batch_context)``),
    with per-request ``estimate``/``estimate_marginal`` built on top of it.
    """

    VERSION = "diffusion-transient-v1"

    def __init__(self, classifier: DiffusionWorkloadClassifier | None = None) -> None:
        self.classifier = classifier or DiffusionWorkloadClassifier()

    def _hard_phase_peak_bytes(self, context: DiffusionBatchContext, phase: DiffusionPhase) -> int:
        per_pixel_cost = _PLACEHOLDER_BYTES_PER_LATENT_PIXEL_PER_BATCH_SLOT[phase]
        return math.ceil(
            context.batch_size * context.latent_pixel_count * per_pixel_cost * context.cfg_parallel_size
        )

    def _hard_envelope_bytes(self, context: DiffusionBatchContext) -> int:
        return max(self._hard_phase_peak_bytes(context, phase) for phase in DiffusionPhase)

    def estimate_batch(
        self,
        context: DiffusionBatchContext,
        profile_store: DiffusionProfileStore | None = None,
    ) -> RequestResourceEstimate:
        """Estimate the whole candidate batch's resource demand -- the M2
        design doc's primary Diffusion interface. A per-request ``estimate``
        is available below for callers that need the
        ``RequestResourceEstimator`` protocol shape, but for Diffusion it
        simply forwards to this method: there is no meaningful
        single-request cost independent of the batch it joins."""
        batch_bucket, pixel_bucket, frame_bucket = self.classifier.bucket_key(
            batch_size=context.batch_size,
            latent_pixel_count=context.latent_pixel_count,
            num_frames=context.num_frames,
        )
        profile = (
            profile_store.get_ceiling(
                batch_size_bucket=batch_bucket,
                latent_pixel_bucket=pixel_bucket,
                frame_count_bucket=frame_bucket,
            )
            if profile_store is not None
            else None
        )

        hard_transient_bytes = self._hard_envelope_bytes(context)
        if profile is not None:
            # A profile is itself an empirical peak observation, so it is
            # already a form of upper bound for its bucket -- but it must
            # never exceed the hard analytical ceiling for THIS request's
            # actual (unbucketed) shape, mirroring
            # evaluate_ar_kv_admission's/Code2Wav's min(calibrated, hard)
            # safety cap.
            quantile_transient_bytes = min(profile.envelope_peak_bytes, hard_transient_bytes)
            persistent_bytes = profile.persistent_bytes_per_request
            fallback_reason = None
            profile_version = profile.profile_version
            sample_count = profile.sample_count
        else:
            quantile_transient_bytes = hard_transient_bytes
            persistent_bytes = _PLACEHOLDER_PERSISTENT_BYTES_PER_REQUEST
            fallback_reason = _UNPROFILED_FALLBACK_REASON
            profile_version = None
            sample_count = 0

        physical = PhysicalResourceDemand(
            immediate_transient_bytes=quantile_transient_bytes,
            quantile_transient_peak_bytes=quantile_transient_bytes,
            hard_transient_peak_bytes=hard_transient_bytes,
            persistent_bytes=persistent_bytes,
            available_dimensions=frozenset(
                {ResourceDimension.TRANSIENT_BYTES, ResourceDimension.PERSISTENT_BYTES}
            ),
        )
        return RequestResourceEstimate(
            logical=LogicalResourceDemand(),
            physical=physical,
            provenance=EstimateProvenance(
                backend="diffusion",
                workload_class=self.classifier.classify(
                    batch_size=context.batch_size,
                    latent_pixel_count=context.latent_pixel_count,
                    num_frames=context.num_frames,
                    cfg_parallel_size=context.cfg_parallel_size,
                ),
                estimator_version=self.VERSION,
                target_coverage=context.target_coverage,
                sample_count=sample_count,
                profile_version=profile_version,
                fallback_reason=fallback_reason,
            ),
        )

    def estimate(
        self,
        context: DiffusionBatchContext,
        profile_store: DiffusionProfileStore | None = None,
    ) -> RequestResourceEstimate:
        """``RequestResourceEstimator``-protocol-compatible alias for
        ``estimate_batch`` -- see that method's docstring for why Diffusion
        has no separate per-request estimate."""
        return self.estimate_batch(context, profile_store)

    def estimate_marginal(
        self,
        request_latent_pixel_count: int,
        request_num_frames: int,
        candidate_batch: DiffusionBatchContext,
        profile_store: DiffusionProfileStore | None = None,
    ) -> MarginalResourceEstimate:
        """The incremental cost of adding one more request to
        ``candidate_batch`` (M2 design doc S9.2). ``candidate_batch`` is the
        batch WITHOUT the new request; this computes the envelope both
        without and with it (padded to whichever of the two latent shapes is
        larger, mirroring how a real batched forward pass pads every member
        to the batch's longest/largest member) and returns their difference,
        floored at zero.

        The "with" batch's per-frame latent pixel count is expressed as a
        1-wide, N-tall grid sized to ``max(existing batch pixel count, new
        request pixel count)`` -- an arbitrary but consistent decomposition,
        since only the product (``latent_pixel_count``) and ``num_frames``
        ever feed the classifier/hard-bound math, never height and width
        individually.
        """
        without_request = self.estimate_batch(candidate_batch, profile_store)

        merged_num_frames = max(candidate_batch.num_frames, request_num_frames)
        merged_pixel_count = max(candidate_batch.latent_pixel_count, request_latent_pixel_count)
        merged_height = math.ceil(merged_pixel_count / merged_num_frames)
        with_batch = DiffusionBatchContext(
            batch_size=candidate_batch.batch_size + 1,
            latent_height=merged_height,
            latent_width=1,
            num_frames=merged_num_frames,
            cfg_parallel_size=candidate_batch.cfg_parallel_size,
            workload_class=candidate_batch.workload_class,
            target_coverage=candidate_batch.target_coverage,
            profile_version=candidate_batch.profile_version,
            sample_count=candidate_batch.sample_count,
        )
        with_request = self.estimate_batch(with_batch, profile_store)

        delta = with_request.physical.quantile_transient_peak_bytes - without_request.physical.quantile_transient_peak_bytes
        return MarginalResourceEstimate(
            without_request=without_request,
            with_request=with_request,
            delta_transient_bytes=max(0, delta),
        )
