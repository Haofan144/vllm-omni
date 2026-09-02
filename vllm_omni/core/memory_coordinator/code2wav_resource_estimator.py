# SPDX-License-Identifier: Apache-2.0
"""Code2Wav/acoustic-decoder resource model (Milestone 2c).

Unlike the Talker (an ``LLM_AR`` stage reusing ``ARResourceEstimator``
unchanged, see ``tts_resource_estimator.py``), a Code2Wav-style decoder stage
(``StageExecutionType.LLM_GENERATION``) is architecturally a different
backend and needs a different resource shape entirely:

* Its real per-step cost is a batch-envelope function of how many codec
  frames are in flight this step, padded to the batch's longest member
  (``cost ~= f(batch_size * max_frames_in_batch)``), not a KV-block count --
  see ``qwen3_tts_code2wav.py``'s ``batched_chunked_decode`` call, which pads
  every request in a forward call to ``max(request_lengths)``.
* Its per-request chunk size (``codec_chunk_frames`` and friends) is a
  deploy-time config constant, not a data-dependent unknown the way an AR
  request's *output length* is -- so unlike the Talker, this estimator does
  not need an "unknown future length" quantile model. The real uncertainty
  is the batch-envelope cost as concurrent batch composition varies, which
  is exactly the M2 design doc's Diffusion "batch envelope" pattern (S9.2):
  profile discrete (batch_size, frame_count) buckets and, for any
  unprofiled/未测量 combination, round UP to the nearest profiled bucket on
  both dimensions rather than linearly extrapolating.
* It has genuine per-request PERSISTENT state (``_decoder_state_cache``'s
  sliding-window/ICL overlap tensors, populated only in async-streaming
  mode) that lives outside vLLM's KV-block accounting entirely and today has
  no capacity model at all (just a soft 512-entry log warning) -- this
  estimator's ``persistent_bytes`` dimension is what governs it.

Placeholder cost constants
---------------------------
``_PLACEHOLDER_BYTES_PER_FRAME_PER_BATCH_SLOT`` and
``_PLACEHOLDER_BYTES_PER_PERSISTENT_ENTRY`` below are NOT measured data. No
real-GPU profiling of Code2Wav's actual per-frame/per-session byte cost has
been run yet (a follow-up M2c step). They exist so this module's data
structures, bucket/ceiling-lookup logic, and fallback hierarchy are complete
and testable now; every estimate this module produces carries a
``fallback_reason`` of ``"unprofiled_placeholder_constant"`` until a real
``Code2WavEnvelopeProfile`` (built from actual observations, mirroring how
``analyze_ar_shadow_trace.py`` builds AR profiles from real traces) replaces
them. Treat any absolute byte number from this module as structurally sound
but unvalidated -- do not use it to size a real deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.core.memory_coordinator.resource_estimator import (
    EstimateProvenance,
    LogicalResourceDemand,
    PhysicalResourceDemand,
    RequestResourceEstimate,
    ResourceDimension,
)
from vllm_omni.core.memory_coordinator.resource_profile import ProfileFingerprint

# PLACEHOLDER: not measured. A deliberately round, conservative-leaning guess
# for the per-frame, per-batch-slot transient (activation/workspace) cost of
# one Code2Wav forward call, in bytes. See module docstring.
_PLACEHOLDER_BYTES_PER_FRAME_PER_BATCH_SLOT = 4 * 1024 * 1024  # 4 MiB
# PLACEHOLDER: not measured. A guess for the per-request persistent
# sliding-window/ICL overlap state size (`_decoder_state_cache` entry) an
# async-streaming Code2Wav session holds across calls, in bytes.
_PLACEHOLDER_BYTES_PER_PERSISTENT_ENTRY = 8 * 1024 * 1024  # 8 MiB

_UNPROFILED_FALLBACK_REASON = "unprofiled_placeholder_constant"


@dataclass(frozen=True)
class Code2WavRequestContext:
    """Estimator input for one request's Code2Wav decode step.

    ``batch_size``/``batch_max_frame_count`` describe the *candidate batch*
    this request would join -- the padding-to-batch-max cost (see module
    docstring) means a request's real cost depends on its batchmates, not
    only on itself, mirroring the M2 design doc's diffusion
    ``estimate_marginal(request_context, candidate_batch_context)`` shape
    rather than AR's purely-per-request KV accounting.
    """

    frame_count: int
    batch_size: int
    batch_max_frame_count: int
    persistent_state_active: bool = False
    workload_class: str = "unknown"
    target_coverage: float = 0.95
    profile_version: str | None = None
    sample_count: int = 0

    def __post_init__(self) -> None:
        if self.frame_count < 0 or self.batch_max_frame_count < 0:
            raise ValueError("Code2Wav frame counts must be non-negative")
        if self.batch_size < 1:
            raise ValueError("Code2Wav batch_size must be at least 1")
        if self.frame_count > self.batch_max_frame_count:
            raise ValueError("frame_count cannot exceed batch_max_frame_count")


class Code2WavWorkloadClassifier:
    """Buckets on batch composition, not prompt/output tokens -- a Code2Wav
    request has no "prompt"/"max_tokens" in the AR sense at all."""

    BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
    FRAME_COUNT_BUCKETS = (25, 50, 100, 200, 300, 600)

    @staticmethod
    def _bucket(value: int, boundaries: tuple[int, ...]) -> int:
        for boundary in boundaries:
            if value <= boundary:
                return boundary
        return boundaries[-1] * 1_000_000  # effectively "overflow", still orderable

    def classify(
        self,
        *,
        batch_size: int,
        frame_count: int,
        persistent_state_active: bool = False,
        **_backend_specific: Any,
    ) -> str:
        if batch_size < 1 or frame_count < 0:
            raise ValueError("workload classifier inputs are invalid")
        batch_bucket = self._bucket(batch_size, self.BATCH_SIZE_BUCKETS)
        frame_bucket = self._bucket(frame_count, self.FRAME_COUNT_BUCKETS)
        return f"code2wav:b{batch_bucket}:f{frame_bucket}:p{int(persistent_state_active)}"

    def bucket_key(self, *, batch_size: int, frame_count: int) -> tuple[int, int]:
        """The numeric (batch_bucket, frame_bucket) pair, for the profile
        store's ceiling search -- string workload_class keys aren't
        orderable, so the store needs the underlying numeric buckets."""
        return (
            self._bucket(batch_size, self.BATCH_SIZE_BUCKETS),
            self._bucket(frame_count, self.FRAME_COUNT_BUCKETS),
        )


@dataclass(frozen=True)
class Code2WavEnvelopeProfile:
    """One profiled (batch_size_bucket, frame_count_bucket) envelope entry.

    ``peak_transient_bytes``/``persistent_bytes_per_request`` are meant to be
    built from real observations (mirroring ``build_ar_output_profiles``),
    not the placeholder constants in this module -- see the module docstring.
    """

    fingerprint: ProfileFingerprint
    batch_size_bucket: int
    frame_count_bucket: int
    sample_count: int
    peak_transient_bytes: int
    persistent_bytes_per_request: int
    profile_version: str = "code2wav-envelope-v1"

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("profile sample_count must be positive")
        if self.peak_transient_bytes < 0 or self.persistent_bytes_per_request < 0:
            raise ValueError("profile byte fields must be non-negative")
        if self.batch_size_bucket < 1 or self.frame_count_bucket < 1:
            raise ValueError("profile bucket boundaries must be positive")


class Code2WavProfileStore:
    """Bucket-ceiling lookup: an unprofiled (batch, frame) combination rounds
    UP to the smallest profiled bucket that is >= it on both dimensions
    (never down, never linearly interpolated/extrapolated), per the M2
    design doc's diffusion batch-envelope fallback rule (S9.2). Returns
    ``None`` when no profiled bucket covers the request at all -- callers
    must fall back to the hard analytical bound, not assume zero cost.
    """

    def __init__(self, profiles: list[Code2WavEnvelopeProfile] | None = None) -> None:
        self._profiles: dict[tuple[int, int], Code2WavEnvelopeProfile] = {
            (profile.batch_size_bucket, profile.frame_count_bucket): profile
            for profile in (profiles or [])
        }

    def get_ceiling(
        self, *, batch_size_bucket: int, frame_count_bucket: int
    ) -> Code2WavEnvelopeProfile | None:
        candidates = [
            profile
            for profile in self._profiles.values()
            if profile.batch_size_bucket >= batch_size_bucket
            and profile.frame_count_bucket >= frame_count_bucket
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p.batch_size_bucket, p.frame_count_bucket))


class Code2WavResourceEstimator:
    """Frame/batch-envelope estimator for a Code2Wav-style decoder stage.

    Deliberately does not produce any ``LogicalResourceDemand`` (KV blocks
    are structurally not applicable to this backend -- see module
    docstring); ``available_dimensions`` marks only the physical dimensions
    this estimator actually models.
    """

    VERSION = "code2wav-transient-v1"

    def __init__(self, classifier: Code2WavWorkloadClassifier | None = None) -> None:
        self.classifier = classifier or Code2WavWorkloadClassifier()

    def estimate(
        self,
        context: Code2WavRequestContext,
        profile_store: Code2WavProfileStore | None = None,
    ) -> RequestResourceEstimate:
        batch_bucket, frame_bucket = self.classifier.bucket_key(
            batch_size=context.batch_size, frame_count=context.batch_max_frame_count
        )
        profile = (
            profile_store.get_ceiling(
                batch_size_bucket=batch_bucket, frame_count_bucket=frame_bucket
            )
            if profile_store is not None
            else None
        )

        hard_transient_bytes = math.ceil(
            context.batch_size
            * context.batch_max_frame_count
            * _PLACEHOLDER_BYTES_PER_FRAME_PER_BATCH_SLOT
        )
        if profile is not None:
            # A profile is itself an empirical peak observation, so it is
            # already a form of upper bound for its bucket -- but it must
            # never exceed the hard analytical ceiling for THIS request's
            # actual (unbucketed) size, mirroring evaluate_ar_kv_admission's
            # min(calibrated, hard) safety cap.
            quantile_transient_bytes = min(profile.peak_transient_bytes, hard_transient_bytes)
            persistent_bytes = profile.persistent_bytes_per_request if context.persistent_state_active else 0
            fallback_reason = None
            profile_version = profile.profile_version
            sample_count = profile.sample_count
        else:
            quantile_transient_bytes = hard_transient_bytes
            persistent_bytes = (
                _PLACEHOLDER_BYTES_PER_PERSISTENT_ENTRY if context.persistent_state_active else 0
            )
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
                backend="code2wav",
                workload_class=self.classifier.classify(
                    batch_size=context.batch_size,
                    frame_count=context.frame_count,
                    persistent_state_active=context.persistent_state_active,
                ),
                estimator_version=self.VERSION,
                target_coverage=context.target_coverage,
                sample_count=sample_count,
                profile_version=profile_version,
                fallback_reason=fallback_reason,
            ),
        )
