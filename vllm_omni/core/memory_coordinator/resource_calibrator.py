# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

from vllm_omni.core.memory_coordinator.resource_observer import ResourceObservation

_EPSILON = 1e-6


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Correction state for one (backend, workload_class, profile_version).

    ``correction`` is a multiplier applied on top of the quantile peak
    estimate. ``1.0`` means "no correction" — this is also the value returned
    when there is not yet enough evidence to trust a learned correction, so a
    caller never needs a separate branch for the cold-start case.
    """

    correction: float
    sample_count: int
    stale: bool


class OnlineCalibrator:
    """EWMA correction of profile-predicted KV peaks against observed peaks.

    Kept separate from ``ARResourceEstimator`` so the estimator stays a pure,
    deterministic function of its context: this class owns the only mutable,
    concurrently-updated state in the M2 estimation path. State is isolated
    per ``(backend, workload_class, profile_version)`` — a profile version
    change starts a fresh correction rather than reusing a ratio computed
    against a different profile's predictions.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        min_correction: float = 0.5,
        max_correction: float = 4.0,
        min_samples: int = 5,
        stale_after_s: float = 900.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 < min_correction <= 1.0 <= max_correction:
            raise ValueError("correction bounds must satisfy min <= 1.0 <= max")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self.alpha = alpha
        self.min_correction = min_correction
        self.max_correction = max_correction
        self.min_samples = min_samples
        self.stale_after_s = stale_after_s
        self._state: dict[tuple[str, str, str | None], tuple[float, int, float]] = {}
        self._lock = Lock()

    @staticmethod
    def _key(
        backend: str, workload_class: str, profile_version: str | None
    ) -> tuple[str, str, str | None]:
        return (backend, workload_class, profile_version)

    def update(self, observation: ResourceObservation, *, backend: str = "ar") -> None:
        """Fold one completed request's prediction/observation pair into the
        EWMA for its (backend, workload_class, profile_version).

        Unknown/fallback estimates (no profile behind the quantile) are not
        folded in: a ratio against a ``max_tokens`` fallback would calibrate
        the correction to worst-case padding rather than to real profile
        error.
        """
        if observation.fallback_reason is not None:
            return
        predicted = observation.predicted_quantile_peak_kv_blocks
        observed = observation.observed_peak_incremental_kv_blocks
        ratio = observed / max(predicted, _EPSILON)
        key = self._key(backend, observation.workload_class, observation.profile_version)
        now = time.monotonic()
        with self._lock:
            previous = self._state.get(key)
            if previous is None:
                correction, sample_count = 1.0, 0
            else:
                correction, sample_count, _ = previous
            correction = self.alpha * ratio + (1.0 - self.alpha) * correction
            correction = min(self.max_correction, max(self.min_correction, correction))
            self._state[key] = (correction, sample_count + 1, now)

    def snapshot(
        self, *, backend: str, workload_class: str, profile_version: str | None
    ) -> CalibrationSnapshot:
        """Return the current correction, or the neutral 1.0 fallback when
        there is not yet enough — or too old — evidence to trust it."""
        key = self._key(backend, workload_class, profile_version)
        with self._lock:
            state = self._state.get(key)
        if state is None:
            return CalibrationSnapshot(correction=1.0, sample_count=0, stale=False)
        correction, sample_count, last_update_s = state
        stale = (time.monotonic() - last_update_s) > self.stale_after_s
        if sample_count < self.min_samples or stale:
            return CalibrationSnapshot(
                correction=1.0, sample_count=sample_count, stale=stale
            )
        return CalibrationSnapshot(
            correction=correction, sample_count=sample_count, stale=False
        )


def uncertainty_multiplier(
    snapshot: CalibrationSnapshot,
    *,
    fallback_reason: str | None,
    min_samples_for_full_confidence: int = 30,
    low_sample_multiplier: float = 1.15,
    stale_multiplier: float = 1.15,
) -> float:
    """Safety margin on top of a (possibly calibrated) quantile estimate.

    M2 design doc S11.3/S11.4: admission demand is the quantile peak scaled
    by both the online calibration ``correction`` *and* this multiplier,
    which is deliberately NOT a single opaque "confidence" float (S11.3
    explicitly rules that out) -- it is a fixed 1.0 unless one of two
    concrete, individually-inspectable conditions holds:

    * ``sample_count < min_samples_for_full_confidence``: the calibrator has
      not yet seen enough completed requests for this workload class to
      trust its correction fully, so demand a modest extra margin. This is a
      *different, coarser* gate than ``OnlineCalibrator``'s own
      ``min_samples`` (which decides whether to apply any correction at
      all) -- a class can clear the calibrator's lower bar and still be in
      this "seen some evidence, not yet a lot" band.
    * ``snapshot.stale``: the calibrator has valid sample history but it is
      old enough that ``snapshot()`` already fell back to a neutral 1.0
      correction; layering an extra margin here (rather than silently
      trusting the neutral value as if nothing were known) reflects that a
      stale profile is closer to "unknown-ish" than to "freshly validated".

    A ``fallback_reason`` (the estimator had no profile at all, so ``quantile
    == hard`` already) needs no additional margin -- the hard bound is
    already the request's own conservative worst case; stacking a multiplier
    on top of it would inflate demand past a value the request could ever
    actually need, which is exactly the "over-conservative capacity loss"
    the M2 design doc's research question 2 is trying to avoid.
    """
    if fallback_reason is not None:
        return 1.0
    multiplier = 1.0
    if snapshot.sample_count < min_samples_for_full_confidence:
        multiplier *= low_sample_multiplier
    if snapshot.stale:
        multiplier *= stale_multiplier
    return multiplier
