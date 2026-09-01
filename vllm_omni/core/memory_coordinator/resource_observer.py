from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

from vllm_omni.core.memory_coordinator.resource_estimator import (
    RequestResourceEstimate,
)


@dataclass(frozen=True)
class ResourceObservation:
    """Completed request-level AR observation for sampled trace analysis.

    Request IDs intentionally live in this bounded trace object, not in metric
    labels. Physical byte observations remain unavailable until phase-aware
    worker instrumentation is added.
    """

    request_id: str
    estimator_version: str
    profile_version: str | None
    workload_class: str
    target_coverage: float
    fallback_reason: str | None
    prompt_tokens: int
    requested_max_tokens: int
    block_size: int
    predicted_immediate_kv_blocks: int
    predicted_quantile_peak_kv_blocks: int
    predicted_hard_peak_kv_blocks: int
    baseline_allocated_kv_blocks: int
    observed_peak_allocated_kv_blocks: int
    observed_peak_incremental_kv_blocks: int
    observed_output_tokens: int
    observed_local_cached_tokens: int
    observed_external_cached_tokens: int
    started_monotonic_s: float
    finished_monotonic_s: float

    @property
    def kv_peak_error(self) -> int:
        return (
            self.predicted_quantile_peak_kv_blocks
            - self.observed_peak_incremental_kv_blocks
        )

    @property
    def kv_underpredicted(self) -> bool:
        return self.kv_peak_error < 0


@dataclass
class _ActiveObservation:
    estimate: RequestResourceEstimate
    started_monotonic_s: float
    prompt_tokens: int = 0
    requested_max_tokens: int = 0
    block_size: int = 1
    baseline_allocated_kv_blocks: int = 0
    peak_allocated_kv_blocks: int = 0
    output_tokens: int = 0
    local_cached_tokens: int = 0
    external_cached_tokens: int = 0


class ResourceObservationCollector:
    """Thread-safe bounded collector for shadow-mode estimator evaluation."""

    def __init__(self, max_completed: int = 2048) -> None:
        if max_completed < 1:
            raise ValueError("max_completed must be positive")
        self._active: dict[str, _ActiveObservation] = {}
        self._completed: deque[ResourceObservation] = deque(maxlen=max_completed)
        self._lock = Lock()

    def begin(
        self,
        request_id: str,
        estimate: RequestResourceEstimate,
        *,
        baseline_allocated_kv_blocks: int = 0,
        prompt_tokens: int = 0,
        requested_max_tokens: int = 0,
        block_size: int = 1,
    ) -> None:
        if min(
            baseline_allocated_kv_blocks,
            prompt_tokens,
            requested_max_tokens,
        ) < 0 or block_size < 1:
            raise ValueError("observation request metadata is invalid")
        with self._lock:
            # Repeated shadow decisions for the same queue head must not reset
            # its lifetime or peak observation.
            self._active.setdefault(
                request_id,
                _ActiveObservation(
                    estimate=estimate,
                    started_monotonic_s=time.monotonic(),
                    baseline_allocated_kv_blocks=baseline_allocated_kv_blocks,
                    peak_allocated_kv_blocks=baseline_allocated_kv_blocks,
                    prompt_tokens=prompt_tokens,
                    requested_max_tokens=requested_max_tokens,
                    block_size=block_size,
                ),
            )

    def observe(
        self,
        request_id: str,
        *,
        allocated_kv_blocks: int,
        output_tokens: int,
        local_cached_tokens: int = 0,
        external_cached_tokens: int = 0,
    ) -> None:
        values = (
            allocated_kv_blocks,
            output_tokens,
            local_cached_tokens,
            external_cached_tokens,
        )
        if any(value < 0 for value in values):
            raise ValueError("observed resource values must be non-negative")
        with self._lock:
            active = self._active.get(request_id)
            if active is None:
                return
            active.peak_allocated_kv_blocks = max(
                active.peak_allocated_kv_blocks, allocated_kv_blocks
            )
            active.output_tokens = max(active.output_tokens, output_tokens)
            active.local_cached_tokens = max(active.local_cached_tokens, local_cached_tokens)
            active.external_cached_tokens = max(
                active.external_cached_tokens, external_cached_tokens
            )

    def finish(self, request_id: str) -> ResourceObservation | None:
        with self._lock:
            active = self._active.pop(request_id, None)
            if active is None:
                return None
            provenance = active.estimate.provenance
            logical = active.estimate.logical
            observation = ResourceObservation(
                request_id=request_id,
                estimator_version=provenance.estimator_version,
                profile_version=provenance.profile_version,
                workload_class=provenance.workload_class,
                target_coverage=provenance.target_coverage,
                fallback_reason=provenance.fallback_reason,
                prompt_tokens=active.prompt_tokens,
                requested_max_tokens=active.requested_max_tokens,
                block_size=active.block_size,
                predicted_immediate_kv_blocks=logical.immediate_kv_blocks,
                predicted_quantile_peak_kv_blocks=logical.quantile_peak_kv_blocks,
                predicted_hard_peak_kv_blocks=logical.hard_peak_kv_blocks,
                baseline_allocated_kv_blocks=active.baseline_allocated_kv_blocks,
                observed_peak_allocated_kv_blocks=active.peak_allocated_kv_blocks,
                observed_peak_incremental_kv_blocks=max(
                    0,
                    active.peak_allocated_kv_blocks
                    - active.baseline_allocated_kv_blocks,
                ),
                observed_output_tokens=active.output_tokens,
                observed_local_cached_tokens=active.local_cached_tokens,
                observed_external_cached_tokens=active.external_cached_tokens,
                started_monotonic_s=active.started_monotonic_s,
                finished_monotonic_s=time.monotonic(),
            )
            self._completed.append(observation)
            return observation

    def drain_completed(self) -> list[ResourceObservation]:
        with self._lock:
            observations = list(self._completed)
            self._completed.clear()
            return observations

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)
