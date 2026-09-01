from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ResourceDimension(StrEnum):
    KV_BLOCKS = "kv_blocks"
    TRANSIENT_BYTES = "transient_bytes"
    PERSISTENT_BYTES = "persistent_bytes"


class AdmissionReason(StrEnum):
    ALLOWED = "allowed"
    DISABLED = "disabled"
    EMPTY_QUEUE = "empty_queue"
    SHADOW_ONLY = "shadow_only"
    KV_IMMEDIATE_INSUFFICIENT = "kv_immediate_insufficient"
    KV_PEAK_RISK = "kv_peak_risk"
    UNKNOWN_WORKLOAD_CLASS = "unknown_workload_class"
    ESTIMATOR_UNAVAILABLE = "estimator_unavailable"
    ESTIMATOR_ERROR = "estimator_error"


@dataclass(frozen=True)
class LogicalResourceDemand:
    """Logical demand with explicit time and safety semantics."""

    immediate_kv_blocks: int = 0
    expected_peak_kv_blocks: int = 0
    quantile_peak_kv_blocks: int = 0
    hard_peak_kv_blocks: int = 0
    slots: int = 1

    def __post_init__(self) -> None:
        values = (
            self.immediate_kv_blocks,
            self.expected_peak_kv_blocks,
            self.quantile_peak_kv_blocks,
            self.hard_peak_kv_blocks,
            self.slots,
        )
        if any(value < 0 for value in values):
            raise ValueError("logical resource demand must be non-negative")
        if not (
            self.immediate_kv_blocks
            <= self.expected_peak_kv_blocks
            <= self.quantile_peak_kv_blocks
            <= self.hard_peak_kv_blocks
        ):
            raise ValueError("KV demand must satisfy immediate <= expected <= quantile <= hard")


@dataclass(frozen=True)
class PhysicalResourceDemand:
    """Per-device physical demand.

    ``available_dimensions`` distinguishes an estimated zero from a resource
    dimension for which no model exists yet.
    """

    immediate_transient_bytes: int = 0
    quantile_transient_peak_bytes: int = 0
    hard_transient_peak_bytes: int = 0
    persistent_bytes: int = 0
    available_dimensions: frozenset[ResourceDimension] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        values = (
            self.immediate_transient_bytes,
            self.quantile_transient_peak_bytes,
            self.hard_transient_peak_bytes,
            self.persistent_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("physical resource demand must be non-negative")
        if not (
            self.immediate_transient_bytes
            <= self.quantile_transient_peak_bytes
            <= self.hard_transient_peak_bytes
        ):
            raise ValueError("transient demand must satisfy immediate <= quantile <= hard")


@dataclass(frozen=True)
class EstimateProvenance:
    backend: str
    workload_class: str
    estimator_version: str
    target_coverage: float
    sample_count: int = 0
    profile_version: str | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1]")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")


@dataclass(frozen=True)
class RequestResourceEstimate:
    logical: LogicalResourceDemand
    physical: PhysicalResourceDemand
    provenance: EstimateProvenance

    @property
    def kv_blocks(self) -> int:
        """Compatibility view for the original M2 experiment harness."""
        return self.logical.quantile_peak_kv_blocks

    @property
    def transient_bytes(self) -> int:
        return self.physical.quantile_transient_peak_bytes

    @property
    def persistent_bytes(self) -> int:
        return self.physical.persistent_bytes


@dataclass(frozen=True)
class ARRequestResourceContext:
    num_prompt_tokens: int
    max_tokens: int
    block_size: int
    num_computed_tokens: int = 0
    num_generated_tokens: int = 0
    allocated_kv_blocks: int = 0
    reusable_cached_tokens: int = 0
    next_scheduled_tokens: int = 1
    sliding_window_tokens: int | None = None
    expected_output_tokens: int | None = None
    quantile_output_tokens: int | None = None
    workload_class: str = "unknown"
    target_coverage: float = 1.0
    profile_version: str | None = None
    sample_count: int = 0

    def __post_init__(self) -> None:
        integer_values = (
            self.num_prompt_tokens,
            self.max_tokens,
            self.block_size,
            self.num_computed_tokens,
            self.num_generated_tokens,
            self.allocated_kv_blocks,
            self.reusable_cached_tokens,
            self.next_scheduled_tokens,
        )
        if any(value < 0 for value in integer_values):
            raise ValueError("AR resource context values must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: AdmissionReason
    estimate: RequestResourceEstimate | None = None
    required: int = 0
    available: int = 0
    shadow_would_defer: bool = False
    correction: float = 1.0


@runtime_checkable
class RequestResourceEstimator(Protocol):
    def estimate(self, context: ARRequestResourceContext) -> RequestResourceEstimate: ...


class ARResourceEstimator:
    """AR KV estimator with explicit immediate/peak/hard semantics.

    Without an output-length profile, expected and quantile demand fall back
    to ``max_tokens`` and provenance reports that conservative fallback.
    """

    VERSION = "ar-kv-v2"

    def __init__(self, block_size: int | None = None) -> None:
        if block_size is not None and block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size

    @staticmethod
    def _blocks(tokens: int, block_size: int) -> int:
        return math.ceil(tokens / block_size) if tokens > 0 else 0

    def estimate(
        self,
        context: ARRequestResourceContext | None = None,
        *,
        num_prompt_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> RequestResourceEstimate:
        # Keep the original keyword form for existing experiment tooling.
        if context is None:
            if self.block_size is None:
                raise ValueError("block_size is required when no context is supplied")
            context = ARRequestResourceContext(
                num_prompt_tokens=max(0, num_prompt_tokens or 0),
                max_tokens=max(0, max_tokens or 0),
                block_size=self.block_size,
            )

        hard_output = max(context.num_generated_tokens, context.max_tokens)
        expected_output = context.expected_output_tokens
        quantile_output = context.quantile_output_tokens
        fallback_reason = None
        if expected_output is None or quantile_output is None:
            expected_output = hard_output
            quantile_output = hard_output
            fallback_reason = "output_length_profile_unavailable"
        expected_output = min(hard_output, max(context.num_generated_tokens, expected_output))
        quantile_output = min(hard_output, max(expected_output, quantile_output))

        reusable_tokens = min(context.num_prompt_tokens, context.reusable_cached_tokens)
        uncached_prompt = context.num_prompt_tokens - reusable_tokens

        def incremental_peak(output_tokens: int) -> int:
            total_tokens = uncached_prompt + output_tokens
            if context.sliding_window_tokens is not None:
                total_tokens = min(total_tokens, context.sliding_window_tokens)
            return max(
                0,
                self._blocks(total_tokens, context.block_size) - context.allocated_kv_blocks,
            )

        remaining_now = max(
            0,
            context.num_prompt_tokens
            + context.num_generated_tokens
            - context.num_computed_tokens
            - reusable_tokens,
        )
        immediate_tokens = min(
            remaining_now or context.next_scheduled_tokens,
            context.next_scheduled_tokens,
        )
        immediate_blocks = self._blocks(immediate_tokens, context.block_size)
        expected_blocks = max(immediate_blocks, incremental_peak(expected_output))
        quantile_blocks = max(expected_blocks, incremental_peak(quantile_output))
        hard_blocks = max(quantile_blocks, incremental_peak(hard_output))

        return RequestResourceEstimate(
            logical=LogicalResourceDemand(
                immediate_kv_blocks=immediate_blocks,
                expected_peak_kv_blocks=expected_blocks,
                quantile_peak_kv_blocks=quantile_blocks,
                hard_peak_kv_blocks=hard_blocks,
            ),
            physical=PhysicalResourceDemand(),
            provenance=EstimateProvenance(
                backend="ar",
                workload_class=context.workload_class,
                estimator_version=self.VERSION,
                profile_version=context.profile_version,
                target_coverage=context.target_coverage,
                sample_count=context.sample_count,
                fallback_reason=fallback_reason,
            ),
        )


def evaluate_ar_kv_admission(
    estimate: RequestResourceEstimate,
    *,
    free_kv_blocks: int,
    enforce: bool,
    correction: float = 1.0,
    uncertainty_multiplier: float = 1.0,
) -> AdmissionDecision:
    """Evaluate an estimate without reading scheduler state.

    ``required`` is the admission demand — the quantile peak scaled by the
    online calibration ``correction`` and a fixed ``uncertainty_multiplier`` —
    not the raw quantile estimate. Both default to ``1.0`` (no correction),
    so callers without a calibrator get the original uncorrected behavior.
    The scaled demand is capped at ``hard_peak_kv_blocks``: calibration may
    make a systematically-underestimating profile more conservative, but it
    must never demand more than the request's own worst-case bound.
    """
    if free_kv_blocks < 0:
        raise ValueError("free_kv_blocks must be non-negative")
    if correction <= 0 or uncertainty_multiplier <= 0:
        raise ValueError("correction and uncertainty_multiplier must be positive")
    required = min(
        estimate.logical.hard_peak_kv_blocks,
        math.ceil(estimate.logical.quantile_peak_kv_blocks * correction * uncertainty_multiplier),
    )
    would_defer = required > free_kv_blocks
    if not would_defer:
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionReason.ALLOWED,
            estimate=estimate,
            required=required,
            available=free_kv_blocks,
            correction=correction,
        )
    return AdmissionDecision(
        allowed=not enforce,
        reason=AdmissionReason.KV_PEAK_RISK,
        estimate=estimate,
        required=required,
        available=free_kv_blocks,
        shadow_would_defer=not enforce,
        correction=correction,
    )
