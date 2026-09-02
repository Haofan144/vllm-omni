# Milestone 2 — Resource Estimator Output Contract (M2e freeze)

Status: 2026-09-02. This freezes the shape of what a Milestone 2 resource
estimator hands to a caller, so Milestone 3 (per-device global planner) has
a stable input to build on without depending on any scheduler-internal
state. It does not freeze internal estimator logic (bucket boundaries,
placeholder byte constants, EWMA parameters) — those keep evolving as real
profiling data replaces placeholders. What is frozen is the *shape*:
dataclass fields, invariants, and the meaning of "unavailable" vs. "zero".

## 1. The three frozen dataclasses

All three live in `vllm_omni/core/memory_coordinator/resource_estimator.py`
and are backend-agnostic — AR, TTS, Code2Wav, and Diffusion estimators all
produce exactly this shape, never a backend-specific subclass or a
differently-named sibling type.

```
RequestResourceEstimate
├── logical: LogicalResourceDemand
│   ├── immediate_kv_blocks: int        (>= 0)
│   ├── expected_peak_kv_blocks: int    (>= immediate)
│   ├── quantile_peak_kv_blocks: int    (>= expected)
│   ├── hard_peak_kv_blocks: int        (>= quantile)
│   └── slots: int
├── physical: PhysicalResourceDemand
│   ├── immediate_transient_bytes: int       (>= 0)
│   ├── quantile_transient_peak_bytes: int   (>= immediate)
│   ├── hard_transient_peak_bytes: int       (>= quantile)
│   ├── persistent_bytes: int                (>= 0)
│   └── available_dimensions: frozenset[ResourceDimension]
└── provenance: EstimateProvenance
    ├── backend: str
    ├── workload_class: str
    ├── estimator_version: str
    ├── target_coverage: float           (0 < x <= 1)
    ├── sample_count: int                (>= 0)
    ├── profile_version: str | None
    └── fallback_reason: str | None
```

**Invariants** (enforced by `__post_init__`, not just documented):
non-negativity on every numeric field; `immediate <= expected <= quantile
<= hard` for KV blocks; `immediate <= quantile <= hard` for transient bytes.
A caller may rely on these without re-validating them.

**"Unavailable" vs. "zero"**: a backend that does not model a given
dimension at all (e.g. Code2Wav/Diffusion have no KV-block accounting) must
report it as `LogicalResourceDemand()` defaults (all zero) *and* must not
list `ResourceDimension.KV_BLOCKS` in `PhysicalResourceDemand
.available_dimensions`. A caller that needs to distinguish "this backend
genuinely uses 0 blocks" from "this backend has no KV concept" must check
`available_dimensions`, never infer it from the numeric value alone. Every
estimator in this milestone (AR, TTS, Code2Wav, Diffusion) follows this
rule; it is part of the frozen contract, not an implementation detail.

## 2. Per-backend estimator inputs (not frozen the same way)

Unlike the output, each backend's *input* context type is backend-specific
by design (M2 design doc §6: `RequestResourceContext` carries "backend-
specific metadata" alongside common fields) — Milestone 3 is expected to
consume `RequestResourceEstimate`, not to construct these contexts itself:

| Backend | Estimator | Context type | Primary interface |
|---|---|---|---|
| AR (text LLM, TTS Talker) | `ARResourceEstimator` | `ARRequestResourceContext` | `estimate(context)` |
| Code2Wav / acoustic decoder | `Code2WavResourceEstimator` | `Code2WavRequestContext` | `estimate(context, profile_store)` |
| Diffusion | `DiffusionResourceEstimator` | `DiffusionBatchContext` | `estimate_batch(context, profile_store)`, `estimate(...)` alias, `estimate_marginal(...)` |

The TTS Talker stage reuses `ARResourceEstimator` unchanged and only swaps
in `TTSWorkloadClassifier` for the workload-class key (see
`tts_resource_estimator.py`'s module docstring) — it is not a fourth
estimator type.

Diffusion's primary interface is batch-level, not per-request, because a
diffusion stage has no scheduler-mediated per-request admission today (see
§4 below); `estimate_marginal` is the shape Milestone 3 (or a future
diffusion-admission milestone) would call to ask "does one more request fit
in this candidate batch," returning a `MarginalResourceEstimate` with
`delta_transient_bytes` floored at zero.

## 3. Fallback hierarchy — what each backend actually implements today

The M2 design doc §11.4 describes a five-level hierarchy (exact profile →
parent workload bucket → backend/model conservative profile → hard
analytical bound → single-request/defer). No estimator in this milestone
implements all five levels distinctly; each collapses the middle three
levels into a single "profiled bucket via ceiling-rounding, or hard bound"
step, which is the sound and tested subset actually shipped:

| Backend | Level 1 (exact) | Levels 2–3 (coarser profile) | Level 4 (hard bound) | Level 5 (defer) |
|---|---|---|---|---|
| AR | `ARProfileStore.get(workload_class)` exact match | Not implemented — no parent-bucket walk | `max_tokens`-derived `hard_peak_kv_blocks`, always computed | Scheduler-level (bounded bypass / enforce mode), not the estimator's concern |
| Code2Wav | `Code2WavProfileStore.get_ceiling` (nearest bucket >= request, both dims) | Ceiling search *is* the coarser-bucket step — collapses levels 1–3 into one monotonic lookup | `_PLACEHOLDER_BYTES_*`-derived hard bound, always computed | N/A — observation-only, no gating yet |
| Diffusion | `DiffusionProfileStore.get_ceiling` (nearest bucket >= request, all 3 dims) | Same collapse as Code2Wav | `_PLACEHOLDER_BYTES_PER_LATENT_PIXEL_*`-derived envelope, always computed | N/A — no scheduler call site yet (see §4) |

**What this means for Milestone 3**: `fallback_reason` on
`EstimateProvenance` is the single signal a consumer should branch on, not
an assumption about which of the five design-doc levels fired. Today it is
either `None` (a profile ceiling-matched) or one fixed string per backend
(`"output_length_profile_unavailable"` for AR,
`"unprofiled_placeholder_constant"` for Code2Wav and Diffusion). A future
change that adds a real parent-bucket level would introduce a new
`fallback_reason` value rather than changing this field's type or the
caller's branching shape.

**Never silently zero**: an unknown workload class or an
estimator-construction error (Attribute/Type/Value/OverflowError caught at
every scheduler call site — see `_dynamic_hbm_resource_admission_decision`,
`_sample_code2wav_resource_observation`) always degrades to a hard-bound
estimate or a logged-and-skipped observation, never to a zero-cost estimate
or an unhandled exception reaching the scheduler.

## 4. Scheduler integration status per backend (as of this freeze)

| Backend | Scheduler call site | Mode |
|---|---|---|
| AR | `OmniARScheduler.schedule()` via `_dynamic_hbm_resource_admission_decision` | Shadow (default) or enforce, gated by `DynamicHBMConfig.resource_admission_mode` |
| Code2Wav | `OmniGenerationScheduler.schedule()` via `_sample_code2wav_resource_observation` | Observation-only — predicts, never gates. No physical-memory budget signal exists yet to enforce against (M2c open item) |
| Diffusion | **None** | `StageExecutionType.DIFFUSION` stages do not run through `OmniSchedulerMixin`/`VLLMScheduler` at all (`_resolve_scheduler` maps `DIFFUSION` to no scheduler class); batching is a fixed `diffusion_batch_size` set at engine/stage-runtime construction time, not a per-request admission decision. `DiffusionResourceEstimator` is therefore usable today only as a library call (e.g. from an experiment harness or a future admission mechanism), not wired into any live scheduling loop. |

This is a structural fact about the current engine, not an oversight of
this milestone: there is no per-request-arriving-at-a-full-batch admission
decision to hook for Diffusion yet, because there is no scheduler in the
loop. Wiring one in is out of scope for M2 and would need its own design
(likely: whatever future milestone gives Diffusion dynamic, per-request
batch composition instead of a fixed `diffusion_batch_size`).

## 5. Uncertainty multiplier (M2e)

`uncertainty_multiplier()` in `resource_calibrator.py` returns a
configurable safety-margin float, applied only on top of a
profile-backed (non-fallback) estimate — never on top of an
already-conservative hard-bound fallback (M2 design doc §11.3's rule that
demand should not exceed what a request could ever actually need). It
composes two independent, inspectable conditions rather than a single
opaque confidence score:

- `sample_count < resource_uncertainty_min_samples` (default 30, distinct
  from `OnlineCalibrator`'s own smaller `min_samples` gate for whether to
  trust a correction at all) → `resource_uncertainty_low_sample_multiplier`
  (default 1.15).
- `snapshot.stale` → `resource_uncertainty_stale_multiplier` (default
  1.15).

Both multiply together when both conditions hold. All three are
`DynamicHBMConfig` fields, tunable per deployment without a code change.
Wired into the AR admission path
(`_dynamic_hbm_resource_admission_decision` → `evaluate_ar_kv_admission`'s
`uncertainty_multiplier` argument); Code2Wav and Diffusion do not yet call
it, since neither has an enforce-mode admission path to apply it to (§4).

## 6. What is explicitly NOT frozen by this document

- Bucket boundaries (`ARWorkloadClassifier.PROMPT_BUCKETS`,
  `Code2WavWorkloadClassifier.BATCH_SIZE_BUCKETS`,
  `DiffusionWorkloadClassifier.LATENT_PIXEL_BUCKETS`, etc.) — expected to be
  retuned as real trace data accumulates.
- Placeholder byte-cost constants in `code2wav_resource_estimator.py` and
  `diffusion_resource_estimator.py` — explicitly unvalidated; replacing them
  with measured values does not change any consumer's code. The real-GPU
  measurement path for Code2Wav already exists
  (`measure_code2wav_forward_peak_bytes`, `Code2WavObservation`,
  `Code2WavObservationJSONLWriter`, `build_code2wav_envelope_profiles`, and
  an opt-in `VLLM_OMNI_QWEN3_CODE2WAV_MEMORY_TRACE_PATH` hook in
  `qwen3_tts_code2wav.py`'s forward pass) — only the actual calibration run
  is outstanding, blocked by host environment issues at freeze time (see
  `MILESTONE2_PHASE_PLAN.md` §2c), not by missing infrastructure.
- `estimator_version`/`profile_version` string values — expected to bump as
  the above change; consumers should treat them as opaque compatibility
  tokens, never parse them.
- Diffusion's scheduler wiring (§4) — deferred to a future milestone by
  design, not a gap in this freeze.
