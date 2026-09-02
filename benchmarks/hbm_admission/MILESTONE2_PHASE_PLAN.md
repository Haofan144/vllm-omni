# Milestone 2 — Heterogeneity-Aware Resource Modeling: Phase Plan

Status snapshot: 2026-09-02 (updated). Tracks the phased execution plan for
Milestone 2 against the original design doc. Phase 0–3 complete; Phase 2's
Code2Wav real-GPU calibration sub-item has its instrumentation/collector/
profile-builder built and tested, but the actual calibration run is blocked
by host environment issues (see Phase 2c) rather than incomplete. Phase 4
partially complete (uncertainty multiplier + contract freeze done;
fallback-hierarchy depth intentionally not extended beyond what's documented
in `M2_ESTIMATOR_CONTRACT.md`).

## Phase 0 — Bounded bypass (head-of-line blocking mitigation)

**Status: Done.**

- Problem: strict FCFS admission means one waiting request whose estimated
  demand doesn't fit can freeze the entire queue even when later, smaller
  requests would fit.
- Design: bounded bypass scan. When the head-of-line waiting request is
  rejected by the resource admission check, scan up to
  `resource_admission_bypass_scan_limit` subsequent waiting requests for one
  that fits, and admit it out of order instead of stalling.
- Aging guard: once the head-of-line request has waited at least
  `resource_admission_bypass_aging_ms`, stop bypassing it even if a smaller
  request would otherwise qualify, so a large request cannot be starved
  indefinitely by an unbroken stream of smaller ones.
- Implementation: `DynamicHBMConfig.resource_admission_bypass_scan_limit` /
  `resource_admission_bypass_aging_ms` (`config.py`);
  `_dynamic_hbm_bounded_bypass_waiting` in `omni_scheduler_mixin.py`; queue
  substitution + reverse-order fix in `omni_ar_scheduler.py`'s `schedule()`.
- Validation: `H13_bounded_bypass_effectiveness.py` (5 checks: bypass scan,
  aging cutoff, scan-limit bound, no queue pollution), plus 8 targeted unit
  tests in `test_dynamic_hbm_admission_gate.py`. All passing.
- Commit: `4b794d45`.

## Safety gap fix (inserted before Phase 2 continued)

**Status: Done.**

- Problem: the AR resource estimator assumed every request/stage was an AR
  (`LLM_AR`) workload. An unknown or unsupported `StageExecutionType` would
  silently fall through rather than fail closed.
- Fix: `_dynamic_hbm_ar_estimator_applies()` / later
  `_dynamic_hbm_code2wav_estimator_applies()` in `omni_scheduler_mixin.py` —
  explicit execution-type gates that both fail closed (estimator does not
  apply) on any unrecognized `StageExecutionType`. New
  `AdmissionReason.UNSUPPORTED_EXECUTION_TYPE` for observability.
- Commit: `8fa29f91`.

## Phase 1 — AR resource estimator: real-GPU evidence

**Status: Done, with one documented limitation.**

- Goal: validate that the offline-profile + online-EWMA-calibrated AR KV
  estimator (already implemented pre-Phase-0) actually reduces excess KV
  block reservation on real hardware, not just in unit tests.
- Method: real vLLM-Omni server + GPU, shadow-mode admission (estimator
  computes counterfactual decisions but never gates), traced via JSONL and
  analyzed with `analyze_ar_shadow_trace.py` (`ARProfileStore` coverage +
  excess-block-reduction vs. worst-case-reservation baseline).
- Result (scaled-up run, `M2_ar_shadow_trace_scaleup_v1`): 135 real requests,
  41 held-out samples, 95.1% profile coverage, 98.98% excess-block reduction
  vs. worst-case.
- **Limitation**: evidence covers a single workload class only (plain
  text-LLM AR generation). Not yet validated across TTS/other workload
  classes.
- Follow-up attempt: real enforce-mode (B0 baseline vs. enforce/profile+EWMA)
  online comparison via `run_ar_admission_mode_experiment.py`. Root-caused
  "no differentiation" (v1) to under-stressed concurrency, and the
  subsequent "identical failures in both arms" (v2) to an unrelated CUDA
  stability issue at extreme `gpu_memory_utilization=0.11`
  (`torch.AcceleratorError: device-side assert triggered`, occurring
  identically in both arms and independent of admission policy). Decision:
  stop further enforce-mode attempts given the cost/reliability tradeoff;
  shadow-mode results stand as the Phase 1 deliverable.
- Commit (script): part of `4b794d45`'s history /
  `run_ar_admission_mode_experiment.py` (tracked, no pending changes).

## Phase 2 — TTS / Code2Wav resource modeling

**Status: Partially done.** Structure and observation-only wiring complete;
real calibration and enforce mode not started.

### 2a. TTS workload classifier (talker/AR stage) — Done

- `TTSWorkloadClassifier(ARWorkloadClassifier)` in
  `tts_resource_estimator.py`: extends the bucket key with
  `task_type`/`ref_audio_present`, prefixes `"tts:"`, explicitly excludes
  `language` from bucketing (too high-cardinality, not cost-relevant).
- Pluggability: `DynamicHBMConfig.ar_workload_classifier` — qualname string
  (e.g. `"vllm_omni.core.memory_coordinator.tts_resource_estimator.
  TTSWorkloadClassifier"`) resolved via `resolve_obj_by_qualname`, so the
  backend-agnostic `ARResourceEstimator` KV-block math is unchanged; only the
  workload-class key differs per stage.
- Tests: `test_tts_resource_estimator.py` (6 tests),
  `test_dynamic_hbm_admission_gate.py` classifier-pluggability tests.
- Commit: `6d800fbc`.

### 2b. Code2Wav resource estimator (waveform decode stage) — structure done, calibration not started

- New data structures in `code2wav_resource_estimator.py`:
  `Code2WavRequestContext`, `Code2WavWorkloadClassifier` (numeric bucket
  boundaries for orderable ceiling search), `Code2WavEnvelopeProfile`,
  `Code2WavProfileStore` (`get_ceiling` — conservative, never rounds down),
  `Code2WavResourceEstimator` (caps estimate at hard bound;
  `available_dimensions` excludes `KV_BLOCKS` since Code2Wav is
  `LLM_GENERATION`, not AR).
- Cost constants (`_PLACEHOLDER_BYTES_PER_FRAME_PER_BATCH_SLOT`,
  `_PLACEHOLDER_BYTES_PER_PERSISTENT_ENTRY`) are **explicitly marked as
  unvalidated placeholders** — decision was to write the structure first
  rather than block on real GPU profiling.
- Scheduler wiring: `_sample_code2wav_resource_observation()` called from
  `OmniGenerationScheduler.schedule()` — **observation-only**, no
  admit/defer gating, because no real physical-memory budget signal exists
  yet for the generation-stage decode path.
- Tests: `test_code2wav_resource_estimator.py` (12 tests), 8 Code2Wav wiring
  tests in `test_dynamic_hbm_admission_gate.py`.
- Commits: `6f1d4ab4` (estimator structure), `8d8d9f6b` (scheduler wiring).

### 2c. Remaining Phase 2 work

**Status: Infrastructure done and tested; the real-GPU calibration run itself
is blocked by host-level environment issues, not by anything in this
codebase.**

- Real-GPU profiling infrastructure is now built: `Code2WavObservation`
  (one flat per-forward-call record — batch size, batch max frame count,
  persistent-state flag, measured peak transient bytes — unlike AR's
  per-request-lifetime `ResourceObservation`, since Code2Wav's real cost is
  a per-forward-call quantity with no lifetime to track),
  `measure_code2wav_forward_peak_bytes()` (wraps a decode call with
  `torch.cuda.reset_peak_memory_stats`/`max_memory_allocated`, measuring the
  delta against the pre-call allocation level so it doesn't double-count
  resident state), `Code2WavObservationJSONLWriter`/
  `read_code2wav_observations_jsonl` (same bounded-flush/atexit-flush shape
  as the AR writer), and `build_code2wav_envelope_profiles()` (buckets
  observations by `Code2WavWorkloadClassifier.bucket_key`, takes the MAX
  observed peak per bucket — an empirical peak is a lower bound on the true
  worst case, never a mean/median). All in
  `code2wav_resource_estimator.py`, covered by 4 new unit tests in
  `test_code2wav_resource_estimator.py` (observation validation, max-not-mean
  bucketing, min-samples respected, a built profile correctly feeding the
  estimator and clearing `fallback_reason`).
- Wired into the real model: `qwen3_tts_code2wav.py`'s `forward()` now
  measures `batched_chunked_decode`'s peak bytes and appends one
  `Code2WavObservation` per forward call when
  `VLLM_OMNI_QWEN3_CODE2WAV_MEMORY_TRACE_PATH` is set (opt-in, mirroring the
  existing `VLLM_OMNI_QWEN3_CODE2WAV_BATCH_STATS` convention — every measured
  call pays a CUDA synchronize + stat-reset cost, so this must never run by
  default). Verified this instrumentation does not break the model: the
  pre-existing `test_qwen3_tts_code2wav.py` suite (36 tests) still passes
  unchanged with the new code path present but disabled.
- **What could not be completed this session**: an actual real-GPU
  calibration run to replace the placeholder byte constants
  (`_PLACEHOLDER_BYTES_PER_FRAME_PER_BATCH_SLOT`,
  `_PLACEHOLDER_BYTES_PER_PERSISTENT_ENTRY`) with measured values. Two
  environment blockers were hit in sequence, both external to this
  session's code changes: (1) the installed vLLM (0.28.0) vs. the repo's
  target vLLM (0.27.0) version drift goes deeper than the already-known
  `error_response` import-path move — `serving_chat.py` also eagerly
  imports `get_history_tool_calls_cnt`, a symbol removed (not renamed) in
  0.28, which crashes the server at startup regardless of endpoint; (2)
  switching to `.venv-vllm027` (a pre-existing venv with the matched vLLM
  0.27.0 + vllm_omni installed) hit severe host-level I/O contention instead
  — plain `import vllm` and the server's own module-loading stalled for
  minutes at a time in kernel `D`-state (`folio_wait_bit_common`, host load
  average 50-64) on the shared Ceph RBD-backed filesystem, well before any
  GPU work started, and did not clear within two ~10-minute attempts.
  Neither blocker is a defect in the Code2Wav instrumentation or estimator
  code itself.
- **How to complete this once the environment cooperates**: launch the
  server with `.venv-vllm027`'s python and
  `VLLM_OMNI_QWEN3_CODE2WAV_MEMORY_TRACE_PATH` set (exact recipe: same
  model/deploy-config args as `M2_ar_shadow_trace_scaleup_v1`'s
  `metadata.json`, i.e. `../huggingface/Qwen3-TTS-12Hz-0.6B-CustomVoice` +
  `vllm_omni/deploy/qwen3_tts.yaml`), send a real request batch (`vllm bench
  serve --backend openai-audio-speech --endpoint /v1/audio/speech
  --dataset-name seed-tts-text`, as `run_dynamic_hbm_experiment.py`'s
  `benchmark_command` already does), then call
  `build_code2wav_envelope_profiles()` on the resulting JSONL and
  `Code2WavProfileStore.write_jsonl`-equivalent persistence (not yet
  written — mirror `ARProfileStore.write_jsonl` if/when needed) to produce a
  real profile for `Code2WavProfileStore`.
- A real enforce mode for Code2Wav remains blocked on a separate, unrelated
  question — defining and measuring a genuine physical-memory budget signal
  for the generation/decode path (unlike AR's KV-block pool, there is no
  existing hard budget to enforce against yet) — independent of whether the
  placeholder constants get replaced.

## Phase 3 — Diffusion estimator

**Status: Done (design/structure), with a documented scheduler-wiring gap.**

- Extended the unified resource model
  (`LogicalResourceDemand`/`PhysicalResourceDemand`/`EstimateProvenance`/
  `RequestResourceEstimate`) to `StageExecutionType.DIFFUSION` in the new
  `diffusion_resource_estimator.py`: `DiffusionBatchContext`,
  `DiffusionWorkloadClassifier` (3D orderable bucket key: batch size ×
  latent-pixel-count × frame-count, plus `cfg_parallel_size`/`vae_mode` in
  the string label), `DiffusionPhaseEnvelopeProfile` (per-`DiffusionPhase`
  peaks — condition-encoding/denoising/VAE-encode/VAE-decode/
  post-processing — combined as an envelope/MAX, not a sum, per M2 design
  doc S9.3), `DiffusionProfileStore` (`get_ceiling` — same never-round-down
  3D ceiling search as Code2Wav's 2D version), `DiffusionResourceEstimator`
  (`estimate_batch` as the primary interface per S9.2, `estimate` as a
  protocol-compatible alias, `estimate_marginal` returning a
  `MarginalResourceEstimate` with `delta_transient_bytes` floored at zero).
- `denoising_steps` is deliberately excluded from both the context and the
  classifier (S9.1: step count affects wall-clock, not peak HBM, since every
  step reuses the same latent buffers — asserted by a dedicated test rather
  than left as an implicit omission).
- Cost constants (`_PLACEHOLDER_BYTES_PER_LATENT_PIXEL_PER_BATCH_SLOT`) are
  **explicitly marked as unvalidated placeholders**, mirroring Code2Wav's
  documented approach — structure first, real-GPU calibration later.
- **Scheduler-wiring gap** (structural, not an oversight): confirmed
  `StageExecutionType.DIFFUSION` stages do not run through
  `OmniSchedulerMixin`/`VLLMScheduler` at all — `_resolve_scheduler` maps
  `DIFFUSION` to no scheduler class, and diffusion batching is a fixed
  `diffusion_batch_size` set at engine/stage-runtime construction time, not
  a per-request admission decision. There is therefore no `schedule()` call
  site to hook observation or admission into today, unlike Code2Wav (which
  does run through `OmniGenerationScheduler`). `DiffusionResourceEstimator`
  is usable today as a library call (experiment harness, or a future
  admission mechanism once diffusion gets dynamic per-request batching) but
  is not wired into any live scheduling loop. See
  `M2_ESTIMATOR_CONTRACT.md` §4 for the full per-backend integration matrix.
- Tests: `test_diffusion_resource_estimator.py` (20 tests) covering context
  validation, classifier bucket ordering/labels, hard-bound scaling
  (batch/frames/CFG), envelope-is-max-not-sum, profile ceiling
  never-rounds-down, profile-capped-at-hard-bound, and marginal-estimate
  monotonicity.
- Commit: pending (this session).

## Phase 4 — Fallback hierarchy completion & estimator-contract freeze (M2e)

**Status: Partially done.** Uncertainty multiplier implemented and wired;
contract frozen and documented; fallback-hierarchy depth intentionally
matched to what Code2Wav/Diffusion's batch-envelope design actually needs,
not extended to literally implement all five levels the design doc
describes.

- Uncertainty multiplier: `uncertainty_multiplier()` in
  `resource_calibrator.py` — a configurable safety margin (two independently
  inspectable conditions, `sample_count < resource_uncertainty_min_samples`
  and `snapshot.stale`, each with its own configurable multiplier) applied
  only on top of a profile-backed estimate, never stacked on an
  already-conservative hard-bound fallback. Wired into
  `_dynamic_hbm_resource_admission_decision`'s call to
  `evaluate_ar_kv_admission` via three new `DynamicHBMConfig` fields
  (`resource_uncertainty_min_samples`,
  `resource_uncertainty_low_sample_multiplier`,
  `resource_uncertainty_stale_multiplier`), all validated in
  `__post_init__` and covered by `test_config_protocol.py`. Code2Wav and
  Diffusion do not call it yet since neither has an enforce-mode admission
  path to apply it to.
- Fallback hierarchy: audited what AR/Code2Wav/Diffusion actually implement
  against the design doc's five-level hierarchy (exact profile → parent
  bucket → backend/model conservative profile → hard bound →
  single-request/defer) and found all three collapse levels 1–3 into one
  monotonic ceiling-rounding step (documented in
  `M2_ESTIMATOR_CONTRACT.md` §3) rather than implementing a literal
  parent-bucket walk. This was a deliberate scope decision this session,
  not an oversight: adding a real parent-bucket level with no data to
  populate it would be speculative structure, the kind of unvalidated
  complexity the M2 design doc's own placeholder-labeling discipline
  argues against. Revisit once real trace volume shows the collapsed
  version's coverage is insufficient for some workload class.
- Final estimator-contract freeze: `M2_ESTIMATOR_CONTRACT.md` — the frozen
  `RequestResourceEstimate`/`LogicalResourceDemand`/`PhysicalResourceDemand`/
  `EstimateProvenance` shape and invariants, per-backend estimator/context
  table, fallback-hierarchy-as-implemented table, per-backend scheduler
  integration status, and an explicit "not frozen" list (bucket boundaries,
  placeholder constants, version strings, Diffusion's scheduler wiring) so
  Milestone 3 knows what it can build on vs. what will still change under
  it.
- Commit: pending (this session).
