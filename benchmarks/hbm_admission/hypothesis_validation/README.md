# HBM dynamic-monitoring hypothesis validation (H1–H9)

Layered experiments that test whether the **Milestone 1 reactive HBM safety**
design actually works — measurement accuracy, shared-device propagation,
watermark control law, no-eviction admission control, gradual recovery, and
fail-closed behaviour.

Each `H*.py` script:

* drives the **real** production classes (`BudgetAllocator`,
  `OmniCoordinator`, `OmniSchedulerMixin`, `RankMemoryReporter`) — see
  `_harness.py` for how,
* prints a PASS/FAIL line per check,
* writes `results/H<n>_result.json` and `results/H<n>_result.md` with the
  hypothesis, expectation, observations, per-check verdicts, and an analysis of
  whether the result matched the expectation and — if not — why.

## Run

```bash
# everything (H1 needs a CUDA device; H2-H7 are GPU-free and deterministic)
python benchmarks/hbm_admission/hypothesis_validation/run_all.py

# skip the GPU experiment
python benchmarks/hbm_admission/hypothesis_validation/run_all.py --skip-h1

# one at a time
python benchmarks/hbm_admission/hypothesis_validation/H3_high_watermark_multiplicative_decrease.py
```

## Files

| file | what |
|---|---|
| `_harness.py` | vLLM import shim + report builders + allocator/coordinator/scheduler drivers + result IO |
| `H1_measurement_accuracy.py` | real GPU: reporter vs NVML, staircase, latency, idle |
| `H2_shared_device_identity.py` | coordinator groups by `(node_id, device_uuid)` |
| `H3_high_watermark_multiplicative_decrease.py` | immediate ×`scale_down_ratio` at high watermark |
| `H4_critical_watermark_halts_admission.py` | one-step to cap 0 at critical; scheduler clamp; KV path |
| `H5_running_requests_not_evicted.py` | cap 0 blocks admission, keeps running set, self-heals |
| `H6_recovery_after_pressure_release.py` | RECOVERING hold then additive climb; re-arm |
| `H7_fail_closed_on_faults.py` | disconnect / stale / missing-rank / stale-generation / KV guard |
| `H8_no_pressure_overhead.py` | real-server paired A/B test for no-pressure throughput and P99 latency overhead |
| `H9_oom_success_effectiveness.py` | calibrated real-pressure C/D test for OOM/failure reduction or request-success improvement |
| `H10_resource_aware_admission.py` | Milestone 2: per-request KV-block estimate (`ARResourceEstimator`) vs the uniform slot-count admission gate |
| `H11_ar_profile_prediction.py` | Milestone 2: observation → workload-class P95 profile → estimator JSONL pipeline and synthetic holdout coverage |
| `H12_ewma_calibration_effectiveness.py` | Milestone 2: `OnlineCalibrator` corrects a systematically-underestimating profile via EWMA, end to end through the real scheduler mixin |
| `../../build_dataset/seed_tts_long/` | long-text (2–4 sentence) seed-tts set for the H9 danger cell — bigger per-request KV footprint |
| `run_all.py` | run H1–H7 and H10–H12, print verdict table |
| `ANALYSIS.md` | consolidated results + analysis + what is / isn't established |

## Result summary (this run, commit `ce7d544d`, RTX A6000)

| ID | Verdict |
|---|---|
| H1 | **PARTIAL** — constant ~1.2 pp NVML-vs-`mem_get_info` offset (fixed, safe direction); shape/latency/monotonicity/idle all pass |
| H2–H7 | **MATCHES EXPECTATION** — all checks pass |
| H8 | **MATCHES EXPECTATION** — no measurable no-pressure overhead (`benchmarks/results/H8_no_pressure_20260828_a6000/`) |
| H9 | **NOT RUN** — v1 calibration (`benchmarks/results/H9_calibration_20260828_a6000/`) found no danger cell (talker KV pool over-provisioned; all 9 cells `failure_rate == 0.0`). Rerun with the Scheme A recipe below. |
| H10 | **MATCHES EXPECTATION** — all checks pass (Milestone 2 AR resource estimator; GPU-free) |
| H11 | **MATCHES EXPECTATION** — all checks pass (Milestone 2 workload-class profile pipeline; GPU-free, synthetic distributions) |
| H12 | **MATCHES EXPECTATION** — all checks pass (Milestone 2 `OnlineCalibrator` EWMA correction; GPU-free) |

See `ANALYSIS.md` for the full write-up, including the H1 root cause and fix
options, and the list of things these experiments do **not** cover (real ZMQ
transport, monitoring overhead H8, OOM-benefit H9).

## End-to-end H8/H9

H8 and H9 launch fresh real servers and are intentionally not part of
`run_all.py`. H8 uses paired no-pressure A/B repetitions:

```bash
python benchmarks/hbm_admission/hypothesis_validation/H8_no_pressure_overhead.py run \
  --output-dir benchmarks/results/H8_no_pressure --repeats 10
```

H9 must be calibrated before its formal comparison. Keep calibration output
separate from the frozen formal result.

### The danger-cell problem (v1 calibration)

The first calibration sweep
(`benchmarks/results/H9_calibration_20260828_a6000/`, concurrency {16,24,32} ×
critical-target {0.93,0.95,0.97}) found **no danger cell** — every cell had
`failure_rate == 0.0`, so `recommended_cells` was empty and the formal C/D
comparison never ran.

Root cause: with the stock `qwen3_tts.yaml` the talker's KV pool is ~11.8 GiB /
110k tokens (~27x headroom for a 4096-token request), physically pre-allocated
at startup. vLLM issues no new large allocations during decode, so the external
`memory_pressure.py` sidecar only eats *unused* transient headroom and never
starves vLLM itself. Raising `--pressure-critical-target` or lowering
`--pressure-reserve-mib` does not change this.

### Danger-cell recipe (Scheme A — shrink the KV pool)

Make the talker's KV pool a real bottleneck, then load it past capacity:

1. **Shrink stage-0 KV pool** with `--stage-overrides` (via `--server-extra-arg`,
   forwarded end to end). `gpu_memory_utilization: 0.11` on stage 0 ⇒ KV pool
   ≈ 2 GiB ≈ 20k tokens ≈ ~5x headroom. `0.10` on stage 1 (no KV) frees HBM back
   to the card so the sidecar can still reach the critical watermark.
2. **Long-text dataset** `benchmarks/build_dataset/seed_tts_long` — 2–4 sentence
   paragraphs ⇒ longer audio ⇒ more KV blocks pinned per running request.
3. **Long critical window** `--pressure-critical-seconds 45` so the danger
   overlaps the benchmark steady state.
4. **High concurrency** (`--concurrencies 32 40 48 56`, `--max-num-seqs 48`).

**`--server-extra-arg` must use the `=` form** — argparse rejects a flag-like
value (`--stage-overrides`) supplied as a separate token:

```bash
python benchmarks/hbm_admission/hypothesis_validation/H9_oom_success_effectiveness.py calibrate \
  --output-dir benchmarks/results/H9_calibration_v2 \
  --dataset-path benchmarks/build_dataset/seed_tts_long \
  --concurrencies 32 40 48 56 --critical-targets 0.88 0.90 \
  --max-num-seqs 48 --pressure-critical-seconds 45 --pressure-reserve-mib 256 \
  --num-prompts 320 --repeats 3 \
  --server-extra-arg=--stage-overrides \
  '--server-extra-arg={"0":{"gpu_memory_utilization":0.11},"1":{"gpu_memory_utilization":0.10}}'

# freeze the qualifying cell, then:
python benchmarks/hbm_admission/hypothesis_validation/H9_oom_success_effectiveness.py run \
  --output-dir benchmarks/results/H9_formal_v2 \
  --dataset-path benchmarks/build_dataset/seed_tts_long \
  --critical-target <p> --concurrency <c> --max-num-seqs 48 \
  --pressure-critical-seconds 45 --pressure-reserve-mib 256 \
  --num-prompts 320 --repeats 20 --fixed-cap 4 \
  --server-extra-arg=--stage-overrides \
  '--server-extra-arg={"0":{"gpu_memory_utilization":<frozen>},"1":{"gpu_memory_utilization":0.10}}'
```

**Fallback ladder** if the sweep still finds no cell in the 0.30–0.80 band:
1. stage-0 `gpu_memory_utilization` 0.11 → 0.09; add `--concurrencies 64`.
2. `{"0":{"hbm_limit_gb":2.0,"hbm_admission_guard":false}}` — hard KV cap with
   the static admission guard disabled so the dynamic-off arm can actually
   overflow (`hbm_limit_gb` alone auto-enables that guard).
3. longer `seed_tts_long` paragraphs; `--pressure-critical-seconds 60`.
4. if a cell exceeds 0.80, the sidecar OOMs, or the dynamic-on arm also fails:
   stage-0 `gpu_memory_utilization` up a notch, or `--pressure-reserve-mib 512`,
   or shorter paragraphs.
