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
| `run_all.py` | run H1–H7, print verdict table |
| `ANALYSIS.md` | consolidated results + analysis + what is / isn't established |

## Result summary (this run, commit `ce7d544d`, RTX A6000)

| ID | Verdict |
|---|---|
| H1 | **PARTIAL** — constant ~1.2 pp NVML-vs-`mem_get_info` offset (fixed, safe direction); shape/latency/monotonicity/idle all pass |
| H2–H7 | **MATCHES EXPECTATION** — all checks pass |

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
separate from the frozen formal result:

```bash
python benchmarks/hbm_admission/hypothesis_validation/H9_oom_success_effectiveness.py calibrate \
  --output-dir benchmarks/results/H9_calibration
python benchmarks/hbm_admission/hypothesis_validation/H9_oom_success_effectiveness.py run \
  --output-dir benchmarks/results/H9_formal --critical-target 0.95 \
  --concurrency 24 --repeats 20
```
