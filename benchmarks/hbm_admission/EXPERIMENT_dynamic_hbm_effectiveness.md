# Dynamic HBM effectiveness experiment

## Claim under test

This experiment tests one end-to-end claim:

> Dynamic HBM monitoring detects shared-device pressure accurately and in
> time, applies a safety decision that stops new admission, reduces
> OOM/server crashes relative to the same uncontrolled workload, and restores
> capacity after pressure is released.

It is deliberately stricter than the Milestone-1 pressure smoke test.  A run
where neither controlled nor uncontrolled service can OOM is useful for
control-chain testing, but is not evidence of OOM prevention.

## Experimental controls

The formal comparison uses the same model, dataset, request count,
concurrency, pressure profile, watermarks, and GPU for every arm:

| Arm | Dynamic controller | External pressure | Purpose |
| --- | --- | --- | --- |
| C | off | on | uncontrolled counterfactual |
| D | on | on | dynamic safety controller |
| E | off, fixed cap | on | conservative static baseline |

Adjacent repetitions reverse arm order.  Each case starts a fresh server and
waits for device memory to return below the release threshold.  The pressure
sidecar has a nonzero reserve and must not itself report allocation OOM.

The current base runner starts pressure after the first HTTP response.  This
ensures the model is initialized and business traffic is active before the
shared GPU crosses the high and critical watermarks.

## Phase 1: calibration

Calibration is performed with arm C only.  Sweep critical pressure targets
and/or workload concurrency until the uncontrolled arm has a reproducible but
not deterministic failure probability.  Select a point satisfying:

```text
0.30 <= uncontrolled OOM-or-server-failure rate <= 0.80
```

The sidecar's `allocation_oom_count` must remain zero.  Do not use calibration
runs in the formal result.

**A pressure sidecar alone cannot create the danger cell on the stock deploy.**
`qwen3_tts.yaml` gives the talker an ~11.8 GiB / 110k-token KV pool (~27x
headroom for a 4096-token request), physically pre-allocated at startup.  vLLM
issues no new large allocations during decode, so the sidecar only consumes
*unused* transient headroom and never starves vLLM.  The first real sweep
(concurrency {16,24,32} x critical-target {0.93,0.95,0.97}) produced
`failure_rate == 0.0` in every cell for exactly this reason.

**Scheme A — shrink the talker KV pool, then overload it.**  Pass
`--stage-overrides` through `--server-extra-arg` (forwarded end to end) to drop
stage-0 `gpu_memory_utilization` so the KV pool becomes a genuine bottleneck,
use the long-text dataset, and stretch the critical window:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_effectiveness.py \
  calibrate --output-dir benchmarks/results/hbm_effectiveness_calibration_v2 \
  --dataset-path benchmarks/build_dataset/seed_tts_long \
  --critical-targets 0.88 0.90 --concurrencies 32 40 48 56 \
  --max-num-seqs 48 --pressure-critical-seconds 45 --pressure-reserve-mib 256 \
  --num-prompts 320 --repeats 3 \
  --server-extra-arg=--stage-overrides \
  '--server-extra-arg={"0":{"gpu_memory_utilization":0.11},"1":{"gpu_memory_utilization":0.10}}'
```

Note the `=` form: argparse rejects `--server-extra-arg --stage-overrides`
(flag-like value as a separate token).

Fallback ladder if no cell lands in `[0.30, 0.80]`:

1. stage-0 `gpu_memory_utilization` 0.11 -> 0.09; add `--concurrencies 64`.
2. `{"0":{"hbm_limit_gb":2.0,"hbm_admission_guard":false}}` — a hard KV cap with
   the static admission guard disabled (`hbm_limit_gb` alone auto-enables that
   guard, which would protect arm C and erase the C-vs-D contrast).
3. longer `seed_tts_long` paragraphs; `--pressure-critical-seconds 60`.
4. a cell above 0.80, sidecar OOM, or arm D also failing -> stage-0
   `gpu_memory_utilization` up a notch, `--pressure-reserve-mib 512`, or shorter
   paragraphs.

The desired final HBM increment must be caused by admitted inference work; a
sidecar that OOMs does not demonstrate admission-control effectiveness.

## Phase 2: frozen formal comparison

Freeze one calibrated pressure target and workload before running the formal
comparison.  Carry the **same** `--dataset-path`, `--stage-overrides`,
`--max-num-seqs`, and `--pressure-*` values that qualified the cell.  Use at
least 20 repetitions:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_effectiveness.py \
  formal --output-dir benchmarks/results/hbm_effectiveness_formal_v2 \
  --dataset-path benchmarks/build_dataset/seed_tts_long \
  --critical-target <p> --concurrency <c> --repeats 20 \
  --max-num-seqs 48 --pressure-critical-seconds 45 --pressure-reserve-mib 256 \
  --num-prompts 320 --fixed-cap 4 \
  --server-extra-arg=--stage-overrides \
  '--server-extra-arg={"0":{"gpu_memory_utilization":<frozen>},"1":{"gpu_memory_utilization":0.10}}'
```

The command writes the underlying per-case artifacts plus
`effectiveness_report.json` and `EFFECTIVENESS_SUMMARY.md`.  Re-analysis does
not require a GPU:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_effectiveness.py \
  analyze --output-dir benchmarks/results/hbm_effectiveness_formal
```

## Measurements and acceptance

The analyzer uses independent 200 ms NVML telemetry as the physical-HBM
reference and server events as the controller/action trace.

Required per-run mechanism evidence for D:

1. both shared-GPU stages publish and receive central decisions;
2. the safety cap reaches the configured critical cap (normally zero);
3. `[HBMAdmission] paused` occurs with waiting requests present;
4. `[HBMAdmission] resumed` occurs after the pressure sidecar releases memory;
5. no sidecar allocation OOM occurs.

Formal aggregate gates:

- at least 20 completed/terminal attempts per arm;
- C OOM-or-server-failure rate is at least 30%;
- D OOM-or-server-failure rate is at most 20% of C's rate;
- at least 90% of D attempts contain pause and resume evidence;
- at least 90% of D attempts receive shared-device decisions on every observed
  stage;
- P95 critical-crossing-to-cap latency is at most 1.5 seconds;
- P95 controller-vs-NVML pressure error at cap changes is at most 0.03;
- the pressure sidecar reports no allocation OOM;
- D has greater successful-request throughput/goodput than the safe fixed-cap
  baseline E when both survive.

The latency and accuracy gates require timestamped cap-change log lines.  If a
logging formatter omits timestamps, the report marks those gates unavailable
rather than treating missing evidence as success.

## Interpretation

Passing only the mechanism gates proves control-chain correctness, not OOM
prevention.  Passing C-versus-D but losing to E proves safety benefit but not
the value of dynamic capacity.  The complete claim requires all three:

```text
C fails often -> D survives by pausing admission -> D recovers and outperforms E
```

Calibration and formal results must be kept separate, and failures must remain
in the dataset.  Never aggregate only benchmark JSON files from successful
clients: server exits and missing result files are primary outcome data.
