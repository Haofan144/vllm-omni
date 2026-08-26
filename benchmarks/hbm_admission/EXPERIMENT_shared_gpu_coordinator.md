# Shared-GPU HBM coordinator experiment plan

## Objective

Validate that the centralized coordinator detects two stage processes sharing
one physical GPU, applies one consistent pressure signal to both stages,
prevents OOM under external pressure, and restores concurrency without
oscillation after pressure is released.

The primary workload is Qwen3-TTS because its default deploy places stage 0
(talker) and stage 1 (code2wav) on device `0` as independent stage processes.

## Hypotheses

1. With dynamic HBM disabled, external pressure can cause OOM, preemption, or
   severe tail-latency degradation at high concurrency.
2. With centralized dynamic HBM enabled, both stages receive cap reductions
   when their shared physical GPU crosses the high watermark.
3. At the critical watermark both stages reach `min_num_seqs` within two
   report intervals, without aborting already-running requests.
4. After pressure is released, caps recover additively only after the configured
   stable-sample window and never exceed their configured maxima.
5. With no external pressure, centralized monitoring adds no more than 5%
   throughput overhead and no more than 10% p99 latency overhead.

## Test layers

### Layer 1: deterministic CPU tests

Run:

```bash
.venv/bin/pytest -q \
  tests/core/memory_coordinator \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/distributed/omni_coordinator/test_omni_coord_client_for_stage.py \
  tests/distributed/omni_coordinator/test_omni_coordinator.py \
  --confcutdir=tests
```

Required cases:

- reports with the same `(node_id, device_uuid)` affect both stages;
- equal local `device_id` on different nodes does not create sharing;
- stage-private KV pressure changes only that stage's cap;
- stale and duplicate report/decision generations are ignored;
- stage shutdown and heartbeat timeout remove it from the device consumer set;
- a restarted stage with a new `instance_id` cannot receive an old decision;
- lowering a cap never evicts already-running requests.

### Layer 2: single-GPU functional waveform

Use one A100/H100 and the two-stage Qwen3-TTS deploy. Run only the dynamic arm
first so failures are cheap to diagnose:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_experiment.py \
  --output-dir benchmarks/results/shared_gpu_functional \
  --arms D_dynamic_on_pressure \
  --repeats 1 \
  --concurrency 32 \
  --num-prompts 400 \
  --max-num-seqs 32 \
  --min-num-seqs 2 \
  --sample-interval-ms 500 \
  --report-timeout-ms 1500 \
  --low-watermark 0.72 \
  --high-watermark 0.82 \
  --critical-watermark 0.90 \
  --scale-down-ratio 0.5 \
  --scale-up-step 2 \
  --scale-up-stable-samples 6 \
  --pressure-baseline-seconds 15 \
  --pressure-high-target 0.84 \
  --pressure-high-seconds 20 \
  --pressure-critical-target 0.92 \
  --pressure-critical-seconds 15 \
  --pressure-recovery-target 0.68 \
  --pressure-recovery-seconds 20 \
  --pressure-post-release-seconds 45
```

Expected cap waveform for both stage 0 and stage 1:

```text
baseline: 32
high pressure: 32 -> 16, possibly 16 -> 8 if pressure persists
critical pressure: -> 2
recovery: hold for 6 samples, then 2 -> 4 -> 6 ... -> 32
```

The exact number of multiplicative decreases depends on how long measured
pressure remains above the watermark. Both stages need not have identical caps
at every instant, but a shared-device high-pressure event must produce a
`shared_device_*` decision for both.

Functional acceptance criteria:

- server log contains cap changes for both `stage=0` and `stage=1`;
- both stages contain at least one `reason=shared_device_high_pressure` or
  `reason=shared_device_critical_pressure` event;
- minimum observed cap is `min_num_seqs` during the critical phase;
- the first decrease occurs within `2 * sample_interval + report_timeout`;
- no cap is below 2 or above 32;
- no OOM, traceback, failed request, or server exit;
- caps increase only after six consecutive low-pressure samples.

### Layer 3: controlled comparison

Run all four arms with at least five repeats. Arms are counterbalanced by the
runner to reduce warm-cache and temporal bias.

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_experiment.py \
  --output-dir benchmarks/results/shared_gpu_abcd \
  --repeats 5 \
  --concurrency 32 \
  --num-prompts 1000 \
  --include-fixed-cap 8
```

Arms:

| Arm | Dynamic coordinator | External pressure | Purpose |
| --- | --- | --- | --- |
| A | off | off | unconstrained baseline |
| B | on | off | monitoring overhead |
| C | off | on | unsafe-pressure baseline |
| D | on | on | proposed design |
| E | off, fixed cap 8 | on | conservative static baseline |

Primary metrics:

- successful request ratio and OOM/server-crash count;
- request throughput;
- p50/p95/p99 E2E and audio TTFP;
- minimum cap, number of cap changes, and time-to-decrease/recovery;
- GPU peak pressure and duration above critical watermark.

Comparison criteria:

- D must have zero OOM and at least 99.9% request success;
- if C OOMs or fails requests, D must complete the same workload;
- B throughput must be at least 95% of A;
- B p99 E2E must be at most 110% of A;
- D should outperform E in throughput or p99 latency while retaining E's
  safety; otherwise dynamic coordination has not justified its complexity;
- repeated-run values should be reported as median and bootstrap 95% CI, not
  only a single mean.

### Layer 4: fault injection and soak

Run the following independently:

1. Kill stage 1 after it has reported memory. Verify heartbeat timeout removes
   its memory state and no further decisions target its old instance.
2. Restart stage 1 with the same stage/replica IDs. Verify the new instance
   starts a fresh generation and ignores old decisions.
3. Pause memory reporting while keeping heartbeat alive. The coordinator must
   never interpret missing data as free headroom.
4. Inject duplicate and reversed report generations. Caps must not roll back.
5. Run an 8-hour workload with alternating high/critical/recovery pressure.

Soak acceptance criteria:

- zero OOM and zero server restart;
- coordinator and stage process RSS do not grow monotonically;
- no stale route or old `instance_id` remains after restart;
- cap changes remain bounded and do not oscillate inside the hysteresis band;
- all submitted requests eventually complete or return an explicit error.

## Required artifacts

Retain per case:

- generated deploy YAML and exact command;
- server/client logs;
- benchmark JSON;
- 200 ms GPU telemetry;
- pressure-generator JSONL;
- parsed cap events with stage, replica, generation, reason and pressure;
- process exit codes and request failure counts.

Do not accept a run merely because it has no OOM. A valid dynamic run must also
show cap changes for both shared-GPU stages and subsequent additive recovery.

## Known limitation of the current implementation

The centralized coordinator currently applies the shared device pressure to a
per-stage AIMD allocator. It does not yet compute one aggregate byte/slot budget
and divide it by stage weight or marginal bytes per sequence. These experiments
therefore validate centralized observation and coordinated throttling, not
weighted-fair global allocation.
