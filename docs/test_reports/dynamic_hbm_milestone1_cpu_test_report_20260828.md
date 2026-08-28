# Dynamic HBM Milestone 1 CPU Test Report

## Summary

- Date: 2026-08-28 UTC
- Branch: `dynamic-hbm`
- Base commit: `0d55a4de18f1dd5b0921b4358eee56edef92680a`
- Final result: **324 passed, 1 skipped, 0 failed**
- Warnings: 14 PyTorch deprecation warnings
- GPU/model/benchmark experiments: **not run**

This report covers the CPU-only unit and integration validation for the
Milestone-1 reactive HBM safety foundation. The validated control path is:

```text
rank memory telemetry
  -> replica aggregation
  -> reactive safety state machine
  -> centralized shared-device decision
  -> scheduler SafetyCap/admission gate
  -> budget-applied acknowledgement
```

## Environment

| Item | Value |
| --- | --- |
| Host | `vllm-test` |
| Kernel | Linux 6.8.0-124-generic x86_64 |
| Python | 3.12.3 |
| pytest | 9.1.1 |
| PyTorch | 2.13.0+cu132 |
| pyzmq | 27.1.0 |

Although the installed PyTorch build has CUDA support, all tests in this
report used mocked memory providers, fake schedulers/executors, or localhost
ZMQ. No CUDA device, model weights, or inference server was used.

The repository defaults to xdist (`--dist=loadgroup`). These runs explicitly
used `-o addopts=''` so the safety-control tests ran serially and their
thread/ZMQ behavior remained deterministic.

## Final command

```bash
.venv/bin/pytest -o addopts='' -q \
  tests/core/memory_coordinator \
  tests/core/sched/test_dynamic_hbm_admission_gate.py \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/core/sched/test_omni_scheduler_mixin_shared.py \
  tests/engine/test_dynamic_hbm_control_loop.py \
  tests/engine/test_stage_engine_core_proc.py \
  tests/engine/test_async_omni_engine_stage_init.py \
  tests/distributed/omni_coordinator/test_omni_coord_client_for_stage.py \
  tests/distributed/omni_coordinator/test_omni_coordinator.py \
  tests/distributed/omni_coordinator/test_hbm_failure_semantics.py \
  tests/distributed/omni_coordinator/test_hbm_concurrency.py \
  tests/integration/test_dynamic_hbm_shared_device.py \
  tests/config/test_omni_config.py
```

Final output:

```text
324 passed, 1 skipped, 14 warnings in 16.77s
```

The reported pytest duration excludes part of the process cold-start cost.
The first run spent several minutes in kernel file-page reads while importing
the full vLLM dependency chain from shared storage. Subsequent runs benefited
from the system page cache.

## Test coverage

### 1. Configuration and protocol

Files:

- `tests/core/memory_coordinator/test_config_protocol.py`
- `tests/core/memory_coordinator/test_allocator.py`
- `tests/config/test_omni_config.py`

Validated:

- fail-closed defaults for critical pressure and coordinator disconnect;
- validation of cap, guard, recovery, and sampling settings;
- compatibility of legacy and structured deploy configuration projection;
- separate raw physical-HBM and logical KV pressure;
- guard-adjusted physical pressure;
- baseline completeness and process-reserved delta clamping;
- external-or-unattributed memory clamping;
- fail-safe empty-rank behavior;
- budget-decision serialization with safety state and pressure source.

### 2. Reporter and replica aggregation

Files:

- `tests/core/memory_coordinator/test_reporter.py`
- `tests/core/memory_coordinator/test_aggregator.py`

Validated:

- worker/rank device snapshots;
- resident baseline generation and fields;
- sample timestamps;
- newest-report rank deduplication;
- stage/replica/rank filtering;
- TP/PP expected-rank completeness;
- most-pressured-rank aggregation;
- models without a real KV cache do not report dummy KV pressure;
- delayed polling still consumes an already-completed future;
- timed-out or missing reports do not become low-pressure samples.

### 3. Reactive safety state machine

File:

- `tests/core/memory_coordinator/test_safety_state_machine.py`

Validated:

- multiplicative decrease at high pressure;
- critical physical or KV pressure clamps admission cap to zero;
- pressure-source attribution (`physical_hbm`, `kv`, or both);
- critical cap does not expand while pressure remains high or in hysteresis;
- separate complete-report recovery and stable-low expansion windows;
- incomplete-report grace followed by conservative decrease;
- repeated stale samples never expand capacity;
- fixed HBM guard can trigger an earlier critical transition;
- configured cap remains an absolute upper bound.

### 4. Scheduler SafetyCap and admission

Files:

- `tests/core/sched/test_dynamic_hbm_admission_gate.py`
- `tests/core/sched/test_omni_ar_scheduler_streaming.py`
- `tests/core/sched/test_omni_scheduler_mixin_shared.py`

Validated:

- independent configured, safety, and effective caps;
- high-pressure decisions modify SafetyCap, not configured capacity;
- a critical state cannot carry a nonzero wire cap past the local clamp;
- cap zero blocks new admission;
- running and streaming-occupied slots are retained;
- token budget remains sufficient for already-running requests;
- admission resumes only when occupancy falls below the recovered cap;
- stale decision and stale report generations are rejected;
- unknown safety state is rejected without crashing the scheduler;
- local KV and disconnect guards do not consume central decision generations;
- a fresh central decision can reconcile a local guard;
- token-budget scaling can be disabled;
- AR and generation schedulers share the same dynamic-HBM state ownership.

### 5. EngineCore control loop

Files:

- `tests/engine/test_dynamic_hbm_control_loop.py`
- `tests/engine/test_stage_engine_core_proc.py`
- EngineCore portions of `tests/core/memory_coordinator/test_aggregator.py`

Validated:

- baseline capture is enabled only for dynamic-HBM stages;
- baseline RPC failure is nonfatal;
- periodic and immediate samples use separate rate limits;
- a pending future prevents duplicate collective RPCs;
- KV exhaustion applies a local guard and requests an immediate sample;
- non-KV stages skip local KV handling;
- report-send and decision-poll failure activate disconnect protection;
- decision identity is fenced by instance, stage, and replica;
- an accepted decision is applied and acknowledged once;
- ACK transport failure cannot roll back an applied safety cap;
- existing Omni request fields remain preserved by EngineCore preprocessing.

### 6. Coordinator and client failure semantics

Files:

- `tests/distributed/omni_coordinator/test_omni_coord_client_for_stage.py`
- `tests/distributed/omni_coordinator/test_omni_coordinator.py`
- `tests/distributed/omni_coordinator/test_hbm_failure_semantics.py`
- `tests/distributed/omni_coordinator/test_hbm_concurrency.py`

Validated:

- stage registration, heartbeat, update, reconnect, and shutdown;
- memory-report generation and decision polling;
- shared-device physical pressure propagation;
- replica-local KV pressure isolation;
- different GPU UUIDs do not share physical pressure;
- empty-rank telemetry is dropped without killing the receive thread;
- budget-applied ACK generation and instance fencing;
- unsolicited ACKs are rejected;
- decision-to-apply latency is recorded;
- stale reports never expand the cap;
- a new replica instance clears old memory/ACK state;
- concurrent reports from four shared-GPU consumers preserve coordinator
  receive and periodic thread liveness.

### 7. Runtime lifecycle and CPU control-chain integration

Files:

- `tests/engine/test_async_omni_engine_stage_init.py`
- `tests/integration/test_dynamic_hbm_shared_device.py`

Validated:

- no local coordinator is created when dynamic HBM is disabled;
- exactly one local coordinator is created when any local stage enables it;
- coordinator address propagation to replicas;
- initialization-failure and shutdown cleanup;
- shared critical pressure shrinks both simulated replicas without deleting
  running work;
- cap recovery releases only the available number of admission slots;
- disconnect protection retains running work and accepts later reconciliation;
- local KV protection is reconciled by a fresh central report generation.

## Failures found and fixes made during testing

### 1. Partially initialized scheduler compatibility

Initial scheduler tests exposed direct accesses to
`_last_budget_generation`, `_last_budget_report_generation`, and
`vllm_config` on scheduler objects created with `__new__`.

Fix:

- generation fields now default conservatively to zero through `getattr`;
- logging tolerates a missing model config;
- fully initialized production schedulers retain the same behavior.

### 2. Token budget for running requests

An old assertion expected the token budget to scale only from the new
admission cap. That would under-provision already-running requests after a
shrink.

Fix/validated behavior:

```text
scheduling slots = max(effective admission cap, occupied running slots)
```

For five running requests under a cap of four, a configured token budget of
4096 and configured sequence cap of 16 therefore retains 1280 scheduled
tokens, rather than dropping to the cap-only value of 1024.

### 3. ACK authenticity

The first ACK implementation accepted a generation even when the live
coordinator had never emitted that decision.

Fix:

- an ACK is now accepted only if `(input_addr, decision_generation)` exists in
  the live coordinator's sent-decision table;
- unsolicited, old-instance, and duplicate ACKs do not advance applied state.

### 4. Existing PUB/SUB shutdown timing test

The first full combined run produced:

```text
323 passed, 1 skipped, 1 failed
```

The failure was the existing replica-shutdown PUB/SUB test timing out while
waiting for the empty active-replica broadcast. It passed immediately when
rerun alone, and the unchanged test passed in the final full run. This is
recorded as an existing slow-joiner/timing-sensitive test, not an HBM logic
failure.

## Skip and warnings

### Skip

One existing parameterized configuration test calls `pytest.skip` when a
pipeline model type requires an unavailable Hugging Face config:

```text
tests/config/test_omni_config.py:177
```

This skip is unrelated to dynamic-HBM behavior.

### Warnings

All 14 warnings are the same upstream PyTorch deprecation warning:

```text
torch.jit.script_method is deprecated; use torch.compile or torch.export
```

No warning originated from the new HBM code or tests.

## Static validation

In addition to pytest:

```bash
python -m compileall -q <changed source and test paths>
git diff --check
```

Both completed successfully.

## Out of scope for this report

The following remain intentionally unverified here:

- real CUDA `mem_get_info`, allocated, and reserved readings;
- baseline stability after real model load, CUDA graph capture, sleep, and wake;
- physical OOM prevention under an external memory-pressure process;
- monitoring overhead, throughput, P99 latency, and control-loop latency;
- diffusion engine monitoring outside the StageEngineCore AR/generation path;
- multi-node network delay and process-crash recovery;
- Milestone-2+ prediction, reservation, planner, and cross-stage backpressure.

Those items require GPU or distributed experiments and should be reported
separately from this CPU correctness report.

## Conclusion

The CPU suite validates the intended Milestone-1 safety invariants:

1. physical HBM and logical KV pressure remain distinct;
2. critical, stale, and disconnected states cannot create unsafe expansion;
3. SafetyCap can stop new admission without evicting running work;
4. shared physical pressure propagates only across consumers of the same
   physical device, while KV pressure remains replica-local;
5. instance/report/decision/ACK generations fence stale control messages;
6. local emergency guards can be reconciled by fresh central decisions.

Final CPU status: **PASS**.
