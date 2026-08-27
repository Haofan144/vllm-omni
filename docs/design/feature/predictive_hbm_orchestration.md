# Predictive cross-stage resource orchestration

## 1. Status and purpose

This document is the normative implementation plan for evolving the current
dynamic-HBM prototype into a predictive cross-stage resource orchestrator.
It is intentionally specific enough that independent implementations should
produce the same wire protocol, state transitions, safety behavior, and test
results.

The implementation MUST be delivered incrementally in the milestones defined
in this document. A milestone MUST NOT depend on behavior assigned to a later
milestone. Existing static HBM admission and centralized AIMD behavior MUST
remain available behind configuration flags throughout the migration.

Normative terms have their usual meanings:

- **MUST**: required for correctness or compatibility;
- **SHOULD**: required unless a documented reason and test justify deviation;
- **MUST NOT**: prohibited because it violates the resource model or safety;
- **MAY**: optional and not required for milestone completion.

## 2. Objective

The system prevents future resource over-commitment in a heterogeneous
multi-stage inference pipeline by combining:

1. stage- and backend-specific resource estimation;
2. per-physical-device global planning;
3. predictive downstream demand forecasts;
4. atomic hard reservations;
5. cross-stage admission backpressure; and
6. a fast reactive safety controller for unpredicted pressure.

The primary objective is:

> Maximize completed request throughput, and later SLO goodput, without
> over-committing stage-local logical capacity or shared physical GPU
> transient headroom.

The system allocates **future admissible capacity**. It does not claim to move
or resize already resident model weights or preallocated KV-cache pools.

## 3. Non-goals

The first six milestones MUST NOT introduce:

- physical resizing of an initialized vLLM KV-cache pool;
- request migration between live replicas;
- CPU KV offload or proactive model offload;
- dynamic replica creation or destruction;
- arbitrary graph routing in code paths that currently assume `stage_id + 1`;
- replacement of the upstream vLLM waiting-queue policy;
- a general mathematical solver in the coordinator hot path;
- hard reservation of transfer buffers before transfer accounting is added;
- synchronized absolute monotonic clocks across nodes.

Until an explicit stage graph replaces the current sequential forwarding
assumption, documentation and paper text MUST use **pipeline-aware** or
**stage-chain-aware**, not **arbitrary DAG-aware**.

## 4. Existing implementation to preserve

The current implementation provides:

- `DynamicHBMConfig` in `vllm_omni/core/memory_coordinator/config.py`;
- rank-level physical and process memory telemetry in `reporter.py`;
- replica-level rank aggregation and KV-block telemetry in `aggregator.py`;
- watermark/AIMD control in `allocator.py`;
- shared-device discovery using `(node_id, device_uuid)` in
  `distributed/omni_coordinator/omni_coordinator.py`;
- generation and `instance_id` checks for stale-message rejection;
- scheduler-thread application of `effective_max_num_seqs`; and
- experiment and fault-injection scaffolding under
  `benchmarks/hbm_admission/`.

These mechanisms MUST be reused. Generation checks, instance identity, stale
report handling, configured maximum sequence count, and the rule that a cap
reduction does not evict already-running requests MUST remain intact.

## 5. Terminology and resource model

### 5.1 Device identity

A physical GPU is identified by:

```python
DeviceKey = tuple[str, str]  # (node_id, device_uuid)
```

`device_id` MUST NOT be used as a cluster-wide identity. When a UUID is not
available, a fallback identifier MAY be used only for single-node mode and
MUST include `node_id`.

### 5.2 Physical and logical resources

The implementation MUST NOT combine all resources into one byte scalar.

Physical GPU transient headroom covers memory not already resident:

```text
activations
runtime workspace growth
transfer/staging buffers
CUDA/NCCL transient allocations
untracked-process growth
uncertainty guard
```

Stage-local logical capacity covers already allocated pools or backend-owned
limits:

```text
AR KV-cache blocks
diffusion execution slots/batch envelope
backend-specific persistent request state
```

Free KV blocks MUST NOT be added to `device_free_bytes`. KV tensors are often
already physically allocated, so doing so would double-count capacity.

### 5.3 Resource estimate

The canonical request estimate introduced in Milestone 2 is:

```python
@dataclass(frozen=True)
class RequestResourceEstimate:
    request_class: str
    kv_blocks: int = 0
    transient_bytes_by_device: dict[DeviceKey, int] = field(default_factory=dict)
    transfer_bytes_by_device: dict[DeviceKey, int] = field(default_factory=dict)
    persistent_bytes_by_device: dict[DeviceKey, int] = field(default_factory=dict)
    workspace_class: str | None = None
    slots: int = 1
    confidence: float = 1.0
```

All numeric values MUST be non-negative. `confidence` MUST be in `(0, 1]`.
Unknown cost MUST be represented conservatively by a configured fallback,
not by zero.

### 5.4 Physical transient capacity and emergency headroom

The planner MUST NOT subtract a CONSUMED request prediction directly from the
current `device_free_bytes`: memory already materialized by that request has
already reduced free HBM and would be counted twice.

Each stage process MUST establish a `resident_baseline_bytes` sample after
model/KV-pool initialization and before admitting requests. It includes fixed
weights, preallocated KV tensors, CUDA graphs captured during initialization,
and other process-resident allocations. Recapture requires an explicit stage
restart or a future protocol not defined here; it MUST NOT drift upward during
normal traffic.

For physical device `g`, let:

```text
resident_g = sum(process resident baselines on g)
process_reserved_g = sum(current process-reserved bytes on g)
observed_dynamic_g = sum(max(0, process_reserved - resident_baseline))
external_g = max(0, total_g - observed_free_g - process_reserved_g)
predicted_active_g = sum(predicted transient peaks of CONSUMED reservations)
active_charge_g = max(observed_dynamic_g, predicted_active_g)
transient_capacity_g = max(0, total_g - resident_g - external_g - guard_g)
```

The normal planner constraint is:

```text
active_charge_g + sum(HELD transient reservations on g)
    <= transient_capacity_g
```

This uses a conservative maximum rather than adding observed and predicted
active demand. Allocator-cache growth may make `observed_dynamic_g` remain
high after a request completes; this is safe and SHOULD be exposed in metrics
so its utilization cost is visible.

Separately, the fast safety controller uses instantaneous emergency headroom:

```text
emergency_headroom_g = observed_free_g - emergency_guard_g
```

If emergency headroom is negative, no planner expansion or new hard hold is
allowed and the safety controller immediately tightens admission.

`observed_free_g` MUST be the minimum fresh `device_free_bytes` value reported
for the same `DeviceKey` in the planning snapshot. This is conservative when
multiple colocated processes observe the same device at slightly different
times.

The planner MUST separately retain:

- current physical free bytes;
- current process allocated/reserved bytes for diagnostics;
- untracked bytes when derivable;
- hard transient commitment;
- soft forecast demand; and
- the safety guard.

### 5.5 Logical capacity

For AR replica `i`:

```text
available_kv_blocks_i =
    reported_free_kv_blocks_i - held_kv_blocks_i
```

For a non-AR backend, a provider MUST define its logical capacity and batch
feasibility. Diffusion workspace MUST be modeled as a batch-level envelope;
per-request workspace peaks MUST NOT be blindly summed.

### 5.6 Admission invariant

A hard reservation for request `r` on replica `i` is feasible only if all
constraints hold atomically:

```text
required_kv_blocks_r <= available_kv_blocks_i
required_slots_r <= available_slots_i
for every device g used by replica i:
    active_charge_g
    + held_transient_commitment_g
    + request_transient_bytes_r,g
    + request_transfer_bytes_r,g
    <= transient_capacity_g
```

For tensor-parallel replicas, the reservation MUST be all-or-nothing across
all devices.

## 6. Safety invariants

Every milestone MUST preserve these invariants.

### INV-1: No physical over-grant

For every physical device:

```text
active_charge_g + sum(HELD transient grants on g)
    <= transient_capacity_g
```

### INV-2: No logical over-grant

For every stage replica:

```text
sum(active held KV blocks) <= reported free KV blocks
sum(active held slots) <= reported free slots
```

### INV-3: Atomic multi-device grant

A request has a hard reservation on every required device or on none.

### INV-4: Shrink before expand

Capacity removed from one consumer MUST be acknowledged as applied, or its
old lease MUST expire, before the same capacity can expand another consumer.

### INV-5: Fast safety dominance

The effective stage cap is always:

```python
effective_cap = min(configured_cap, planner_cap, safety_cap)
```

The planner MUST NOT override a tighter safety cap.

### INV-6: Idempotent reservation lifecycle

Duplicate or reordered messages MUST NOT count the same reservation more than
once or resurrect a terminal reservation.

### INV-7: Fail closed

Expired decisions, missing reports, unknown cost classes, and coordinator
disconnects MUST prevent new expansion. They MUST NOT kill running requests.

### INV-8: Single socket owner

Each ZMQ socket MUST be created, used, and closed by one owning thread.

## 7. Target architecture

```text
Orchestrator
  |-- soft pipeline forecasts ------------------------------+
  |-- hard hold / consume / release RPCs ----------------+  |
  |                                                       |  |
  v                                                       v  v
StagePool                                           Coordinator I/O loop
  | candidate replica ordering                      | ROUTER receive/send
  | grant-bound routing                             | heartbeat timers
  v                                                 | planner timers
Stage EngineCore                                    | outgoing queue flush
  | resource reports                                +----------+
  | planner/safety decisions                                   |
  v                                                            v
Stage scheduler                                           Snapshot planner
  | effective=min(configured, planner, safety)            | device allocator
  v                                                       | forecast table
Workers                                                   | reservation ledger
  | rank/device/backend telemetry                         | safety controller
  +-------------------------------------------------------+
```

Expensive planning MAY run outside the I/O thread on an immutable snapshot.
All ROUTER sends MUST return to the owning I/O loop through an in-process
queue. Planner results MUST be rejected if the input snapshot generation is
no longer current enough to apply safely.

### 7.1 Milestone dependency and deliverable summary

Milestones are strictly ordered. An implementation MUST complete the required
tests and completion criteria of every predecessor before starting a dependent
milestone.

| Milestone | Depends on | Primary deliverable | Runtime enforcement |
| --- | --- | --- | --- |
| M0 | current branch | Safety/planner cap separation | Existing AIMD behavior |
| M1 | M0 | Per-device snapshot and global reactive planner | Stage sequence cap |
| M2 | M1 | Backend-aware multidimensional cost model | Planner dry-run/cap |
| M3 | M2 | Atomic reservation ledger and leases | Coordinator accounting only |
| M4 | M3 | Pipeline prediction and cross-stage backpressure | Orchestrator/StagePool gate |
| M5 | M4 | Minimal heterogeneous diffusion support | AR and diffusion admission |
| M6 | M5 | SLO-aware planner weighting | Same safe admission path |
| M7 | M4; normally M6 | Transfer-buffer commitment | Connector lifetime accounting |

M7 may be developed after M4 in a separate branch, but it MUST NOT be merged
into the core paper path until M5 and M6 remain green with it enabled and
disabled.

## 8. Configuration schema

`DynamicHBMConfig` MUST remain backward compatible. Add the following fields
with these defaults:

```python
mode: Literal["reactive", "global_reactive", "predictive"] = "reactive"

planner_interval_ms: int = 1000
decision_lease_ms: int = 3000
revoke_ack_timeout_ms: int = 1500

device_guard_bytes: int = 2 * 1024**3
unknown_request_transient_bytes: int = 512 * 1024**2
unknown_request_kv_blocks: int = 1

soft_forecast_discount: float = 0.4
hard_reservation_horizon_ms: int = 500
reservation_lease_ms: int = 1500

cost_model_type: Literal["static_profile", "profile_ewma"] = "static_profile"
cost_profile_path: str | None = None
ewma_alpha: float = 0.2
uncertainty_multiplier: float = 1.2

min_planner_num_seqs: int = 1
max_cap_change_per_epoch: int = 4
```

Validation MUST enforce:

- every interval and lease is positive;
- `decision_lease_ms >= planner_interval_ms`;
- `revoke_ack_timeout_ms <= decision_lease_ms`;
- `reservation_lease_ms > 0`;
- byte and block fallbacks are non-negative, with unknown KV blocks at least 1;
- discounts and EWMA alpha are in `(0, 1]`;
- uncertainty multiplier is at least 1;
- minimum planner cap is at least 1;
- maximum cap change is at least 1; and
- `predictive` mode requires a usable cost profile or explicit conservative
  fallback values.

Configuration MUST flow through:

- `vllm_omni/core/memory_coordinator/config.py`;
- `vllm_omni/config/model.py`;
- `vllm_omni/config/omni_config.py`;
- `vllm_omni/config/stage_config.py`;
- `vllm_omni/engine/arg_utils.py`; and
- generated deploy configuration used by benchmark runners.

## 9. Wire protocol

All new messages MUST include:

```python
schema_version: int  # initially 2
message_type: str
sender_instance_id: str
generation: int
```

Unknown fields MUST be ignored. Unknown message types MUST be logged and
dropped. Schema version 1 memory reports and budget decisions MUST continue to
work in `reactive` mode.

Do not send an absolute cross-node monotonic expiry. Send `lease_duration_ms`;
the receiver computes local expiry from its own monotonic clock.

### 9.1 Stage resource report

```python
@dataclass(frozen=True)
class DeviceResourceReport:
    node_id: str
    device_uuid: str
    rank: int
    device_total_bytes: int
    device_free_bytes: int
    process_allocated_bytes: int
    process_reserved_bytes: int
    resident_baseline_bytes: int
    pending_peak_bytes: int = 0
    active_transfer_bytes: int = 0

@dataclass(frozen=True)
class StageResourceReport:
    stage_id: int
    replica_id: int
    instance_id: str
    report_generation: int
    observed_admission_generation: int
    devices: tuple[DeviceResourceReport, ...]
    kv_total_blocks: int | None
    kv_free_blocks: int | None
    running_requests: int
    waiting_requests: int
    configured_max_num_seqs: int
    backend_type: str
    current_slots: int | None = None
    max_slots: int | None = None
```

### 9.2 Resource decision

```python
@dataclass(frozen=True)
class StageResourceDecision:
    stage_id: int
    replica_id: int
    instance_id: str
    planner_epoch: int
    decision_generation: int
    based_on_report_generation: int
    phase: Literal["revoke", "grant", "steady"]
    planner_max_num_seqs: int
    safety_max_num_seqs: int
    device_headroom_bytes: dict[str, int]
    lease_duration_ms: int
    reason: str
```

`device_headroom_bytes` is diagnostic/planning state, not permission to resize
an existing KV pool.

### 9.3 Revoke acknowledgement

```python
@dataclass(frozen=True)
class ResourceDecisionAck:
    stage_id: int
    replica_id: int
    instance_id: str
    planner_epoch: int
    decision_generation: int
    applied_planner_max_num_seqs: int
```

The stage MUST send this ACK only after the scheduler thread has applied the
revoke.

### 9.4 Forecast

Soft forecasts do not reserve capacity:

```python
@dataclass(frozen=True)
class PipelineResourceForecast:
    forecast_id: str
    request_id: str
    version: int
    target_stage_id: int
    candidate_replica_ids: tuple[int, ...]
    estimate: RequestResourceEstimate
    expected_arrival_after_ms: int
    confidence: float
    lease_duration_ms: int
```

### 9.5 Hard reservation

```python
class ReservationState(str, Enum):
    HELD = "held"
    CONSUMED = "consumed"
    RELEASED = "released"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass(frozen=True)
class ReservationRequest:
    reservation_id: str
    request_id: str
    version: int
    stage_id: int
    replica_id: int
    estimate: RequestResourceEstimate
    lease_duration_ms: int

@dataclass(frozen=True)
class ReservationReply:
    reservation_id: str
    request_id: str
    version: int
    stage_id: int
    replica_id: int
    accepted: bool
    lease_duration_ms: int
    reason: str
```

`refine` is a higher-version replacement of an existing HELD reservation, not
a separate state. Replacement MUST be atomic: either the new estimate replaces
the old estimate or the old hold remains unchanged.

### 9.6 HELD-to-CONSUMED accounting

HELD transient demand is charged as future demand. CONSUMED transient demand
is moved from the held total into `predicted_active_g`; it is not added to both.
The physical-device active charge is then the conservative maximum of observed
dynamic process memory and predicted active demand as defined in Section 5.4.

For logical KV blocks, HELD blocks are subtracted from the latest reported free
blocks. A CONSUMED reservation remains in `predicted_active_kv_blocks` until a
stage report explicitly covers the corresponding admission generation. After
that report, its blocks are already reflected in reported free-block depletion
and the separate predicted logical charge is removed. To support this
reconciliation:

```python
ReservationConsume:
    reservation_id: str
    version: int
    admission_generation: int

StageResourceReport:
    observed_admission_generation: int
```

The stage increments `admission_generation` when a granted request is accepted
by the stage input path. A report covers all admissions up to
`observed_admission_generation`. Until such coverage, the coordinator keeps the
logical prediction charged, preventing an admission/report race from
temporarily over-granting KV blocks.

The accepted submission returns:

```python
@dataclass(frozen=True)
class StageAdmissionReceipt:
    stage_id: int
    replica_id: int
    instance_id: str
    request_id: str
    reservation_id: str
    admission_generation: int
```

The receipt MUST be created by the receiving stage input path, not guessed by
the orchestrator. A failed submission returns no receipt and leaves the
reservation HELD until it is explicitly cancelled or expires.

## 10. Control-plane threading model

The final coordinator MUST have one I/O thread that owns ROUTER and PUB
sockets. Its loop MUST perform:

1. poll ROUTER input;
2. decode and validate a bounded number of messages;
3. process due heartbeat/report/reservation expiry timers;
4. request a planner run when the planner interval elapses;
5. consume completed planner results;
6. flush revoke, grant, ACK response, and replica-list messages; and
7. stop cleanly before closing sockets.

Planner computation MAY use a worker thread, but it MUST receive an immutable
snapshot and MUST NOT access a ZMQ socket. There MUST be at most one in-flight
planner calculation. If another interval elapses, coalesce it into one pending
run using the newest snapshot.

## 11. Milestone 0: Baseline freeze and behavior-preserving refactor

### Goal

Create a safe refactoring baseline without changing runtime decisions.

### Required implementation

1. Move existing AIMD implementation from `allocator.py` to
   `safety_controller.py` and rename the class to `SafetyController`.
2. Keep this compatibility alias in `allocator.py`:

   ```python
   BudgetAllocator = SafetyController
   ```

3. Add separate scheduler fields:

   ```python
   _configured_max_num_seqs
   _planner_max_num_seqs
   _safety_max_num_seqs
   _effective_max_num_seqs
   ```

   In Milestone 0, planner cap MUST equal configured cap, so behavior is
   unchanged.
4. Centralize cap recomputation in one method:

   ```python
   def _recompute_effective_resource_cap(self) -> None:
       self._effective_max_num_seqs = min(
           self._configured_max_num_seqs,
           self._planner_max_num_seqs,
           self._safety_max_num_seqs,
       )
   ```

5. Preserve the existing occupied-slot rule: a reduced cap prevents new
   admission but does not evict running or streaming requests.
6. Add structured log fields `configured_cap`, `planner_cap`, `safety_cap`,
   and `effective_cap`.

### Files

- Add `vllm_omni/core/memory_coordinator/safety_controller.py`.
- Modify `allocator.py`, `__init__.py`, `omni_ar_scheduler.py`, and tests.

### Tests

- Move existing allocator tests to `test_safety_controller.py` while retaining
  one compatibility import test for `BudgetAllocator`.
- Test every ordering of configured/planner/safety caps.
- Test that cap reduction below occupied slots does not evict requests.
- Run all existing dynamic-HBM tests unchanged.

### Completion criteria

- Existing reactive benchmark emits the same cap sequence for the same report
  sequence.
- No wire schema changes.
- No planner or reservation code exists yet.

## 12. Milestone 1: Per-device snapshots and global reactive planner

### Goal

Replace report-triggered independent allocation with periodic, consistent,
per-device planning while retaining cap-based enforcement.

### Required implementation

1. Add `device_snapshot.py` with:

   ```python
   @dataclass(frozen=True)
   class PhysicalDeviceSnapshot:
       key: DeviceKey
       total_bytes: int
       observed_free_bytes: int
       guard_bytes: int
       fresh_consumer_ids: tuple[ReplicaKey, ...]
       stale_consumer_ids: tuple[ReplicaKey, ...]

   @dataclass(frozen=True)
   class PlannerSnapshot:
       snapshot_generation: int
       captured_at_s: float
       devices: dict[DeviceKey, PhysicalDeviceSnapshot]
       replicas: dict[ReplicaKey, ReplicaMemoryReport]
   ```

2. Reports MUST only update the latest state. They MUST NOT directly run the
   planner.
3. Build a periodic snapshot using only reports received within
   `report_timeout_ms`. A replica with a stale report MUST receive a tighter
   safety cap and MUST NOT receive planner expansion.
4. Add `planner.py` with `GlobalReactivePlanner`. Milestone 1 uses a
   deterministic unit-demand model because request cost modeling belongs to
   Milestone 2.
5. Group consumers by physical `DeviceKey`. A replica spanning multiple
   devices participates in each device group.
6. For a shared device under high pressure, reduce consumers in this order:

   - consumers above their configured minimum;
   - largest current cap first;
   - tie-break by `(stage_id, replica_id)`.

   Reductions MUST respect `max_cap_change_per_epoch` except at critical
   pressure, where safety cap MAY fall directly to `min_num_seqs`.
7. For stable low pressure, expansion uses round-robin fairness across eligible
   consumers, at most `max_cap_change_per_epoch` per epoch.
8. A multi-device replica's planner cap is the minimum cap granted by every
   device it uses.
9. Implement shrink-before-expand:

   - calculate target caps;
   - emit all revoke decisions first;
   - record pending revoke ACKs;
   - emit expansions only after relevant ACKs arrive or old leases expire.

10. Refactor coordinator socket ownership to the single I/O-loop model before
    any periodic thread sends ROUTER messages.

### Files

- Add `device_snapshot.py`, `planner.py`, and `coordinator_loop.py` or refactor
  `omni_coordinator.py` equivalently.
- Modify `protocol.py`, `messages.py`, stage client, coordinator, EngineCore,
  scheduler, and configuration propagation.

### Tests

Add:

- `tests/core/memory_coordinator/test_device_snapshot.py`;
- `tests/core/memory_coordinator/test_planner.py`;
- coordinator integration tests for revoke/ACK/grant ordering;
- test that duplicate device observations use minimum fresh free bytes;
- test that equal local device indices on different nodes do not collide;
- test TP replica cap equals the minimum per-device cap;
- test no expansion is sent before required revoke ACK;
- test lease expiry unblocks expansion;
- test stale consumers never receive expansion;
- test only the I/O thread invokes ROUTER/PUB socket methods.

### Completion criteria

- `mode=reactive` preserves old behavior.
- `mode=global_reactive` uses periodic planning.
- No hard reservation or request-level gating exists.
- For every emitted decision trace, INV-1 through INV-5 hold.

## 13. Milestone 2: Backend-aware resource cost and capacity models

### Goal

Introduce the multidimensional resource model without changing cross-stage
request flow.

### Required implementation

1. Add `cost_model.py` with:

   ```python
   class StageCostModel(Protocol):
       def classify(self, request_features: RequestResourceFeatures) -> str: ...
       def estimate(self, request_features: RequestResourceFeatures) -> RequestResourceEstimate: ...
       def observe(self, observation: CostObservation) -> None: ...
   ```

2. Add `capacity_model.py` with backend interfaces:

   ```python
   class StageCapacityProvider(Protocol):
       def report_capacity(self) -> BackendCapacityReport: ...
       def can_fit(self, estimate: RequestResourceEstimate) -> bool: ...
   ```

3. Implement `ARCostModel`:

   - estimate KV blocks from prompt tokens plus configured/estimated output
     tokens and the actual cache block size;
   - use conservative fallback output length when max output is unknown;
   - represent activation/runtime growth as transient bytes;
   - never infer KV usage from changes in physical free HBM.

4. Implement `ARCapacityProvider` using actual scheduler KV block totals/free
   counts.
5. Each worker/backend provider MUST capture and report
   `resident_baseline_bytes` after initialization and before request admission.
   The baseline MUST be stable for the worker instance and reset when
   `instance_id` changes.
6. Implement a minimal `DiffusionCostModel` using a static profile keyed by at
   least model/stage, resolution or latent shape, batch size, dtype, and
   execution mode.
7. Implement a minimal `DiffusionCapacityProvider` that reports current batch,
   maximum configured batch/slots, and workspace class. Workspace feasibility
   MUST use a batch-envelope lookup, not sum per-request peaks.
8. Implement `StaticProfileCostModel` first. Implement EWMA only as a
   multiplicative correction:

   ```text
   corrected = profile_estimate * ewma_correction * uncertainty_multiplier
   ```

   Clamp correction to configured safe bounds. Unknown classes use conservative
   fallback estimates.
9. Add a versioned YAML profile schema:

   ```yaml
   schema_version: 1
   stages:
     "0":
       backend: ar
       transient_bytes_per_class: {}
       output_token_fallback: 512
     "1":
       backend: diffusion
       workspace_envelopes: {}
   ```

10. Extend reports with backend capacity, but keep old fields for compatibility.
11. Planner feasibility APIs MUST operate on a complete resource vector.

### Files

- Add `cost_model.py`, `capacity_model.py`, and a profile schema/example under
  `benchmarks/hbm_admission/specs/`.
- Modify worker/model-runner reporting hooks, EngineCore report construction,
  protocol, aggregator, configuration, and benchmark artifact collection.

### Tests

- Exact KV-block calculation for short, long, and unknown-output AR requests.
- Unknown AR cost is conservative and nonzero.
- KV free blocks do not increase physical headroom.
- resident baseline is stable within an instance and resets across instances.
- active physical accounting uses `max(observed_dynamic, predicted_active)` and
  never their sum.
- Diffusion batch-envelope lookup and unknown-class fallback.
- EWMA bounds and stability.
- TP estimate contains every rank device.
- Profile validation rejects missing schema version, negative values, and
  duplicate classes.

### Completion criteria

- Planner dry-run logs show separate KV/logical and physical/transient demand.
- At least one AR and one diffusion profile can be loaded and validated.
- No request is blocked on a reservation yet.

## 14. Milestone 3: Forecast table, hard reservation ledger, and leases

### Goal

Provide a correct coordinator-side accounting substrate before changing
orchestrator request flow.

### Required implementation

1. Add `forecast_table.py`. Forecasts are soft, discounted planning inputs and
   MUST NOT consume hard capacity.
2. Add `reservation_ledger.py` with HELD and CONSUMED active states and
   RELEASED/CANCELLED/EXPIRED terminal outcomes.
3. Key reservations by `reservation_id`; also index by request, target stage,
   replica, instance, and device.
4. A request/version pair MUST be idempotent:

   - duplicate same-version request returns the original result;
   - lower version is rejected as stale;
   - higher version atomically replaces the old estimate if feasible;
   - failed replacement leaves the old reservation unchanged.

5. Hard hold across all replica devices and logical pools MUST execute under a
   single ledger transaction/lock.
6. `consume` changes HELD to CONSUMED and moves physical demand from held to
   predicted-active accounting without charging both. Logical demand remains
   conservatively charged until a stage report covers its
   `admission_generation`, as specified in Section 9.6.
7. `release`, `cancel`, and `expire` remove active accounting exactly once.
8. A stage instance restart invalidates reservations bound to the old
   `instance_id`.
9. Lease duration is transmitted; local monotonic clocks determine expiry.
10. Terminal records SHOULD remain as bounded tombstones for at least one
    maximum message-retry window to prevent resurrection by duplicates.
11. Add query-only debug snapshots for tests and benchmark artifacts. They MUST
    not expose mutable ledger internals.

### Files

- Add `forecast_table.py` and `reservation_ledger.py`.
- Modify protocol, coordinator message dispatch, configuration, and metrics.

### Tests

- hold/consume/release happy path;
- duplicate messages;
- reversed versions;
- atomic refine success and rollback on failure;
- atomic TP success/failure;
- expiry using a fake clock;
- instance restart cleanup;
- request cancellation cleanup;
- admission-generation reconciliation removes the temporary logical charge
  only after a covering report;
- HELD-to-CONSUMED transition does not double-charge physical demand;
- bounded tombstone garbage collection;
- concurrent attempts cannot violate INV-1 or INV-2.

### Completion criteria

- Ledger is fully tested through coordinator APIs.
- Runtime orchestrator does not use it yet.
- A deterministic trace replay produces identical ledger snapshots.

## 15. Milestone 4: Predictive pipeline reservation and backpressure

### Goal

Use downstream predictions to gate cross-stage submission before resource
pressure materializes.

### Required implementation

1. Add `ResourceCoordClientForOrchestrator`. It MUST own its socket in one
   thread/event loop and provide async APIs:

   ```python
   async def publish_forecast(...)
   async def cancel_forecast(...)
   async def try_hold(...)
   async def refine_hold(...)
   async def consume(...)
   async def release(...)
   async def cancel(...)
   ```

2. Extend `OrchestratorRequestState` with:

   ```python
   resource_features_by_stage
   forecast_ids_by_stage
   reservation_ids_by_stage
   reservation_versions_by_stage
   reservation_replica_by_stage
   deferred_resource_forwards
   ```

3. On ingress, derive request features and publish soft forecasts for known
   future stages. Soft forecasts MUST NOT block stage-0 submission by default.
4. Before `_forward_to_next_stage_unguarded` submits to the next StagePool:

   - materialize actual next-stage input;
   - refine its resource features from actual token/tensor/audio/image shape;
   - obtain ordered candidate replicas from StagePool;
   - atomically try a hard hold for candidates in order;
   - bind the request to the accepted replica;
   - submit only to that replica using a reservation-aware submission API;
   - receive a `StageAdmissionReceipt` from the target stage;
   - mark reservation CONSUMED with the receipt's admission generation only
     after successful submit;
   - cancel the hold and binding on submit failure.

5. StagePool remains responsible for candidate ordering, liveness, affinity,
   and load balancing. Coordinator remains responsible for feasibility and
   atomic accounting.
6. Add explicit StagePool APIs rather than modifying private bindings directly:

   ```python
   async def candidate_replica_ids(...)
   def bind_granted_replica(request_id, replica_id)
   def release_granted_binding(request_id)
   async def submit_reserved_initial(...) -> StageAdmissionReceipt
   async def submit_reserved_update(...) -> StageAdmissionReceipt
   ```

   Existing `submit_initial` and `submit_update` signatures MUST remain
   compatible for non-predictive modes.

7. If no candidate receives a grant, place the forward in an orchestrator-owned
   deferred queue. Do not submit it to the vLLM scheduler.
8. Deferred forwards MUST be retried on capacity update, reservation expiry,
   or a bounded timer. They MUST preserve per-request ordering.
9. Backpressure is per target stage and route/request class. It MUST NOT pause
   unrelated requests that terminate earlier or use a different route.
10. Existing `_cleanup_request_ids` MUST release/cancel all forecasts and
    reservations for normal completion, abort, error, CFG companion cleanup,
    duplex cleanup, and dead replica handling.
11. Streaming updates for an already CONSUMED request MUST reuse its binding.
    Additional resource growth requires versioned refinement before submission.
12. The first implementation MUST NOT scan or reorder the vLLM scheduler
    waiting queue.

### Files

- Add an orchestrator resource client under
  `distributed/omni_coordinator/`.
- Modify `orchestrator.py`, `stage_pool.py`, coordinator dispatch, messages,
  config, request cleanup, and tests.

### Tests

- ingress creates soft forecast but no hard charge;
- downstream submit is impossible before hard grant;
- accepted grant fixes the replica binding;
- admission receipt originates at the target stage and drives CONSUMED
  reconciliation;
- submit failure cancels the hold;
- no-capacity request is deferred, not failed and not submitted;
- capacity update retries deferred request;
- unrelated route proceeds while another route is backpressured;
- abort/finish/error/dead replica release all state;
- CFG parent/companions do not leak or double-count;
- streaming refinement is versioned and idempotent;
- coordinator disconnect fails closed for new forwards while running requests
  continue.

### Completion criteria

- Predictive mode operates end-to-end on the primary TTS pipeline.
- Every request can be traced through forecast, hold, consume, and release.
- Fault injection shows zero leaked active reservations after cleanup timeout.

## 16. Milestone 5: Heterogeneous-stage enforcement and evaluation

### Goal

Demonstrate that the mechanism is not specific to an AR-to-AR pipeline.

### Required implementation

1. Select one supported pipeline containing a diffusion stage. Record the exact
   model and deploy configuration in the benchmark specification.
2. Wire `DiffusionCostModel` and `DiffusionCapacityProvider` into the actual
   diffusion worker/runner and scheduler boundary.
3. Planner decision for diffusion MUST control an existing safe actuator such
   as maximum admitted jobs or scheduler batch slots. It MUST NOT claim to
   shrink a workspace already used by a running batch.
4. Hard reservation must cover persistent per-request state and batch-envelope
   feasibility.
5. Add backend-neutral metrics for grant wait, stage queue time, predicted
   cost, actual peak, and fallback safety events.
6. Extend the benchmark matrix with:

   - primary TTS pipeline;
   - one AR-to-diffusion or multimodal diffusion pipeline;
   - steady, burst, mixed-size, and external-pressure workloads;
   - no control, static cap, centralized AIMD, global reactive, and predictive
     arms.

7. Add prediction-error injection at `-30, -20, -10, 0, +10, +20, +30` percent.

### Tests

- diffusion profile class selection;
- batch-envelope feasibility;
- cap affects only new jobs;
- prediction underestimation triggers safety without over-grant;
- overestimation remains safe and only reduces utilization;
- mixed AR/diffusion reports preserve per-device identity;
- no logical capacity is converted into duplicate physical free bytes.

### Completion criteria

- Both selected pipelines complete predictive experiments.
- Under injected underestimation, the safety controller prevents new expansion
  and the server remains alive.
- Results separately report resident bytes, physical free HBM, KV blocks,
  transient commitment, and soft forecast.

## 17. Milestone 6: SLO-aware planning

### Goal

Optimize end-to-end SLO goodput after predictive safety is established.

### Required implementation

1. Add optional request deadline and priority at the API/request boundary.
2. Propagate the original request timestamp, deadline duration, and priority
   through `OrchestratorRequestState`. Do not propagate absolute monotonic time
   across nodes.
3. Add per-stage service-time estimators keyed by request class. Use bounded
   EWMA or quantile estimates; never use zero for unknown service time.
4. Compute remaining slack at the orchestrator/coordinator using one clock
   domain:

   ```text
   slack = deadline_remaining - estimated_remaining_pipeline_time
   ```

5. Use a bounded urgency function. Do not use unbounded `1 / slack`:

   ```text
   urgency = priority_weight * min(max_urgency, exp(-slack / tau))
   ```

6. Add configurable policy for requests predicted to miss SLO:

   - `best_effort`;
   - `deprioritize`; or
   - `reject_before_stage0`.

   Default MUST be `best_effort` for compatibility.
7. Planner weight SHOULD combine queue demand, bounded urgency, and estimated
   completions per resource unit. It MUST retain a minimum stage share to avoid
   pipeline starvation.
8. Report SLO goodput separately from raw throughput.

### Tests

- bounded urgency at negative, zero, and large positive slack;
- priority ordering without starvation;
- unknown service-time fallback;
- expired requests follow configured policy;
- disabling SLO mode reproduces Milestone 5 planning decisions;
- multi-stage deadline propagation and cleanup.

### Completion criteria

- Predictive resource safety works identically with SLO weighting disabled.
- SLO-aware mode improves or matches SLO goodput on at least one burst/mixed
  workload without increasing OOM or request failure rate.

## 18. Milestone 7: Transfer-buffer commitment (optional extension)

### Goal

Account for connector staging and in-flight tensors when they are a material
source of peak HBM.

### Required implementation

1. Add a backend-neutral transfer descriptor containing request ID, source and
   target stage/replica, tensor shape, dtype, logical payload bytes, and actual
   staging-buffer bytes.
2. Instrument connector adapters at the point where buffer size is known.
   Orchestrator-side serialization time with unknown payload size is not
   sufficient.
3. Add IN_FLIGHT accounting as an orthogonal flag or sub-record associated with
   a CONSUMED reservation. Do not add it before actual connector ownership and
   lifetime are observable.
4. Charge source and destination according to real buffer lifetimes. Release
   each charge on completion, failure, timeout, or request cancellation.
5. Verify zero-copy paths do not invent a duplicate staging allocation.

### Tests

- exact tensor byte calculation;
- copied versus zero-copy connector behavior;
- source/destination lifetime boundaries;
- transfer failure and timeout cleanup;
- duplicate completion event idempotency.

### Completion criteria

- Transfer commitment materially improves peak prediction on at least one
  measured workload; otherwise keep it experimental and exclude it from the
  core paper claim.

## 19. Metrics and observability

The following metrics MUST be emitted with stage, replica, node, device, mode,
and reason labels where applicable:

```text
hbm_observed_free_bytes
hbm_guard_bytes
hbm_hard_transient_commitment_bytes
hbm_soft_forecast_bytes
hbm_planner_headroom_bytes
kv_free_blocks
kv_held_blocks
planner_cap
safety_cap
effective_cap
planner_epoch_total
planner_duration_seconds
planner_revoke_total
planner_grant_total
planner_revoke_ack_seconds
reservation_hold_total
reservation_reject_total
reservation_expire_total
reservation_active
reservation_wait_seconds
forecast_error_ratio
stage_resource_queue_seconds
safety_fallback_total
```

Logs for every reservation transition MUST include request ID, reservation ID,
version, target stage/replica, planner epoch, state transition, resource vector,
and reason. Production log level MAY sample successful transitions, but
benchmark mode MUST retain a complete trace.

## 20. Benchmark and paper evaluation matrix

Required controller arms:

```text
A: no control
B: static cap
C: local/reactive AIMD
D: centralized shared-pressure AIMD
E: global reactive planner
F: global predictive reservation
G: predictive + SLO-aware weighting (Milestone 6)
O: oracle costs for selected experiments only
```

Required workload dimensions:

- steady and burst arrivals;
- short/long or small/large mixed requests;
- shared-GPU and dedicated-GPU placement;
- external HBM pressure;
- stage service-rate imbalance;
- prediction error;
- report loss and delayed decisions;
- stage restart and request cancellation.

Primary metrics:

- request success and OOM/server-crash count;
- raw throughput;
- SLO goodput after Milestone 6;
- p50/p95/p99 end-to-end latency and modality-specific first-output latency;
- physical peak HBM and duration above critical watermark;
- logical KV utilization;
- reservation wait and stage queue time;
- planner/safety intervention count; and
- GPU utilization.

Each main comparison SHOULD use at least five repetitions and report median
with bootstrap 95% confidence interval. A run without OOM is not sufficient;
it MUST show that the intended planner, reservation, and safety transitions
actually occurred.

## 21. Failure behavior

### Coordinator unavailable

- Existing running requests continue.
- New predictive hard holds fail closed or remain deferred.
- Stage scheduler uses the last unexpired decision.
- On decision expiry, planner cap falls to configured fail-safe minimum while
  safety cap remains active.

### Stage report stale

- No expansion.
- Existing hard holds bound to a live instance remain until lease expiry.
- New holds targeting the stale instance are rejected.

### Stage restart

- New `instance_id` resets report and decision generations.
- Old hard reservations are cancelled or expire.
- Old decisions and ACKs are rejected.

### Orchestrator restart

- Forecasts expire naturally.
- HELD reservations expire by lease.
- CONSUMED reservations require stage-report reconciliation or conservative
  lease retention; the coordinator MUST NOT immediately interpret missing
  orchestrator state as free capacity.

### Request cancellation

- Cancel forecasts.
- Cancel HELD reservations.
- Release CONSUMED reservations after stage abort/cleanup acknowledgement or a
  conservative timeout.

## 22. Backward compatibility and rollout

Modes behave as follows:

```text
enabled=false:
    no dynamic-HBM behavior

mode=reactive:
    existing centralized/local AIMD wire path and behavior

mode=global_reactive:
    periodic planner + safety controller, no request reservation gate

mode=predictive:
    resource model + forecast + hard reservation + backpressure
```

New fields MUST have defaults. Existing deploy YAML files MUST continue to
parse. Old stage clients MUST not be selected for predictive reservations; the
coordinator MUST detect supported schema/capabilities during registration.

Each registration SHOULD advertise:

```python
capabilities: tuple[str, ...] = (
    "resource_report_v2",
    "resource_decision_v2",
    "revoke_ack",
    "reservation_v1",
)
```

Mixed-version deployments MAY run in reactive mode. Predictive mode MUST reject
or exclude a replica missing required capabilities.

## 23. Required test commands

At the end of every milestone, run at minimum:

```bash
pytest -q -c /dev/null \
  tests/core/memory_coordinator \
  tests/distributed/omni_coordinator \
  tests/core/sched/test_omni_ar_scheduler_streaming.py \
  tests/config/test_omni_config.py \
  tests/benchmarks/test_dynamic_hbm_experiment.py
```

If the normal repository pytest configuration is usable, also run the same
selection without `-c /dev/null`. Milestone-specific orchestrator, StagePool,
diffusion, and connector tests MUST be added to this command as those features
are introduced.

Before committing:

```bash
git diff --check
python -m compileall -q vllm_omni/core/memory_coordinator
```

## 24. Implementation constraints for coding agents

An implementation agent following this document MUST:

1. implement only one milestone at a time;
2. preserve all previous milestone tests;
3. avoid changing upstream scheduler semantics unless the milestone explicitly
   requires it;
4. keep protocol parsing backward compatible;
5. use explicit device, replica, instance, generation, epoch, and version
   identities—never infer identity from list position alone;
6. use fake clocks for lease/expiry unit tests;
7. avoid sleeps in deterministic unit tests;
8. use integer bytes and block counts, not floating-point GiB internally;
9. never add free KV-block bytes to physical free HBM;
10. never issue a partial multi-device hard grant;
11. never expand before required revokes are acknowledged or expired;
12. never access a ZMQ socket outside its owning thread;
13. never hard-reserve long-horizon early forecasts;
14. never route a granted request to a replica different from the grant;
15. never release a CONSUMED reservation merely because an orchestrator message
    is temporarily missing; and
16. document any deviation from this specification with a failing/passing test
    that demonstrates why the change is necessary.

## 25. Definition of project completion

The core project is complete after Milestone 6 when:

- reactive mode remains backward compatible;
- physical and logical resources are modeled separately;
- the coordinator plans per physical device from consistent snapshots;
- multi-device reservations are atomic;
- shrink-before-expand is enforced with ACK/lease semantics;
- soft forecasts and hard reservations have separate accounting;
- pipeline forwarding is gated by replica-bound hard grants;
- all completion and fault paths release commitments safely;
- at least one AR/audio and one diffusion-containing pipeline are evaluated;
- prediction-error experiments demonstrate safety fallback; and
- SLO-aware mode reports end-to-end goodput without weakening memory safety.

Milestone 7 is optional and belongs in the core paper only if measured transfer
buffers materially affect peak-memory prediction or scheduling decisions.
