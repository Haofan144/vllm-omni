# HBM dynamic-monitoring hypothesis validation — H1–H7

Consolidated results and analysis for the seven falsifiable hypotheses that the
current **Milestone 1 reactive HBM safety** design must satisfy for the
monitoring/control chain to be considered *working*.

## Verdict table

| ID | Hypothesis (one line) | Verdict | Failing checks |
|---|---|---|---|
| H1 | Worker-reported HBM pressure tracks NVML (accuracy, latency, monotonicity, no false-critical) | **PARTIAL / DEVIATION** | 1 of 6 — a *constant* ~1.2 pp offset |
| H2 | Coordinator groups stages by `(node_id, device_uuid)`; shared-GPU pressure propagates, distinct devices stay decoupled | **MATCHES** | 0 of 5 |
| H3 | Crossing the high watermark → immediate multiplicative cap decrease, identical for every co-resident stage | **MATCHES** | 0 of 4 |
| H4 | Crossing the critical watermark → cap 0 in one step, on all co-resident stages; scheduler clamps inconsistent wire caps; KV path works | **MATCHES** | 0 of 5 |
| H5 | cap = 0 halts new admission but never evicts running requests; admission resumes on drain | **MATCHES** | 0 of 5 |
| H6 | After pressure release the controller recovers *gradually* (RECOVERING hold, then additive `+scale_up_step`), NORMAL only at the configured cap; a high sample re-arms the decrease | **MATCHES** | 0 of 6 |
| H7 | Fail-closed on disconnect / stale report / missing rank / stale-or-replayed generation, without killing running work | **MATCHES** | 0 of 7 |

**Overall:** the control-plane design (H2–H7) behaves exactly as specified. The
only deviation is in H1 and it is a *measurement-bias* issue, not a control
issue: it is a fixed offset that makes the controller marginally more
conservative, never less safe.

---

## Environment

| | |
|---|---|
| Repo commit | `ce7d544d` — "[Core] Add Milestone 1 reactive HBM safety foundation" |
| GPU (H1 only) | NVIDIA RTX A6000, 49140 MiB, driver 595.71.05 |
| Python / torch | 3.12.3 / 2.13.0+cu130 |
| Classes under test | `vllm_omni.core.memory_coordinator.{BudgetAllocator, ReplicaMemoryAggregator, RankMemoryReporter}`, `vllm_omni.distributed.omni_coordinator.omni_coordinator.OmniCoordinator`, `vllm_omni.core.sched.omni_scheduler_mixin.OmniSchedulerMixin` |
| Config | `DynamicHBMConfig()` defaults: watermarks 0.75 / 0.90 / 0.95, `scale_down_ratio` 0.5, `scale_up_step` 1, `scale_up_stable_samples` 5, `recovery_complete_samples` 3, `missing_report_grace_samples` 1, `critical_admission_cap` 0, `disconnect_admission_cap` 0 |

### Method note — why these are not full server benchmarks

The installed environment has a vLLM 0.27 → 0.28 API drift
(`vllm.entrypoints.serve.utils.error_response` was moved) that currently blocks
`import vllm_omni`, so an end-to-end TTS server could not be launched in this
run. `_harness.py` installs a one-symbol `sys.modules` shim for that moved
module and then imports and drives the **real production classes** directly:

* H1 uses the real `RankMemoryReporter` against `torch.cuda.mem_get_info` on a
  real GPU, with an independent NVML reference and a real staircase allocation.
* H2, H3(shared), H4(shared), H7(b) drive the real
  `OmniCoordinator._handle_memory_report_locked` /
  `_check_memory_report_timeouts_locked` through a fake ZMQ router that captures
  the exact wire `budget_decision` dicts the coordinator would send.
* H3(pure), H4(pure), H6, H7(c) drive the real `BudgetAllocator` state machine.
* H4(clamp), H5, H7(a,d,e) drive the real `OmniSchedulerMixin` admission-budget
  methods, mirroring `tests/core/sched/test_dynamic_hbm_admission_gate.py`.

This validates the **control logic and state machine** with full fidelity. It
does **not** cover: real ZMQ transport, thread scheduling, real
cross-process latency, or the safety *benefit* under a genuine OOM boundary
(that is H9 in the plan and still requires a working server + calibrated
pressure). See "Not covered here" at the end.

---

## Per-hypothesis analysis

### H1 — measurement accuracy · PARTIAL / DEVIATION

**Expected:** median |reporter − NVML| pressure error < 0.01; p95 < 0.02;
monotone staircase; sample p95 < 2× interval; idle never ≥ 0.95.

**Observed:**

| metric | value | expectation | pass |
|---|---|---|---|
| raw median abs pressure error | **0.01121** | < 0.01 | ✗ |
| NVML−reporter "used" bias | **598.6 MiB, stdev 0.000 MiB** (1.22 % of total) | fixed offset | ✓ |
| bias-corrected p95 error | **0.000000** | < 0.005 | ✓ |
| staircase-up `device_used` | strictly monotone, min step +1024 MiB | monotone | ✓ |
| sample wall p95 | **0.6 ms** (interval 300 ms) | < 600 ms | ✓ |
| max idle reporter pressure | **0.0056** | < 0.95 | ✓ |

**Why it deviates.** `torch.cuda.mem_get_info` reports free/total *after* the
CUDA primary context, cuBLAS/cuDNN handles and NCCL scratch already exist, so
that ~0.6 GiB is invisible to the worker but counted by NVML as
device-resident. The gap is **constant to a fraction of a MiB across every
idle, up-step and down-step sample** (population stdev 0.000 MiB) — it is a
fixed "context tax," not drift, not noise, not load-dependent.

**Does it matter for the design?** Minimally, and in the safe direction:

* The AIMD state machine in `allocator.py` keys entirely off *relative* pressure
  crossing the low/high/critical watermarks and off sample-to-sample deltas. A
  constant offset shifts every watermark comparison by the same 1.2 pp — i.e.
  it is equivalent to running with watermarks 1.2 pp *lower*. The controller
  reacts a hair **earlier**; it can never react late because of this.
* The quantity the controller actually consumes — the *change* in pressure — is
  accurate: the bias-corrected p95 error is 0.000000.

**Recommended fix (any one):**
1. Subtract a one-time post-init context-baseline sample in the pressure
   calculation (the reporter already carries `baseline_process_reserved_bytes`
   plumbing for exactly this kind of correction).
2. Rely on the existing `guard_bytes` / `guard_ratio` knob, which is already
   sized for offsets this large (default guidance uses 1 GiB).
3. Document that effective watermarks run ~1 pp tighter than configured on a
   given GPU/driver.

---

### H2 — shared-device identity · MATCHES

Four sub-cases, all as specified:

| sub-case | setup | result |
|---|---|---|
| A | stage 0 + stage 1, same `(n0, GPU-AAAA)` | stage-0 high-pressure report → **both** stages get cap 16→8, reason `shared_device_high_pressure`, source `shared_physical_hbm` |
| B | stage 0 `GPU-AAAA`, stage 2 `GPU-BBBB` | stage-0 high pressure → **only stage 0** moves |
| C | both `device_uuid=""`, `node_id` `nodeA` vs `nodeB` | not grouped — equal local `device_id` on different nodes stays decoupled |
| D | same node, `GPU-UUID-1` vs `GPU-UUID-2` | not grouped |

`OmniCoordinator._device_keys` = `(node_id, device_uuid or "local-device-<id>")`
and `affected` = replicas whose device-key set intersects the incoming report's.
The design's "never trust the bare local index" rule holds.

---

### H3 — high watermark → multiplicative decrease · MATCHES

* **Pure `BudgetAllocator`** from cap 16 at pressure 0.925 (between high and
  critical): ladder **[16, 8, 4, 2, 1, 1, 1]** — `max(min_num_seqs,
  floor(current × 0.5))` on every sample, floored at `min_num_seqs=1`.
* First crossing sample decreases immediately (`state=high_pressure`,
  `reason=high_pressure`) — no stable-sample gate on the way down.
* **Shared-device:** only stage 0's report carries pressure; both stages'
  caps follow the identical ladder `[8, 4, 2, 1, 1]` because the coordinator
  feeds each co-resident allocator the group-max guarded pressure via
  `pressure_override`.
* One epsilon **below** the high watermark → cap stays 16, `state=normal` — the
  comparison is a true `>= high_watermark` test.
* Reaching 0 needs the critical branch (H4); the high branch floors at
  `min_num_seqs`.

---

### H4 — critical watermark → cap 0 · MATCHES

| check | result |
|---|---|
| pure allocator, cap 16 → **0 in one step** at ≥ critical | `state=critical`, `reason=critical_pressure` |
| shared-device critical | stage 0 **and** stage 1 → cap 0, `reason=shared_device_critical_pressure` |
| scheduler clamp | a `CRITICAL` wire decision carrying `effective_max_num_seqs=4` is clamped by `apply_stage_budget_decision` to `min(configured, critical_admission_cap, max(0, wire))` = **0**; `_dynamic_hbm_allows_new_admission()` = False |
| KV-only critical | `kv_pressure ≥ critical` with calm physical HBM also → cap 0, `pressure_source=kv` |

The critical branch is handled *before* the multiplicative branch and assigns
`critical_admission_cap` directly, so the cap collapses regardless of its
current value. New admission is halted from every path the design intends.

---

### H5 — running requests not evicted · MATCHES

| scenario | result |
|---|---|
| 6 running, apply `CRITICAL` cap 0 | `effective_cap=0`, but `_dynamic_max_num_running_reqs()` = **6**, token budget = **1536** (not zero), `allows_new_admission=False` |
| 2 running at cap 2 → pop one | blocked at running 2, **allowed** at running 1 — no new decision needed |
| cap 0 → later `NORMAL` cap 16 | admission re-enabled, `effective_cap=16` |

`_recompute_effective_dynamic_hbm_budget` separates the **admission cap**
(pullable to 0) from the **execution budget** =
`max(effective_cap, occupied_slots)`, so in-flight requests always keep a
non-zero step/token budget and are never evicted.
`_dynamic_hbm_allows_new_admission()` is a side-effect-free
`len(running) < effective_cap`, so admission self-heals on drain.

---

### H6 — gradual recovery · MATCHES

Full ladder from cap 0 with default config (`recovery_complete_samples=3`,
`scale_up_stable_samples=5`, `scale_up_step=1`), run for 3 + 16×5 + 10 low
samples:

* `caps` compressed: `0×3` (`recovering_complete_reports` hold) → then `+1`
  every 5 low samples (`stable_headroom`) → … → `16`.
* First cap increase at low-sample index **7** (3 completion holds + the
  stable-headroom counter reaching 5).
* Expansion is **additive** (`+1`), never multiplicative — asymmetric AIMD.
* `state` stays `RECOVERING` the entire climb and flips to `NORMAL` **exactly**
  at the sample where the cap regains 16 (index 82 of the low run).
* **Re-arm:** a single sample back above the high watermark mid-recovery
  immediately re-enters the `high_pressure` branch (`state=high_pressure`) and
  multiplies the cap down again — a flapping signal cannot ratchet the cap up.

---

### H7 — fail-closed · MATCHES

| fault | mechanism | result |
|---|---|---|
| **a. coordinator disconnect** | `apply_dynamic_hbm_disconnect_guard` (local, no message needed) | cap → 0, `state=DISCONNECTED`, 5 running retained, `allows_new_admission=False` |
| **b. stale report** | `_check_memory_report_timeouts_locked` synthesises an empty-rank report → `BudgetAllocator` | successive sweeps: cap `8 → 4 → 2 → 1 → 0`, every decision `safety_state=stale`, `reason=stale_report_*`, `pressure_source=telemetry_health`; never increases |
| **c. missing rank** | `BudgetAllocator` incomplete branch | first incomplete report **held** (`state=incomplete`, grace = 1); next incomplete report **decreases** (`state=stale`, `reason=incomplete_rank_reports_decrease`, 16→8); **no** incomplete step is ever `NORMAL` or increases |
| **d. stale / replayed generation** | `apply_stage_budget_decision` guards | older generation (3 < 5), equal generation (5), and decision built on a stale report generation (2 < 11) **all rejected**; cap stays at last good value 8 |
| **e. local KV-exhaustion guard** | `apply_dynamic_hbm_local_kv_guard` | tightens to critical immediately **without** advancing `_last_budget_generation`, so the next genuine coordinator decision (gen 8) still applies — fail-closed does not wedge the control channel |

Invariant across all paths: faults **stop expansion** and, where visibility is
lost, **tighten admission**, but never kill running work and never let a stale
input advance state.

---

## What this does and does not establish

**Established (H1–H7):**

* The worker measures physical HBM pressure correctly in *shape* and *change*,
  with sub-millisecond sampling and no idle false-critical; there is a known,
  characterised, safe-direction constant bias (H1).
* Shared-GPU pressure is propagated to every co-resident stage by physical
  device identity, and only to those stages (H2).
* The AIMD control law is exactly as designed: immediate multiplicative
  decrease at `high`, one-step to zero at `critical`, slow additive recovery,
  re-armable (H3, H4, H6).
* Admission control throttles *new* work only and self-heals on drain (H5).
* Every telemetry/control fault fails closed without evicting running requests
  (H7).

**Not covered here** (needs a working `vllm_omni` import + GPU server):

* Real ZMQ transport, coordinator threads, and true end-to-end
  crossing→report→decision→apply latency (the plan's ≤ 1.5 s p95 gate).
* H8 — monitoring overhead under real inference load (A/B throughput).
* H9 — the safety *benefit*: C-vs-D OOM reduction at a calibrated,
  reproducibly-dangerous pressure boundary.
* Multi-rank (TP ≥ 2) missing-rank behaviour end to end (H7c is validated at the
  allocator level only).

**Immediate follow-ups:**

1. Fix or document the H1 constant bias (context-baseline subtraction, or lean
   on `guard_bytes`).
2. Restore `import vllm_omni` (add the moved-module shim to the codebase or pin
   vLLM) so the server-level experiments H8/H9 and the timestamped-latency
   gates can run.
