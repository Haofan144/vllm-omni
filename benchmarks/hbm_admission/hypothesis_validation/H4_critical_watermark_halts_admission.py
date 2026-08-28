#!/usr/bin/env python3
"""H4 - crossing the critical watermark takes the cap to zero (stop new admission).

Hypothesis
----------
When pressure reaches ``critical_watermark`` the safety cap goes straight to
``critical_admission_cap`` (default 0) in a single step - not via the
multiplicative ladder - the scheduler then admits no new request, and every
co-resident stage on the shared GPU does the same. A wire decision that claims a
nonzero cap while asserting the CRITICAL state is clamped to 0 by the scheduler.

Method
------
1. Pure state machine: one report at >= critical_watermark from cap 16 must
   yield cap 0 in one step (reason=critical_pressure, state=critical).
2. Shared-device path: stage 0 hits critical; assert stage 0 AND stage 1 both
   receive cap 0 with a ``shared_device_critical_pressure`` reason.
3. Scheduler clamp: drive the REAL ``OmniSchedulerMixin`` with a CRITICAL
   decision that (inconsistently) carries effective_max_num_seqs=4; assert the
   applied safety cap is 0 and ``_dynamic_hbm_allows_new_admission()`` is False.
4. KV-driven critical: a report with kv_pressure >= critical (physical calm)
   must also drive cap 0, tagged as a kv pressure source.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402
from _harness import SafetyState  # noqa: E402

CFG = h.DynamicHBMConfig(enabled=True)


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. pure state machine one-step to zero --------------------- #
    trace = h.run_allocator_trace(CFG, 16, [
        {"label": "idle", "hbm": 0.30},
        {"label": "critical", "hbm": CFG.critical_watermark + 1e-3},
    ])
    obs["pure"] = [(t.label, t.cap_out, t.state, t.reason) for t in trace]
    checks.append(h.Check(
        "critical report drives cap 16 -> 0 in ONE step",
        trace[1].cap_out == 0 and trace[1].state == "critical" and trace[1].reason == "critical_pressure",
        f"{obs['pure'][1]}",
    ))

    # ---- 2. shared-device propagation of critical ------------------ #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-C")
    c.report(addr="s1", stage_id=1, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-C")
    c.clear_sent()
    c.advance_clock(CFG.sample_interval_ms / 1000.0)
    out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=CFG.critical_watermark + 1e-3, device_uuid="GPU-C")
    by_stage = {d["stage_id"]: d for d in out}
    obs["shared"] = [(d["stage_id"], d["reason"], d["safety_state"], d["effective_max_num_seqs"]) for d in out]
    checks.append(h.Check(
        "critical fans out to every co-resident stage as cap 0",
        set(by_stage) == {0, 1} and all(d["effective_max_num_seqs"] == 0 for d in out)
        and all(d["safety_state"] == "critical" for d in out),
        f"{obs['shared']}",
    ))
    checks.append(h.Check(
        "shared critical decision reason is shared_device_critical_pressure",
        all(d["reason"] == "shared_device_critical_pressure" for d in out),
        f"reasons={[d['reason'] for d in out]}",
    ))

    # ---- 3. scheduler clamps an inconsistent nonzero wire cap ------ #
    sch = h.SchedulerHarness(cap=16)
    applied = sch.apply(generation=1, cap=4, state=SafetyState.CRITICAL, pressure=0.97)
    obs["scheduler_clamp"] = {
        "applied": applied,
        "safety_cap": sch.safety_cap,
        "effective_cap": sch.effective_cap,
        "effective_tokens": sch.effective_tokens,
        "allows_new_admission": sch.allows_new_admission(),
        "state": sch.state.value,
    }
    checks.append(h.Check(
        "scheduler clamps CRITICAL wire cap=4 down to 0 and blocks admission",
        applied and sch.safety_cap == 0 and sch.effective_cap == 0 and not sch.allows_new_admission(),
        f"{obs['scheduler_clamp']}",
    ))

    # ---- 4. KV-driven critical ------------------------------------ #
    trace_kv = h.run_allocator_trace(CFG, 16, [
        {"label": "idle", "hbm": 0.30, "kv": 0.10},
        {"label": "kv_critical", "hbm": 0.40, "kv": CFG.critical_watermark + 1e-3},
    ])
    obs["kv_critical"] = [(t.label, t.cap_out, t.state, t.reason, t.pressure_source) for t in trace_kv]
    checks.append(h.Check(
        "KV pressure >= critical also drives cap 0 (source=kv)",
        trace_kv[1].cap_out == 0 and trace_kv[1].state == "critical" and trace_kv[1].pressure_source == "kv",
        f"{obs['kv_critical'][1]}",
    ))

    analysis = (
        "MATCHES EXPECTATION. allocator.py handles critical before the multiplicative branch and "
        "sets cap = critical_admission_cap directly, so 16 collapses to 0 on the first sample at "
        "or above critical_watermark, independent of the current cap. The coordinator propagates "
        "this to the whole shared-device group (reason prefixed shared_device_) using the group's "
        "max guarded pressure. apply_stage_budget_decision in omni_scheduler_mixin.py has an extra "
        "belt-and-braces clamp: in the CRITICAL state it takes min(configured, "
        "critical_admission_cap, max(0, wire_cap)), so a malformed wire decision that pairs "
        "state=critical with a nonzero cap still lands at 0 and _dynamic_hbm_allows_new_admission() "
        "returns False. Because pressure = max(guarded_physical, kv), a KV-pool exhaustion alone "
        "trips the same path with pressure_source=kv. New admission is therefore halted from every "
        "angle the design intends; H5 covers that running requests are not killed."
    )

    h.write_result(
        "H4",
        "Critical watermark -> cap 0, admission halted",
        "At/above critical watermark the safety cap goes straight to critical_admission_cap (0) in "
        "one step, on every co-resident stage, and the scheduler admits nothing new; a KV-only "
        "critical does the same.",
        "one-step 16->0; shared fan-out to 0; scheduler clamps inconsistent wire cap; KV path works.",
        obs,
        checks,
        analysis,
    )
    print(f"H4 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
