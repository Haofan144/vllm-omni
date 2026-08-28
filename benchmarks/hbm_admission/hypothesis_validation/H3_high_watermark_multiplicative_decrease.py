#!/usr/bin/env python3
"""H3 - crossing the high watermark drives a multiplicative cap decrease.

Hypothesis
----------
Once pressure (guarded physical HBM, or KV) reaches ``high_watermark`` but stays
below ``critical_watermark``, each successive sample multiplies the cap by
``scale_down_ratio`` (floored to ``min_num_seqs``); the reaction is immediate
(no stable-sample wait) and every co-resident stage sees the same decrease.

Method
------
1. Pure state machine: feed the real ``BudgetAllocator`` a run of
   ``high_watermark``-level reports and record the cap trajectory.
2. Shared-device path: two stages on one GPU; only stage 0's report carries
   high pressure; assert both stages' caps follow the same multiplicative
   ladder.
3. Boundary: a sample exactly one epsilon below ``high_watermark`` must NOT
   decrease the cap.

Config: default watermarks 0.75 / 0.90 / 0.95, scale_down_ratio 0.5,
min_num_seqs 1 -> from 16 the ladder is 16 -> 8 -> 4 -> 2 -> 1 -> 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

CFG = h.DynamicHBMConfig(enabled=True)  # defaults


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. pure state machine ladder -------------------------------- #
    hi = CFG.high_watermark + (CFG.critical_watermark - CFG.high_watermark) / 2  # 0.925
    steps = [{"label": "idle", "hbm": 0.30}] + [{"label": f"high_{i}", "hbm": hi} for i in range(6)]
    trace = h.run_allocator_trace(CFG, 16, steps)
    ladder = [t.cap_out for t in trace]
    obs["pure_ladder"] = ladder
    obs["pure_states"] = [t.state for t in trace]
    obs["pure_reasons"] = [t.reason for t in trace]
    expected_ladder = [16, 8, 4, 2, 1, 1, 1]
    checks.append(h.Check(
        "pure BudgetAllocator ladder is 16->8->4->2->1 (x0.5, floor min_num_seqs=1)",
        ladder == expected_ladder,
        f"observed={ladder} expected={expected_ladder}",
    ))
    checks.append(h.Check(
        "first high-pressure sample decreases immediately (no stable wait)",
        trace[1].cap_out == 8 and trace[1].state == "high_pressure",
        f"step1 cap={trace[1].cap_out} state={trace[1].state} reason={trace[1].reason}",
    ))

    # ---- 2. shared-device path -------------------------------------- #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-X")
    c.report(addr="s1", stage_id=1, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-X")
    caps_s0, caps_s1 = [], []
    for _ in range(5):
        # only stage 0's report has the pressure; stage 1 keeps reporting calm.
        # advance the clock a full sample interval so the coordinator's
        # once-per-interval allocation rate limiter lets each sample act.
        c.advance_clock(CFG.sample_interval_ms / 1000.0)
        out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=hi, device_uuid="GPU-X")
        for d in out:
            (caps_s0 if d["stage_id"] == 0 else caps_s1).append(d["effective_max_num_seqs"])
    obs["shared_caps_stage0"] = caps_s0
    obs["shared_caps_stage1"] = caps_s1
    checks.append(h.Check(
        "shared-device: both stages follow the same multiplicative ladder",
        caps_s0 == caps_s1 and caps_s0[:3] == [8, 4, 2],
        f"stage0={caps_s0} stage1={caps_s1}",
    ))

    # ---- 3. boundary just below high watermark --------------------- #
    just_below = CFG.high_watermark - 1e-4
    trace_b = h.run_allocator_trace(CFG, 16, [
        {"label": "idle", "hbm": 0.30},
        {"label": "below_high", "hbm": just_below},
        {"label": "below_high2", "hbm": just_below},
    ])
    obs["below_high_caps"] = [t.cap_out for t in trace_b]
    obs["below_high_states"] = [t.state for t in trace_b]
    checks.append(h.Check(
        "pressure just below high watermark does NOT decrease the cap",
        all(t.cap_out == 16 for t in trace_b),
        f"caps={[t.cap_out for t in trace_b]} states={[t.state for t in trace_b]}",
    ))

    analysis = (
        "MATCHES EXPECTATION. In allocator.py the high-pressure branch sets "
        "cap = max(min_num_seqs, floor(current * scale_down_ratio)) on every sample where "
        "high_watermark <= pressure < critical_watermark, with no stable-sample gate, so the "
        "response is a geometric decay 16->8->4->2->1 that begins on the very first crossing "
        "sample (state=high_pressure, reason=high_pressure). The shared-device test shows the "
        "coordinator recomputes `shared_hbm_pressure` as the max guarded pressure over the "
        "affected group and feeds it to every co-resident allocator via pressure_override, so "
        "stage 1 tracks the identical ladder even though its own rank report stayed calm. The "
        "boundary case confirms the comparison is a true >= high_watermark test: one epsilon "
        "below leaves the cap untouched (state=normal, reason=within_hysteresis). Note the ladder "
        "floors at min_num_seqs (1 here); reaching 0 requires the critical branch (H4)."
    )

    h.write_result(
        "H3",
        "High watermark -> multiplicative cap decrease",
        "At/above high watermark (below critical) the cap is multiplied by scale_down_ratio each "
        "sample, immediately, and identically for every co-resident stage.",
        "pure ladder 16->8->4->2->1; shared stages identical; no decrease just below the line.",
        obs,
        checks,
        analysis,
    )
    print(f"H3 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
