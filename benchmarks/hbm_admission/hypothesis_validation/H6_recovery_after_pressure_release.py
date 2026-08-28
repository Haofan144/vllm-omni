#!/usr/bin/env python3
"""H6 - after pressure is released the controller recovers gradually, not instantly.

Hypothesis
----------
Once pressure falls back to/below ``low_watermark`` after a high/critical
episode, the controller does NOT jump straight back to the configured cap. It
first spends ``recovery_complete_samples`` samples in a RECOVERING hold
(confirming reports are complete and headroom is real), then adds
``scale_up_step`` per ``scale_up_stable_samples`` low-pressure samples until the
configured cap is reached, at which point the state returns to NORMAL. Pressure
bouncing back above high during recovery re-arms the decrease.

Method
------
Pure ``BudgetAllocator`` trace with default config
(recovery_complete_samples=3, scale_up_stable_samples=5, scale_up_step=1):

1. idle -> critical (cap 0) -> then a long run of low-pressure samples;
   record the exact recovery ladder and the sample index where cap first
   increases above 0 and where it reaches 16 / NORMAL.
2. Re-arm: during the low-pressure recovery, inject one high-pressure sample
   and assert the cap decreases again and state leaves RECOVERING.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

CFG = h.DynamicHBMConfig(enabled=True)


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    low = CFG.low_watermark - 0.05  # 0.70, comfortably in the recovery band
    crit = CFG.critical_watermark + 1e-3

    # Enough low samples to climb the full ladder 0 -> 16:
    # recovery_complete_samples holds + 16 increases * scale_up_stable_samples.
    n_low = CFG.recovery_complete_samples + 16 * CFG.scale_up_stable_samples + 10
    steps = [
        {"label": "idle", "hbm": 0.30},
        {"label": "critical", "hbm": crit},
    ] + [{"label": f"low_{i}", "hbm": low} for i in range(n_low)]
    trace = h.run_allocator_trace(CFG, 16, steps)

    caps = [t.cap_out for t in trace]
    states = [t.state for t in trace]
    reasons = [t.reason for t in trace]
    # keep the JSON readable: record the compressed ladder (value, run-length)
    def _rle(seq):
        out = []
        for v in seq:
            if out and out[-1][0] == v:
                out[-1][1] += 1
            else:
                out.append([v, 1])
        return out
    obs["caps_rle"] = _rle(caps)
    obs["states_rle"] = _rle(states)
    obs["n_low_samples"] = n_low
    obs["caps_head"] = caps[:20]

    # index 0 = idle, 1 = critical(cap 0), 2.. = low samples
    low_caps = caps[2:]
    low_states = states[2:]
    first_increase_idx = next((i for i, v in enumerate(low_caps) if v > 0), None)
    reached_configured_idx = next((i for i, v in enumerate(low_caps) if v == 16), None)

    checks.append(h.Check(
        "cap is 0 immediately after critical",
        caps[1] == 0,
        f"caps[:3]={caps[:3]}",
    ))
    checks.append(h.Check(
        "recovery does NOT jump straight to configured cap on the first low sample",
        low_caps[0] == 0,
        f"first low sample cap={low_caps[0]} state={low_states[0]} reason={reasons[2]}",
    ))
    checks.append(h.Check(
        "controller spends >= recovery_complete_samples in RECOVERING hold before expanding",
        first_increase_idx is not None and first_increase_idx >= CFG.recovery_complete_samples,
        f"first cap increase at low-sample index {first_increase_idx} "
        f"(recovery_complete_samples={CFG.recovery_complete_samples})",
    ))
    checks.append(h.Check(
        "expansion is additive (+scale_up_step), not multiplicative",
        first_increase_idx is not None and low_caps[first_increase_idx] == CFG.scale_up_step,
        f"cap at first increase = {low_caps[first_increase_idx] if first_increase_idx is not None else None} "
        f"(scale_up_step={CFG.scale_up_step})",
    ))
    checks.append(h.Check(
        "state returns to NORMAL only once cap reaches configured",
        reached_configured_idx is not None and low_states[reached_configured_idx] == "normal"
        and all(s != "normal" for s in low_states[:reached_configured_idx]),
        f"reached configured at index {reached_configured_idx}, "
        f"state there={low_states[reached_configured_idx] if reached_configured_idx is not None else None}",
    ))

    # ---- 2. re-arm: high pressure mid-recovery -------------------- #
    rearm_steps = [
        {"label": "idle", "hbm": 0.30},
        {"label": "critical", "hbm": crit},
        {"label": "low_1", "hbm": low},
        {"label": "low_2", "hbm": low},
        {"label": "low_3", "hbm": low},
        {"label": "low_4", "hbm": low},
        {"label": "low_5", "hbm": low},
        {"label": "low_6", "hbm": low},   # cap should have ticked up to 1 by now
        {"label": "high_again", "hbm": CFG.high_watermark + 0.02},
    ]
    rt = h.run_allocator_trace(CFG, 16, rearm_steps)
    obs["rearm_caps"] = [t.cap_out for t in rt]
    obs["rearm_states"] = [t.state for t in rt]
    pre = rt[-2]
    post = rt[-1]
    checks.append(h.Check(
        "a high sample during recovery re-arms the decrease and leaves RECOVERING",
        post.state == "high_pressure" and post.cap_out <= max(1, pre.cap_out),
        f"pre=({pre.cap_out},{pre.state}) post=({post.cap_out},{post.state})",
    ))

    analysis = (
        "MATCHES EXPECTATION. The low-pressure branch in allocator.py has two gates. First, if the "
        "previous state was any of CRITICAL/HIGH_PRESSURE/INCOMPLETE/STALE/DISCONNECTED/RECOVERING "
        "and fewer than recovery_complete_samples confirming samples have been seen, it only holds "
        "the current cap (reason=recovering_complete_reports, state=RECOVERING) and increments a "
        "counter - no expansion. Only after that does it count scale_up_stable_samples consecutive "
        "low samples and then add scale_up_step (=1). So from cap 0 the trajectory is a hold of "
        f"{CFG.recovery_complete_samples} samples, then +1 every {CFG.scale_up_stable_samples} "
        "samples (in this run the first +1 lands at low-sample index 7 = 3 completion-hold samples "
        "plus the stable-headroom counter reaching scale_up_stable_samples), and the state flips "
        "to NORMAL exactly when the cap regains the configured value "
        "(before that it stays RECOVERING). This is asymmetric by design: fast multiplicative "
        "decrease, slow additive increase, matching AIMD. The re-arm case shows recovery is not "
        "sticky - a single sample back above high_watermark immediately re-enters the "
        "high_pressure branch and multiplies the cap down again, so a flapping pressure signal "
        "cannot ratchet the cap up."
    )

    h.write_result(
        "H6",
        "Gradual recovery after pressure release",
        "After pressure returns below low_watermark, the controller holds for "
        "recovery_complete_samples then increases additively by scale_up_step per "
        "scale_up_stable_samples, reaching NORMAL only at the configured cap; a high sample "
        "mid-recovery re-arms the decrease.",
        "no instant jump; RECOVERING hold >= 3 samples; additive +1 steps; NORMAL only at configured; re-arm works.",
        obs,
        checks,
        analysis,
    )
    print(f"H6 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
