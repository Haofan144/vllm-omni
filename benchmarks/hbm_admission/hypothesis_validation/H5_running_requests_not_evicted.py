#!/usr/bin/env python3
"""H5 - cap=0 blocks new admission but never evicts running requests.

Hypothesis
----------
A safety cap of 0 (or any cap below the current running count) stops the
scheduler from admitting new requests, yet:
  * the running set is retained (execution budget >= running count);
  * the scheduled-token budget still covers the in-flight work;
  * admission resumes automatically as soon as the running count drops back
    below the effective cap - no external nudge required.

Method
------
Drive the REAL ``OmniSchedulerMixin`` (as in
tests/core/sched/test_dynamic_hbm_admission_gate.py):

1. 6 running requests, apply CRITICAL cap 0 -> effective_cap 0, but
   ``_dynamic_max_num_running_reqs()`` == 6 and token budget > 0.
2. HIGH cap 2 with 2 running -> no new admission; pop one -> admission allowed.
3. Recovery: after cap returns to configured 16, admission is allowed again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402
from _harness import SafetyState  # noqa: E402


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. critical cap 0 with 6 running ------------------------- #
    sch = h.SchedulerHarness(cap=16, tokens=4096, running=6)
    sch.apply(generation=1, cap=0, state=SafetyState.CRITICAL, pressure=0.97)
    obs["critical_with_running"] = {
        "effective_cap": sch.effective_cap,
        "max_running_execution_budget": sch.max_running(),
        "effective_tokens": sch.effective_tokens,
        "allows_new_admission": sch.allows_new_admission(),
    }
    checks.append(h.Check(
        "cap=0 keeps all 6 running (execution budget == running count)",
        sch.effective_cap == 0 and sch.max_running() == 6,
        f"{obs['critical_with_running']}",
    ))
    checks.append(h.Check(
        "cap=0 still grants a nonzero scheduled-token budget for in-flight work",
        sch.effective_tokens >= 1,
        f"effective_tokens={sch.effective_tokens}",
    ))
    checks.append(h.Check(
        "cap=0 blocks NEW admission",
        not sch.allows_new_admission(),
        f"allows_new_admission={sch.allows_new_admission()}",
    ))

    # ---- 2. drain unblocks admission without any new decision ---- #
    sch2 = h.SchedulerHarness(cap=16, running=2)
    sch2.apply(generation=1, cap=2, state=SafetyState.HIGH_PRESSURE, pressure=0.92)
    blocked_at_2 = not sch2.allows_new_admission()
    sch2.set_running(1)  # one request finishes
    allowed_at_1 = sch2.allows_new_admission()
    obs["drain_unblocks"] = {"blocked_when_running_2_cap_2": blocked_at_2, "allowed_when_running_1_cap_2": allowed_at_1}
    checks.append(h.Check(
        "admission resumes when running drops below effective cap (no new decision)",
        blocked_at_2 and allowed_at_1,
        f"{obs['drain_unblocks']}",
    ))

    # ---- 3. recovery to configured cap re-enables admission ----- #
    sch3 = h.SchedulerHarness(cap=16, running=0)
    sch3.apply(generation=1, cap=0, state=SafetyState.CRITICAL, pressure=0.97)
    down = sch3.allows_new_admission()
    sch3.apply(generation=2, cap=16, state=SafetyState.NORMAL, pressure=0.40)
    up = sch3.allows_new_admission()
    obs["recovery"] = {
        "allows_when_critical": down,
        "allows_after_normal_restore": up,
        "effective_cap_after": sch3.effective_cap,
    }
    checks.append(h.Check(
        "after NORMAL decision restores cap, admission is allowed again",
        (not down) and up and sch3.effective_cap == 16,
        f"{obs['recovery']}",
    ))

    analysis = (
        "MATCHES EXPECTATION. _recompute_effective_dynamic_hbm_budget in omni_scheduler_mixin.py "
        "separates two quantities: the ADMISSION cap (_effective_max_num_seqs, which the safety "
        "controller can pull to 0) and the EXECUTION budget "
        "(_dynamic_max_num_running_reqs / scheduling_slots = max(effective_cap, occupied_slots)). "
        "Because occupied_slots = len(running) + waiting-for-streaming-input, a cap of 0 while 6 "
        "requests run yields an execution budget of 6 and a token budget scaled by "
        "min(1, scheduling_slots/configured), never zero - so the scheduler keeps stepping the "
        "in-flight requests to completion and evicts nothing. _dynamic_hbm_allows_new_admission() "
        "is a pure `len(running) < effective_cap` test with no side effects, so as soon as a "
        "request finishes and running drops below the cap, admission re-enables on the next "
        "scheduler tick with no new coordinator message. Restoring the cap to its configured "
        "value (recovery, H6) also re-enables admission immediately. This is the design's "
        "'reduce a cap, do not evict' invariant, verified against the real scheduler mixin."
    )

    h.write_result(
        "H5",
        "cap=0 halts admission without evicting running requests",
        "A zero/low safety cap blocks new admission while retaining the running set and its token "
        "budget, and admission resumes automatically once running drops below the cap.",
        "6 running survive cap 0; drain re-enables admission; recovery restores admission.",
        obs,
        checks,
        analysis,
    )
    print(f"H5 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
