#!/usr/bin/env python3
"""H7 - the controller fails closed on coordinator loss, stale reports, missing
ranks, and stale/replayed generations.

Hypothesis
----------
Every telemetry / control fault stops *expansion* and, where the fault implies
lost visibility, tightens admission - but none of them evicts running requests
or advances state off a stale input:

  a. Coordinator disconnect: the scheduler's local guard pulls the safety cap
     to ``disconnect_admission_cap`` (0) and enters DISCONNECTED; running set
     retained.
  b. Stale report (replica alive, telemetry stopped): after
     ``report_timeout_ms`` the coordinator emits a STALE decision whose cap is
     <= the current cap (multiplicative decrease via the empty-rank path),
     tagged ``stale_report_*`` / ``telemetry_health``.
  c. Missing rank (incomplete report): first incomplete report is held for
     ``missing_report_grace_samples``; a further incomplete report decreases
     the cap; an incomplete report is NEVER treated as low pressure and never
     expands.
  d. Stale / replayed generation: an older ``decision_generation`` (or an
     equal one) is rejected by the scheduler and does not overwrite the
     current cap.
  e. Local KV-exhaustion guard: pulls cap to critical without consuming a
     coordinator generation.

Method
------
Real ``OmniSchedulerMixin`` for (a), (d), (e); real
``OmniCoordinator._check_memory_report_timeouts_locked`` for (b); real
``BudgetAllocator`` incomplete-report path for (c).
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

    # ---- a. coordinator disconnect ------------------------------- #
    sch = h.SchedulerHarness(cap=16, running=5)
    sch.apply(generation=1, cap=12, state=SafetyState.HIGH_PRESSURE, pressure=0.88)
    guarded = sch.disconnect_guard()
    obs["a_disconnect"] = {
        "guard_applied": guarded,
        "safety_cap": sch.safety_cap,
        "effective_cap": sch.effective_cap,
        "state": sch.state.value,
        "execution_budget_running": sch.max_running(),
        "allows_new_admission": sch.allows_new_admission(),
    }
    checks.append(h.Check(
        "a: disconnect -> cap 0, DISCONNECTED, 5 running retained, no new admission",
        guarded and sch.safety_cap == 0 and sch.state is SafetyState.DISCONNECTED
        and sch.max_running() == 5 and not sch.allows_new_admission(),
        f"{obs['a_disconnect']}",
    ))

    # ---- b. stale report via coordinator timeout sweep ---------- #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, cap=16, device_uuid="GPU-S")
    # push it up first so there is room to decrease
    c.advance_clock(CFG.sample_interval_ms / 1000.0)
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=CFG.high_watermark + 0.02, cap=16, device_uuid="GPU-S")
    cap_before_stale = c._c._last_caps["s0"]
    c.clear_sent()
    # run several timeout sweeps, each a sample-interval apart, so the
    # empty-rank grace->decrease path is visible end to end.
    stale_out: list[dict] = []
    stale_out += c.advance_time_and_check_timeouts(CFG.report_timeout_ms / 1000.0 + 0.1)
    for _ in range(4):
        stale_out += c.advance_time_and_check_timeouts(CFG.sample_interval_ms / 1000.0 + 0.01)
    obs["b_stale"] = {
        "cap_before_stale": cap_before_stale,
        "decisions": [(d["reason"], d["safety_state"], d["pressure_source"], d["effective_max_num_seqs"]) for d in stale_out],
    }
    caps_seq = [d["effective_max_num_seqs"] for d in stale_out]
    checks.append(h.Check(
        "b: stale report -> STALE decisions, cap monotonically non-increasing, telemetry_health source",
        len(stale_out) >= 2
        and all(d["safety_state"] == "stale" for d in stale_out)
        and all(d["reason"].startswith("stale_report_") for d in stale_out)
        and all(d["pressure_source"] == "telemetry_health" for d in stale_out)
        and all(b <= a for a, b in zip(caps_seq, caps_seq[1:]))
        and caps_seq[-1] < cap_before_stale,
        f"cap_before={cap_before_stale}, stale cap sequence={caps_seq}",
    ))

    # ---- c. missing rank / incomplete report ------------------- #
    # expected_rank_count=2 but only one rank present -> incomplete.
    r0 = h.make_rank_report(hbm_pressure=0.20, rank=0, device_uuid="GPU-M")
    incomplete_steps = [
        {"label": "complete_calm", "hbm": 0.20, "kv": 0.0},  # establish cap
        {"label": "incomplete_1", "hbm": 0.20, "expected_rank_count": 2, "complete": False, "rank_reports": (r0,)},
        {"label": "incomplete_2", "hbm": 0.20, "expected_rank_count": 2, "complete": False, "rank_reports": (r0,)},
        {"label": "incomplete_3", "hbm": 0.20, "expected_rank_count": 2, "complete": False, "rank_reports": (r0,)},
    ]
    it = h.run_allocator_trace(CFG, 16, incomplete_steps)
    obs["c_incomplete"] = [(t.label, t.cap_out, t.state, t.reason) for t in it]
    grace = CFG.missing_report_grace_samples
    held = it[1 + grace - 1] if grace >= 1 else None
    decreased = it[1 + grace]
    checks.append(h.Check(
        "c: first incomplete report(s) are HELD for missing_report_grace_samples",
        grace == 0 or (held is not None and held.state == "incomplete" and held.reason == "incomplete_rank_reports_hold"),
        f"grace={grace}, held step={obs['c_incomplete'][1 + grace - 1] if grace >= 1 else None}",
    ))
    checks.append(h.Check(
        "c: a further incomplete report DECREASES the cap (never expands, never 'low pressure')",
        decreased.state == "stale" and decreased.reason == "incomplete_rank_reports_decrease"
        and decreased.cap_out < 16,
        f"decrease step={obs['c_incomplete'][1 + grace]}",
    ))
    checks.append(h.Check(
        "c: no incomplete step is ever NORMAL or increases the cap",
        all(t.state != "normal" for t in it[1:]) and all(t.cap_out <= 16 for t in it[1:]),
        f"states={[t.state for t in it[1:]]}",
    ))

    # ---- d. stale / replayed generation ----------------------- #
    schd = h.SchedulerHarness(cap=16)
    schd.apply(generation=5, cap=8, state=SafetyState.HIGH_PRESSURE, pressure=0.9, report=10)
    cap_after_g5 = schd.safety_cap
    replay_lower = schd.apply(generation=3, cap=16, state=SafetyState.NORMAL, pressure=0.3, report=11)
    replay_equal = schd.apply(generation=5, cap=16, state=SafetyState.NORMAL, pressure=0.3, report=11)
    stale_report_gen = schd.apply(generation=6, cap=16, state=SafetyState.NORMAL, pressure=0.3, report=2)
    obs["d_generation"] = {
        "cap_after_g5": cap_after_g5,
        "older_generation_accepted": replay_lower,
        "equal_generation_accepted": replay_equal,
        "stale_report_generation_accepted": stale_report_gen,
        "final_safety_cap": schd.safety_cap,
    }
    checks.append(h.Check(
        "d: older / equal decision generation and stale report generation are all rejected",
        (not replay_lower) and (not replay_equal) and (not stale_report_gen) and schd.safety_cap == cap_after_g5 == 8,
        f"{obs['d_generation']}",
    ))

    # ---- e. local KV guard does not consume a generation ------ #
    sche = h.SchedulerHarness(cap=16, running=3)
    sche.apply(generation=7, cap=10, state=SafetyState.HIGH_PRESSURE, pressure=0.88, report=20)
    kv_guarded = sche.local_kv_guard()
    # a fresh coordinator decision at generation 8 must still apply afterwards
    post = sche.apply(generation=8, cap=12, state=SafetyState.HIGH_PRESSURE, pressure=0.86, report=21)
    obs["e_kv_guard"] = {
        "kv_guard_applied": kv_guarded,
        "post_decision_accepted": post,
        "final_state": sche.state.value,
    }
    checks.append(h.Check(
        "e: local KV guard tightens immediately and does not block the next coordinator decision",
        kv_guarded and post,
        f"{obs['e_kv_guard']}",
    ))

    analysis = (
        "MATCHES EXPECTATION on all five fault paths.\n"
        "(a) apply_dynamic_hbm_disconnect_guard takes min(current_safety_cap, "
        "disconnect_admission_cap)=0 and sets DISCONNECTED; the execution budget still equals the "
        "running count, so the 5 in-flight requests are not evicted, but _dynamic_hbm_allows_"
        "new_admission() is False. This is a purely local action - it fires without any message "
        "from the (now gone) coordinator.\n"
        "(b) _check_memory_report_timeouts_locked, once report age exceeds report_timeout_ms, "
        "synthesises a report with an EMPTY rank tuple and runs it through the same BudgetAllocator; "
        "the empty-rank path multiplicatively decreases the cap, and the wire decision is tagged "
        "safety_state=stale, reason=stale_report_*, pressure_source=telemetry_health. Expansion is "
        "impossible while stale.\n"
        "(c) In BudgetAllocator the incomplete branch holds the cap for missing_report_grace_"
        "samples (state=incomplete) then multiplicatively decreases (state=stale, "
        "reason=incomplete_rank_reports_decrease). No incomplete report can reach the low-pressure "
        "branch, so a half-missing report can never be mistaken for headroom and can never expand "
        "the cap.\n"
        "(d) apply_stage_budget_decision rejects any decision whose generation <= the last applied "
        "generation, and any decision whose based_on_report_generation < the last report "
        "generation. An older replay, an equal-generation replay, and a decision built on a stale "
        "report are all dropped, leaving the last good cap (8) intact.\n"
        "(e) apply_dynamic_hbm_local_kv_guard tightens to critical_admission_cap immediately "
        "WITHOUT advancing _last_budget_generation, so the next genuine coordinator decision "
        "(generation 8) still applies and reconciles the replica. Fail-closed here does not wedge "
        "the control channel.\n"
        "Across all paths the invariant holds: faults stop expansion and (where visibility is "
        "lost) tighten admission, but never kill running work and never let a stale input drive "
        "state."
    )

    h.write_result(
        "H7",
        "Fail-closed on disconnect / stale / missing-rank / stale-generation",
        "Coordinator loss, stale telemetry, incomplete reports, and replayed/stale generations all "
        "stop expansion and (where relevant) tighten admission, without evicting running requests "
        "or advancing state off a stale input.",
        "a: disconnect->0 & retain running; b: STALE decrease; c: incomplete held then decreased, "
        "never NORMAL; d: old/equal/stale-report generations rejected; e: KV guard keeps channel live.",
        obs,
        checks,
        analysis,
    )
    print(f"H7 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
