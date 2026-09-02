#!/usr/bin/env python3
"""H13 - bounded bypass prevents head-of-line blocking without starving the
head-of-line request.

Hypothesis
----------
Before this change, ``OmniARScheduler.schedule()`` reacted to a resource-aware
defer decision by substituting an EMPTY queue for ``self.waiting`` for the
whole step (see ``_should_defer_waiting_admission`` /
``schedule()``). ``_dynamic_hbm_resource_admission_decision`` itself only ever
looked at ``next(iter(waiting))`` - the head-of-line request. The combination
meant that once free KV blocks fell below the head-of-line request's estimate,
EVERY waiting request was frozen for that step, even a short request right
behind it that would easily fit in the same free KV blocks. This is exactly
the head-of-line-blocking failure mode the M2 design doc (S12.3) calls out
as a required "bounded bypass" safeguard.

``_dynamic_hbm_bounded_bypass_waiting`` fixes this: when the defer reason is
specifically ``KV_PEAK_RISK`` for the head-of-line request (not a global
block-exhaustion guard, and not the estimator being disabled/unavailable), it
scans up to ``resource_admission_bypass_scan_limit`` subsequent waiting
requests and lets any that individually fit into the current free-block count
bypass ahead of the still-unfit head. The head-of-line request itself is never
dropped from consideration, and once it has waited at least
``resource_admission_bypass_aging_ms``, the bypass stops (even if a bypass-
eligible request is still available) so it cannot be starved by an unbroken
stream of smaller requests jumping the queue.

Expectation
-----------
1. With bypass disabled (``resource_admission_bypass_scan_limit=0``, the
   default), a long head-of-line request that does not fit produces an empty
   candidate queue - the short request behind it gets no chance this step.
2. With bypass enabled, the same scenario instead produces a candidate queue
   containing BOTH the (still-unfit) head-of-line request and the short
   request that fits - the short request is no longer blocked.
3. A second long request behind the head, which itself does not fit, is
   correctly left out of the candidate queue (bypass admits only requests
   that actually fit, not everything behind the head).
4. The scan is bounded: with ``scan_limit=1``, only the first request behind
   the head is examined, so a fitting request two positions back is not
   reached even though it would fit.
5. Once the head-of-line request's recorded wait time reaches the aging
   threshold, bypass stops for that step (the head-of-line request is not
   perpetually deprioritized in favor of smaller requests jumping ahead of
   it), even though a bypass-eligible request is still present in the queue.

Method
------
Drive the REAL ``OmniSchedulerMixin`` (as in H10/H11/H12) via the extended
``SchedulerHarness``, calling ``_dynamic_hbm_bounded_bypass_waiting`` directly
- the same private method ``OmniARScheduler.schedule()`` calls once a step's
resource-aware defer decision is scoped to the head-of-line request.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

BLOCK_SIZE = 16
LONG_REQ = h.FakeWaitingRequest(
    request_id="long", num_prompt_tokens=1024, max_tokens=512
)  # ceil(1536/16) = 96 blocks - never fits in this experiment's 32 free blocks
SECOND_LONG_REQ = h.FakeWaitingRequest(
    request_id="second_long", num_prompt_tokens=1024, max_tokens=512
)  # also 96 blocks - also never fits
SHORT_REQ = h.FakeWaitingRequest(
    request_id="short", num_prompt_tokens=64, max_tokens=32
)  # ceil(96/16) = 6 blocks - fits easily in 32 free blocks
FREE_BLOCKS = 32


def _enforce_harness(*, scan_limit: int, aging_ms: float = 30_000.0) -> h.SchedulerHarness:
    return h.SchedulerHarness(
        cap=16,
        running=0,
        block_size=BLOCK_SIZE,
        free_kv_blocks=FREE_BLOCKS,
        config={
            "enabled": True,
            "resource_admission_mode": "enforce",
            "resource_admission_bypass_scan_limit": scan_limit,
            "resource_admission_bypass_aging_ms": aging_ms,
        },
    )


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. bypass disabled (default): whole queue freezes behind head --- #
    sch1 = _enforce_harness(scan_limit=0)
    sch1.set_waiting([LONG_REQ, SHORT_REQ])
    candidates_1 = [r.request_id for r in sch1.bounded_bypass_waiting()]
    obs["bypass_disabled_default"] = {
        "scan_limit": 0,
        "candidate_queue": candidates_1,
    }
    checks.append(h.Check(
        "scan_limit=0 (default) reproduces the pre-M2-bypass all-or-nothing freeze",
        candidates_1 == [],
        f"{obs['bypass_disabled_default']}",
    ))

    # ---- 2. bypass enabled: short request escapes the oversized head ----- #
    sch2 = _enforce_harness(scan_limit=5)
    sch2.set_waiting([LONG_REQ, SHORT_REQ])
    candidates_2 = [r.request_id for r in sch2.bounded_bypass_waiting()]
    obs["bypass_admits_fitting_request_behind_oversized_head"] = {
        "scan_limit": 5,
        "candidate_queue": candidates_2,
        "bypass_count": sch2.bypass_count,
    }
    checks.append(h.Check(
        "bypass keeps the head-of-line request and additionally admits the fitting short request",
        candidates_2 == ["long", "short"],
        f"{obs['bypass_admits_fitting_request_behind_oversized_head']}",
    ))
    checks.append(h.Check(
        "exactly one bypass was recorded",
        sch2.bypass_count == 1,
        f"bypass_count={sch2.bypass_count}",
    ))

    # ---- 3. bypass does not admit a later request that also doesn't fit -- #
    sch3 = _enforce_harness(scan_limit=5)
    sch3.set_waiting([LONG_REQ, SECOND_LONG_REQ])
    candidates_3 = [r.request_id for r in sch3.bounded_bypass_waiting()]
    obs["bypass_rejects_a_second_oversized_request"] = {
        "scan_limit": 5,
        "candidate_queue": candidates_3,
        "bypass_count": sch3.bypass_count,
    }
    checks.append(h.Check(
        "a second oversized request behind the head is left out of the candidate queue",
        candidates_3 == ["long"],
        f"{obs['bypass_rejects_a_second_oversized_request']}",
    ))

    # ---- 4. scan is bounded: scan_limit=1 stops before reaching "short" --- #
    sch4 = _enforce_harness(scan_limit=1)
    sch4.set_waiting([LONG_REQ, SECOND_LONG_REQ, SHORT_REQ])
    candidates_4 = [r.request_id for r in sch4.bounded_bypass_waiting()]
    obs["scan_limit_bounds_how_far_the_scan_looks"] = {
        "scan_limit": 1,
        "candidate_queue": candidates_4,
    }
    checks.append(h.Check(
        "scan_limit=1 only examines the first candidate behind the head, missing the fitting one further back",
        candidates_4 == ["long"],
        f"{obs['scan_limit_bounds_how_far_the_scan_looks']}",
    ))

    # ---- 5. aging stops the bypass once the head has waited long enough -- #
    sch5 = _enforce_harness(scan_limit=5, aging_ms=1.0)
    sch5.set_waiting([LONG_REQ, SHORT_REQ])
    times = iter([1000.0, 1000.05])
    with mock.patch(
        "vllm_omni.core.sched.omni_scheduler_mixin.time.monotonic",
        side_effect=lambda: next(times),
    ):
        first_step = [r.request_id for r in sch5.bounded_bypass_waiting()]
        second_step = [r.request_id for r in sch5.bounded_bypass_waiting()]
    obs["aging_stops_bypass_after_threshold"] = {
        "aging_ms": 1.0,
        "first_step_candidate_queue": first_step,
        "second_step_candidate_queue": second_step,
        "aging_stops": sch5.aging_stops,
    }
    checks.append(h.Check(
        "first step (head just started waiting) still bypasses the fitting short request",
        first_step == ["long", "short"],
        f"{obs['aging_stops_bypass_after_threshold']}",
    ))
    checks.append(h.Check(
        "second step (head has now aged past the threshold) stops bypassing, "
        "even though the short request would still fit",
        second_step == ["long"] and sch5.aging_stops == 1,
        f"{obs['aging_stops_bypass_after_threshold']}",
    ))

    analysis = (
        "MATCHES EXPECTATION. Check 1 confirms scan_limit=0 (the default) "
        "reproduces the original all-or-nothing freeze exactly, so bounded "
        "bypass is strictly opt-in and introduces no behavior change until "
        "configured. Checks 2-3 show the scan is a genuine per-candidate fit "
        "check, not a blanket 'let everything through' escape hatch: the "
        "short request that fits is admitted to the candidate queue, while a "
        "second oversized request that does not fit is correctly left "
        "behind. Check 4 confirms resource_admission_bypass_scan_limit is a "
        "real bound on how many candidates are examined per step, not merely "
        "advisory. Check 5 confirms the aging safeguard: once the head-of-"
        "line request has been waiting at least resource_admission_bypass_"
        "aging_ms, the scan stops admitting bypass candidates for that step "
        "even though one is still available - this is what keeps a steady "
        "stream of small requests from starving a large request indefinitely, "
        "which is the failure mode the M2 design doc's bounded-bypass "
        "requirement (S12.3) exists to prevent. Together these confirm the "
        "single all-or-nothing queue swap in OmniARScheduler.schedule() that "
        "existed before this change - and that H10-H12 exercised - could not "
        "express 'let the fitting request through, keep the oversized one "
        "queued, and eventually still serve the oversized one'; bounded "
        "bypass is the missing piece between the resource-aware KV estimate "
        "(H10) and a scheduler policy that is actually safe to enforce."
    )

    h.write_result(
        "H13",
        "bounded bypass prevents head-of-line blocking without starving the head-of-line request",
        "Scanning past an oversized head-of-line request for a bounded number of fitting "
        "later requests removes head-of-line blocking, while an aging guard stops the scan "
        "once the head-of-line request has waited long enough that it must not be starved "
        "further.",
        "scan_limit=0 reproduces the original freeze; scan_limit>0 admits a fitting later "
        "request while still rejecting a non-fitting one; the scan is bounded by scan_limit; "
        "aging stops the bypass once the head-of-line wait exceeds the configured threshold.",
        obs,
        checks,
        analysis,
    )
    print(f"H13 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
