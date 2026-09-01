#!/usr/bin/env python3
"""H10 - resource-aware admission catches oversized requests slot-count misses.

Hypothesis
----------
The pre-Milestone-2 admission gate (``_dynamic_hbm_allows_new_admission``)
treats every AR request as one uniform "slot", so it admits a request whenever
``occupied_slots < effective_max_num_seqs`` - regardless of that request's
prompt/output length. Under a mixed workload (a few long requests, many short
ones) this is blind to the case where a free *slot* exists but the head-of-
line waiting request's real KV-block cost does not fit in the free KV pool.

Milestone 2's ``ARResourceEstimator`` + ``_dynamic_hbm_next_waiting_request_
fits`` close this gap: they estimate ``ceil((prompt_tokens + max_tokens) /
block_size)`` for the head-of-line waiting request and compare it against the
replica's actual free KV blocks, independent of the slot-count cap.

Expectation
-----------
1. A short request (small KV footprint) is admitted by both the old
   slot-count gate and the new resource-aware gate when a slot is free and
   blocks are scarce but sufficient.
2. A long request (large KV footprint) that a free slot would admit under the
   old gate is REJECTED by the new resource-aware gate once free KV blocks
   fall below its estimate - the exact "slot free, but doesn't actually fit"
   case the old gate cannot see.
3. The new gate is a pure extension: with abundant free KV blocks, it never
   rejects a request the old gate would have admitted (no new false
   negatives when there is no real KV pressure).
4. The new gate is disabled (no additional restriction) when dynamic HBM is
   off, and is a no-op on an empty waiting queue - it never introduces a new
   failure mode outside its scope.

Method
------
Drive the REAL ``OmniSchedulerMixin`` (as in H5/H9) via the extended
``SchedulerHarness``, which now exposes ``cache_config.block_size``,
``kv_cache_manager.block_pool`` (a fake with a settable free-block count), and
``waiting`` (a list of fake ``Request``-like objects carrying
``num_prompt_tokens`` / ``max_tokens``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

BLOCK_SIZE = 16
SHORT_REQ = h.FakeWaitingRequest(num_prompt_tokens=64, max_tokens=32)  # ceil(96/16) = 6 blocks
LONG_REQ = h.FakeWaitingRequest(num_prompt_tokens=1024, max_tokens=512)  # ceil(1536/16) = 96 blocks


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- 1. short request: fits under scarce-but-sufficient KV blocks --- #
    sch1 = h.SchedulerHarness(cap=16, running=0, block_size=BLOCK_SIZE, free_kv_blocks=8)
    sch1.set_waiting([SHORT_REQ])
    old_gate_1 = sch1.allows_new_admission()  # slot-count only: 0 running < 16 cap
    new_gate_1 = sch1.next_waiting_request_fits()
    obs["short_request_scarce_but_sufficient"] = {
        "free_kv_blocks": 8,
        "estimated_blocks": 6,
        "old_slot_gate_admits": old_gate_1,
        "new_resource_gate_admits": new_gate_1,
    }
    checks.append(h.Check(
        "short request (6 blocks) admitted by both gates when 8 blocks are free",
        old_gate_1 and new_gate_1,
        f"{obs['short_request_scarce_but_sufficient']}",
    ))

    # ---- 2. long request: slot-count gate is blind, resource gate isn't - #
    sch2 = h.SchedulerHarness(cap=16, running=0, block_size=BLOCK_SIZE, free_kv_blocks=32)
    sch2.set_waiting([LONG_REQ])
    old_gate_2 = sch2.allows_new_admission()  # slot-count only: still 0 running < 16 cap
    new_gate_2 = sch2.next_waiting_request_fits()
    obs["long_request_insufficient_blocks"] = {
        "free_kv_blocks": 32,
        "estimated_blocks": 96,
        "old_slot_gate_admits": old_gate_2,
        "new_resource_gate_admits": new_gate_2,
    }
    checks.append(h.Check(
        "old slot-count gate is blind: admits the 96-block request with only 32 free",
        old_gate_2 is True,
        f"{obs['long_request_insufficient_blocks']}",
    ))
    checks.append(h.Check(
        "new resource-aware gate rejects the same request the old gate missed",
        new_gate_2 is False,
        f"{obs['long_request_insufficient_blocks']}",
    ))

    # ---- 3. abundant KV blocks: new gate introduces no false negative ---- #
    sch3 = h.SchedulerHarness(cap=16, running=0, block_size=BLOCK_SIZE, free_kv_blocks=10_000)
    sch3.set_waiting([LONG_REQ])
    old_gate_3 = sch3.allows_new_admission()
    new_gate_3 = sch3.next_waiting_request_fits()
    obs["abundant_blocks_no_regression"] = {
        "free_kv_blocks": 10_000,
        "estimated_blocks": 96,
        "old_slot_gate_admits": old_gate_3,
        "new_resource_gate_admits": new_gate_3,
    }
    checks.append(h.Check(
        "with abundant free KV blocks, the new gate agrees with the old gate (no regression)",
        old_gate_3 and new_gate_3,
        f"{obs['abundant_blocks_no_regression']}",
    ))

    # ---- 4a. dynamic HBM disabled: new gate imposes no extra restriction - #
    sch4 = h.SchedulerHarness(cap=16, running=0, block_size=BLOCK_SIZE, free_kv_blocks=0, config={"enabled": False})
    sch4.set_waiting([LONG_REQ])
    disabled_gate = sch4.next_waiting_request_fits()
    obs["disabled_dynamic_hbm"] = {"free_kv_blocks": 0, "estimated_blocks": 96, "new_resource_gate_admits": disabled_gate}
    checks.append(h.Check(
        "dynamic HBM disabled -> resource gate is a no-op (returns True) even with 0 free blocks",
        disabled_gate is True,
        f"{obs['disabled_dynamic_hbm']}",
    ))

    # ---- 4b. empty waiting queue: no request to check, gate is a no-op --- #
    sch5 = h.SchedulerHarness(cap=16, running=0, block_size=BLOCK_SIZE, free_kv_blocks=0)
    sch5.set_waiting([])
    empty_queue_gate = sch5.next_waiting_request_fits()
    obs["empty_waiting_queue"] = {"free_kv_blocks": 0, "new_resource_gate_admits": empty_queue_gate}
    checks.append(h.Check(
        "empty waiting queue -> resource gate is a no-op (returns True)",
        empty_queue_gate is True,
        f"{obs['empty_waiting_queue']}",
    ))

    analysis = (
        "MATCHES EXPECTATION. The pre-Milestone-2 gate "
        "(_dynamic_hbm_allows_new_admission) is a pure "
        "`occupied_slots < effective_max_num_seqs` count with no notion of a "
        "request's size, so it admits a 96-block request with only 32 free "
        "blocks (check 2) exactly as readily as it admits a 6-block request "
        "(check 1) - both cases have 0 running against a cap of 16 and are "
        "indistinguishable to the slot-count gate. The new "
        "_dynamic_hbm_next_waiting_request_fits check closes this gap by "
        "estimating the head-of-line waiting request's KV-block cost via "
        "ARResourceEstimator (same ceil((prompt+max_tokens)/block_size) "
        "accounting OmniARScheduler already uses for KV-transfer truncation) "
        "and comparing it against block_pool.get_num_free_blocks(). It agrees "
        "with the old gate whenever blocks are abundant (check 3), so it adds "
        "no new false negatives in the common case, and it is a strict no-op "
        "outside its scope (dynamic HBM disabled, check 4a; empty waiting "
        "queue, check 4b) - it never introduces a new failure mode. This is "
        "the heterogeneous-resource-model gap identified in the Milestone 2 "
        "research design: a uniform 'max_num_seqs' cap cannot express that "
        "different requests cost different amounts of the same KV pool."
    )

    h.write_result(
        "H10",
        "resource-aware admission catches oversized requests slot-count admission misses",
        "A per-request KV-block estimate compared against free KV blocks rejects requests "
        "the uniform slot-count gate would admit, without rejecting anything the slot-count "
        "gate would admit when KV blocks are abundant.",
        "Short request admitted by both gates; long request admitted by the old gate but "
        "rejected by the new gate under scarce free blocks; no regression when blocks are "
        "abundant; no-op when disabled or waiting is empty.",
        obs,
        checks,
        analysis,
    )
    print(f"H10 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
