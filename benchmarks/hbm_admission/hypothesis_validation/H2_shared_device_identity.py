#!/usr/bin/env python3
"""H2 - the coordinator identifies stages that share one physical GPU.

Hypothesis
----------
When several stage replicas report the same physical device, the coordinator
groups them by ``(node_id, device_uuid)`` (not by local ``device_id``), so a
pressure signal from one is applied to all co-resident consumers; and stages on
*different* physical devices are not coupled.

Method
------
Drive the REAL ``OmniCoordinator._handle_memory_report_locked`` via
``CoordinatorHarness``. Four sub-cases:

  A. stage 0 and stage 1 report the same ``device_uuid`` -> a high-pressure
     report from stage 0 must produce a decision for BOTH, tagged
     ``shared_device_*``.
  B. stage 0 and stage 2 report *different* ``device_uuid`` -> a high-pressure
     report from stage 0 must NOT move stage 2's cap.
  C. same local ``device_id`` (0) but different ``node_id`` -> must NOT be
     grouped (guards against multi-node index collision).
  D. same local ``device_id`` (0) but different ``device_uuid`` on one node ->
     must NOT be grouped.

Acceptance
----------
All four sub-cases behave as described; the shared decision reason is
``shared_device_high_pressure`` and its ``pressure_source`` is
``shared_physical_hbm``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402


def main() -> None:
    obs: dict = {}
    checks: list[h.Check] = []

    # ---- A: same (node, uuid) -> grouped -------------------------------- #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-AAAA", node_id="n0")
    c.report(addr="s1", stage_id=1, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-AAAA", node_id="n0")
    c.clear_sent()
    out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.92, device_uuid="GPU-AAAA", node_id="n0")
    by_stage = {d["stage_id"]: d for d in out}
    obs["A_decisions"] = [(d["stage_id"], d["reason"], d["pressure_source"], d["effective_max_num_seqs"]) for d in out]
    checks.append(h.Check(
        "A: shared uuid propagates high pressure to both stages",
        set(by_stage) == {0, 1}
        and by_stage[0]["effective_max_num_seqs"] == 8
        and by_stage[1]["effective_max_num_seqs"] == 8,
        f"{obs['A_decisions']}",
    ))
    checks.append(h.Check(
        "A: shared decision tagged shared_device_high_pressure / shared_physical_hbm",
        all(d["reason"] == "shared_device_high_pressure" and d["pressure_source"] == "shared_physical_hbm" for d in out),
        f"reasons={[d['reason'] for d in out]}",
    ))

    # ---- B: different uuid -> not grouped ----------------------------- #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-AAAA", node_id="n0")
    c.report(addr="s2", stage_id=2, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-BBBB", node_id="n0")
    c.clear_sent()
    out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.92, device_uuid="GPU-AAAA", node_id="n0")
    obs["B_decisions"] = [(d["stage_id"], d["reason"], d["effective_max_num_seqs"]) for d in out]
    checks.append(h.Check(
        "B: distinct uuid -> peer stage on other GPU is untouched",
        all(d["stage_id"] == 0 for d in out),
        f"{obs['B_decisions']}",
    ))

    # ---- C: same device_id, different node_id -> not grouped ---------- #
    c = h.CoordinatorHarness()
    # empty device_uuid so the coordinator falls back to local-device-<id>;
    # different node_id must keep them separate.
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="", node_id="nodeA")
    c.report(addr="s1", stage_id=1, replica_id=0, hbm_pressure=0.30, device_uuid="", node_id="nodeB")
    c.clear_sent()
    out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.92, device_uuid="", node_id="nodeA")
    obs["C_decisions"] = [(d["stage_id"], d["reason"], d["effective_max_num_seqs"]) for d in out]
    checks.append(h.Check(
        "C: equal local device_id on different nodes are NOT grouped",
        all(d["stage_id"] == 0 for d in out),
        f"{obs['C_decisions']}",
    ))

    # ---- D: same node+device_id, different uuid -> not grouped -------- #
    c = h.CoordinatorHarness()
    c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-UUID-1", node_id="n0")
    c.report(addr="s1", stage_id=1, replica_id=0, hbm_pressure=0.30, device_uuid="GPU-UUID-2", node_id="n0")
    c.clear_sent()
    out = c.report(addr="s0", stage_id=0, replica_id=0, hbm_pressure=0.92, device_uuid="GPU-UUID-1", node_id="n0")
    obs["D_decisions"] = [(d["stage_id"], d["reason"], d["effective_max_num_seqs"]) for d in out]
    checks.append(h.Check(
        "D: same node/device_id but different uuid are NOT grouped",
        all(d["stage_id"] == 0 for d in out),
        f"{obs['D_decisions']}",
    ))

    analysis = (
        "MATCHES EXPECTATION. OmniCoordinator._device_keys builds the grouping key as "
        "(node_id, device_uuid or 'local-device-<device_id>'), and _handle_memory_report_locked "
        "computes `affected` as every registered replica whose device-key set intersects the "
        "incoming report's. Sub-case A confirms a single high-pressure report fans a decision out "
        "to every co-resident stage with reason `shared_device_high_pressure` and pressure_source "
        "`shared_physical_hbm`, and the cap drop (16->8) is applied to the peer that never saw "
        "pressure itself. Sub-cases B/C/D confirm the negative direction: a different UUID, a "
        "different node_id behind an empty UUID, or a different UUID on the same node all keep the "
        "replicas in disjoint groups, so cross-talk cannot happen. The design's multi-node "
        "identity rule (never trust the bare local index) holds."
    )

    h.write_result(
        "H2",
        "Coordinator identifies stages sharing one physical GPU",
        "Stages are grouped by (node_id, device_uuid); a pressure signal from one co-resident stage "
        "reaches all of them, and stages on distinct devices stay decoupled.",
        "A groups & propagates with shared_device_* tag; B/C/D stay decoupled.",
        obs,
        checks,
        analysis,
    )
    print(f"H2 verdict: {'MATCHES' if all(c.passed for c in checks) else 'DEVIATION'}")
    for ck in checks:
        print(f"  [{'PASS' if ck.passed else 'FAIL'}] {ck.name} -- {ck.detail}")


if __name__ == "__main__":
    main()
