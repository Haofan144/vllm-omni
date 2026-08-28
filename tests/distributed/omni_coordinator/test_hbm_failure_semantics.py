# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import asdict

import pytest
from vllm.v1.utils import get_engine_client_zmq_addr

from vllm_omni.core.memory_coordinator import RankMemoryReport, ReplicaMemoryReport
from vllm_omni.distributed.omni_coordinator import OmniCoordClientForStage, OmniCoordinator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _coordinator() -> OmniCoordinator:
    router = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    pub = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    return OmniCoordinator(router, pub, heartbeat_timeout=1000.0)


def _client(coordinator, name: str, stage: int) -> OmniCoordClientForStage:
    return OmniCoordClientForStage(
        coordinator.router_zmq_addr,
        f"tcp://stage:{name}",
        f"tcp://stage:{name}-out",
        stage,
        replica_id=0,
    )


def _report(
    stage: int,
    *,
    hbm: float,
    kv: float,
    node: str = "node",
    uuid: str = "GPU-shared",
) -> dict:
    rank = RankMemoryReport(
        stage_id=stage,
        replica_id=0,
        rank=0,
        device_id=0,
        timestamp_monotonic_s=time.monotonic(),
        device_total_bytes=1000,
        device_free_bytes=round(1000 * (1 - hbm)),
        process_allocated_bytes=100,
        process_reserved_bytes=120,
        node_id=node,
        device_uuid=uuid,
    )
    return asdict(
        ReplicaMemoryReport(
            stage_id=stage,
            replica_id=0,
            timestamp_monotonic_s=rank.timestamp_monotonic_s,
            rank_reports=(rank,),
            expected_rank_count=1,
            kv_total_blocks=100,
            kv_free_blocks=round(100 * (1 - kv)),
            running_requests=0,
            waiting_requests=4,
            configured_max_num_seqs=16,
        )
    )


def _wait_for(client, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    decisions = []
    while time.monotonic() < deadline:
        decisions.extend(client.poll_budget_decisions())
        matches = [decision for decision in decisions if predicate(decision)]
        if matches:
            return matches[-1]
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for HBM decision; received={decisions!r}")


def _close(coordinator, *clients) -> None:
    for client in clients:
        if not client._closed:
            client.close()
    coordinator.close()


def test_kv_pressure_is_replica_local_on_shared_gpu() -> None:
    coordinator = _coordinator()
    first = _client(coordinator, "kv-a", 0)
    second = _client(coordinator, "kv-b", 1)
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    try:
        first.send_memory_report(_report(0, hbm=0.5, kv=0.96), config)
        second.send_memory_report(_report(1, hbm=0.5, kv=0.2), config)
        critical = _wait_for(first, lambda d: d.safety_state == "critical")
        normal = _wait_for(second, lambda d: d.pressure < 0.9)
        assert critical.pressure_source == "kv"
        assert critical.effective_max_num_seqs == 0
        assert normal.effective_max_num_seqs == 16
    finally:
        _close(coordinator, first, second)


def test_physical_pressure_does_not_cross_device_uuid() -> None:
    coordinator = _coordinator()
    first = _client(coordinator, "gpu-a", 0)
    second = _client(coordinator, "gpu-b", 1)
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    try:
        first.send_memory_report(_report(0, hbm=0.96, kv=0.2, uuid="GPU-A"), config)
        second.send_memory_report(_report(1, hbm=0.5, kv=0.2, uuid="GPU-B"), config)
        critical = _wait_for(first, lambda d: d.safety_state == "critical")
        normal = _wait_for(second, lambda d: d.pressure < 0.9)
        assert critical.effective_max_num_seqs == 0
        assert normal.effective_max_num_seqs == 16
    finally:
        _close(coordinator, first, second)


def test_budget_ack_is_fenced_and_latency_is_recorded() -> None:
    coordinator = _coordinator()
    client = _client(coordinator, "ack", 0)
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    try:
        client.send_memory_report(_report(0, hbm=0.96, kv=0.2), config)
        decision = _wait_for(client, lambda d: d.safety_state == "critical")
        client.send_budget_applied(
            decision_generation=decision.decision_generation,
            applied_safety_cap=0,
            effective_cap=0,
            occupied_slots=3,
            applied_monotonic_s=time.monotonic(),
        )
        deadline = time.monotonic() + 2
        while client._input_addr not in coordinator._last_applied_decisions and time.monotonic() < deadline:
            time.sleep(0.01)
        ack = coordinator._last_applied_decisions[client._input_addr]
        assert ack.occupied_slots == 3
        assert coordinator._decision_apply_latency_ms[client._input_addr] >= 0

        client.send_budget_applied(
            decision_generation=decision.decision_generation,
            applied_safety_cap=8,
            effective_cap=8,
            occupied_slots=0,
            applied_monotonic_s=time.monotonic(),
        )
        time.sleep(0.05)
        assert coordinator._last_applied_decisions[client._input_addr].effective_cap == 0
    finally:
        _close(coordinator, client)


def test_unsolicited_budget_ack_is_rejected() -> None:
    coordinator = _coordinator()
    client = _client(coordinator, "unsolicited-ack", 0)
    try:
        client.send_budget_applied(
            decision_generation=999,
            applied_safety_cap=16,
            effective_cap=16,
            occupied_slots=0,
            applied_monotonic_s=time.monotonic(),
        )
        time.sleep(0.05)
        assert client._input_addr not in coordinator._last_applied_decisions
    finally:
        _close(coordinator, client)


def test_stale_report_never_expands_cap() -> None:
    coordinator = _coordinator()
    client = _client(coordinator, "stale", 0)
    config = {
        "enabled": True,
        "sample_interval_ms": 1,
        "report_timeout_ms": 2,
        "missing_report_grace_samples": 0,
    }
    try:
        client.send_memory_report(_report(0, hbm=0.92, kv=0.2), config)
        high = _wait_for(client, lambda d: d.effective_max_num_seqs == 8)
        stale = _wait_for(client, lambda d: d.safety_state == "stale" and d.decision_generation > high.decision_generation)
        assert stale.effective_max_num_seqs <= high.effective_max_num_seqs
        assert stale.pressure_source == "telemetry_health"
    finally:
        _close(coordinator, client)


def test_new_instance_clears_old_ack_and_memory_state() -> None:
    coordinator = _coordinator()
    client = _client(coordinator, "restart", 0)
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    try:
        client.send_memory_report(_report(0, hbm=0.96, kv=0.2), config)
        decision = _wait_for(client, lambda d: d.safety_state == "critical")
        client.send_budget_applied(
            decision_generation=decision.decision_generation,
            applied_safety_cap=0,
            effective_cap=0,
            occupied_slots=0,
            applied_monotonic_s=time.monotonic(),
        )
        deadline = time.monotonic() + 2
        while client._input_addr not in coordinator._last_applied_decisions and time.monotonic() < deadline:
            time.sleep(0.01)
        old_addr = client._input_addr
        client.close()

        replacement = OmniCoordClientForStage(
            coordinator.router_zmq_addr,
            old_addr,
            "tcp://stage:restart-out-2",
            0,
            replica_id=0,
        )
        try:
            replacement.send_memory_report(_report(0, hbm=0.5, kv=0.2), config)
            _wait_for(replacement, lambda d: d.pressure < 0.9)
            assert coordinator._memory_instances[old_addr] == replacement._instance_id
            assert old_addr not in coordinator._last_applied_decisions
        finally:
            replacement.close()
    finally:
        if not coordinator._closed:
            coordinator.close()
