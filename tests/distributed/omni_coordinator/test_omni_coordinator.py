# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from dataclasses import asdict

import pytest
import zmq
from vllm.v1.utils import get_engine_client_zmq_addr

from vllm_omni.core.memory_coordinator import RankMemoryReport, ReplicaMemoryReport
from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForStage,
    OmniCoordinator,
    ReplicaStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _recv_replica_list(sub: zmq.Socket, timeout_ms: int = 2000) -> dict | None:
    """Receive ReplicaList JSON from SUB socket. Returns None on timeout."""
    sub.setsockopt(zmq.RCVTIMEO, timeout_ms)
    try:
        data = sub.recv()
        return json.loads(data.decode("utf-8"))
    except zmq.Again:
        return None


def _wait_for_replica_list(
    sub: zmq.Socket,
    expected_count: int,
    timeout: float = 3.0,
) -> dict | None:
    """Wait until received ReplicaList with expected_count active replicas."""
    start = time.time()
    while time.time() - start < timeout:
        msg = _recv_replica_list(sub, timeout_ms=500)
        if msg is not None and len(msg.get("replicas", [])) == expected_count:
            return msg
    return None


def _drain_sub_messages(sub: zmq.Socket, max_seconds: float = 0.4) -> None:
    """Drain queued SUB messages for a short window."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        _recv_replica_list(sub, timeout_ms=50)


def _memory_report(stage_id: int, replica_id: int, pressure: float) -> dict:
    rank = RankMemoryReport(
        stage_id=stage_id,
        replica_id=replica_id,
        rank=0,
        device_id=0,
        timestamp_monotonic_s=time.monotonic(),
        device_total_bytes=1000,
        device_free_bytes=round(1000 * (1 - pressure)),
        process_allocated_bytes=100,
        process_reserved_bytes=120,
        node_id="shared-node",
        device_uuid="GPU-shared",
    )
    return asdict(
        ReplicaMemoryReport(
            stage_id=stage_id,
            replica_id=replica_id,
            timestamp_monotonic_s=rank.timestamp_monotonic_s,
            rank_reports=(rank,),
            expected_rank_count=1,
            kv_total_blocks=100,
            kv_free_blocks=50,
            running_requests=0,
            waiting_requests=4,
            configured_max_num_seqs=16,
        )
    )


def test_central_hbm_coordinator_updates_all_shared_gpu_consumers():
    router_addr = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    pub_addr = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    coordinator = OmniCoordinator(router_addr, pub_addr, heartbeat_timeout=1000.0)
    first = OmniCoordClientForStage(
        coordinator.router_zmq_addr,
        "tcp://stage:hbm-0",
        "tcp://stage:hbm-0-out",
        0,
        replica_id=0,
    )
    second = OmniCoordClientForStage(
        coordinator.router_zmq_addr,
        "tcp://stage:hbm-1",
        "tcp://stage:hbm-1-out",
        1,
        replica_id=0,
    )
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    first.send_memory_report(_memory_report(0, 0, 0.5), config)
    second.send_memory_report(_memory_report(1, 0, 0.92), config)

    deadline = time.time() + 2
    first_decisions = []
    second_decisions = []
    while time.time() < deadline:
        first_decisions.extend(first.poll_budget_decisions())
        second_decisions.extend(second.poll_budget_decisions())
        if first_decisions and second_decisions and first_decisions[-1].effective_max_num_seqs == 8:
            break
        time.sleep(0.01)

    assert first_decisions[-1].effective_max_num_seqs == 8
    assert second_decisions[-1].effective_max_num_seqs == 8
    assert first_decisions[-1].reason.startswith("shared_device_")

    first.close()
    second.close()
    coordinator.close()


def test_omni_coordinator_wait_for_shutdown_unblocks_on_close():
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    waiter = threading.Thread(target=coordinator.wait_for_shutdown)
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()

    coordinator.close()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()


def test_omni_coordinator_pub_coalescing_on_rapid_queue_updates():
    """Rapid updates should be coalesced into fewer PUB messages."""
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(coordinator.pub_zmq_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    time.sleep(0.3)  # PUB/SUB slow-joiner

    client = OmniCoordClientForStage(
        coordinator.router_zmq_addr,
        "tcp://stage:coalesce",
        "tcp://stage:coalesce-out",
        0,
    )

    # Wait for initial registration broadcast and clear any queued messages.
    msg = _wait_for_replica_list(sub, expected_count=1)
    assert msg is not None
    _drain_sub_messages(sub)

    # Burst many queue updates in a short period.
    update_count = 80
    for i in range(update_count):
        client.update_info(queue_length=i)

    # With publish_min_interval=0.1s, received messages over ~1s should be
    # much smaller than update_count (coalescing effect).
    window_s = 1.1
    deadline = time.time() + window_s
    recv_count = 0
    while time.time() < deadline:
        if _recv_replica_list(sub, timeout_ms=100) is not None:
            recv_count += 1

    assert recv_count < update_count // 2, (
        f"expected coalesced PUB traffic, got {recv_count} for {update_count} updates"
    )

    client.close()
    coordinator.close()
    sub.close(0)
    sub_ctx.term()


def test_omni_coordinator_registration_broadcast():
    """Verify that after multiple OmniCoordClientForStage replicas register,
    OmniCoordinator publishes a ReplicaList containing all registered replicas.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(coordinator.pub_zmq_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    # ZMQ PUB/SUB slow-joiner: allow SUB to connect before clients register.
    time.sleep(0.3)

    # Create 3 stage clients; each auto-registers on init.
    clients = [
        OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:10001", "tcp://stage:10001-out", 0),
        OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:10002", "tcp://stage:10002-out", 0),
        OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:10003", "tcp://stage:10003-out", 1),
    ]

    msg = _wait_for_replica_list(sub, expected_count=3)
    assert msg is not None, "Expected ReplicaList with 3 replicas"
    assert len(msg["replicas"]) == 3
    assert isinstance(msg["timestamp"], (int, float))

    input_addrs = {rep["input_addr"] for rep in msg["replicas"]}
    assert "tcp://stage:10001" in input_addrs
    assert "tcp://stage:10002" in input_addrs
    assert "tcp://stage:10003" in input_addrs

    for c in clients:
        c.close()
    coordinator.close()
    sub.close(0)
    sub_ctx.term()


def test_omni_coordinator_heartbeat_timeout_handling():
    """Verify that when a stage replica stops sending heartbeats,
    OmniCoordinator marks it as unhealthy and excludes it from the active list.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=5.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(coordinator.pub_zmq_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    time.sleep(0.3)

    # A and B: real clients that send heartbeats every 5s.
    client_a = OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:a", "tcp://stage:a-out", 0)
    client_b = OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:b", "tcp://stage:b-out", 0)

    # C: raw DEALER that sends only registration, no heartbeat.
    dealer_ctx = zmq.Context.instance()
    dealer_c = dealer_ctx.socket(zmq.DEALER)
    dealer_c.connect(coordinator.router_zmq_addr)
    reg_event = {
        "input_addr": "tcp://stage:c",
        "output_addr": "tcp://stage:c-out",
        "stage_id": 0,
        "event_type": "update",
        "status": ReplicaStatus.UP.value,
        "queue_length": 0,
    }
    dealer_c.send(json.dumps(reg_event).encode("utf-8"))

    msg = _wait_for_replica_list(sub, expected_count=3)
    assert msg is not None, "Expected initial 3 replicas"
    assert len(msg["replicas"]) == 3

    # Wait for heartbeat timeout (timeout=5s, check interval ~2.5s).
    time.sleep(8.0)

    # Receive the update (C should be ERROR and excluded from active list).
    msg_after_timeout = _wait_for_replica_list(sub, expected_count=2, timeout=5.0)
    assert msg_after_timeout is not None, "Expected ReplicaList with 2 replicas after timeout"
    replicas = msg_after_timeout.get("replicas", [])
    input_addrs = {rep["input_addr"] for rep in replicas}

    assert "tcp://stage:a" in input_addrs
    assert "tcp://stage:b" in input_addrs
    assert "tcp://stage:c" not in input_addrs

    client_a.close()
    client_b.close()
    dealer_c.close(0)
    coordinator.close()
    sub.close(0)
    dealer_ctx.term()
    sub_ctx.term()


def test_omni_coordinator_replica_shutdown_handling():
    """Verify that when a stage replica sends status='down',
    OmniCoordinator removes it from the active list and broadcasts an updated list.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(coordinator.pub_zmq_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    time.sleep(0.3)  # PUB/SUB slow-joiner

    client = OmniCoordClientForStage(coordinator.router_zmq_addr, "tcp://stage:shutdown", "tcp://stage:shutdown-out", 0)

    msg = _wait_for_replica_list(sub, expected_count=1)
    assert msg is not None
    assert len(msg["replicas"]) == 1
    assert msg["replicas"][0]["input_addr"] == "tcp://stage:shutdown"

    # Send down status (simulating graceful shutdown).
    client.update_info(status=ReplicaStatus.DOWN)

    # Receive updated list (should have 0 active replicas).
    msg = _wait_for_replica_list(sub, expected_count=0)
    assert msg is not None
    assert len(msg["replicas"]) == 0

    client.close()
    coordinator.close()
    sub.close(0)
    sub_ctx.term()
