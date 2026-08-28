# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import asdict

import pytest
from vllm.v1.utils import get_engine_client_zmq_addr

from vllm_omni.core.memory_coordinator import RankMemoryReport, ReplicaMemoryReport
from vllm_omni.distributed.omni_coordinator import OmniCoordClientForStage, OmniCoordinator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _report(stage: int, pressure: float) -> dict:
    rank = RankMemoryReport(
        stage_id=stage,
        replica_id=0,
        rank=0,
        device_id=0,
        timestamp_monotonic_s=time.monotonic(),
        device_total_bytes=1000,
        device_free_bytes=round(1000 * (1 - pressure)),
        process_allocated_bytes=100,
        process_reserved_bytes=120,
        node_id="concurrency-node",
        device_uuid="GPU-shared",
    )
    return asdict(
        ReplicaMemoryReport(
            stage_id=stage,
            replica_id=0,
            timestamp_monotonic_s=rank.timestamp_monotonic_s,
            rank_reports=(rank,),
            expected_rank_count=1,
            kv_total_blocks=100,
            kv_free_blocks=50,
            running_requests=0,
            waiting_requests=1,
            configured_max_num_seqs=16,
        )
    )


def test_concurrent_reports_from_shared_gpu_consumers_keep_coordinator_live() -> None:
    router = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    pub = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
    coordinator = OmniCoordinator(router, pub, heartbeat_timeout=1000.0)
    clients = [
        OmniCoordClientForStage(
            coordinator.router_zmq_addr,
            f"tcp://stage:concurrent-{stage}",
            f"tcp://stage:concurrent-{stage}-out",
            stage,
            replica_id=0,
        )
        for stage in range(4)
    ]
    config = {"enabled": True, "sample_interval_ms": 1, "report_timeout_ms": 1000}
    errors = []

    def publish(client, stage):
        try:
            for index in range(20):
                pressure = 0.92 if index % 2 else 0.5
                client.send_memory_report(_report(stage, pressure), config)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(client, stage)) for stage, client in enumerate(clients)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert not errors

        deadline = time.monotonic() + 3
        while len(coordinator._memory_reports) < len(clients) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(coordinator._memory_reports) == len(clients)
        assert coordinator._recv_thread.is_alive()
        assert coordinator._periodic_thread.is_alive()
    finally:
        for client in clients:
            client.close()
        coordinator.close()
