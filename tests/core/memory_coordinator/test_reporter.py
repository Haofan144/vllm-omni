# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import RankMemoryReporter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_rank_memory_reporter_collects_worker_device_snapshot() -> None:
    reporter = RankMemoryReporter(
        stage_id=2,
        replica_id=4,
        rank=1,
        device_id=7,
        device_memory_provider=lambda: (200, 1000),
        process_memory_provider=lambda: (300, 400),
        clock=lambda: 123.5,
    )

    report = reporter.report()

    assert report.stage_id == 2
    assert report.replica_id == 4
    assert report.rank == 1
    assert report.device_id == 7
    assert report.timestamp_monotonic_s == 123.5
    assert report.process_allocated_bytes == 300
    assert report.process_reserved_bytes == 400
    assert report.hbm_pressure == pytest.approx(0.8)
