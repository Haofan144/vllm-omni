"""Shared harness for the H1-H7 HBM dynamic-monitoring hypothesis experiments.

Every ``H*.py`` script imports from here.  The harness does two jobs:

1.  Install a ``sys.modules`` shim for one vLLM API that drifted between
    vllm 0.27 (what vllm-omni expects) and the installed vllm 0.28
    (``vllm.entrypoints.serve.utils.error_response`` was moved to
    ``vllm.entrypoints.serve.exception_handling.error_response``).  Without
    this shim ``import vllm_omni`` fails outright and none of the real
    coordinator classes can be exercised.  The shim only re-exports the moved
    module; it changes no behaviour of the code under test.

2.  Provide small helpers to build ``ReplicaMemoryReport`` objects, drive the
    real ``BudgetAllocator`` state machine, drive the real
    ``OmniSchedulerMixin`` admission budget, and write JSON/Markdown result
    files next to the scripts.

The classes actually under test are the production classes:
``vllm_omni.core.memory_coordinator.BudgetAllocator`` / ``ReplicaMemoryAggregator``
/ ``RankMemoryReporter`` and ``vllm_omni.core.sched.omni_scheduler_mixin.OmniSchedulerMixin``.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure the repo root (which contains the ``vllm_omni`` package) is importable
# regardless of the caller's working directory, and is not shadowed by this
# script's own directory being first on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SELF_DIR = str(Path(__file__).resolve().parent)
if _SELF_DIR in sys.path:
    sys.path.remove(_SELF_DIR)

# --------------------------------------------------------------------------- #
# 1. vLLM drift shim - must run before importing vllm_omni.
# --------------------------------------------------------------------------- #
_MISSING = "vllm.entrypoints.serve.utils.error_response"
_MOVED_TO = "vllm.entrypoints.serve.exception_handling.error_response"
if _MISSING not in sys.modules:
    try:
        sys.modules[_MISSING] = importlib.import_module(_MOVED_TO)
    except Exception:  # pragma: no cover - environment dependent
        # Last resort: a stub with the one symbol vllm-omni imports.
        stub = SimpleNamespace(create_error_response=lambda *a, **k: None)
        sys.modules[_MISSING] = stub  # type: ignore[assignment]

# Now the real imports.
from vllm_omni.core.memory_coordinator import (  # noqa: E402
    BudgetAllocator,
    DynamicHBMConfig,
    RankMemoryReport,
    RankMemoryReporter,
    ReplicaMemoryAggregator,
    ReplicaMemoryReport,
    SafetyState,
)

__all__ = [
    "BudgetAllocator",
    "DynamicHBMConfig",
    "FakeBlockPool",
    "FakeKVCacheManager",
    "FakeWaitingRequest",
    "RankMemoryReport",
    "RankMemoryReporter",
    "ReplicaMemoryAggregator",
    "ReplicaMemoryReport",
    "SafetyState",
    "RESULTS_DIR",
    "make_replica_report",
    "run_allocator_trace",
    "SchedulerHarness",
    "write_result",
    "approx",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2a. Report builders.
# --------------------------------------------------------------------------- #
_TOTAL = 48 * 1024**3  # pretend-A6000 total bytes for synthetic reports


def make_rank_report(
    *,
    hbm_pressure: float,
    rank: int = 0,
    stage_id: int = 0,
    replica_id: int = 0,
    device_uuid: str = "GPU-AAAA",
    node_id: str = "node-0",
    total_bytes: int = _TOTAL,
    ts: float = 1.0,
    process_reserved_bytes: int | None = None,
    baseline_process_reserved_bytes: int | None = None,
) -> RankMemoryReport:
    free = round(total_bytes * (1.0 - hbm_pressure))
    reserved = process_reserved_bytes if process_reserved_bytes is not None else (total_bytes - free)
    return RankMemoryReport(
        stage_id=stage_id,
        replica_id=replica_id,
        rank=rank,
        device_id=rank,
        timestamp_monotonic_s=ts,
        device_total_bytes=total_bytes,
        device_free_bytes=free,
        process_allocated_bytes=int(reserved * 0.9),
        process_reserved_bytes=reserved,
        node_id=node_id,
        device_uuid=device_uuid,
        baseline_process_reserved_bytes=baseline_process_reserved_bytes,
        sample_started_monotonic_s=ts,
        sample_finished_monotonic_s=ts,
    )


def make_replica_report(
    *,
    hbm_pressure: float = 0.0,
    kv_pressure: float = 0.0,
    cap: int = 16,
    complete: bool = True,
    expected_rank_count: int = 1,
    rank_reports: tuple[RankMemoryReport, ...] | None = None,
    running: int = 0,
    waiting: int = 0,
    report_generation: int = 0,
    stage_id: int = 0,
    replica_id: int = 0,
    device_uuid: str = "GPU-AAAA",
    kv_total_blocks: int = 10_000,
) -> ReplicaMemoryReport:
    if rank_reports is None:
        rank_reports = (
            make_rank_report(
                hbm_pressure=hbm_pressure,
                stage_id=stage_id,
                replica_id=replica_id,
                device_uuid=device_uuid,
            ),
        )
    return ReplicaMemoryReport(
        stage_id=stage_id,
        replica_id=replica_id,
        timestamp_monotonic_s=rank_reports[0].timestamp_monotonic_s if rank_reports else 1.0,
        rank_reports=rank_reports,
        expected_rank_count=expected_rank_count,
        kv_total_blocks=kv_total_blocks,
        kv_free_blocks=round(kv_total_blocks * (1.0 - kv_pressure)),
        running_requests=running,
        waiting_requests=waiting,
        configured_max_num_seqs=cap,
        report_generation=report_generation,
    )


# --------------------------------------------------------------------------- #
# 2b. Allocator trace driver.
# --------------------------------------------------------------------------- #
@dataclass
class TraceStep:
    label: str
    pressure: float
    cap_out: int
    state: str
    reason: str
    pressure_source: str
    generation: int


def run_allocator_trace(
    config: DynamicHBMConfig,
    initial_cap: int,
    steps: list[dict[str, Any]],
) -> list[TraceStep]:
    """Feed a sequence of synthetic reports through the real BudgetAllocator.

    Each step dict: {label, hbm, kv?, cap?, complete?, expected_rank_count?}.
    """
    allocator = BudgetAllocator(config, initial_cap)
    out: list[TraceStep] = []
    for i, s in enumerate(steps):
        report = make_replica_report(
            hbm_pressure=s.get("hbm", 0.0),
            kv_pressure=s.get("kv", 0.0),
            cap=s.get("cap", initial_cap),
            complete=s.get("complete", True),
            expected_rank_count=s.get("expected_rank_count", 1),
            rank_reports=s.get("rank_reports"),
            report_generation=i + 1,
        )
        d = allocator.allocate(report, pressure_override=s.get("pressure_override"))
        out.append(
            TraceStep(
                label=s["label"],
                pressure=round(d.pressure, 4),
                cap_out=d.effective_max_num_seqs,
                state=d.safety_state,
                reason=d.reason,
                pressure_source=d.pressure_source,
                generation=d.generation,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 2c. Scheduler-mixin harness (real OmniSchedulerMixin admission budget).
# --------------------------------------------------------------------------- #
class FakeWaitingRequest:
    """Minimal stand-in for ``vllm.v1.request.Request`` fields the AR
    resource estimator reads (see ``ARResourceEstimator.estimate``)."""

    def __init__(
        self,
        *,
        num_prompt_tokens: int,
        max_tokens: int,
        request_id: str = "fake",
        num_output_tokens: int = 0,
    ) -> None:
        self.num_prompt_tokens = num_prompt_tokens
        self.max_tokens = max_tokens
        self.request_id = request_id
        self.num_output_tokens = num_output_tokens
        self.prefill_stats = None


class FakeBlockPool:
    """Minimal stand-in for ``KVCacheManager.block_pool``."""

    def __init__(self, free_blocks: int) -> None:
        self._free = free_blocks

    def get_num_free_blocks(self) -> int:
        return self._free

    def set_free_blocks(self, n: int) -> None:
        self._free = n


class _FakeAllocatedBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self._block_ids = block_ids

    def get_block_ids(self) -> list[list[int]]:
        return [self._block_ids]


class FakeKVCacheManager:
    """Stand-in for ``KVCacheManager`` exposing ``get_blocks`` so
    ``_get_request_allocated_kv_blocks`` can report real observed ground
    truth instead of silently defaulting to 0 (its behavior when
    ``get_blocks`` is absent, as with the plain ``SimpleNamespace`` other
    harness scenarios use)."""

    def __init__(self, block_pool: FakeBlockPool) -> None:
        self.block_pool = block_pool
        self._allocated: dict[str, int] = {}

    def set_allocated_blocks(self, request_id: str, num_blocks: int) -> None:
        self._allocated[request_id] = num_blocks

    def get_blocks(self, request_id: str) -> _FakeAllocatedBlocks:
        return _FakeAllocatedBlocks(list(range(self._allocated.get(request_id, 0))))


class SchedulerHarness:
    """Thin real-mixin wrapper mirroring tests/core/sched/test_dynamic_hbm_admission_gate.py."""

    def __init__(
        self,
        *,
        cap: int = 16,
        tokens: int = 4096,
        running: int = 0,
        config: dict | None = None,
        block_size: int = 16,
        free_kv_blocks: int | None = None,
        track_allocated_blocks: bool = False,
    ):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

        class _S(OmniSchedulerMixin):
            def __init__(self):
                self.max_num_running_reqs = cap
                self.max_num_scheduled_tokens = tokens
                self.running = [object()] * running
                self.policy = SchedulingPolicy.FCFS
                self.waiting = []
                self.num_waiting_for_streaming_input = 0
                from vllm_omni.config.stage_config import StageExecutionType

                self.vllm_config = SimpleNamespace(
                    model_config=SimpleNamespace(
                        stage_id=0,
                        dynamic_hbm=config or {"enabled": True},
                        async_chunk=False,
                        stage_pipeline_config=SimpleNamespace(
                            execution_type=StageExecutionType.LLM_AR
                        ),
                    )
                )
                self.cache_config = SimpleNamespace(block_size=block_size)
                self.block_pool = FakeBlockPool(
                    free_kv_blocks if free_kv_blocks is not None else 1 << 30
                )
                self.kv_cache_manager = (
                    FakeKVCacheManager(self.block_pool)
                    if track_allocated_blocks
                    else SimpleNamespace(block_pool=self.block_pool)
                )
                self._init_dynamic_hbm_scheduling_state()

        self._s = _S()

    # ---- passthroughs -------------------------------------------------- #
    @property
    def configured_cap(self) -> int:
        return self._s._configured_max_num_seqs

    @property
    def safety_cap(self) -> int:
        return self._s._safety_max_num_seqs

    @property
    def effective_cap(self) -> int:
        return self._s._effective_max_num_seqs

    @property
    def effective_tokens(self) -> int:
        return self._s._effective_max_num_scheduled_tokens

    @property
    def state(self) -> SafetyState:
        return self._s._dynamic_hbm_safety_state

    def allows_new_admission(self) -> bool:
        return self._s._dynamic_hbm_allows_new_admission()

    def next_waiting_request_fits(self) -> bool:
        return self._s._dynamic_hbm_next_waiting_request_fits()

    def resource_admission_decision(self):
        return self._s._dynamic_hbm_resource_admission_decision()

    def bounded_bypass_waiting(self) -> list[FakeWaitingRequest]:
        """Run the bounded-bypass queue filter and return the resulting
        candidate order as a plain list (the requests ``super().schedule()``
        would get a chance to admit this step)."""
        return list(self._s._dynamic_hbm_bounded_bypass_waiting())

    @property
    def bypass_count(self) -> int:
        return self._s._resource_admission_bypass_count

    @property
    def bypassed_requests(self) -> int:
        return self._s._resource_admission_bypassed_requests

    @property
    def aging_stops(self) -> int:
        return self._s._resource_admission_aging_stops

    @property
    def head_of_line_since(self) -> dict[str, float]:
        return self._s._resource_admission_head_of_line_since

    def finish_resource_observation(self, request: FakeWaitingRequest) -> None:
        self._s._finish_resource_observation(request)

    @property
    def calibrator(self):
        return self._s._resource_calibrator

    def max_running(self) -> int:
        return self._s._dynamic_max_num_running_reqs()

    def set_running(self, n: int) -> None:
        self._s.running = [object()] * n
        self._s._recompute_effective_dynamic_hbm_budget()

    def set_waiting(self, requests: list[FakeWaitingRequest]) -> None:
        self._s.waiting = requests

    def set_free_kv_blocks(self, n: int) -> None:
        self._s.block_pool.set_free_blocks(n)

    def set_allocated_blocks(self, request_id: str, num_blocks: int) -> None:
        self._s.kv_cache_manager.set_allocated_blocks(request_id, num_blocks)

    def apply(self, *, generation: int, cap: int, state: SafetyState, pressure: float = 0.92, report: int = 1) -> bool:
        return self._s.apply_stage_budget_decision(
            generation=generation,
            based_on_report_generation=report,
            effective_max_num_seqs=cap,
            pressure=pressure,
            reason=state.value,
            safety_state=state.value,
            pressure_source="physical_hbm",
        )

    def disconnect_guard(self) -> bool:
        return self._s.apply_dynamic_hbm_disconnect_guard()

    def local_kv_guard(self) -> bool:
        return self._s.apply_dynamic_hbm_local_kv_guard()


# --------------------------------------------------------------------------- #
# 2c-bis. OmniCoordinator harness with a fake ZMQ router.
# --------------------------------------------------------------------------- #
class _FakeRouter:
    """Captures send_multipart frames instead of touching a real socket."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, dict]] = []

    def send_multipart(self, frames, flags=0):  # noqa: ANN001
        route, payload = frames
        self.sent.append((route, json.loads(payload.decode("utf-8"))))


class CoordinatorHarness:
    """Drive the REAL OmniCoordinator shared-device logic without ZMQ / threads.

    Exercises ``_handle_memory_report_locked`` and
    ``_check_memory_report_timeouts_locked`` directly, exactly as the recv /
    periodic loops would, and returns the ``budget_decision`` wire dicts the
    coordinator tried to send.
    """

    def __init__(self) -> None:
        import threading as _t

        from vllm_omni.distributed.omni_coordinator.omni_coordinator import OmniCoordinator

        self._c = OmniCoordinator.__new__(OmniCoordinator)
        c = self._c
        c._router = _FakeRouter()
        c._lock = _t.Lock()
        c._replicas = {}
        c._stage_routes = {}
        c._memory_reports = {}
        c._memory_instances = {}
        c._memory_report_generations = {}
        c._budget_generations = {}
        c._budget_allocators = {}
        c._last_caps = {}
        c._last_reasons = {}
        c._memory_configs = {}
        c._memory_received_at = {}
        c._last_allocation_at = {}
        c._last_pressures = {}
        c._last_applied_decisions = {}
        c._decision_sent_at = {}
        c._decision_apply_latency_ms = {}
        self._gen: dict[str, int] = {}

    @property
    def sent(self) -> list[dict]:
        return [payload for _route, payload in self._c._router.sent]

    def clear_sent(self) -> None:
        self._c._router.sent.clear()

    def report(
        self,
        *,
        addr: str,
        stage_id: int,
        replica_id: int,
        instance_id: str = "inst-A",
        hbm_pressure: float = 0.0,
        kv_pressure: float = 0.0,
        cap: int = 16,
        device_uuid: str = "GPU-AAAA",
        node_id: str = "node-0",
        expected_rank_count: int = 1,
        rank_reports: tuple[RankMemoryReport, ...] | None = None,
        running: int = 0,
        waiting: int = 0,
        config: DynamicHBMConfig | None = None,
    ) -> list[dict]:
        """Deliver one memory report from `addr` and return decisions emitted this call."""
        cfg = config or DynamicHBMConfig(enabled=True)
        if rank_reports is None:
            rank_reports = (
                make_rank_report(
                    hbm_pressure=hbm_pressure,
                    stage_id=stage_id,
                    replica_id=replica_id,
                    device_uuid=device_uuid,
                    node_id=node_id,
                ),
            )
        rep = make_replica_report(
            cap=cap,
            complete=len(rank_reports) == expected_rank_count,
            expected_rank_count=expected_rank_count,
            rank_reports=rank_reports,
            kv_pressure=kv_pressure,
            running=running,
            waiting=waiting,
            stage_id=stage_id,
            replica_id=replica_id,
            device_uuid=device_uuid,
        )
        gen = self._gen.get(addr, 0) + 1
        self._gen[addr] = gen
        data = {
            "message_type": "memory_report",
            "input_addr": addr,
            "instance_id": instance_id,
            "report_generation": gen,
            "dynamic_hbm": {
                k: getattr(cfg, k)
                for k in cfg.__dataclass_fields__
            },
            "report": {
                "stage_id": rep.stage_id,
                "replica_id": rep.replica_id,
                "timestamp_monotonic_s": rep.timestamp_monotonic_s,
                "rank_reports": tuple(
                    {f: getattr(r, f) for f in r.__dataclass_fields__} for r in rep.rank_reports
                ),
                "expected_rank_count": rep.expected_rank_count,
                "kv_total_blocks": rep.kv_total_blocks,
                "kv_free_blocks": rep.kv_free_blocks,
                "running_requests": rep.running_requests,
                "waiting_requests": rep.waiting_requests,
                "configured_max_num_seqs": rep.configured_max_num_seqs,
                "report_generation": gen,
                "trigger_reason": "periodic",
            },
        }
        before = len(self._c._router.sent)
        self._c._handle_memory_report_locked(data, routing_identity=addr.encode())
        return [p for _r, p in self._c._router.sent[before:]]

    def advance_clock(self, seconds: float) -> None:
        """Rewind all internal 'received/allocated at' stamps so `seconds` appear to pass.

        The coordinator rate-limits allocations to one per ``sample_interval_ms``;
        call this between successive ``report(...)`` calls to let each report be
        acted on, mirroring real wall-clock spacing between samples.
        """
        for d in (
            self._c._memory_received_at,
            self._c._last_allocation_at,
        ):
            for k in list(d):
                d[k] -= seconds

    def advance_time_and_check_timeouts(self, seconds: float) -> list[dict]:
        """Simulate `seconds` elapsing with no new reports, then run the timeout sweep."""
        self.advance_clock(seconds)
        before = len(self._c._router.sent)
        self._c._check_memory_report_timeouts_locked()
        return [p for _r, p in self._c._router.sent[before:]]


# --------------------------------------------------------------------------- #
# 2d. Result IO.
# --------------------------------------------------------------------------- #
def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def write_result(
    hyp_id: str,
    title: str,
    hypothesis: str,
    expectation: str,
    observations: dict[str, Any],
    checks: list[Check],
    analysis: str,
) -> dict[str, Any]:
    all_pass = all(c.passed for c in checks)
    payload = {
        "hypothesis_id": hyp_id,
        "title": title,
        "hypothesis": hypothesis,
        "expectation": expectation,
        "verdict": "MATCHES_EXPECTATION" if all_pass else "DEVIATION",
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "observations": observations,
        "analysis": analysis,
    }
    (RESULTS_DIR / f"{hyp_id}_result.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {hyp_id}: {title}",
        "",
        f"**Hypothesis.** {hypothesis}",
        "",
        f"**Expectation.** {expectation}",
        "",
        f"**Verdict.** {'MATCHES EXPECTATION' if all_pass else 'DEVIATION FROM EXPECTATION'}",
        "",
        "## Checks",
        "",
        "| # | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    lines += ["", "## Observations", "", "```json", json.dumps(observations, indent=2), "```", "", "## Analysis", "", analysis, ""]
    (RESULTS_DIR / f"{hyp_id}_result.md").write_text("\n".join(lines))
    return payload
