# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import logging
import threading
from dataclasses import asdict
from time import time
from typing import Any

import zmq

from vllm_omni.core.memory_coordinator import (
    BudgetAllocator,
    DynamicHBMConfig,
    RankMemoryReport,
    ReplicaMemoryReport,
)

from .messages import BudgetAppliedEvent, ReplicaEvent, ReplicaInfo, ReplicaList, ReplicaStatus

logger = logging.getLogger(__name__)


class OmniCoordinator:
    """Coordinator for stage replicas and hub clients.

    This service receives replica events from :class:`OmniCoordClientForStage`
    via a ZMQ ROUTER socket and publishes active replica lists to
    :class:`OmniCoordClientForHub` via a PUB socket.

    The coordinator maintains an in-memory registry of all known replicas,
    including their status, queue length, and heartbeat timestamps. A
    background thread periodically checks for heartbeat timeouts and marks
    unhealthy replicas as ``ReplicaStatus.ERROR``.
    """

    def __init__(
        self,
        router_zmq_addr: str,
        pub_zmq_addr: str,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        """Initialize coordinator and start background service loops.

        Args:
            router_zmq_addr: ZMQ address to bind the ROUTER socket.
            pub_zmq_addr: ZMQ address to bind the PUB socket.
            heartbeat_timeout: Seconds before a replica is considered
                unhealthy if no heartbeat / update is received.
        """
        self._heartbeat_timeout = heartbeat_timeout

        # Dedicated ZMQ context for this coordinator instance.
        self._ctx = zmq.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(router_zmq_addr)
        # Recover the actual bound address (port=0 → real port).
        self.router_zmq_addr = self._router.getsockopt_string(zmq.LAST_ENDPOINT)
        self._pub_zmq_addr = pub_zmq_addr  # keep original for internal use

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(pub_zmq_addr)
        # Recover the actual bound address (port=0 → real port).
        self.pub_zmq_addr = self._pub.getsockopt_string(zmq.LAST_ENDPOINT)

        self._replicas: dict[str, ReplicaInfo] = {}
        self._stage_routes: dict[str, bytes] = {}
        self._memory_reports: dict[str, ReplicaMemoryReport] = {}
        self._memory_instances: dict[str, str] = {}
        self._memory_report_generations: dict[str, int] = {}
        self._budget_generations: dict[str, int] = {}
        self._budget_allocators: dict[str, BudgetAllocator] = {}
        self._last_caps: dict[str, int] = {}
        self._last_reasons: dict[str, str] = {}
        self._memory_configs: dict[str, DynamicHBMConfig] = {}
        self._memory_received_at: dict[str, float] = {}
        self._last_allocation_at: dict[str, float] = {}
        self._last_pressures: dict[str, float] = {}
        self._last_applied_decisions: dict[str, BudgetAppliedEvent] = {}
        self._decision_sent_at: dict[tuple[str, int], float] = {}
        self._decision_apply_latency_ms: dict[str, float] = {}
        self._lock = threading.Lock()
        self._pub_lock = threading.Lock()

        self._publish_min_interval: float = 0.1  # seconds
        self._pending_broadcast: bool = False
        self._pending_lock = threading.Lock()

        self._running = True
        self._closed = False
        self._stop_event = threading.Event()

        self._router.setsockopt(zmq.RCVTIMEO, 100)

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._periodic_thread = threading.Thread(target=self._periodic_loop, daemon=True)
        self._periodic_thread.start()

    def get_active_replicas(self) -> ReplicaList:
        """Return a :class:`ReplicaList` of active (UP) replicas only."""
        with self._lock:
            active = [rep for rep in self._replicas.values() if rep.status == ReplicaStatus.UP]
        return ReplicaList(replicas=active, timestamp=time())

    def add_new_replica(self, event: ReplicaEvent) -> None:
        """Add a new replica based on an incoming event."""
        with self._lock:
            self._add_new_replica_locked(event)
        self._schedule_broadcast()

    def update_replica_info(self, event: ReplicaEvent) -> None:
        """Update an existing replica based on an incoming event."""
        with self._lock:
            self._update_replica_info_locked(event)
        self._schedule_broadcast()

    def remove_replica(self, event: ReplicaEvent) -> None:
        """Mark a replica as removed / down based on an incoming event.

        This marks the replica's status as DOWN or ERROR (depending on the
        event) but keeps it in the internal registry. It is removed from the
        *active* replica list published to hubs.
        """
        with self._lock:
            self._remove_replica_locked(event)
        self._schedule_broadcast()

    def publish_replica_list_update(self) -> bool:
        """Publish the current active replica list to all subscribers.

        Returns:
            True if the PUB send succeeded, False if it was dropped (e.g.
            socket not ready when using ``zmq.NOBLOCK``).
        """
        active_list = self.get_active_replicas()
        payload = asdict(active_list)
        data = json.dumps(payload).encode("utf-8")

        with self._pub_lock:
            try:
                # PUB socket is best-effort; drop update if not ready.
                self._pub.send(data, flags=zmq.NOBLOCK)
                return True
            except (zmq.Again, zmq.ZMQError):
                # Silently ignore send failures; next update will catch up.
                return False

    def _schedule_broadcast(self) -> None:
        """Request a broadcast to be flushed by the periodic loop.

        All broadcast requests are coalesced via ``_pending_broadcast`` and
        flushed at most once per ``_publish_min_interval``.
        """
        with self._pending_lock:
            self._pending_broadcast = True

    def _mark_replica_error_locked(self, info: ReplicaInfo) -> None:
        """Mark replica as ERROR (e.g. after heartbeat timeout)."""
        info.status = ReplicaStatus.ERROR

    def _clear_memory_state(self, input_addr: str) -> None:
        self._stage_routes.pop(input_addr, None)
        self._memory_reports.pop(input_addr, None)
        self._memory_instances.pop(input_addr, None)
        self._memory_report_generations.pop(input_addr, None)
        self._budget_generations.pop(input_addr, None)
        self._budget_allocators.pop(input_addr, None)
        self._last_caps.pop(input_addr, None)
        self._last_reasons.pop(input_addr, None)
        self._memory_configs.pop(input_addr, None)
        self._memory_received_at.pop(input_addr, None)
        self._last_allocation_at.pop(input_addr, None)
        self._last_pressures.pop(input_addr, None)
        self._last_applied_decisions.pop(input_addr, None)
        self._decision_apply_latency_ms.pop(input_addr, None)
        for key in [key for key in self._decision_sent_at if key[0] == input_addr]:
            self._decision_sent_at.pop(key, None)

    def _check_heartbeat_timeouts(self) -> None:
        """Mark replicas as ERROR if their heartbeat has timed out."""
        now = time()
        timed_out = False
        gc_ttl = 600.0  # 10 minutes

        with self._lock:
            to_delete: list[str] = []

            for input_addr, info in self._replicas.items():
                if info.status == ReplicaStatus.UP and now - info.last_heartbeat > self._heartbeat_timeout:
                    self._mark_replica_error_locked(info)
                    self._clear_memory_state(input_addr)
                    timed_out = True
                elif info.status in (ReplicaStatus.DOWN, ReplicaStatus.ERROR) and now - info.last_heartbeat > gc_ttl:
                    to_delete.append(input_addr)

            for input_addr in to_delete:
                del self._replicas[input_addr]
        if timed_out:
            # Replica liveness changed; request broadcast.
            self._schedule_broadcast()

    def close(self) -> None:
        """Shut down background threads and close all ZMQ sockets."""
        if self._closed:
            raise RuntimeError("Coordinator already closed")

        self._closed = True
        self._running = False
        self._stop_event.set()

        # Wait for threads to exit before closing sockets.
        for thread in (self._recv_thread, self._periodic_thread):
            thread.join(timeout=1.0)

        try:
            self._router.close(0)
        except zmq.ZMQError:
            pass

        try:
            self._pub.close(0)
        except zmq.ZMQError:
            pass

        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass

    def wait_for_shutdown(self) -> None:
        """Block until the coordinator is asked to stop.

        ``OmniCoordinatorRuntime`` runs the coordinator in a child process.
        The child needs a small, explicit blocking lifecycle API so it does
        not return immediately after construction.
        """
        self._stop_event.wait()

    def _parse_replica_event(self, data: dict[str, Any]) -> ReplicaEvent | None:
        """Parse wire payload dict into ReplicaEvent. Returns None if invalid."""
        try:
            return ReplicaEvent(
                input_addr=str(data["input_addr"]),
                output_addr=str(data["output_addr"]),
                stage_id=int(data["stage_id"]),
                event_type=str(data["event_type"]),
                status=ReplicaStatus(data.get("status")),
                queue_length=data.get("queue_length"),
                replica_id=int(data.get("replica_id", 0)),
                instance_id=str(data.get("instance_id", "")),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _recv_loop(self) -> None:
        """Background loop that receives and processes replica events."""
        while self._running:
            try:
                frames = self._router.recv_multipart()
            except zmq.Again:
                self._check_memory_report_timeouts()
                # RCVTIMEO expired, loop to recheck _running.
                continue
            except zmq.ZMQError:
                # Socket likely closed or context terminated.
                break

            if not frames:
                continue

            routing_identity = frames[0]
            payload = frames[-1]
            try:
                data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in replica event, dropping: %s", e)
                continue
            if data.get("message_type") == "memory_report":
                self._handle_memory_report(data, routing_identity)
                continue
            if data.get("message_type") == "budget_applied":
                self._handle_budget_applied(data)
                continue

            event = self._parse_replica_event(data)
            if event is None:
                logger.warning("Malformed replica event, dropping")
                continue

            self._stage_routes[event.input_addr] = routing_identity
            self._handle_event(event)

    def _handle_budget_applied(self, data: dict[str, Any]) -> None:
        try:
            event = BudgetAppliedEvent(**data)
        except (TypeError, ValueError) as exc:
            logger.warning("Dropping malformed HBM budget ACK: %s", exc)
            return
        with self._lock:
            registered = self._replicas.get(event.input_addr)
            if registered is None or registered.instance_id != event.instance_id:
                return
            previous = self._last_applied_decisions.get(event.input_addr)
            if previous is not None and event.decision_generation <= previous.decision_generation:
                return
            sent_at = self._decision_sent_at.get((event.input_addr, event.decision_generation))
            if sent_at is None:
                # Do not let a fabricated or out-of-window ACK advance the
                # applied generation. Only decisions emitted by this live
                # coordinator instance are acknowledgeable.
                return
            self._decision_apply_latency_ms[event.input_addr] = max(0.0, (time() - sent_at) * 1000)
            self._last_applied_decisions[event.input_addr] = event

    @staticmethod
    def _device_keys(report: ReplicaMemoryReport) -> set[tuple[str, str]]:
        return {(rank.node_id, rank.device_uuid or f"local-device-{rank.device_id}") for rank in report.rank_reports}

    def _handle_memory_report(self, data: dict[str, Any], routing_identity: bytes) -> None:
        """Update the central view and send decisions for all shared-device consumers."""
        with self._lock:
            self._handle_memory_report_locked(data, routing_identity)

    def _handle_memory_report_locked(self, data: dict[str, Any], routing_identity: bytes) -> None:
        try:
            input_addr = str(data["input_addr"])
            instance_id = str(data["instance_id"])
            report_generation = int(data["report_generation"])
            raw_report = dict(data["report"])
            raw_report["rank_reports"] = tuple(RankMemoryReport(**rank) for rank in raw_report.get("rank_reports", ()))
            report = ReplicaMemoryReport(**raw_report)
            config = DynamicHBMConfig.from_value(data.get("dynamic_hbm"))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Dropping malformed memory report: %s", exc)
            return
        if not config.enabled:
            return

        registered = self._replicas.get(input_addr)
        if registered is not None and (
            registered.stage_id != report.stage_id
            or registered.replica_id != report.replica_id
            or (registered.instance_id and registered.instance_id != instance_id)
        ):
            logger.warning("Dropping memory report whose identity does not match registration: %s", input_addr)
            return

        previous_instance = self._memory_instances.get(input_addr)
        if previous_instance != instance_id:
            self._memory_report_generations[input_addr] = 0
            self._budget_generations[input_addr] = 0
            self._budget_allocators.pop(input_addr, None)
        if report_generation <= self._memory_report_generations.get(input_addr, 0):
            return

        incoming_devices = self._device_keys(report)
        if not incoming_devices:
            logger.warning(
                "Dropping memory report without rank device telemetry: stage=%d replica=%d generation=%d",
                report.stage_id,
                report.replica_id,
                report_generation,
            )
            return

        self._stage_routes[input_addr] = routing_identity
        self._memory_instances[input_addr] = instance_id
        self._memory_report_generations[input_addr] = report_generation
        self._memory_reports[input_addr] = report
        self._memory_configs[input_addr] = config
        self._memory_received_at[input_addr] = time()
        self._budget_allocators.setdefault(
            input_addr,
            BudgetAllocator(config, report.configured_max_num_seqs),
        )
        if report_generation == 1:
            logger.info(
                "[HBMCoordinator] registered memory stream stage=%d replica=%d devices=%s",
                report.stage_id,
                report.replica_id,
                sorted(self._device_keys(report)),
            )

        affected = [
            addr for addr, candidate in self._memory_reports.items() if incoming_devices & self._device_keys(candidate)
        ]
        # Device HBM is shared; KV blocks are private to a stage replica and
        # must not throttle unrelated consumers on the same physical GPU.
        # Apply each consumer's safety guard to all reports for its shared
        # devices so a stricter stage policy cannot be bypassed by a peer.
        for addr in affected:
            candidate = self._memory_reports[addr]
            allocator = self._budget_allocators.get(addr)
            route = self._stage_routes.get(addr)
            if allocator is None or route is None:
                continue
            candidate_config = self._memory_configs[addr]
            shared_hbm_pressure = max(
                self._memory_reports[shared_addr].hbm_pressure_with_guard(
                    guard_bytes=candidate_config.guard_bytes,
                    guard_ratio=candidate_config.guard_ratio,
                )
                for shared_addr in affected
            )
            effective_pressure = max(shared_hbm_pressure, candidate.kv_pressure)
            now = time()
            due = now - self._last_allocation_at.get(addr, 0.0) >= candidate_config.sample_interval_ms / 1000
            pressure_crossing = (
                effective_pressure >= candidate_config.high_watermark
                and self._last_pressures.get(addr, 0.0) < candidate_config.high_watermark
            )
            if not due and not pressure_crossing:
                continue
            decision = allocator.allocate(candidate, pressure_override=effective_pressure)
            self._last_allocation_at[addr] = now
            self._last_pressures[addr] = effective_pressure
            # Every emitted decision gets a generation. Pressure state can
            # change while the cap remains at its minimum; consumers still
            # need that update to leave critical admission mode.
            generation = self._budget_generations.get(addr, 0) + 1
            self._budget_generations[addr] = generation
            reason = decision.reason if len(affected) == 1 else f"shared_device_{decision.reason}"
            wire = {
                "message_type": "budget_decision",
                "stage_id": candidate.stage_id,
                "replica_id": candidate.replica_id,
                "instance_id": self._memory_instances[addr],
                "decision_generation": generation,
                "based_on_report_generation": self._memory_report_generations[addr],
                "effective_max_num_seqs": decision.effective_max_num_seqs,
                "pressure": effective_pressure,
                "reason": reason,
                "safety_state": decision.safety_state,
                "pressure_source": (
                    "shared_physical_hbm"
                    if shared_hbm_pressure >= candidate.kv_pressure
                    else "kv"
                ),
                "physical_hbm_pressure": shared_hbm_pressure,
                "kv_pressure": candidate.kv_pressure,
                "report_age_ms": max(0.0, (now - self._memory_received_at[addr]) * 1000),
            }
            try:
                self._router.send_multipart([route, json.dumps(wire).encode("utf-8")], flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                logger.warning("Dropping HBM budget decision for %s", addr)
            else:
                self._decision_sent_at[(addr, generation)] = time()
                previous_cap = self._last_caps.get(addr)
                previous_reason = self._last_reasons.get(addr)
                self._last_caps[addr] = decision.effective_max_num_seqs
                self._last_reasons[addr] = reason
                if previous_cap != decision.effective_max_num_seqs or previous_reason != reason:
                    logger.info(
                        "[HBMCoordinator] central decision stage=%d replica=%d cap=%d "
                        "pressure=%.4f reason=%s generation=%d",
                        candidate.stage_id,
                        candidate.replica_id,
                        decision.effective_max_num_seqs,
                        effective_pressure,
                        reason,
                        generation,
                    )

    def _check_memory_report_timeouts(self) -> None:
        """Fail safe when a live replica stops publishing memory samples."""
        with self._lock:
            self._check_memory_report_timeouts_locked()

    def _check_memory_report_timeouts_locked(self) -> None:
        now = time()
        for addr, received_at in list(self._memory_received_at.items()):
            config = self._memory_configs.get(addr)
            report = self._memory_reports.get(addr)
            allocator = self._budget_allocators.get(addr)
            route = self._stage_routes.get(addr)
            if config is None or report is None or allocator is None or route is None:
                continue
            if now - received_at < config.report_timeout_ms / 1000:
                continue
            # Advance at most once per sample interval; an empty rank set uses
            # BudgetAllocator's grace/decrease policy.
            if now - self._last_allocation_at.get(addr, 0.0) < config.sample_interval_ms / 1000:
                continue
            stale = ReplicaMemoryReport(
                stage_id=report.stage_id,
                replica_id=report.replica_id,
                timestamp_monotonic_s=report.timestamp_monotonic_s,
                rank_reports=(),
                expected_rank_count=report.expected_rank_count,
                kv_total_blocks=report.kv_total_blocks,
                kv_free_blocks=report.kv_free_blocks,
                running_requests=report.running_requests,
                waiting_requests=report.waiting_requests,
                configured_max_num_seqs=report.configured_max_num_seqs,
            )
            decision = allocator.allocate(stale)
            self._last_allocation_at[addr] = now
            generation = self._budget_generations.get(addr, 0) + 1
            self._budget_generations[addr] = generation
            self._last_caps[addr] = decision.effective_max_num_seqs
            wire = {
                "message_type": "budget_decision",
                "stage_id": report.stage_id,
                "replica_id": report.replica_id,
                "instance_id": self._memory_instances[addr],
                "decision_generation": generation,
                "based_on_report_generation": self._memory_report_generations[addr],
                "effective_max_num_seqs": decision.effective_max_num_seqs,
                "pressure": config.high_watermark,
                "reason": f"stale_report_{decision.reason}",
                "safety_state": "stale",
                "pressure_source": "telemetry_health",
                "physical_hbm_pressure": 1.0,
                "kv_pressure": report.kv_pressure,
                "report_age_ms": max(0.0, (now - received_at) * 1000),
            }
            try:
                self._router.send_multipart([route, json.dumps(wire).encode("utf-8")], flags=zmq.NOBLOCK)
            except (zmq.Again, zmq.ZMQError):
                logger.warning("Dropping stale-report HBM decision for %s", addr)
            else:
                self._decision_sent_at[(addr, generation)] = time()

    def _periodic_loop(self) -> None:
        """Periodic loop to check heartbeat timeouts and flush broadcasts.

        Heartbeat timeouts are checked on their original cadence, while all
        broadcast requests are coalesced and flushed at most once per
        ``_publish_min_interval``. The heartbeat-check tick also schedules a
        keepalive broadcast so late-joining hubs (which miss any PUB sends
        that happened before their SUB connected) catch up within at most
        ``heartbeat_interval`` seconds.
        """
        heartbeat_interval = max(1.0, min(self._heartbeat_timeout / 2.0, 5.0))
        loop_interval = self._publish_min_interval

        last_heartbeat_check = 0.0
        while self._running:
            now = time()

            if now - last_heartbeat_check >= heartbeat_interval:
                self._check_heartbeat_timeouts()
                # Keepalive broadcast: ZMQ PUB doesn't queue for late
                # subscribers, so an OmniCoordClientForHub that connects
                # after the initial UP events miss them entirely and would
                # never see the current replica list otherwise. Scheduling a
                # broadcast on every heartbeat tick caps that staleness at
                # ``heartbeat_interval`` without flooding the wire.
                self._schedule_broadcast()
                last_heartbeat_check = now

            with self._pending_lock:
                has_pending_broadcast = self._pending_broadcast

            if not has_pending_broadcast:
                if self._stop_event.wait(timeout=loop_interval):
                    break
                continue

            # Publish outside lock. Clear pending only on success.
            if self.publish_replica_list_update():
                with self._pending_lock:
                    self._pending_broadcast = False

            if self._stop_event.wait(timeout=loop_interval):
                break

    def _handle_event(self, event: ReplicaEvent) -> None:
        """Dispatch an incoming event to the appropriate handler."""
        try:
            input_addr = event.input_addr

            # Heartbeat: refresh last_heartbeat and queue_length. The stage
            # client refreshes queue_length just-in-time via its
            # ``_on_heartbeat`` hook, so heartbeats are the only periodic
            # source of live load for LeastQueueLengthBalancer; failing to
            # propagate it here would let the policy route on stale data.
            # If previously ERROR, promote back to UP and broadcast once.
            if event.event_type == "heartbeat":
                promote = False
                queue_changed = False
                with self._lock:
                    info = self._replicas.get(input_addr)
                    if info is not None:
                        info.last_heartbeat = time()
                        if event.queue_length is not None and info.queue_length != event.queue_length:
                            info.queue_length = event.queue_length
                            queue_changed = True
                        if info.status == ReplicaStatus.ERROR:
                            info.status = ReplicaStatus.UP
                            promote = True
                if promote or queue_changed:
                    self._schedule_broadcast()
                return

            # Check-and-act under single lock to avoid TOCTOU race (duplicate
            # registration when concurrent events arrive for the same replica).
            with self._lock:
                if input_addr not in self._replicas:
                    self._add_new_replica_locked(event)
                else:
                    if event.status == ReplicaStatus.DOWN:
                        self._remove_replica_locked(event)
                    else:
                        self._update_replica_info_locked(event)

            # Any non-heartbeat state change that affects the active list
            # is coalesced and flushed via the periodic loop.
            self._schedule_broadcast()
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Dropping malformed event: %s", e)

    def _add_new_replica_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        if not input_addr:
            raise KeyError("input_addr required")
        stage_id = event.stage_id
        if stage_id < 0:
            raise KeyError("stage_id required and must be non-negative")

        now = time()
        info = ReplicaInfo(
            input_addr=input_addr,
            output_addr=event.output_addr,
            stage_id=stage_id,
            status=event.status,
            queue_length=event.queue_length,
            last_heartbeat=now,
            registered_at=now,
            replica_id=event.replica_id,
            instance_id=event.instance_id,
        )
        self._replicas[input_addr] = info

    def _update_replica_info_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        info = self._replicas[input_addr]

        if event.status is not None:
            info.status = event.status

        if event.queue_length is not None:
            info.queue_length = event.queue_length
        if event.instance_id and event.instance_id != info.instance_id:
            self._clear_memory_state(input_addr)
            info.instance_id = event.instance_id
        info.replica_id = event.replica_id

    def _remove_replica_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        info = self._replicas.get(input_addr)
        if info is None:
            return

        info.status = ReplicaStatus.DOWN
        self._clear_memory_state(input_addr)
