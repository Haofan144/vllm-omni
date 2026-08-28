# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReplicaStatus(str, Enum):
    """Enumeration for stage replica status."""

    UP = "up"  # Replica is ready and available
    DOWN = "down"  # Replica is shutdown gracefully
    ERROR = "error"  # Replica encountered an error or timeout


@dataclass
class ReplicaEvent:
    """Wire payload from OmniCoordClientForStage to OmniCoordinator.

    Schema for Stage → Coordinator events over ZMQ:
    input_addr, output_addr, stage_id, status, queue_length, event_type.
    """

    input_addr: str  # Stage replica input ZMQ address (e.g., "tcp://host:port")
    output_addr: str  # Stage replica output ZMQ address (e.g., "tcp://host:port")
    stage_id: int  # Stage ID
    event_type: str  # "update" | "heartbeat"
    status: ReplicaStatus  # Current status
    queue_length: int  # Current queue length
    replica_id: int = 0
    instance_id: str = ""


@dataclass(frozen=True)
class MemoryReportEvent:
    """Stage replica memory report sent to the central coordinator."""

    input_addr: str
    stage_id: int
    replica_id: int
    instance_id: str
    report_generation: int
    report: dict
    dynamic_hbm: dict


@dataclass(frozen=True)
class BudgetDecisionEvent:
    """Coordinator decision delivered to exactly one stage replica."""

    message_type: str
    stage_id: int
    replica_id: int
    instance_id: str
    decision_generation: int
    based_on_report_generation: int
    effective_max_num_seqs: int
    pressure: float
    reason: str
    safety_state: str = "normal"
    pressure_source: str = "none"
    physical_hbm_pressure: float = 0.0
    kv_pressure: float = 0.0
    report_age_ms: float = 0.0


@dataclass(frozen=True)
class BudgetAppliedEvent:
    """Scheduler acknowledgement that a safety decision became effective."""

    message_type: str
    input_addr: str
    stage_id: int
    replica_id: int
    instance_id: str
    decision_generation: int
    applied_safety_cap: int
    effective_cap: int
    occupied_slots: int
    applied_monotonic_s: float


@dataclass
class ReplicaInfo:
    """Metadata for a single stage replica.

    This type is stored in OmniCoordinator's internal registry and is also
    published to hubs via :class:`ReplicaList`.
    """

    input_addr: str  # Stage replica input ZMQ address (e.g., "tcp://host:port")
    output_addr: str  # Stage replica output ZMQ address (e.g., "tcp://host:port")
    stage_id: int  # Stage ID of this replica
    status: ReplicaStatus  # Current status of the replica
    queue_length: int  # Current queue length of this replica
    last_heartbeat: float  # Timestamp of the last heartbeat received (seconds)
    registered_at: float  # Timestamp when the replica was registered (seconds)
    replica_id: int = 0
    instance_id: str = ""


@dataclass
class ReplicaList:
    """Container for replica list updates.

    OmniCoordinator publishes a :class:`ReplicaList` whenever its view of
    active replicas changes. OmniCoordClientForHub caches the latest value
    and exposes it to AsyncOmni and the load balancer.
    """

    replicas: list[ReplicaInfo]
    timestamp: float  # Time when the list was last updated (seconds)
