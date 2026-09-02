from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.memory_coordinator import (
    ARRequestResourceContext,
    ARProfileStore,
    ARWorkloadClassifier,
    AdmissionDecision,
    AdmissionReason,
    BudgetAllocator,
    Code2WavRequestContext,
    Code2WavResourceEstimator,
    Code2WavWorkloadClassifier,
    DynamicHBMConfig,
    OnlineCalibrator,
    ReplicaMemoryReport,
    ProfileFingerprint,
    ResourceObservationCollector,
    ResourceObservationJSONLWriter,
    SafetyState,
    evaluate_ar_kv_admission,
    uncertainty_multiplier,
)
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.core.sched.output import (
    OmniChunkRecvHandle,
    OmniNewRequestData,
    OmniSchedulerOutput,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_STATS_INTERVAL_S = 1.0

# Upper bound on how long a request may wait for stage input before the
# scheduler force-fails it.  Defends against stuck consumer-side requests when
# the producer drops a payload, send fails, or recv never arrives.  Override
# per-deployment via VLLM_OMNI_INPUT_WAIT_TIMEOUT_S; set exactly 0 to disable
# the safety net. Negative and non-finite values are rejected at startup.
#
# Scope: both transports.  The full-payload path measures time parked in
# WAITING_FOR_INPUT (``OmniSchedulingCoordinator._waiting_since``); the
# async-chunk path measures time stalled in WAITING_FOR_CHUNK
# (``OmniChunkTransferAdapter._waiting_since``, reset on each chunk arrival).
# One knob, because the question it answers -- "how long may a request wait for
# stage input?" -- does not depend on which transport carries it.
_INPUT_WAIT_TIMEOUT_DEFAULT = 600.0
_INPUT_WAIT_TIMEOUT_RAW = os.environ.get("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "600")
try:
    DEFAULT_INPUT_WAIT_TIMEOUT_S: float = float(_INPUT_WAIT_TIMEOUT_RAW)
except ValueError:
    logger.warning(
        "Invalid VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=%r; falling back to %s seconds.",
        _INPUT_WAIT_TIMEOUT_RAW,
        _INPUT_WAIT_TIMEOUT_DEFAULT,
    )
    DEFAULT_INPUT_WAIT_TIMEOUT_S = _INPUT_WAIT_TIMEOUT_DEFAULT

if not math.isfinite(DEFAULT_INPUT_WAIT_TIMEOUT_S):
    # nan compares false against everything, so every deadline check silently
    # passes; inf makes the deadline unreachable. Both disable the net while
    # looking like a number, which is the failure this item exists to stop.
    raise ValueError(
        f"VLLM_OMNI_INPUT_WAIT_TIMEOUT_S={_INPUT_WAIT_TIMEOUT_RAW!r} is not finite. "
        "Use a positive number of seconds, or exactly 0 to disable the deadline."
    )
if DEFAULT_INPUT_WAIT_TIMEOUT_S < 0:
    # Fail at startup rather than substituting a default. `-1` is a common idiom
    # for "no limit", so silently turning it into 600 would give an operator who
    # meant "never time out" mysterious failures ten minutes in.
    raise ValueError(
        f"VLLM_OMNI_INPUT_WAIT_TIMEOUT_S={_INPUT_WAIT_TIMEOUT_RAW!r} is negative. "
        "Use a positive number of seconds, or exactly 0 to disable the deadline."
    )
elif DEFAULT_INPUT_WAIT_TIMEOUT_S == 0:
    # Zero stays a supported opt-out, but it is no longer silent: it disables
    # the deadline on BOTH transports, so a request whose stage input never
    # arrives waits forever. Say which nets just went away.
    logger.warning(
        "VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0: stage-input deadlines are DISABLED for "
        "both the full-payload path (WAITING_FOR_INPUT) and the async-chunk path "
        "(WAITING_FOR_CHUNK). A request whose input never arrives will wait "
        "indefinitely instead of failing."
    )


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    # ------------------------------------------------------------------ #
    #  Shared scheduler/output helpers (lift the AR / generation duplicates)
    # ------------------------------------------------------------------ #

    def _init_dynamic_hbm_scheduling_state(self) -> None:
        """Initialize dynamic admission state for any Omni scheduler."""
        self._dynamic_hbm_config = DynamicHBMConfig.from_value(
            getattr(self.vllm_config.model_config, "dynamic_hbm", None)
        )
        self._configured_max_num_seqs = self.max_num_running_reqs
        self._configured_max_num_scheduled_tokens = self.max_num_scheduled_tokens
        self._safety_max_num_seqs = self._configured_max_num_seqs
        self._safety_max_num_scheduled_tokens = self._configured_max_num_scheduled_tokens
        self._effective_max_num_seqs = self._configured_max_num_seqs
        self._effective_max_num_scheduled_tokens = self._configured_max_num_scheduled_tokens
        self._last_budget_generation = 0
        self._last_budget_report_generation = 0
        self._dynamic_hbm_critical = False
        self._dynamic_hbm_safety_state = SafetyState.NORMAL
        self._resource_admission_counts: dict[str, int] = {}
        self._last_resource_admission_decision: AdmissionDecision | None = None
        # First monotonic time each request was observed as an un-fitting
        # head-of-line request. Cleared once the request is admitted, leaves
        # the queue, or ages out. Powers the bounded-bypass aging guard.
        self._resource_admission_head_of_line_since: dict[str, float] = {}
        # Code2Wav/LLM_GENERATION observation-only telemetry (M2c). Unlike
        # the AR admission decision above, this never gates scheduling: no
        # real "free physical bytes" budget exists anywhere in the scheduler
        # to compare a predicted transient/persistent-byte cost against, so
        # building an admit/defer decision here would mean fabricating a
        # budget rather than measuring one. This purely records what
        # Code2WavResourceEstimator would have predicted for the head-of-line
        # waiting request each tick, for later offline analysis (the same
        # role H11's shadow-trace analysis played for the AR estimator before
        # a real KV-block budget made enforce mode meaningful there).
        self._code2wav_workload_classifier = Code2WavWorkloadClassifier()
        self._code2wav_resource_estimator = Code2WavResourceEstimator(
            self._code2wav_workload_classifier
        )
        self._code2wav_observation_count = 0
        self._code2wav_observation_errors = 0
        self._resource_admission_bypass_count = 0
        self._resource_admission_bypassed_requests = 0
        self._resource_admission_aging_stops = 0
        self._resource_observation_collector = ResourceObservationCollector()
        self._resource_calibrator = OnlineCalibrator()
        classifier_path = self._dynamic_hbm_config.ar_workload_classifier
        self._ar_workload_classifier = (
            resolve_obj_by_qualname(classifier_path)() if classifier_path else ARWorkloadClassifier()
        )
        self._dynamic_hbm_ar_profile_store: ARProfileStore | None = None
        self._dynamic_hbm_ar_profile_loaded = False
        self._resource_observation_writer: ResourceObservationJSONLWriter | None = None
        observation_path = self._dynamic_hbm_config.resource_observation_path
        if observation_path:
            stage_id = int(getattr(self.vllm_config.model_config, "stage_id", 0))
            replica_id = int(os.environ.get("VLLM_OMNI_REPLICA_ID", "0"))
            values = {
                "stage_id": stage_id,
                "replica_id": replica_id,
                "pid": os.getpid(),
            }
            if "{" in observation_path:
                resolved_path = observation_path.format(**values)
            else:
                path = os.path.abspath(observation_path)
                root, extension = os.path.splitext(path)
                resolved_path = (
                    f"{root}.stage-{stage_id}.replica-{replica_id}.pid-{os.getpid()}"
                    f"{extension or '.jsonl'}"
                )
            self._resource_observation_writer = ResourceObservationJSONLWriter(
                resolved_path,
                flush_size=self._dynamic_hbm_config.resource_observation_flush_size,
            )
        self._dynamic_hbm_allocator = (
            BudgetAllocator(self._dynamic_hbm_config, self._configured_max_num_seqs)
            if self._dynamic_hbm_config.enabled
            else None
        )

    def update_replica_memory_report(self, report: ReplicaMemoryReport) -> None:
        """Apply a process-local decision when no central coordinator exists."""
        config = getattr(self, "_dynamic_hbm_config", None)
        allocator = getattr(self, "_dynamic_hbm_allocator", None)
        if not config or not config.enabled or allocator is None:
            return
        stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
        if report.stage_id != stage_id:
            return
        if not hasattr(self, "_configured_max_num_seqs"):
            self._configured_max_num_seqs = report.configured_max_num_seqs
        if not hasattr(self, "_configured_max_num_scheduled_tokens"):
            self._configured_max_num_scheduled_tokens = getattr(
                self,
                "max_num_scheduled_tokens",
                report.configured_max_num_seqs,
            )
        if not hasattr(self, "_safety_max_num_seqs"):
            self._safety_max_num_seqs = self._configured_max_num_seqs
        decision = allocator.allocate(report)
        self.apply_stage_budget_decision(
            generation=getattr(self, "_last_budget_generation", 0) + 1,
            effective_max_num_seqs=decision.effective_max_num_seqs,
            pressure=decision.pressure,
            reason=decision.reason,
            safety_state=decision.safety_state,
            pressure_source=decision.pressure_source,
        )

    def _recompute_effective_dynamic_hbm_budget(self) -> None:
        """Combine configured and reactive-safety limits.

        A future predictive planner can add its own cap to this minimum
        without granting the safety controller authority to expand past it.
        """
        configured_seqs = self._configured_max_num_seqs
        configured_tokens = getattr(
            self,
            "_configured_max_num_scheduled_tokens",
            getattr(self, "max_num_scheduled_tokens", configured_seqs),
        )
        self._effective_max_num_seqs = min(
            configured_seqs,
            self._safety_max_num_seqs,
        )
        if self._dynamic_hbm_config.scale_token_budget:
            occupied_slots = len(getattr(self, "running", ())) + getattr(
                self,
                "num_waiting_for_streaming_input",
                0,
            )
            scheduling_slots = max(self._effective_max_num_seqs, occupied_slots)
            ratio = min(1.0, scheduling_slots / configured_seqs)
            self._safety_max_num_scheduled_tokens = max(
                1,
                self._effective_max_num_seqs,
                int(configured_tokens * ratio),
            )
        else:
            self._safety_max_num_scheduled_tokens = configured_tokens
        self._effective_max_num_scheduled_tokens = min(
            configured_tokens,
            self._safety_max_num_scheduled_tokens,
        )

    def apply_stage_budget_decision(
        self,
        *,
        generation: int,
        effective_max_num_seqs: int,
        pressure: float,
        reason: str,
        based_on_report_generation: int = 0,
        safety_state: str = SafetyState.NORMAL.value,
        pressure_source: str = "none",
    ) -> bool:
        """Apply a monotonic central dynamic-HBM decision."""
        last_budget_generation = getattr(self, "_last_budget_generation", 0)
        last_report_generation = getattr(self, "_last_budget_report_generation", 0)
        if generation <= last_budget_generation:
            return False
        if based_on_report_generation and based_on_report_generation < last_report_generation:
            return False
        try:
            state = SafetyState(safety_state)
        except ValueError:
            logger.warning("Dropping dynamic-HBM decision with unknown safety state %r", safety_state)
            return False
        normal_floor = self._dynamic_hbm_config.min_num_seqs
        floor = (
            self._dynamic_hbm_config.critical_admission_cap
            if state in {
                SafetyState.CRITICAL,
                SafetyState.STALE,
                SafetyState.DISCONNECTED,
                SafetyState.RECOVERING,
            }
            else normal_floor
        )
        if state is SafetyState.CRITICAL:
            new_cap = min(
                self._configured_max_num_seqs,
                self._dynamic_hbm_config.critical_admission_cap,
                max(0, int(effective_max_num_seqs)),
            )
        else:
            new_cap = min(
                self._configured_max_num_seqs,
                max(floor, int(effective_max_num_seqs)),
            )
        previous_cap = self._effective_max_num_seqs
        self._safety_max_num_seqs = new_cap
        self._dynamic_hbm_safety_state = state
        self._dynamic_hbm_critical = state is SafetyState.CRITICAL
        self._dynamic_hbm_pressure_source = pressure_source
        self._recompute_effective_dynamic_hbm_budget()
        self._last_budget_generation = generation
        self._last_budget_report_generation = max(
            last_report_generation,
            based_on_report_generation,
        )
        if previous_cap != new_cap:
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
            stage_id = getattr(model_config, "stage_id", 0)
            replica_id = int(os.environ.get("VLLM_OMNI_REPLICA_ID", "0"))
            logger.info(
                "[HBMCoordinator] stage=%d replica=%d cap=%d->%d pressure=%.4f "
                "reason=%s generation=%d mode=centralized",
                stage_id,
                replica_id,
                previous_cap,
                new_cap,
                pressure,
                reason,
                generation,
            )
        return True

    def apply_dynamic_hbm_disconnect_guard(self) -> bool:
        """Fail closed locally when the central safety stream is unavailable."""
        config = getattr(self, "_dynamic_hbm_config", None)
        if not config or not config.enabled or not config.fail_closed_on_disconnect:
            return False
        self._safety_max_num_seqs = min(
            getattr(self, "_safety_max_num_seqs", self._configured_max_num_seqs),
            config.disconnect_admission_cap,
        )
        self._dynamic_hbm_safety_state = SafetyState.DISCONNECTED
        self._dynamic_hbm_critical = False
        self._dynamic_hbm_pressure_source = "telemetry_health"
        self._recompute_effective_dynamic_hbm_budget()
        return True

    def apply_dynamic_hbm_local_kv_guard(self) -> bool:
        """Stop admission immediately when the local KV pool is exhausted.

        This local safety action deliberately does not consume a central
        decision generation. The next fresh coordinator decision can still
        reconcile the replica after its report observes the KV state.
        """
        config = getattr(self, "_dynamic_hbm_config", None)
        if not config or not config.enabled:
            return False
        self._safety_max_num_seqs = min(
            getattr(self, "_safety_max_num_seqs", self._configured_max_num_seqs),
            config.critical_admission_cap,
        )
        self._dynamic_hbm_safety_state = SafetyState.CRITICAL
        self._dynamic_hbm_critical = True
        self._dynamic_hbm_pressure_source = "kv"
        self._recompute_effective_dynamic_hbm_budget()
        return True

    def _dynamic_hbm_allows_new_admission(self) -> bool:
        if not getattr(getattr(self, "_dynamic_hbm_config", None), "enabled", False):
            return True
        occupied_slots = len(self.running) + getattr(self, "num_waiting_for_streaming_input", 0)
        return occupied_slots < self._effective_max_num_seqs

    def _dynamic_hbm_execution_type(self) -> Any:
        """This stage's ``StageExecutionType``, or ``None`` if undeterminable."""
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        stage_pipeline_config = getattr(model_config, "stage_pipeline_config", None)
        return getattr(stage_pipeline_config, "execution_type", None)

    def _dynamic_hbm_ar_estimator_applies(self) -> bool:
        """Whether this stage's requests are genuine AR text/output tokens.

        ``ARResourceEstimator`` is only meaningful for an ``LLM_AR`` stage.
        Other execution types (e.g. ``LLM_GENERATION`` Code2Wav/decoder
        stages) still expose ``request.num_prompt_tokens``/``max_tokens`` --
        every vLLM ``Request`` has them -- but those fields hold codec-frame
        or other backend-specific placeholder counts, not text tokens, so
        applying AR KV-block accounting to them would be silently wrong
        rather than loudly unsupported. Stages whose execution type cannot be
        determined are treated as unsupported (fail closed to "don't apply
        the AR estimator") rather than assumed AR.
        """
        execution_type = self._dynamic_hbm_execution_type()
        if execution_type is None:
            return False
        from vllm_omni.config.stage_config import StageExecutionType

        return execution_type == StageExecutionType.LLM_AR

    def _dynamic_hbm_code2wav_estimator_applies(self) -> bool:
        """Whether this stage is a Code2Wav-style ``LLM_GENERATION`` decoder.

        Symmetric to ``_dynamic_hbm_ar_estimator_applies``: an execution type
        that cannot be determined is treated as unsupported, not assumed to
        be a generation stage.
        """
        execution_type = self._dynamic_hbm_execution_type()
        if execution_type is None:
            return False
        from vllm_omni.config.stage_config import StageExecutionType

        return execution_type == StageExecutionType.LLM_GENERATION

    @staticmethod
    def _code2wav_frame_count(request: Any) -> int:
        """A request's codec-frame count for this step.

        Mirrors ``OmniGenerationScheduler.schedule()``'s own
        ``required_tokens = max(len(request.prompt_token_ids), 1)`` --
        Code2Wav's forward pass derives frame count directly from
        ``prompt_token_ids`` length (flattened codec ids, ``n // q``; see
        ``qwen3_tts_code2wav.py``'s ``forward()``), so the scheduler's own
        placeholder-token accounting already IS the frame count, not a
        separate concept requiring new bookkeeping.
        """
        return max(len(getattr(request, "prompt_token_ids", ()) or ()), 1)

    def _code2wav_resource_context(self, request: Any) -> Code2WavRequestContext | None:
        """Build estimator input for the head-of-line waiting Code2Wav
        request, describing the candidate batch it would join (the already-
        ``running`` requests plus itself) -- see ``Code2WavRequestContext``'s
        docstring on why cost depends on batchmates, not only the request
        itself.
        """
        try:
            frame_count = self._code2wav_frame_count(request)
            running = getattr(self, "running", ())
            batch_max_frame_count = frame_count
            for other in running:
                batch_max_frame_count = max(batch_max_frame_count, self._code2wav_frame_count(other))
            persistent_state_active = bool(
                getattr(request, "resumable", False)
                or getattr(request, "streaming_queue", None) is not None
            )
            return Code2WavRequestContext(
                frame_count=frame_count,
                batch_size=len(running) + 1,
                batch_max_frame_count=batch_max_frame_count,
                persistent_state_active=persistent_state_active,
                workload_class=self._code2wav_workload_classifier.classify(
                    batch_size=len(running) + 1,
                    frame_count=frame_count,
                    persistent_state_active=persistent_state_active,
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Code2Wav resource context build failed: %s", exc)
            return None

    def _sample_code2wav_resource_observation(self) -> None:
        """Observation-only telemetry for one ``LLM_GENERATION`` schedule()
        tick: predicts the head-of-line waiting request's transient/
        persistent byte cost and records it, without ever gating admission.

        Called unconditionally from ``OmniGenerationScheduler.schedule()``;
        declines immediately (and cheaply) for any stage that isn't a
        Code2Wav-style generation stage or has an empty waiting queue, so it
        adds no real cost to schedulers that don't apply.
        """
        if not self._dynamic_hbm_code2wav_estimator_applies():
            return
        waiting = getattr(self, "waiting", None)
        if not waiting:
            return
        request = next(iter(waiting), None)
        if request is None:
            return
        try:
            context = self._code2wav_resource_context(request)
            if context is None:
                return
            self._code2wav_resource_estimator.estimate(context)
            self._code2wav_observation_count += 1
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            self._code2wav_observation_errors += 1
            logger.warning("Code2Wav resource observation failed: %s", exc)

    def _ar_resource_estimator(self) -> Any:
        """Lazily build (and cache) the AR KV-block estimator for this replica.

        ``block_size`` mirrors the fallback chain already used by
        ``OmniARScheduler._mark_request_for_kv_transfer`` (cache_config first,
        then scheduler_config) since neither is guaranteed set at
        ``_init_dynamic_hbm_scheduling_state`` time.
        """
        estimator = getattr(self, "_dynamic_hbm_ar_estimator", None)
        if estimator is not None:
            return estimator
        block_size = None
        cache_config = getattr(self, "cache_config", None)
        if cache_config is not None and hasattr(cache_config, "block_size"):
            block_size = cache_config.block_size
        else:
            scheduler_config = getattr(self, "scheduler_config", None)
            if scheduler_config is not None and hasattr(scheduler_config, "block_size"):
                block_size = scheduler_config.block_size
        if not block_size:
            return None
        from vllm_omni.core.memory_coordinator import ARResourceEstimator

        estimator = ARResourceEstimator(block_size)
        self._dynamic_hbm_ar_estimator = estimator
        return estimator

    def _get_request_allocated_kv_blocks(self, request: Any) -> int:
        """Return logical allocated capacity without summing KV groups."""
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_blocks"):
            return 0
        try:
            block_ids = kv_cache_manager.get_blocks(request.request_id).get_block_ids()
            return max((len(ids) for ids in block_ids), default=0)
        except (AttributeError, KeyError):
            return 0

    def _ar_profile_fingerprint(self, block_size: int) -> ProfileFingerprint:
        model_config = self.vllm_config.model_config
        model_id = getattr(model_config, "model", None)
        if model_id is None:
            hf_config = getattr(model_config, "hf_config", None)
            model_id = getattr(hf_config, "_name_or_path", "unknown")
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        return ProfileFingerprint(
            model_id=str(model_id),
            device_type=str(self._dynamic_hbm_config.resource_profile_device_type),
            dtype=str(getattr(model_config, "dtype", "unknown")),
            tp_size=max(
                1,
                int(getattr(parallel_config, "tensor_parallel_size", 1)),
            ),
            block_size=block_size,
            execution_mode="async" if getattr(model_config, "async_chunk", False) else "default",
        )

    def _ar_resource_profile_store(self, block_size: int) -> ARProfileStore | None:
        if self._dynamic_hbm_ar_profile_loaded:
            return self._dynamic_hbm_ar_profile_store
        self._dynamic_hbm_ar_profile_loaded = True
        path = self._dynamic_hbm_config.resource_profile_path
        if not path:
            return None
        try:
            store = ARProfileStore.read_jsonl(path)
            store.require_fingerprint(self._ar_profile_fingerprint(block_size))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Unable to load AR resource profile %s; using hard fallback: %s",
                path,
                exc,
            )
            self._resource_admission_counts["profile_load_error"] = (
                self._resource_admission_counts.get("profile_load_error", 0) + 1
            )
            return None
        self._dynamic_hbm_ar_profile_store = store
        return store

    @staticmethod
    def _tts_classify_extra_kwargs(request: Any) -> dict[str, Any]:
        """Extract the TTS-specific classifier dimensions from a request's
        ``additional_information``, when present.

        ``TTSWorkloadClassifier.classify`` reads ``task_type``/
        ``ref_audio_present``; ``ARWorkloadClassifier.classify`` ignores
        unknown keyword arguments, so it is safe to always build and pass
        this dict regardless of which classifier is configured -- the
        scheduler mixin does not need to know which one is active.

        Values in ``additional_information`` follow the batch-friendly,
        single-element-list convention every TTS request-builder in
        ``serving_speech.py`` uses (e.g. ``tts_params["task_type"] =
        [request.task_type]``): unwrap a one-element list before reading it.
        Missing/differently-shaped fields (e.g. a non-TTS model's
        ``additional_information``) degrade to the classifier's own
        defaults rather than raising.
        """
        payload = getattr(request, "additional_information", None)
        if payload is None:
            return {}
        try:
            info = deserialize_additional_information(payload)
        except (AttributeError, KeyError, TypeError, ValueError):
            return {}

        def _unwrap(key: str) -> Any:
            value = info.get(key)
            if isinstance(value, list):
                return value[0] if value else None
            return value

        task_type = _unwrap("task_type")
        ref_audio_present = bool(_unwrap("ref_audio") or _unwrap("ref_code_length"))
        return {
            "task_type": task_type if isinstance(task_type, str) else None,
            "ref_audio_present": ref_audio_present,
        }

    def _ar_resource_context(self, request: Any) -> ARRequestResourceContext | None:
        """Build estimator input from scheduler state without hiding it in the estimator."""
        estimator = self._ar_resource_estimator()
        if estimator is None or estimator.block_size is None:
            return None

        generated = getattr(request, "num_output_tokens", None)
        if generated is None:
            generated = len(getattr(request, "output_token_ids", ()))
        streaming = bool(
            getattr(request, "resumable", False)
            or getattr(request, "streaming_queue", None) is not None
        )
        workload_class = self._ar_workload_classifier.classify(
            prompt_tokens=max(0, int(getattr(request, "num_prompt_tokens", 0))),
            max_tokens=max(0, int(getattr(request, "max_tokens", 0))),
            streaming=streaming,
            **self._tts_classify_extra_kwargs(request),
        )
        profile = None
        store = self._ar_resource_profile_store(estimator.block_size)
        if store is not None:
            candidate = store.get(workload_class)
            if (
                candidate is not None
                and candidate.sample_count
                >= self._dynamic_hbm_config.resource_profile_min_samples
            ):
                profile = candidate
        return ARRequestResourceContext(
            num_prompt_tokens=max(0, int(getattr(request, "num_prompt_tokens", 0))),
            max_tokens=max(0, int(getattr(request, "max_tokens", 0))),
            block_size=estimator.block_size,
            num_computed_tokens=max(0, int(getattr(request, "num_computed_tokens", 0))),
            num_generated_tokens=max(0, int(generated)),
            allocated_kv_blocks=self._get_request_allocated_kv_blocks(request),
            # Prefix-cache hits are only known for a request vLLM's own
            # scheduler has actually scheduled: they are computed by
            # KVCacheManager.get_computed_blocks (populating
            # request.prefill_stats), which has side effects — it creates
            # block references and, in "full" KV-event mode, emits
            # BlockStored events — and is meant to run at most once per
            # prefill. This admission check runs on the still-waiting
            # head-of-line request *before* that scheduling step, so there is
            # no side-effect-free way to look up the real hit count here
            # without duplicating KVCacheCoordinator.find_longest_cache_hit's
            # internal (and cache/config-version-fragile) prefix-hash walk.
            # Zero is therefore a structural lower bound on reuse, not a
            # placeholder pending a follow-up: it only ever overestimates a
            # request's new-token KV cost, never underestimates it.
            reusable_cached_tokens=0,
            next_scheduled_tokens=max(
                1,
                min(
                    max(
                        1,
                        int(getattr(request, "num_tokens", 0))
                        - int(getattr(request, "num_computed_tokens", 0)),
                    ),
                    int(getattr(self, "_effective_max_num_scheduled_tokens", 1)),
                ),
            ),
            expected_output_tokens=(
                profile.p50_output_tokens if profile is not None else None
            ),
            quantile_output_tokens=(
                profile.output_tokens_at(
                    self._dynamic_hbm_config.resource_target_coverage
                )
                if profile is not None
                else None
            ),
            workload_class=workload_class,
            target_coverage=self._dynamic_hbm_config.resource_target_coverage,
            profile_version=(profile.profile_version if profile is not None else None),
            sample_count=(profile.sample_count if profile is not None else 0),
        )

    def _record_resource_admission_decision(
        self,
        decision: AdmissionDecision,
        request: Any | None = None,
    ) -> None:
        self._last_resource_admission_decision = decision
        key = decision.reason.value
        self._resource_admission_counts[key] = self._resource_admission_counts.get(key, 0) + 1
        if decision.shadow_would_defer:
            self._resource_admission_counts["shadow_would_defer"] = (
                self._resource_admission_counts.get("shadow_would_defer", 0) + 1
            )
        request_id = getattr(request, "request_id", None)
        if request_id is not None and decision.estimate is not None:
            self._resource_observation_collector.begin(
                request_id,
                decision.estimate,
                baseline_allocated_kv_blocks=self._get_request_allocated_kv_blocks(
                    request
                ),
                prompt_tokens=max(
                    0, int(getattr(request, "num_prompt_tokens", 0))
                ),
                requested_max_tokens=max(
                    0, int(getattr(request, "max_tokens", 0))
                ),
                block_size=max(
                    1,
                    int(getattr(self._ar_resource_estimator(), "block_size", 1)),
                ),
            )

    def _sample_resource_observations(self) -> None:
        """Sample logical KV ground truth after a scheduler allocation step."""
        for request in getattr(self, "running", ()):
            prefill_stats = getattr(request, "prefill_stats", None)
            self._resource_observation_collector.observe(
                request.request_id,
                allocated_kv_blocks=self._get_request_allocated_kv_blocks(request),
                output_tokens=max(0, int(getattr(request, "num_output_tokens", 0))),
                local_cached_tokens=max(
                    0,
                    int(getattr(prefill_stats, "num_local_cached_tokens", 0)),
                ),
                external_cached_tokens=max(
                    0,
                    int(getattr(prefill_stats, "num_external_cached_tokens", 0)),
                ),
            )

    def drain_resource_observations(self) -> list[Any]:
        """Drain bounded request-level traces for offline M2 evaluation."""
        return self._resource_observation_collector.drain_completed()

    def _finish_resource_observation(self, request: Any) -> None:
        """Capture and persist one request before its KV blocks are released.

        Normal AR completion uses ``OmniARScheduler._free_request`` while
        abort/error cleanup uses ``finish_requests``.  Keeping the common
        finalization here prevents either lifecycle path from silently losing
        shadow observations.  Repeated calls are harmless because ``finish``
        removes the request from the collector's active set.
        """
        # A few lightweight tests construct schedulers through ``__new__``;
        # production schedulers always initialize the collector in the mixin.
        collector = getattr(self, "_resource_observation_collector", None)
        if collector is None:
            return
        request_id = request.request_id
        prefill_stats = getattr(request, "prefill_stats", None)
        collector.observe(
            request_id,
            allocated_kv_blocks=self._get_request_allocated_kv_blocks(request),
            output_tokens=max(0, int(getattr(request, "num_output_tokens", 0))),
            local_cached_tokens=max(
                0,
                int(getattr(prefill_stats, "num_local_cached_tokens", 0)),
            ),
            external_cached_tokens=max(
                0,
                int(getattr(prefill_stats, "num_external_cached_tokens", 0)),
            ),
        )
        observation = collector.finish(request_id)
        if observation is not None:
            self._resource_calibrator.update(observation)
            if self._resource_observation_writer is not None:
                self._resource_observation_writer.append(observation)

    def _dynamic_hbm_resource_admission_decision(
        self, request: Any | None = None, *, record: bool = True
    ) -> AdmissionDecision:
        """Return an explainable AR resource decision for one request.

        Defaults to the head-of-line waiting request when ``request`` is not
        given, preserving the original single-request call contract used by
        the H10/H11/H12 experiment harness. ``_dynamic_hbm_bounded_bypass_waiting``
        also calls this per-candidate while scanning past a head-of-line
        request that does not fit, so ``record`` lets that scan evaluate
        candidates without polluting ``_resource_admission_counts`` for
        requests that are not actually the step's admission decision.

        The default ``shadow`` mode observes the counterfactual decision but
        cannot block scheduling. This keeps M2 model validation separate from
        the later global commitment policy.
        """
        config = getattr(self, "_dynamic_hbm_config", None)
        if not config or not config.enabled or config.resource_admission_mode == "off":
            decision = AdmissionDecision(True, AdmissionReason.DISABLED)
            if record:
                self._record_resource_admission_decision(decision)
            return decision
        if request is None:
            waiting = getattr(self, "waiting", None)
            request = next(iter(waiting), None) if waiting else None
        if request is None:
            decision = AdmissionDecision(True, AdmissionReason.EMPTY_QUEUE)
            if record:
                self._record_resource_admission_decision(decision)
            return decision
        if not self._dynamic_hbm_ar_estimator_applies():
            # ARResourceEstimator/ARWorkloadClassifier read request.num_prompt_tokens
            # / request.max_tokens as AR text-token counts. A non-AR generation
            # stage (e.g. a Code2Wav decoder) still exposes those same Request
            # attributes, but they hold codec-frame-derived placeholder counts --
            # feeding them to the AR estimator would silently produce a
            # nonsensical (not an error) admission signal instead of failing
            # loudly. Decline before ever building an AR context for this stage.
            decision = AdmissionDecision(True, AdmissionReason.UNSUPPORTED_EXECUTION_TYPE)
            if record:
                self._record_resource_admission_decision(decision)
            return decision
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        context = self._ar_resource_context(request)
        estimator = self._ar_resource_estimator()
        if (
            block_pool is None
            or not hasattr(block_pool, "get_num_free_blocks")
            or context is None
            or estimator is None
        ):
            decision = AdmissionDecision(True, AdmissionReason.ESTIMATOR_UNAVAILABLE)
            if record:
                self._record_resource_admission_decision(decision)
            return decision
        try:
            estimate = estimator.estimate(context)
            snapshot = self._resource_calibrator.snapshot(
                backend="ar",
                workload_class=context.workload_class,
                profile_version=context.profile_version,
            )
            margin = uncertainty_multiplier(
                snapshot,
                fallback_reason=estimate.provenance.fallback_reason,
                min_samples_for_full_confidence=config.resource_uncertainty_min_samples,
                low_sample_multiplier=config.resource_uncertainty_low_sample_multiplier,
                stale_multiplier=config.resource_uncertainty_stale_multiplier,
            )
            decision = evaluate_ar_kv_admission(
                estimate,
                free_kv_blocks=block_pool.get_num_free_blocks(),
                enforce=config.resource_admission_mode == "enforce",
                correction=snapshot.correction,
                uncertainty_multiplier=margin,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("AR resource estimation failed; leaving admission to safety guard: %s", exc)
            decision = AdmissionDecision(True, AdmissionReason.ESTIMATOR_ERROR)
        if record:
            self._record_resource_admission_decision(decision, request)
        return decision

    def _dynamic_hbm_next_waiting_request_fits(self) -> bool:
        """Whether the head-of-line waiting AR request's estimated KV-block
        cost fits in the replica's currently free KV blocks.

        This refines admission beyond the uniform "1 request = 1 slot" count
        in ``_dynamic_hbm_allows_new_admission``: a long-prompt/long-output
        request can be rejected even when a free slot exists, instead of
        being admitted and only failing later under KV pressure. Disabled
        (returns True, i.e. no additional restriction) unless dynamic HBM is
        enabled and both the block size and KV pool free-block count are
        resolvable.
        """
        # Compatibility helper used by H10: report the counterfactual fit even
        # when production rollout is in shadow mode.
        decision = self._dynamic_hbm_resource_admission_decision()
        return not decision.shadow_would_defer and decision.reason is not AdmissionReason.KV_PEAK_RISK

    def _dynamic_hbm_bounded_bypass_waiting(self) -> Any:
        """Filter ``self.waiting`` for one scheduling step under resource-aware
        enforcement, instead of freezing the entire queue behind a
        head-of-line request that does not fit (M2 design doc S12.3).

        Returns a new ``RequestQueue`` (same type/policy as ``self.waiting``)
        containing, in original FIFO order: the head-of-line request itself
        (so the caller's normal admission check still applies to it and it is
        never silently dropped), plus up to ``resource_admission_bypass_scan_limit``
        subsequent requests whose individual resource estimate fits the
        currently free KV blocks. Requests that do not fit, and requests
        beyond the scan limit, are left out of the returned queue -- callers
        restore them to the front of ``self.waiting`` afterward, exactly as
        the pre-M2 empty-queue swap already did.

        Only called when the head-of-line request itself is the reason for
        deferral (``KV_PEAK_RISK``); an empty queue, a disabled estimator, or
        a global KV exhaustion guard call this scan pointless, so callers
        should keep using the original all-or-nothing swap for those cases.
        """
        waiting = self.waiting
        scan_limit = self._dynamic_hbm_config.resource_admission_bypass_scan_limit
        bypassed = create_request_queue(self.policy)
        if scan_limit <= 0:
            return bypassed

        head_of_line_since = self._resource_admission_head_of_line_since
        now = time.monotonic()
        aging_ms = self._dynamic_hbm_config.resource_admission_bypass_aging_ms
        live_ids = set()
        scanned = 0
        for index, candidate in enumerate(waiting):
            live_ids.add(candidate.request_id)
            if index == 0:
                bypassed.add_request(candidate)
                started_at = head_of_line_since.setdefault(candidate.request_id, now)
                if (now - started_at) * 1000.0 >= aging_ms:
                    self._resource_admission_aging_stops += 1
                    break
                continue
            if scanned >= scan_limit:
                break
            scanned += 1
            decision = self._dynamic_hbm_resource_admission_decision(candidate, record=False)
            # ``reason`` reflects whether the estimate itself fits the
            # currently free KV blocks; ``allowed``/``shadow_would_defer``
            # instead encode the enforcement policy (shadow mode always sets
            # ``allowed=True`` even for a request that does not fit), so the
            # fit check must key off ``reason`` here.
            if decision.reason is not AdmissionReason.KV_PEAK_RISK:
                bypassed.add_request(candidate)
                self._resource_admission_bypass_count += 1
                self._resource_admission_bypassed_requests += 1

        stale_ids = set(head_of_line_since) - live_ids
        for stale_id in stale_ids:
            head_of_line_since.pop(stale_id, None)
        return bypassed

    def _dynamic_max_num_running_reqs(self) -> int:
        configured_cap = self.max_num_running_reqs
        dynamic_cap = getattr(self, "_effective_max_num_seqs", configured_cap)
        occupied_slots = len(self.running) + getattr(
            self,
            "num_waiting_for_streaming_input",
            0,
        )
        return min(configured_cap, max(dynamic_cap, occupied_slots))

    def _init_omni_io_scheduling_state(self) -> None:
        """Initialize scheduler state shared by AR and generation stages."""
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = (
            OmniChunkTransferAdapter(self.vllm_config) if getattr(model_config, "async_chunk", False) else None
        )
        self.input_coordinator = (
            OmniSchedulingCoordinator(stage_id=getattr(model_config, "stage_id", 0))
            if uses_full_payload_input_coordinator(model_config)
            else None
        )
        self._latest_omni_connector_output = None
        # Optional per-stage pooling-output decoder hook (dotted path in
        # model_config); applied worker-side before IPC.
        self._pooling_output_decoder = None
        _decoder_path = getattr(model_config, "pooling_output_decoder", None)
        if _decoder_path:
            self._pooling_output_decoder = resolve_obj_by_qualname(str(_decoder_path))

    def _maybe_decode_pooling_output(self, request: Request, pooler_output: Any) -> Any:
        """Apply the stage's pooling-output decoder hook to the pooler tensor
        before IPC, or pass it through unchanged when none is configured.
        Decoder exceptions propagate; callers fail the request with
        FinishReason.ERROR rather than emitting an empty success."""
        if self._pooling_output_decoder is None:
            return pooler_output
        if pooler_output is None or getattr(request, "pooling_params", None) is None:
            return pooler_output
        if not isinstance(pooler_output, torch.Tensor):
            return pooler_output
        return self._pooling_output_decoder(
            pooler_output,
            request,
            self.vllm_config.model_config.hf_config,
        )

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    def _replace_streaming_session(self, session: Request, update: StreamingUpdate) -> None:
        """Replace a downstream stage's placeholder with its next payload."""
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is not None:
            adapter.segment_finished_requests.discard(session.request_id)
            watermark = getattr(adapter, "requests_num_chunks_sent", None)
            if watermark is not None:
                watermark.pop(session.external_req_id, None)
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        # In-flight outputs from the previous segment were optimistically
        # scheduled (async lookahead). Mark them stale so update_from_output
        # drops them instead of underflowing num_output_placeholders
        # (vLLM 0.27 a0c092ee72 removed async_tokens_to_discard). Seed in
        # SCHEDULED-token units — num_in_flight_tokens matches what each
        # pre-replacement frame will drain, so the counter reaches exactly
        # zero; a placeholder-based seed swallowed valid new-segment frames
        # whenever placeholder counts diverged from scheduled counts.
        # num_in_flight_tokens already includes any undrained stale share.
        # Assign instead of accumulating so callers that fenced the same
        # rollover before entering this helper do not count it twice.
        session.num_stale_output_tokens = int(getattr(session, "num_in_flight_tokens", 0) or 0)
        session.num_output_placeholders = 0
        session.spec_token_ids = []
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = new_prompt
        session.additional_information = update.additional_information or None
        session.model_intermediate_buffer = getattr(
            update,
            "model_intermediate_buffer",
            None,
        )
        session.update_block_hashes()
        session.num_prompt_tokens = len(new_prompt)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
            self._enqueue_waiting_request(session)
        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _release_replaced_streaming_prompt_cache(self, session: Request) -> None:
        """Discard cache state that belongs to a replaced prompt."""
        # A prompt replacement is not a normal streaming extension: none of
        # the old KV blocks or encoder state is valid for the new prompt. Use
        # the scheduler's block-free path so an in-flight GPU step is fenced
        # correctly before the blocks return to the pool.
        self._free_request_blocks(session)
        self.encoder_cache_manager.free(session)
        getattr(self, "_inflight_prefills", set()).discard(session)

    def _reset_ready_async_chunk_replacements(self) -> None:
        """Release stale cache state after an async-chunk prompt rollover."""
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None:
            return
        replaced_ids = getattr(adapter, "replaced_streaming_prompt_ids", None)
        ready_ids = getattr(adapter, "requests_with_ready_chunks", None)
        if not replaced_ids or not ready_ids:
            return

        for request_id in tuple(replaced_ids & ready_ids):
            request = self.requests.get(request_id)
            if request is None:
                replaced_ids.discard(request_id)
                continue
            # The streaming update may already have fenced this same in-flight
            # frame. Seed idempotently so the replacement does not count it twice.
            request.num_stale_output_tokens = int(getattr(request, "num_in_flight_tokens", 0) or 0)
            request.num_output_placeholders = 0
            request.spec_token_ids = []
            self._release_replaced_streaming_prompt_cache(request)
            watermark = getattr(adapter, "requests_num_chunks_sent", None)
            if watermark is not None:
                watermark.pop(request.external_req_id, None)
            # Consume this marker after the one-time cache reset. The separate
            # ready-chunk marker remains until scheduler admission succeeds.
            replaced_ids.discard(request_id)

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        """Drain ``self._latest_omni_connector_output`` into the coordinator.

        Called at the top of every ``schedule()`` cycle.  Identical between
        AR and generation schedulers except for the ``model_mode`` argument
        forwarded to ``update_request_metadata``.
        """
        connector_output = getattr(self, "_latest_omni_connector_output", None)
        self._latest_omni_connector_output = None
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
            )
        input_coordinator.process_pending_full_payload_inputs(
            self.waiting,
            connector_output.stage_recv_req_ids if connector_output else set(),
        )

    def _process_pending_omni_inputs(self, model_mode: str) -> None:
        """Apply pending connector inputs, timeouts, and async chunks."""
        self._consume_pending_connector_output(model_mode)
        self._process_pending_input_timeouts()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting,
                self.running,
                scheduler_requests=self.requests,
            )
            self._reset_ready_async_chunk_replacements()
            self._process_pending_chunk_timeouts()
            self._log_failed_chunk_sends()

    def _restore_omni_wait_queues(self) -> None:
        """Restore requests temporarily parked by Omni input gates."""
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.restore_queues(
                self.waiting,
                self.running,
                scheduler_requests=self.requests,
            )
        if self.input_coordinator:
            self.input_coordinator.restore_queues(self.waiting)

    def _process_pending_input_timeouts(self) -> None:
        """Force-fail requests waiting on the full-payload coordinator too long.

        Called at the top of every ``schedule()`` cycle, right after
        ``_consume_pending_connector_output``.  Without this hook, a request
        whose producer dropped a payload would sit in the
        full-payload-input wait state indefinitely (the runner mixin
        protects ``_pending_load_reqs`` from prune sweeps).

        Reads ``_waiting_since`` timestamps maintained by the input
        coordinator and delegates to the base scheduler's
        ``finish_requests`` to mark expired requests FINISHED_ERROR.
        Disabled when ``DEFAULT_INPUT_WAIT_TIMEOUT_S`` is <= 0.

        Scope: only covers ``input_coordinator`` (full-payload path).
        Async-chunk requests park in ``chunk_transfer_adapter`` instead and
        are handled by ``_process_pending_chunk_timeouts`` below, which shares
        this timeout.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        timed_out_ids = input_coordinator.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after waiting > %.0fs for connector input: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _log_failed_chunk_sends(self) -> None:
        """Surface chunks the sender gave up on (R1.2 of #4855).

        ``save_loop`` pops a task with ``popleft()`` and never re-queues it, and
        ``connector.put`` can report failure without raising, so a dropped chunk
        is a give-up rather than a retry. Until this change the only trace was a
        warning that printed ``None`` for the request id, because the task dict
        has no ``request_id`` key.

        This only reports. Failing the request from here does not work: the
        producer is not the ``final_output`` stage, so an abort synthesized by
        ``finish_requests`` is not turned into a client-visible termination by
        ``Orchestrator._route_output`` (see the ``final_output`` guards there).
        Making a producer-side drop client-visible needs an orchestrator-side
        path like ``_fail_request_dead_stage``; the consumer's R1.1 deadline is
        what actually ends the request today.
        """
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None:
            return
        failures = adapter.collect_failed_send_request_ids()
        for request_id, reason in failures.items():
            if request_id in self.requests:
                logger.error(
                    "[OmniScheduler] req=%s: chunk send gave up (%s); the consumer will "
                    "fail on the stage-input deadline",
                    request_id,
                    reason,
                )
            else:
                # Drained but no longer scheduled here: the request finished or
                # was aborted between the failed send and this sweep. Still say
                # so -- collect_* empties the map, so staying quiet would make
                # this the one drop path in a change whose point is visibility.
                logger.warning(
                    "[OmniScheduler] req=%s: chunk send gave up (%s) after the request "
                    "left this scheduler (finished or aborted)",
                    request_id,
                    reason,
                )

    def _process_pending_chunk_timeouts(self) -> None:
        """Force-fail requests stalled in ``WAITING_FOR_CHUNK`` too long.

        The async-chunk counterpart of ``_process_pending_input_timeouts``,
        called right after ``chunk_transfer_adapter.process_pending_chunks``.
        Until now the chunk path had no safety net at all: the scheduler-side
        comment on the full-payload net said a similar guard "belongs in the
        chunk adapter", while the worker mixin assumed the full-payload net
        already covered it.  Only the first was true, so a dropped terminal
        chunk or a producer that exhausted its send retries parked the request
        forever (vllm-project/vllm-omni#3833).  Since ``async_chunk: true`` is
        the default for the flagship TTS pipelines, that was the default
        streaming path.

        Shares ``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S`` with the full-payload net:
        one knob for "how long may a request wait for stage input", regardless
        of which transport carries it.  Disabled when the value is <= 0.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is None or not getattr(adapter, "receives_chunks", False):
            return
        timed_out_ids = adapter.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after stalling > %.0fs waiting for a chunk: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _capture_omni_connector_output(self, model_runner_output: Any) -> None:
        """Stash the model runner's omni_connector_output for next schedule().

        Called at the tail of every ``update_from_output()`` -- identical
        between AR and generation schedulers.  Only stashes the output;
        applying the metadata is the responsibility of
        ``_consume_pending_connector_output()`` at the start of the next
        ``schedule()`` cycle.  Applying it twice (once here, once on
        consume) is unsafe under ``update_request_metadata`` in
        generation mode, which resets ``prompt_token_ids`` /
        ``_output_token_ids`` / ``num_computed_tokens`` and would
        clobber any progress between the two calls.
        """
        omni_output = getattr(model_runner_output, "omni_connector_output", None)
        if omni_output is None:
            return
        self._latest_omni_connector_output = omni_output

    def _wrap_omni_scheduler_output(
        self,
        base: SchedulerOutput,
        *,
        finished_requests_needing_kv_transfer: dict | None = None,
        pending_input_registrations: list[OmniChunkRecvHandle] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_input_registrations is None:
            pending_input_registrations = input_coordinator.pending_input_registrations if input_coordinator else []
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_input_registrations=pending_input_registrations,
        )

    def _rewrap_scheduled_new_reqs(self, scheduler_output: SchedulerOutput) -> None:
        """Attach Omni payloads without reconstructing existing Omni entries."""
        scheduler_output.scheduled_new_reqs = [  # type: ignore[assignment]
            data
            if isinstance(data, OmniNewRequestData)
            else OmniNewRequestData.from_base(data, self.requests.get(data.req_id))
            for data in scheduler_output.scheduled_new_reqs
        ]

    def _postprocess_omni_schedule_output(
        self,
        scheduler_output: SchedulerOutput,
        *,
        include_cached_payloads: bool = False,
    ) -> None:
        """Enrich new requests and apply async-chunk output bookkeeping."""
        self._rewrap_scheduled_new_reqs(scheduler_output)
        if not self.chunk_transfer_adapter:
            return
        if include_cached_payloads:
            self.chunk_transfer_adapter.postprocess_scheduler_output(
                scheduler_output,
                self.requests,
            )
        else:
            self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output)

    def _make_omni_engine_output(
        self,
        request: Request,
        *,
        new_token_ids: list[int],
        finish_reason: Any = None,
        new_logprobs: Any = None,
        new_prompt_logprobs_tensors: Any = None,
        pooling_output: Any = None,
        multimodal_output: Any = None,
        stop_reason: Any = None,
        prefill_stats: Any = None,
        kv_transfer_params: Any = None,
        routed_experts: Any = None,
        num_nans_in_logits: int = 0,
        is_segment_finished: bool | None = False,
        new_prompt_len_snapshot: int | None = None,
    ) -> OmniEngineCoreOutput:
        """Build the common request-output envelope used by LLM schedulers."""
        return OmniEngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=new_token_ids,
            finish_reason=finish_reason,
            new_logprobs=new_logprobs,
            new_prompt_logprobs_tensors=new_prompt_logprobs_tensors,
            pooling_output=pooling_output,
            multimodal_output=multimodal_output,
            stop_reason=stop_reason,
            events=request.take_events(),
            prefill_stats=prefill_stats,
            kv_transfer_params=kv_transfer_params,
            trace_headers=request.trace_headers,
            routed_experts=routed_experts,
            num_nans_in_logits=num_nans_in_logits,
            is_segment_finished=is_segment_finished,
            new_prompt_len_snapshot=new_prompt_len_snapshot,
        )

    def _append_request_output(
        self,
        outputs: dict[int, list[EngineCoreOutput]],
        request: Request,
        **output_fields: Any,
    ) -> None:
        outputs[request.client_index].append(
            OmniSchedulerMixin._make_omni_engine_output(
                self,
                request,
                **output_fields,
            )
        )

    def _handle_failed_kv_load_outputs(
        self,
        failed_request_ids: set[str] | None,
        outputs: dict[int, list[EngineCoreOutput]],
    ) -> list[Request]:
        """Finish unrecoverable KV loads and emit their terminal outputs."""
        if not failed_request_ids or self.recompute_kv_load_failures:
            return []
        requests = [self.requests[req_id] for req_id in failed_request_ids]
        self.finish_requests(failed_request_ids, RequestStatus.FINISHED_ERROR)
        for request in requests:
            OmniSchedulerMixin._append_request_output(
                self,
                outputs,
                request,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
            )
        return requests

    def _attach_finished_request_sets(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        *,
        synthesize_abort_outputs: bool,
    ) -> None:
        """Attach finished IDs while keeping AR's synthetic-abort policy explicit."""
        finished_req_ids = self.finished_req_ids_dict
        if not finished_req_ids:
            return
        for client_index, finished_set in finished_req_ids.items():
            output = engine_core_outputs.get(client_index)
            if output is None:
                output = EngineCoreOutputs()
                engine_core_outputs[client_index] = output
            if synthesize_abort_outputs:
                emitted = {item.request_id for item in output.outputs}
                output.outputs.extend(
                    EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT)
                    for req_id in finished_set
                    if req_id not in emitted
                )
            output.finished_requests = finished_set
        finished_req_ids.clear()

    def _remove_stopped_requests_from_queues(
        self,
        stopped_running_reqs: set[Request],
        stopped_preempted_reqs: set[Request],
    ) -> None:
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

    def _aggregate_kv_connector_stats(
        self,
        kv_connector_output: Any,
    ) -> KVConnectorStats | None:
        stats = kv_connector_output.kv_connector_stats if kv_connector_output else None
        if self.connector:
            scheduler_stats = self.connector.get_kv_connector_stats()
            if scheduler_stats is not None and not scheduler_stats.is_empty():
                stats = stats.aggregate(scheduler_stats) if stats is not None else scheduler_stats
        return stats

    def _publish_kv_cache_events(self) -> None:
        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)
        if events:
            self.kv_event_publisher.publish(KVEventBatch(ts=time.time(), events=events))

    def _attach_scheduler_stats(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        spec_decoding_stats: SpecDecodingStats | None,
        kv_connector_stats: KVConnectorStats | None,
        cudagraph_stats: CUDAGraphStat | None,
        perf_stats: PerfStats | None,
    ) -> None:
        stats = self.make_stats(
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )
        if stats is None:
            return
        if (output := next(iter(engine_core_outputs.values()), None)) is None:
            engine_core_outputs[0] = output = EngineCoreOutputs()
        output.scheduler_stats = stats

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        """Finish requests and clean all Omni-owned queue/coordinator state."""
        if isinstance(request_ids, str):
            target_request_ids = {request_ids}
        elif request_ids is None:
            target_request_ids = set(self.requests)
        else:
            if isinstance(request_ids, Iterator):
                request_ids = tuple(request_ids)
            target_request_ids = set(request_ids)

        # Capture the final logical peak/output before upstream frees KV block
        # tables. Requests never observed in M2 shadow mode are harmless no-ops.
        for request_id in target_request_ids:
            request = self.requests.get(request_id)
            if request is None:
                continue
            self._finish_resource_observation(request)

        skipped_waiting_ids = {r.request_id for r in getattr(self, "skipped_waiting", ())}
        pre_adapter_streaming_wait_ids = {
            rid
            for rid in target_request_ids & skipped_waiting_ids
            if (req := self.requests.get(rid)) is not None
            and (
                req.status == RequestStatus.WAITING_FOR_STREAMING_REQ
                or (getattr(req, "resumable", False) and req.status == RequestStatus.FINISHED_STOPPED)
            )
        }

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        self._realign_request_status_to_queues(
            request_ids,
            pre_adapter_streaming_wait_ids=pre_adapter_streaming_wait_ids,
        )
        finished = super().finish_requests(request_ids, finished_status)
        self._purge_finished_from_running(target_request_ids)

        for request in finished:
            self._free_input_coordinator_request(request.request_id)
        return finished

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        now = time.monotonic()
        if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
            return None
        self._last_stats_time = now
        return super().make_stats(*args, **kwargs)

    def _realign_request_status_to_queues(
        self,
        request_ids: str | Iterable[str] | None,
        *,
        pre_adapter_streaming_wait_ids: set[str] | None = None,
    ) -> None:
        """Realign ``request.status`` to actual queue membership.

        ``OmniChunkTransferAdapter._process_chunk_queue`` stamps
        ``requests_origin_status[req.id] = WAITING`` (or ``RUNNING``) when
        first parking a request in a chunk-transfer deque. On the next
        tick, when the chunk arrives, ``_process_chunk_queue`` sets
        ``request.status = target_status`` and continues, but
        ``requests_origin_status`` is left at its first-park value -- no
        hook updates it on the ``waiting → running`` admit transition
        that ``super().schedule()`` later performs. The table stays
        stale until the request makes another deque round-trip.

        If an abort lands in the gap between admit and the next deque
        round-trip, ``chunk_transfer_adapter.finish_requests`` reads the
        stale ``WAITING`` from ``requests_origin_status``, stomps it
        onto ``request.status``, and the upstream
        ``Scheduler.finish_requests`` else branch silently fails to
        remove from ``self.running`` -- the request stays alive in
        ``self.running`` and the worker's ``input_batch`` slot leaks.
        After ``max_num_seqs`` such aborts every new request hangs at
        ``chunks=0`` until the client times out.

        Realign here: if a request lives in ``self.running`` but its
        status is not ``RUNNING``, set it to ``RUNNING``; symmetrically
        flip ``RUNNING → WAITING`` when the request is actually in
        ``self.waiting``. A resumable segment stop still held in
        ``self.skipped_waiting`` is restored to
        ``WAITING_FOR_STREAMING_REQ`` so upstream also balances its
        paused-session counter. Because adapter cleanup may restore a
        connector-owned request's prior status first, ``finish_requests``
        snapshots these counter-bearing rows before invoking the adapter.
        This is a localized safety net for
        ``requests_origin_status`` staleness on the admit transition;
        it does not touch the adapter's invariants and is complementary
        to the chunk-transfer-adapter deque purge that already runs
        inside ``process_pending_chunks`` / ``restore_queues``.

        Note on scope: only the ``async_chunk`` path actually triggers
        the ``requests_origin_status`` staleness this helper repairs.
        When ``async_chunk`` is disabled, no chunk-transfer round-trip
        occurs between admit and finish, so the realignment walk is a
        cheap O(n) no-op over an already-aligned set. The call is kept
        unconditional in ``finish_requests`` to (a) keep the abort path
        uniform and (b) defend any future configuration that re-enables
        chunk transfer from rediscovering the same regression.

        See https://github.com/vllm-project/vllm-omni/pull/3774 and the
        residual-hang reproduction discussed in that PR.
        """
        # Mirror the upstream Scheduler.finish_requests resolution of
        # ``request_ids`` so realignment touches exactly the set that
        # ``super().finish_requests`` will then walk.
        if isinstance(request_ids, str):
            ids_to_align: Iterable[str] = (request_ids,)
        elif request_ids is None:
            ids_to_align = list(self.requests.keys())
        else:
            ids_to_align = list(request_ids)

        if not ids_to_align:
            return

        running_ids = {r.request_id for r in self.running}
        waiting_ids = {r.request_id for r in self.waiting}
        skipped_waiting_ids = {r.request_id for r in getattr(self, "skipped_waiting", ())}

        for rid in ids_to_align:
            req = self.requests.get(rid)
            if req is None:
                continue
            # A persistent Session may be closed after its current segment
            # reached FINISHED_STOPPED but while it is still owned by a live
            # scheduler/connector queue. vLLM skips already-finished requests
            # in finish_requests(), leaking the KV and worker slot. Only
            # recover requests with positive queue ownership; an off-queue
            # terminal can legitimately be waiting for deferred block free.
            resumable_segment_stop = bool(
                getattr(req, "resumable", False) and req.status == RequestStatus.FINISHED_STOPPED
            )
            streaming_wait = (
                rid in pre_adapter_streaming_wait_ids
                if pre_adapter_streaming_wait_ids is not None
                else resumable_segment_stop
            )
            if req.is_finished() and not (resumable_segment_stop or streaming_wait):
                continue
            if rid in skipped_waiting_ids and streaming_wait:
                req.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            elif rid in running_ids and req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING
            elif rid in waiting_ids and (req.status == RequestStatus.RUNNING or resumable_segment_stop):
                req.status = RequestStatus.WAITING

    def _purge_finished_from_running(self, target_request_ids: set[str] | None = None) -> None:
        """Defensive post-finish sweep of ``self.running``.

        Belt-and-suspenders to ``_realign_request_status_to_queues``:
        even after status realignment lets upstream
        ``Scheduler.finish_requests`` pick the right removal branch,
        a future regression or an unexpected ``status`` mid-transition
        could still leave already-finished entries in ``self.running``.
        Sweeping here guarantees the worker's ``input_batch`` slot is
        not pinned by a freed request.

        Complementary to ``_realign_request_status_to_queues``: realign
        is preventive (fix ``status`` before ``super().finish_requests``
        so the right branch fires); this purge is defensive (sweep the
        residue after ``super().finish_requests`` so any stale entries
        are reclaimed).

        A resumable ``FINISHED_STOPPED`` request may legitimately remain
        in ``self.running`` between realtime segments. Preserve such a
        request unless it belongs to this finish call. Other finished or
        untracked entries are stale and can be swept defensively.

        When ``target_request_ids`` is ``None`` or empty, every resumable
        ``FINISHED_STOPPED`` entry is preserved. Production
        ``finish_requests`` always passes its resolved finish set.

        In-place via ``self.running[:] = ...`` for minor consistency
        with idiomatic vLLM scheduler mutation; upstream
        ``Scheduler.finish_requests`` itself rebinds ``self.running``,
        so list identity across the whole call is not preserved -- the
        slice form is just to avoid an extra rebind inside this helper.

        Assumes the upstream V1 invariant that scheduler ticks are
        serialized on a single thread; in-place mutation here is no more
        racy than the rest of the scheduler under that assumption.

        See https://github.com/vllm-project/vllm-omni/pull/3774
        discussion.
        """
        if not self.running:
            return
        target_request_ids = target_request_ids or set()

        def keep_running(req: Request) -> bool:
            if req.request_id not in self.requests:
                return False
            if not req.is_finished():
                return True
            resumable_segment_stop = bool(
                getattr(req, "resumable", False) and req.status == RequestStatus.FINISHED_STOPPED
            )
            return resumable_segment_stop and req.request_id not in target_request_ids

        self.running[:] = [req for req in self.running if keep_running(req)]
