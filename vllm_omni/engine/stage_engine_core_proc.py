"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import asdict
from typing import Any

import vllm.v1.engine.core as _vllm_engine_core_module
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineZmqAddresses,
    SignalCallback,
)

from vllm_omni.core.memory_coordinator import RankMemoryReport, ReplicaMemoryAggregator
from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_init_utils import (
    maybe_apply_cfg_scheduler_patches,
    set_death_signal,
)

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def preprocess_add_request(self, request: OmniEngineCoreRequest) -> tuple[Any, int]:
        """Preserve omni payloads when vLLM builds its scheduler request."""
        scheduler_request, current_wave = super().preprocess_add_request(request)
        scheduler_request.additional_information = request.additional_information
        scheduler_request.external_req_id = getattr(request, "external_req_id", request.request_id)
        return scheduler_request, current_wave

    def _maybe_start_dynamic_hbm_report(self) -> None:
        config = getattr(self.scheduler, "_dynamic_hbm_config", None)
        if config is None or not config.enabled:
            return
        pending = getattr(self, "_dynamic_hbm_report_future", None)
        if pending is not None:
            return
        now = time.monotonic()
        last_sample = getattr(self, "_dynamic_hbm_last_sample_s", float("-inf"))
        if now - last_sample < config.sample_interval_ms / 1000:
            return
        self._dynamic_hbm_last_sample_s = now
        self._dynamic_hbm_report_started_s = now

        try:
            self._dynamic_hbm_report_future = self.model_executor.collective_rpc(
                "report_rank_memory",
                timeout=config.report_timeout_ms / 1000,
                non_block=True,
            )
        except Exception:
            logger.exception("[HBMCoordinator] failed to start replica rank memory report")

    def _maybe_finish_dynamic_hbm_report(self) -> None:
        future = getattr(self, "_dynamic_hbm_report_future", None)
        if future is None:
            return
        config = self.scheduler._dynamic_hbm_config
        timed_out = (
            time.monotonic() - getattr(self, "_dynamic_hbm_report_started_s", time.monotonic())
            >= config.report_timeout_ms / 1000
        )
        if not future.done() and not timed_out:
            return
        self._dynamic_hbm_report_future = None

        parallel_config = self.vllm_config.parallel_config
        expected_rank_count = parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
        stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
        replica_id = int(os.environ.get("VLLM_OMNI_REPLICA_ID", "0"))
        aggregator = getattr(self, "_dynamic_hbm_aggregator", None)
        if aggregator is None:
            aggregator = ReplicaMemoryAggregator(
                stage_id=stage_id,
                replica_id=replica_id,
                expected_rank_count=expected_rank_count,
            )
            self._dynamic_hbm_aggregator = aggregator

        rank_reports: list[RankMemoryReport] = []
        try:
            if not timed_out:
                payloads = future.result()
                rank_reports = [RankMemoryReport(**payload) for payload in payloads if payload is not None]
            else:
                future.cancel()
                logger.warning("[HBMCoordinator] rank memory report timed out")
        except Exception:
            logger.exception("[HBMCoordinator] failed to finish replica rank memory report")

        free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        total_blocks = getattr(getattr(self.scheduler, "kv_cache_config", None), "num_blocks", None)
        report = aggregator.aggregate(
            rank_reports,
            kv_total_blocks=total_blocks,
            kv_free_blocks=free_blocks,
            running_requests=len(self.scheduler.running),
            waiting_requests=len(self.scheduler.waiting),
            configured_max_num_seqs=self.scheduler._configured_max_num_seqs,
        )
        coord_client = getattr(self, "_dynamic_hbm_coord_client", None)
        if coord_client is None:
            # Standalone EngineCore remains backward compatible.
            self.scheduler.update_replica_memory_report(report)
            return
        try:
            generation = coord_client.send_memory_report(asdict(report), asdict(config))
            if generation == 1:
                logger.info(
                    "[HBMCoordinator] stage=%d replica=%d first central memory report sent ranks=%d pressure=%.4f",
                    report.stage_id,
                    report.replica_id,
                    len(report.rank_reports),
                    report.pressure,
                )
        except Exception:
            logger.exception("[HBMCoordinator] failed to send central memory report")

    def _apply_dynamic_hbm_decisions(self) -> None:
        client = getattr(self, "_dynamic_hbm_coord_client", None)
        if client is None:
            return
        try:
            decisions = client.poll_budget_decisions()
        except Exception:
            logger.exception("[HBMCoordinator] failed to receive central budget decisions")
            return
        for decision in decisions:
            if decision.instance_id != client._instance_id:
                continue
            if decision.stage_id != client._stage_id or decision.replica_id != client._replica_id:
                continue
            self.scheduler.apply_stage_budget_decision(
                generation=decision.decision_generation,
                effective_max_num_seqs=decision.effective_max_num_seqs,
                pressure=decision.pressure,
                reason=decision.reason,
                based_on_report_generation=decision.based_on_report_generation,
            )

    def step(self):
        self._apply_dynamic_hbm_decisions()
        self._maybe_finish_dynamic_hbm_report()
        self._maybe_start_dynamic_hbm_report()
        result = super().step()
        self._maybe_finish_dynamic_hbm_report()
        return result

    def step_with_batch_queue(self):
        self._apply_dynamic_hbm_decisions()
        self._maybe_finish_dynamic_hbm_report()
        self._maybe_start_dynamic_hbm_report()
        result = super().step_with_batch_queue()
        self._maybe_finish_dynamic_hbm_report()
        return result

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Patch the decoder type so process_input_sockets (started
            # during __init__) decodes OmniEngineCoreRequest (which
            # carries additional_information) instead of the base
            # EngineCoreRequest.  Must happen BEFORE __init__ because
            # the IO thread creates MsgpackDecoder(EngineCoreRequest)
            # during __init__.
            _vllm_engine_core_module.EngineCoreRequest = OmniEngineCoreRequest
            logger.debug(
                "[StageEngineCoreProc] Patched EngineCoreRequest -> OmniEngineCoreRequest: %s",
                _vllm_engine_core_module.EngineCoreRequest,
            )

            # CFG pairing scheduler patches must land before EngineCore builds
            # its Scheduler; gated on the stage's logits_processors and its
            # default sampling extra_args.
            maybe_apply_cfg_scheduler_patches(kwargs.get("vllm_config"))

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                scheduler = getattr(engine_core, "scheduler", None)
                if scheduler is None:
                    raise RuntimeError("EngineCore scheduler is not initialized")
                coord_client = create_stage_coord_client(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    replica_id=max(int(omni_replica_id), 0),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )
                engine_core._dynamic_hbm_coord_client = coord_client

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()
