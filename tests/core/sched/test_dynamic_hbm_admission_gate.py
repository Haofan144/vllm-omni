# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.core.memory_coordinator import DynamicHBMConfig, SafetyState
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    def __init__(self, *, cap: int = 16, tokens: int = 4096, running: int = 0, config=None):
        self.max_num_running_reqs = cap
        self.max_num_scheduled_tokens = tokens
        self.running = [object()] * running
        self.policy = SchedulingPolicy.FCFS
        self.waiting = []
        self.num_waiting_for_streaming_input = 0
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                stage_id=0,
                dynamic_hbm=config or {"enabled": True},
                async_chunk=False,
                stage_pipeline_config=SimpleNamespace(execution_type=StageExecutionType.LLM_AR),
            )
        )
        self._init_dynamic_hbm_scheduling_state()


def _apply(scheduler: _Scheduler, *, generation: int, cap: int, state: SafetyState, report=1) -> bool:
    pressure = 0.96 if state is SafetyState.CRITICAL else 0.92
    return scheduler.apply_stage_budget_decision(
        generation=generation,
        based_on_report_generation=report,
        effective_max_num_seqs=cap,
        pressure=pressure,
        reason=state.value,
        safety_state=state.value,
        pressure_source="physical_hbm",
    )


def test_scheduler_initializes_independent_safety_cap() -> None:
    scheduler = _Scheduler()
    assert scheduler._configured_max_num_seqs == 16
    assert scheduler._safety_max_num_seqs == 16
    assert scheduler._effective_max_num_seqs == 16
    assert scheduler._dynamic_hbm_safety_state is SafetyState.NORMAL


def test_high_decision_updates_only_safety_budget() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=8, state=SafetyState.HIGH_PRESSURE)
    assert scheduler._configured_max_num_seqs == 16
    assert scheduler._safety_max_num_seqs == 8
    assert scheduler._effective_max_num_seqs == 8
    assert scheduler._effective_max_num_scheduled_tokens == 2048


def test_critical_state_clamps_inconsistent_wire_cap_to_zero() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=4, state=SafetyState.CRITICAL)
    assert scheduler._safety_max_num_seqs == 0
    assert scheduler._effective_max_num_seqs == 0
    assert scheduler._effective_max_num_scheduled_tokens == 1
    assert not scheduler._dynamic_hbm_allows_new_admission()


def test_running_requests_are_retained_in_execution_budget() -> None:
    scheduler = _Scheduler(running=6)
    _apply(scheduler, generation=1, cap=0, state=SafetyState.CRITICAL)
    assert scheduler._effective_max_num_seqs == 0
    assert scheduler._dynamic_max_num_running_reqs() == 6
    assert scheduler._effective_max_num_scheduled_tokens == 1536
    assert len(scheduler.running) == 6


def test_admission_resumes_only_below_effective_cap() -> None:
    scheduler = _Scheduler(running=2)
    _apply(scheduler, generation=1, cap=2, state=SafetyState.HIGH_PRESSURE)
    assert not scheduler._dynamic_hbm_allows_new_admission()
    scheduler.running.pop()
    assert scheduler._dynamic_hbm_allows_new_admission()


def test_stale_decision_and_unknown_state_are_rejected() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=2, cap=8, state=SafetyState.HIGH_PRESSURE, report=5)
    assert not _apply(scheduler, generation=1, cap=16, state=SafetyState.NORMAL, report=6)
    assert not scheduler.apply_stage_budget_decision(
        generation=3,
        based_on_report_generation=6,
        effective_max_num_seqs=16,
        pressure=0.5,
        reason="invalid",
        safety_state="not-a-state",
    )
    assert scheduler._effective_max_num_seqs == 8


def test_older_report_cannot_expand_cap() -> None:
    scheduler = _Scheduler()
    assert _apply(scheduler, generation=1, cap=4, state=SafetyState.HIGH_PRESSURE, report=5)
    assert not _apply(scheduler, generation=2, cap=16, state=SafetyState.NORMAL, report=4)
    assert scheduler._effective_max_num_seqs == 4


def test_local_kv_guard_does_not_consume_central_generation() -> None:
    scheduler = _Scheduler()
    assert scheduler.apply_dynamic_hbm_local_kv_guard()
    assert scheduler._last_budget_generation == 0
    assert scheduler._dynamic_hbm_safety_state is SafetyState.CRITICAL
    assert scheduler._effective_max_num_seqs == 0
    assert _apply(scheduler, generation=1, cap=1, state=SafetyState.RECOVERING)
    assert scheduler._effective_max_num_seqs == 1


@pytest.mark.parametrize("fail_closed", [True, False])
def test_disconnect_guard_respects_policy(fail_closed) -> None:
    scheduler = _Scheduler(config={"enabled": True, "fail_closed_on_disconnect": fail_closed})
    applied = scheduler.apply_dynamic_hbm_disconnect_guard()
    assert applied is fail_closed
    expected = 0 if fail_closed else 16
    assert scheduler._effective_max_num_seqs == expected
    assert scheduler._last_budget_generation == 0


def test_token_scaling_can_be_disabled() -> None:
    scheduler = _Scheduler(config={"enabled": True, "scale_token_budget": False})
    _apply(scheduler, generation=1, cap=4, state=SafetyState.HIGH_PRESSURE)
    assert scheduler._effective_max_num_scheduled_tokens == 4096


class _BlockPool:
    def __init__(self, free_blocks: int):
        self.free_blocks = free_blocks

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


class _WaitingRequest:
    request_id = "request-0"
    num_prompt_tokens = 64
    max_tokens = 32
    num_computed_tokens = 0
    num_output_tokens = 0


def _resource_scheduler(mode: str, free_blocks: int) -> _Scheduler:
    scheduler = _Scheduler(
        config={"enabled": True, "resource_admission_mode": mode},
    )
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=_BlockPool(free_blocks))
    scheduler.waiting = [_WaitingRequest()]
    return scheduler


def test_resource_admission_defaults_to_shadow() -> None:
    scheduler = _resource_scheduler("shadow", free_blocks=2)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.shadow_would_defer
    assert scheduler._resource_admission_counts["shadow_would_defer"] == 1


def test_resource_admission_can_be_explicitly_enforced() -> None:
    scheduler = _resource_scheduler("enforce", free_blocks=2)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert not decision.allowed
    assert decision.reason.value == "kv_peak_risk"


def test_resource_admission_off_is_noop() -> None:
    scheduler = _resource_scheduler("off", free_blocks=0)
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.reason.value == "disabled"


def test_resource_admission_declines_on_non_ar_execution_type() -> None:
    # A Code2Wav/decoder-style LLM_GENERATION stage still exposes
    # num_prompt_tokens/max_tokens on its Request objects (every vLLM Request
    # has them), but those hold codec-frame placeholder counts, not text
    # tokens. The AR estimator must decline rather than silently misapply KV
    # accounting to them.
    scheduler = _resource_scheduler("enforce", free_blocks=2)
    scheduler.vllm_config.model_config.stage_pipeline_config.execution_type = (
        StageExecutionType.LLM_GENERATION
    )
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.reason.value == "unsupported_execution_type"


def test_resource_admission_declines_when_execution_type_unknown() -> None:
    # Fail closed: a stage whose execution type cannot be determined at all
    # (e.g. a minimal/legacy model_config missing stage_pipeline_config) must
    # not be assumed AR by default.
    scheduler = _resource_scheduler("enforce", free_blocks=2)
    del scheduler.vllm_config.model_config.stage_pipeline_config
    decision = scheduler._dynamic_hbm_resource_admission_decision()
    assert decision.allowed
    assert decision.reason.value == "unsupported_execution_type"


class _WaitingRequestWithStalePrefillStats(_WaitingRequest):
    """A still-waiting request that already carries a populated
    ``prefill_stats`` (e.g. left over from a prior preemption). Real
    prefix-cache hits for the *next* scheduling attempt are only computed by
    ``KVCacheManager.get_computed_blocks`` when vLLM's own scheduler actually
    schedules the request — a side-effecting call this pre-scheduling
    admission check must not trigger. This value must never be read as if it
    reflected the upcoming attempt's reuse."""

    prefill_stats = SimpleNamespace(
        num_local_cached_tokens=48, num_external_cached_tokens=16
    )


def test_resource_context_ignores_prefill_stats_before_real_scheduling() -> None:
    scheduler = _resource_scheduler("shadow", free_blocks=1_000)
    request = _WaitingRequestWithStalePrefillStats()
    scheduler.waiting = [request]
    context = scheduler._ar_resource_context(request)
    assert context is not None
    assert context.reusable_cached_tokens == 0


class _SizedRequest:
    """A waiting request whose block_size=16 KV cost is
    ceil((num_prompt_tokens + max_tokens) / 16) blocks (no profile loaded,
    so the estimator falls back to the hard max_tokens bound)."""

    num_computed_tokens = 0
    num_output_tokens = 0

    def __init__(self, request_id: str, *, prompt_tokens: int, max_tokens: int):
        self.request_id = request_id
        self.num_prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens


def _bypass_scheduler(
    *, free_blocks: int, scan_limit: int, aging_ms: float = 30_000.0
) -> _Scheduler:
    scheduler = _Scheduler(
        config={
            "enabled": True,
            "resource_admission_mode": "enforce",
            "resource_admission_bypass_scan_limit": scan_limit,
            "resource_admission_bypass_aging_ms": aging_ms,
        },
    )
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=_BlockPool(free_blocks))
    return scheduler


def test_bypass_scan_limit_zero_returns_empty_queue() -> None:
    # scan_limit=0 is the default and must reproduce the pre-bypass
    # all-or-nothing behavior: the returned queue is always empty regardless
    # of whether later requests would fit.
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=0)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    small = _SizedRequest("small", prompt_tokens=8, max_tokens=8)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(small)
    bypassed = scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert list(bypassed) == []


def test_bypass_admits_a_later_request_that_fits_past_an_oversized_head() -> None:
    # 8 free blocks: "huge" needs far more than that and cannot fit; "small"
    # needs 1 block and fits. The head-of-line request is always included
    # (so its own admission decision still runs), and the fitting later
    # request should be pulled into the same step's candidate queue.
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=5)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    small = _SizedRequest("small", prompt_tokens=8, max_tokens=8)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(small)
    bypassed = scheduler._dynamic_hbm_bounded_bypass_waiting()
    ids = [req.request_id for req in bypassed]
    assert ids == ["huge", "small"]
    assert scheduler._resource_admission_bypass_count == 1
    assert scheduler._resource_admission_bypassed_requests == 1


def test_bypass_does_not_admit_a_later_request_that_also_does_not_fit() -> None:
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=5)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    also_huge = _SizedRequest("also_huge", prompt_tokens=1000, max_tokens=1000)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(also_huge)
    bypassed = scheduler._dynamic_hbm_bounded_bypass_waiting()
    ids = [req.request_id for req in bypassed]
    assert ids == ["huge"]
    assert scheduler._resource_admission_bypass_count == 0


def test_bypass_respects_scan_limit() -> None:
    # scan_limit=1: only the first candidate past the head is scanned, even
    # though the second candidate past the head would also fit.
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=1)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    also_huge = _SizedRequest("also_huge", prompt_tokens=1000, max_tokens=1000)
    small = _SizedRequest("small", prompt_tokens=8, max_tokens=8)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(also_huge)
    scheduler.waiting.add_request(small)
    bypassed = scheduler._dynamic_hbm_bounded_bypass_waiting()
    ids = [req.request_id for req in bypassed]
    assert ids == ["huge"]


def test_bypass_scan_does_not_pollute_admission_counts() -> None:
    # Candidates scanned past the head are evaluated with record=False so
    # the step's admission-decision telemetry still reflects only the real
    # head-of-line decision, not every candidate probed during the scan.
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=5)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    small = _SizedRequest("small", prompt_tokens=8, max_tokens=8)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(small)
    scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert sum(scheduler._resource_admission_counts.values()) == 0


def test_bypass_aging_stops_after_threshold(monkeypatch) -> None:
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=5, aging_ms=1.0)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    small = _SizedRequest("small", prompt_tokens=8, max_tokens=8)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler.waiting.add_request(small)

    times = iter([1000.0, 1000.05])

    def fake_monotonic():
        return next(times)

    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_scheduler_mixin.time.monotonic", fake_monotonic
    )
    # First call seeds head_of_line_since["huge"] = 1000.0 and, since
    # (now - started_at) == 0 < aging_ms, still bypasses "small".
    first = scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert [req.request_id for req in first] == ["huge", "small"]
    # Second call: 50ms later >= aging_ms=1ms, so aging stops the bypass scan
    # for this step even though "small" would still fit.
    second = scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert [req.request_id for req in second] == ["huge"]
    assert scheduler._resource_admission_aging_stops == 1


def test_bypass_clears_head_of_line_state_once_request_leaves_queue() -> None:
    scheduler = _bypass_scheduler(free_blocks=8, scan_limit=5)
    huge = _SizedRequest("huge", prompt_tokens=1000, max_tokens=1000)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(huge)
    scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert "huge" in scheduler._resource_admission_head_of_line_since

    # "huge" is gone next step (admitted, finished, or aborted elsewhere);
    # its aging timer must not leak.
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.waiting.add_request(
        _SizedRequest("other", prompt_tokens=8, max_tokens=8)
    )
    scheduler._dynamic_hbm_bounded_bypass_waiting()
    assert "huge" not in scheduler._resource_admission_head_of_line_since


def test_default_workload_classifier_is_ar() -> None:
    scheduler = _Scheduler()
    assert type(scheduler._ar_workload_classifier).__name__ == "ARWorkloadClassifier"


def test_ar_workload_classifier_config_resolves_configured_classifier() -> None:
    scheduler = _Scheduler(
        config={
            "enabled": True,
            "ar_workload_classifier": (
                "vllm_omni.core.memory_coordinator.tts_resource_estimator."
                "TTSWorkloadClassifier"
            ),
        }
    )
    assert type(scheduler._ar_workload_classifier).__name__ == "TTSWorkloadClassifier"


class _RequestWithAdditionalInformation(_WaitingRequest):
    def __init__(self, additional_information):
        self.additional_information = additional_information


def test_tts_classify_extra_kwargs_unwraps_list_convention() -> None:
    request = _RequestWithAdditionalInformation(
        {"task_type": ["CustomVoice"], "ref_audio": [["wav", 24000]]}
    )
    extra = OmniSchedulerMixin._tts_classify_extra_kwargs(request)
    assert extra == {"task_type": "CustomVoice", "ref_audio_present": True}


def test_tts_classify_extra_kwargs_detects_ref_audio_via_code_length_only() -> None:
    # A precomputed-profile request may carry only ref_code_length (no raw
    # ref_audio waveform) -- see serving_speech.py's precomputed_speakers path.
    request = _RequestWithAdditionalInformation(
        {"task_type": ["Base"], "ref_code_length": [42]}
    )
    extra = OmniSchedulerMixin._tts_classify_extra_kwargs(request)
    assert extra == {"task_type": "Base", "ref_audio_present": True}


def test_tts_classify_extra_kwargs_defaults_when_absent() -> None:
    request = _WaitingRequest()  # no additional_information attribute at all
    extra = OmniSchedulerMixin._tts_classify_extra_kwargs(request)
    assert extra == {}


def test_tts_classify_extra_kwargs_defaults_when_fields_missing() -> None:
    request = _RequestWithAdditionalInformation({})
    extra = OmniSchedulerMixin._tts_classify_extra_kwargs(request)
    assert extra == {"task_type": None, "ref_audio_present": False}


def test_tts_classify_extra_kwargs_tolerates_non_list_value() -> None:
    # Not every model wraps every field in a list (e.g. a differently-shaped
    # additional_information from a non-Qwen3-TTS model) -- a bare scalar
    # must not crash the extraction.
    request = _RequestWithAdditionalInformation({"task_type": "CustomVoice"})
    extra = OmniSchedulerMixin._tts_classify_extra_kwargs(request)
    assert extra == {"task_type": "CustomVoice", "ref_audio_present": False}


class _Code2WavRequest:
    def __init__(self, request_id: str, num_frames: int, *, resumable: bool = False):
        self.request_id = request_id
        self.prompt_token_ids = [1] * num_frames
        self.resumable = resumable
        self.streaming_queue = None


def _generation_scheduler(execution_type=None) -> _Scheduler:
    scheduler = _Scheduler(config={"enabled": True})
    if execution_type is not None:
        scheduler.vllm_config.model_config.stage_pipeline_config.execution_type = execution_type
    return scheduler


def test_code2wav_estimator_applies_only_to_llm_generation() -> None:
    ar_scheduler = _generation_scheduler(StageExecutionType.LLM_AR)
    assert not ar_scheduler._dynamic_hbm_code2wav_estimator_applies()
    generation_scheduler = _generation_scheduler(StageExecutionType.LLM_GENERATION)
    assert generation_scheduler._dynamic_hbm_code2wav_estimator_applies()


def test_code2wav_estimator_applies_fails_closed_when_execution_type_unknown() -> None:
    scheduler = _generation_scheduler()
    del scheduler.vllm_config.model_config.stage_pipeline_config
    assert not scheduler._dynamic_hbm_code2wav_estimator_applies()
    assert not scheduler._dynamic_hbm_ar_estimator_applies()


def test_code2wav_resource_context_reflects_batch_composition() -> None:
    scheduler = _generation_scheduler(StageExecutionType.LLM_GENERATION)
    scheduler.running = [_Code2WavRequest("running-0", 200)]
    context = scheduler._code2wav_resource_context(_Code2WavRequest("head", 25))
    assert context is not None
    assert context.frame_count == 25
    assert context.batch_size == 2  # 1 running + the head-of-line candidate
    assert context.batch_max_frame_count == 200  # padded to the batch's longest member


def test_code2wav_resource_context_flags_persistent_state() -> None:
    scheduler = _generation_scheduler(StageExecutionType.LLM_GENERATION)
    context = scheduler._code2wav_resource_context(
        _Code2WavRequest("head", 25, resumable=True)
    )
    assert context is not None
    assert context.persistent_state_active is True


def test_sample_code2wav_observation_declines_for_ar_stage() -> None:
    scheduler = _generation_scheduler(StageExecutionType.LLM_AR)
    scheduler.waiting = [_Code2WavRequest("head", 25)]
    scheduler._sample_code2wav_resource_observation()
    assert scheduler._code2wav_observation_count == 0


def test_sample_code2wav_observation_declines_on_empty_queue() -> None:
    scheduler = _generation_scheduler(StageExecutionType.LLM_GENERATION)
    scheduler.waiting = []
    scheduler._sample_code2wav_resource_observation()
    assert scheduler._code2wav_observation_count == 0


def test_sample_code2wav_observation_records_without_gating() -> None:
    scheduler = _generation_scheduler(StageExecutionType.LLM_GENERATION)
    scheduler.waiting = [_Code2WavRequest("head", 25)]
    scheduler._sample_code2wav_resource_observation()
    assert scheduler._code2wav_observation_count == 1
    assert scheduler._code2wav_observation_errors == 0
    # Purely observational: nothing about admission/defer state exists to
    # assert on here -- there is no admit/defer decision to make.
    assert not hasattr(scheduler, "_code2wav_admission_counts")
