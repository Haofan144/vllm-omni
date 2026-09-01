# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import (
    ARRequestResourceContext,
    ARResourceEstimator,
    ARProfileStore,
    ARWorkloadClassifier,
    AdmissionReason,
    LogicalResourceDemand,
    ProfileFingerprint,
    ResourceDimension,
    ResourceObservationCollector,
    ResourceObservationJSONLWriter,
    build_ar_output_profiles,
    evaluate_ar_kv_admission,
    read_observations_jsonl,
    write_observations_jsonl,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resource_demand_rejects_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="immediate <= expected <= quantile <= hard"):
        LogicalResourceDemand(
            immediate_kv_blocks=2,
            expected_peak_kv_blocks=1,
            quantile_peak_kv_blocks=1,
            hard_peak_kv_blocks=1,
        )


def test_unprofiled_ar_uses_explicit_hard_fallback() -> None:
    estimate = ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=64,
            max_tokens=32,
            block_size=16,
            next_scheduled_tokens=16,
        )
    )

    assert estimate.logical.immediate_kv_blocks == 1
    assert estimate.logical.expected_peak_kv_blocks == 6
    assert estimate.logical.quantile_peak_kv_blocks == 6
    assert estimate.logical.hard_peak_kv_blocks == 6
    assert estimate.provenance.fallback_reason == "output_length_profile_unavailable"
    assert ResourceDimension.TRANSIENT_BYTES not in estimate.physical.available_dimensions


def test_profiled_ar_separates_quantile_from_hard_bound() -> None:
    estimate = ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=64,
            max_tokens=128,
            block_size=16,
            expected_output_tokens=16,
            quantile_output_tokens=32,
            workload_class="ar:short",
            target_coverage=0.95,
            sample_count=100,
        )
    )

    assert estimate.logical.expected_peak_kv_blocks == 5
    assert estimate.logical.quantile_peak_kv_blocks == 6
    assert estimate.logical.hard_peak_kv_blocks == 12
    assert estimate.provenance.fallback_reason is None


def test_cached_and_allocated_capacity_reduce_incremental_peak() -> None:
    estimate = ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=128,
            max_tokens=32,
            block_size=16,
            reusable_cached_tokens=64,
            allocated_kv_blocks=2,
            expected_output_tokens=16,
            quantile_output_tokens=32,
        )
    )

    # (64 uncached prompt + 32 output) / 16 - 2 allocated = 4 new blocks.
    assert estimate.logical.quantile_peak_kv_blocks == 4
    assert estimate.logical.hard_peak_kv_blocks == 4


def test_sliding_window_caps_peak() -> None:
    estimate = ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=1024,
            max_tokens=1024,
            block_size=16,
            sliding_window_tokens=256,
        )
    )
    assert estimate.logical.hard_peak_kv_blocks == 16


def test_shadow_decision_records_counterfactual_without_blocking() -> None:
    estimate = ARResourceEstimator(16).estimate(num_prompt_tokens=64, max_tokens=32)
    shadow = evaluate_ar_kv_admission(estimate, free_kv_blocks=2, enforce=False)
    enforced = evaluate_ar_kv_admission(estimate, free_kv_blocks=2, enforce=True)

    assert shadow.allowed
    assert shadow.shadow_would_defer
    assert shadow.reason is AdmissionReason.KV_PEAK_RISK
    assert not enforced.allowed
    assert not enforced.shadow_would_defer
    assert enforced.reason is AdmissionReason.KV_PEAK_RISK


def test_legacy_estimator_call_remains_conservative() -> None:
    estimate = ARResourceEstimator(16).estimate(num_prompt_tokens=1024, max_tokens=512)
    assert estimate.kv_blocks == 96


def test_observation_collector_tracks_peak_and_cache_ground_truth() -> None:
    estimate = ARResourceEstimator(16).estimate(num_prompt_tokens=64, max_tokens=32)
    collector = ResourceObservationCollector(max_completed=2)
    collector.begin("r0", estimate)
    collector.begin("r0", estimate)  # repeated shadow tick does not reset it
    collector.observe(
        "r0",
        allocated_kv_blocks=3,
        output_tokens=4,
        local_cached_tokens=16,
    )
    collector.observe(
        "r0",
        allocated_kv_blocks=5,
        output_tokens=12,
        local_cached_tokens=16,
    )
    observation = collector.finish("r0")

    assert observation is not None
    assert observation.observed_peak_allocated_kv_blocks == 5
    assert observation.observed_peak_incremental_kv_blocks == 5
    assert observation.observed_output_tokens == 12
    assert observation.observed_local_cached_tokens == 16
    assert observation.kv_peak_error == 1
    assert not observation.kv_underpredicted
    assert collector.active_count == 0
    assert collector.drain_completed() == [observation]
    assert collector.drain_completed() == []


def test_observation_error_compares_incremental_demand() -> None:
    estimate = ARResourceEstimator().estimate(
        ARRequestResourceContext(
            num_prompt_tokens=64,
            max_tokens=32,
            block_size=16,
            allocated_kv_blocks=4,
        )
    )
    collector = ResourceObservationCollector()
    collector.begin("resumed", estimate, baseline_allocated_kv_blocks=4)
    collector.observe("resumed", allocated_kv_blocks=6, output_tokens=8)
    observation = collector.finish("resumed")

    assert observation is not None
    assert observation.observed_peak_allocated_kv_blocks == 6
    assert observation.observed_peak_incremental_kv_blocks == 2
    assert observation.kv_peak_error == 0


def test_workload_classifier_has_stable_boundaries() -> None:
    classifier = ARWorkloadClassifier()
    assert (
        classifier.classify(prompt_tokens=128, max_tokens=64)
        == "ar:p128:o64:s0"
    )
    assert (
        classifier.classify(prompt_tokens=129, max_tokens=65, streaming=True)
        == "ar:p512:o256:s1"
    )


def test_observation_profile_jsonl_round_trip(tmp_path) -> None:
    classifier = ARWorkloadClassifier()
    workload_class = classifier.classify(prompt_tokens=64, max_tokens=256)
    observations = []
    for index, output_tokens in enumerate([8, 16, 32, 64]):
        estimate = ARResourceEstimator(16).estimate(
            num_prompt_tokens=64, max_tokens=256
        )
        collector = ResourceObservationCollector()
        collector.begin(
            f"r{index}",
            estimate,
            prompt_tokens=64,
            requested_max_tokens=256,
            block_size=16,
        )
        collector.observe(
            f"r{index}",
            allocated_kv_blocks=(64 + output_tokens + 15) // 16,
            output_tokens=output_tokens,
        )
        observation = collector.finish(f"r{index}")
        assert observation is not None
        observations.append(
            observation.__class__(
                **{**observation.__dict__, "workload_class": workload_class}
            )
        )

    observation_path = tmp_path / "observations.jsonl"
    write_observations_jsonl(observations, observation_path)
    loaded_observations = read_observations_jsonl(observation_path)
    assert loaded_observations == observations

    fingerprint = ProfileFingerprint(
        model_id="test",
        device_type="cpu-test",
        dtype="float16",
        tp_size=1,
        block_size=16,
    )
    profiles = build_ar_output_profiles(
        loaded_observations,
        fingerprint=fingerprint,
    )
    assert len(profiles) == 1
    assert profiles[0].p50_output_tokens == 16
    assert profiles[0].p95_output_tokens == 64

    profile_path = tmp_path / "profiles.jsonl"
    ARProfileStore(profiles).write_jsonl(profile_path)
    store = ARProfileStore.read_jsonl(profile_path)
    store.require_fingerprint(fingerprint)
    assert store.get(workload_class) == profiles[0]


def test_profile_fingerprint_mismatch_is_rejected() -> None:
    first = ProfileFingerprint("model", "A6000", "float16", 1, 16)
    second = ProfileFingerprint("model", "H100", "float16", 1, 16)
    observation_estimate = ARResourceEstimator(16).estimate(
        num_prompt_tokens=16, max_tokens=16
    )
    collector = ResourceObservationCollector()
    collector.begin("r", observation_estimate)
    collector.observe("r", allocated_kv_blocks=2, output_tokens=16)
    observation = collector.finish("r")
    assert observation is not None
    profiles = build_ar_output_profiles([observation], fingerprint=first)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ARProfileStore(profiles).require_fingerprint(second)


def test_observation_writer_flushes_in_bounded_batches(tmp_path) -> None:
    estimate = ARResourceEstimator(16).estimate(
        num_prompt_tokens=16, max_tokens=16
    )
    collector = ResourceObservationCollector()
    collector.begin("r", estimate)
    collector.observe("r", allocated_kv_blocks=2, output_tokens=16)
    observation = collector.finish("r")
    assert observation is not None

    path = tmp_path / "trace.jsonl"
    writer = ResourceObservationJSONLWriter(path, flush_size=2)
    writer.append(observation)
    assert not path.exists()
    writer.append(observation)
    assert read_observations_jsonl(path) == [observation, observation]
    writer.append(observation)
    writer.flush()
    assert read_observations_jsonl(path) == [observation] * 3
