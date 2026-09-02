# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.core.memory_coordinator import ARWorkloadClassifier, TTSWorkloadClassifier

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tts_classifier_reuses_ar_bucket_shape() -> None:
    classifier = TTSWorkloadClassifier()
    workload_class = classifier.classify(
        prompt_tokens=64, max_tokens=4096, streaming=False, task_type="CustomVoice"
    )
    assert workload_class == "tts:p128:o4096:s0:CustomVoice:ref0"


def test_tts_classifier_distinguishes_task_type() -> None:
    classifier = TTSWorkloadClassifier()
    custom_voice = classifier.classify(
        prompt_tokens=64, max_tokens=4096, task_type="CustomVoice"
    )
    base = classifier.classify(prompt_tokens=64, max_tokens=4096, task_type="Base")
    assert custom_voice != base


def test_tts_classifier_distinguishes_ref_audio_presence() -> None:
    classifier = TTSWorkloadClassifier()
    with_ref = classifier.classify(
        prompt_tokens=64, max_tokens=4096, task_type="Base", ref_audio_present=True
    )
    without_ref = classifier.classify(
        prompt_tokens=64, max_tokens=4096, task_type="Base", ref_audio_present=False
    )
    assert with_ref != without_ref
    assert with_ref == "tts:p128:o4096:s0:Base:ref1"


def test_tts_classifier_defaults_missing_task_type_to_unknown() -> None:
    classifier = TTSWorkloadClassifier()
    workload_class = classifier.classify(prompt_tokens=64, max_tokens=4096)
    assert workload_class == "tts:p128:o4096:s0:unknown:ref0"


def test_tts_classifier_ignores_unrelated_backend_specific_kwargs() -> None:
    # Any classifier implementation must tolerate extra keyword arguments the
    # scheduler mixin passes uniformly, since it does not know at the call
    # site which classifier is configured.
    classifier = TTSWorkloadClassifier()
    workload_class = classifier.classify(
        prompt_tokens=64,
        max_tokens=4096,
        task_type="CustomVoice",
        some_future_field="ignored",
    )
    assert workload_class == "tts:p128:o4096:s0:CustomVoice:ref0"


def test_ar_classifier_ignores_tts_specific_kwargs() -> None:
    # The reverse direction: ARWorkloadClassifier must not choke when the
    # scheduler mixin passes TTS-shaped extra kwargs for a plain AR stage.
    classifier = ARWorkloadClassifier()
    workload_class = classifier.classify(
        prompt_tokens=64,
        max_tokens=4096,
        task_type="CustomVoice",
        ref_audio_present=True,
    )
    assert workload_class == "ar:p128:o4096:s0"
