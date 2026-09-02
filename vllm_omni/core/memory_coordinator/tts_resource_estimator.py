# SPDX-License-Identifier: Apache-2.0
"""TTS/Audio Talker workload classification (Milestone 2c).

Kept in its own module, separate from ``resource_profile.py``'s AR-only
``ARWorkloadClassifier``, per the M2 design doc's principle of keeping each
backend's resource-modeling logic explicit rather than folding TTS concepts
into AR-named files. The Talker stage of a two-stage TTS pipeline (e.g.
Qwen3-TTS's ``qwen3_tts`` stage 0) is itself an ``LLM_AR`` stage generating
codec/acoustic tokens autoregressively, so it reuses ``ARResourceEstimator``'s
KV-block accounting unchanged -- only the *workload class key* differs, since
a text LLM's ``ARWorkloadClassifier`` has no notion of ``task_type``,
``language``, or reference-audio conditioning, all of which materially affect
a TTS request's real output-length distribution (see the M2 design doc's
"AR 输出长度模型" section: the P50/P95/P99 output-length profile must be
learned per workload class, and a TTS workload class needs these extra
dimensions to be meaningful).

The Code2Wav/acoustic-decoder stage (``LLM_GENERATION``) needs a completely
different resource model (frame/chunk-driven batch envelope, persistent
overlap state -- not KV blocks) and is out of scope for this module; see the
M2 design doc section 8 for that component's separate design.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.core.memory_coordinator.resource_profile import ARWorkloadClassifier


class TTSWorkloadClassifier(ARWorkloadClassifier):
    """Talker workload buckets: AR prompt/output-token buckets plus the TTS
    dimensions known at admission time that the M2 design doc's TTS section
    calls out as affecting output-length distribution: ``task_type``
    (CustomVoice/VoiceDesign/Base -- the deploy yaml notes CustomVoice/
    VoiceDesign are "TTFA-optimal" at a different chunking strategy than
    Base voice-clone) and whether reference-audio conditioning is present
    (changes the Talker's effective prompt/context length distribution).

    ``language`` is deliberately NOT part of the bucket key in this first
    version: with dozens of supported languages, adding it as a bucket
    dimension risks fragmenting samples across too many classes to ever
    reach ``resource_profile_min_samples`` for any one of them, exactly the
    "profile never actually applies, always falls back to hard bound"
    failure the M2 design doc's fallback-hierarchy section warns about.
    Revisit only if real trace data shows language materially changes the
    output-length distribution within a fixed (prompt, output, task_type,
    ref_audio) bucket.
    """

    def classify(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        streaming: bool = False,
        task_type: str | None = None,
        ref_audio_present: bool = False,
        **_backend_specific: Any,
    ) -> str:
        base = super().classify(
            prompt_tokens=prompt_tokens, max_tokens=max_tokens, streaming=streaming
        )
        # super().classify() returns "ar:p{..}:o{..}:s{..}"; re-key the
        # backend prefix to "tts" and append the extra dimensions rather than
        # duplicate the bucketing logic.
        _, _, bucket_suffix = base.partition(":")
        resolved_task_type = task_type or "unknown"
        return f"tts:{bucket_suffix}:{resolved_task_type}:ref{int(bool(ref_audio_present))}"
