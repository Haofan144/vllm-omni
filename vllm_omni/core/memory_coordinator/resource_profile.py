from __future__ import annotations

import json
import math
import atexit
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from threading import Lock

from vllm_omni.core.memory_coordinator.resource_observer import ResourceObservation


@dataclass(frozen=True)
class ProfileFingerprint:
    """Execution identity guarding persisted profiles against unsafe reuse."""

    model_id: str
    device_type: str
    dtype: str
    tp_size: int
    block_size: int
    execution_mode: str = "default"
    backend: str = "ar"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.tp_size < 1 or self.block_size < 1:
            raise ValueError("profile tp_size and block_size must be positive")
        if self.schema_version != 1:
            raise ValueError(f"unsupported profile schema version {self.schema_version}")


class ARWorkloadClassifier:
    """Deterministic, interpretable first-version AR workload buckets."""

    PROMPT_BUCKETS = (128, 512, 2048, 8192)
    OUTPUT_BUCKETS = (64, 256, 1024, 4096)

    @staticmethod
    def _bucket(value: int, boundaries: tuple[int, ...]) -> str:
        for boundary in boundaries:
            if value <= boundary:
                return str(boundary)
        return "overflow"

    def classify(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        streaming: bool = False,
    ) -> str:
        if prompt_tokens < 0 or max_tokens < 0:
            raise ValueError("workload classifier inputs must be non-negative")
        prompt_bucket = self._bucket(prompt_tokens, self.PROMPT_BUCKETS)
        output_bucket = self._bucket(max_tokens, self.OUTPUT_BUCKETS)
        return f"ar:p{prompt_bucket}:o{output_bucket}:s{int(streaming)}"


@dataclass(frozen=True)
class AROutputLengthProfile:
    fingerprint: ProfileFingerprint
    workload_class: str
    sample_count: int
    p50_output_tokens: int
    p95_output_tokens: int
    p99_output_tokens: int
    profile_version: str = "ar-output-v1"

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("profile sample_count must be positive")
        if not (
            0
            <= self.p50_output_tokens
            <= self.p95_output_tokens
            <= self.p99_output_tokens
        ):
            raise ValueError("profile quantiles must satisfy p50 <= p95 <= p99")

    def output_tokens_at(self, coverage: float) -> int:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        if coverage <= 0.50:
            return self.p50_output_tokens
        if coverage <= 0.95:
            return self.p95_output_tokens
        return self.p99_output_tokens


def nearest_rank_quantile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("cannot compute a quantile from no samples")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def build_ar_output_profiles(
    observations: Iterable[ResourceObservation],
    *,
    fingerprint: ProfileFingerprint,
    min_samples: int = 1,
) -> list[AROutputLengthProfile]:
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    grouped: dict[str, list[int]] = {}
    for observation in observations:
        grouped.setdefault(observation.workload_class, []).append(
            observation.observed_output_tokens
        )
    profiles = []
    for workload_class, values in sorted(grouped.items()):
        if len(values) < min_samples:
            continue
        profiles.append(
            AROutputLengthProfile(
                fingerprint=fingerprint,
                workload_class=workload_class,
                sample_count=len(values),
                p50_output_tokens=nearest_rank_quantile(values, 0.50),
                p95_output_tokens=nearest_rank_quantile(values, 0.95),
                p99_output_tokens=nearest_rank_quantile(values, 0.99),
            )
        )
    return profiles


class ARProfileStore:
    def __init__(self, profiles: Iterable[AROutputLengthProfile] = ()) -> None:
        self._profiles = {profile.workload_class: profile for profile in profiles}

    def get(self, workload_class: str) -> AROutputLengthProfile | None:
        return self._profiles.get(workload_class)

    def require_fingerprint(self, expected: ProfileFingerprint) -> None:
        mismatched = [
            profile.workload_class
            for profile in self._profiles.values()
            if profile.fingerprint != expected
        ]
        if mismatched:
            raise ValueError(
                "resource profile fingerprint mismatch for classes: "
                + ", ".join(mismatched[:5])
            )

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as output:
            for profile in sorted(
                self._profiles.values(), key=lambda item: item.workload_class
            ):
                output.write(json.dumps(asdict(profile), sort_keys=True) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> ARProfileStore:
        profiles = []
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    row["fingerprint"] = ProfileFingerprint(**row["fingerprint"])
                    profiles.append(AROutputLengthProfile(**row))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid resource profile at {path}:{line_number}: {exc}"
                    ) from exc
        return cls(profiles)


def write_observations_jsonl(
    observations: Iterable[ResourceObservation], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as output:
        for observation in observations:
            output.write(json.dumps(asdict(observation), sort_keys=True) + "\n")


def read_observations_jsonl(path: str | Path) -> list[ResourceObservation]:
    observations = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                observations.append(ResourceObservation(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid resource observation at {path}:{line_number}: {exc}"
                ) from exc
    return observations


class ResourceObservationJSONLWriter:
    """Bounded-batch append writer for production shadow observations."""

    def __init__(self, path: str | Path, flush_size: int = 1) -> None:
        if flush_size < 1:
            raise ValueError("flush_size must be positive")
        self.path = Path(path)
        self.flush_size = flush_size
        self._pending: list[ResourceObservation] = []
        self._lock = Lock()
        atexit.register(self.flush)

    def append(self, observation: ResourceObservation) -> None:
        with self._lock:
            self._pending.append(observation)
            if len(self._pending) >= self.flush_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            for observation in self._pending:
                output.write(json.dumps(asdict(observation), sort_keys=True) + "\n")
            output.flush()
        self._pending.clear()
