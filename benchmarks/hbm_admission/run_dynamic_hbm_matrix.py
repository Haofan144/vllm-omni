#!/usr/bin/env python3
"""Run a model-independent matrix of dynamic-HBM experiments.

The matrix file describes *what* to serve and benchmark.  This driver expands
workloads, pressure profiles, and controller configurations, then delegates
each cell to ``run_dynamic_hbm_experiment.py``.  Commands are JSON argv arrays;
no shell is involved.

Example::

    python benchmarks/hbm_admission/run_dynamic_hbm_matrix.py \
      --spec benchmarks/hbm_admission/specs/qwen3_tts.yaml \
      --output-dir benchmarks/results/dynamic_hbm
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_dynamic_hbm_experiment.py")

# Only options understood by the lower-level runner may be supplied through a
# spec.  Keeping this explicit catches misspellings before an expensive model
# startup.
VALUE_OPTIONS = {
    "startup_timeout",
    "benchmark_timeout",
    "release_threshold_mib",
    "max_num_seqs",
    "min_num_seqs",
    "sample_interval_ms",
    "report_timeout_ms",
    "missing_report_grace_samples",
    "low_watermark",
    "high_watermark",
    "critical_watermark",
    "scale_down_ratio",
    "scale_up_step",
    "scale_up_stable_samples",
    "pressure_baseline_seconds",
    "pressure_high_target",
    "pressure_high_seconds",
    "pressure_critical_target",
    "pressure_critical_seconds",
    "pressure_recovery_target",
    "pressure_recovery_seconds",
    "pressure_post_release_seconds",
    "pressure_chunk_mib",
    "pressure_reserve_mib",
    "pressure_wait_timeout",
    "include_fixed_cap",
    "num_warmups",
    "safetensors_load_strategy",
    "locale",
    "extra_body",
    "port",
    "device",
}


@dataclass(frozen=True)
class MatrixCell:
    name: str
    workload: str
    pressure_profile: str
    controller: str
    concurrency: int
    num_prompts: int
    request_rate: str
    output_dir: str
    command: tuple[str, ...]


def _slug(value: str) -> str:
    result = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(filter(None, result.split("_"))) or "default"


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("experiment spec must be a YAML mapping")
    return payload


def _named_entries(spec: dict[str, Any], key: str, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = spec.get(key, default)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{key} must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{key}[{index}] must be a mapping")
        item = dict(entry)
        item.setdefault("name", f"{key}_{index}")
        normalized.append(item)
    return normalized


def _resolve_path(spec_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def _append_options(command: list[str], options: dict[str, Any], *, context: str) -> None:
    unknown = set(options) - VALUE_OPTIONS
    if unknown:
        raise ValueError(f"unknown runner options in {context}: {sorted(unknown)}")
    for key, value in options.items():
        if value is None:
            continue
        command.extend((f"--{key.replace('_', '-')}", str(value)))


def _json_argv(value: Any, name: str) -> str:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of argv strings")
    return json.dumps(value)


def build_cells(spec_path: Path, output_dir: Path, python: str = sys.executable) -> list[MatrixCell]:
    spec = _load_mapping(spec_path)
    if "model" not in spec or "deploy_config" not in spec:
        raise ValueError("spec requires model and deploy_config")

    workloads = _named_entries(
        spec,
        "workloads",
        [{"name": "saturation", "concurrency": 32, "num_prompts": 200, "request_rate": "inf"}],
    )
    pressure_profiles = _named_entries(spec, "pressure_profiles", [{"name": "step", "options": {}}])
    controllers = _named_entries(spec, "controllers", [{"name": "default", "options": {}}])
    repeats = int(spec.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be positive")

    common_options = spec.get("runner_options", {})
    if not isinstance(common_options, dict):
        raise ValueError("runner_options must be a mapping")
    env = spec.get("env", {})
    valid_env = isinstance(env, dict) and all(
        isinstance(key, str) and isinstance(value, (str, int, float))
        for key, value in env.items()
    )
    if not valid_env:
        raise ValueError("env must be a string-to-scalar mapping")

    cells: list[MatrixCell] = []
    for workload, pressure, controller in itertools.product(workloads, pressure_profiles, controllers):
        concurrency = int(workload.get("concurrency", 32))
        num_prompts = int(workload.get("num_prompts", 200))
        request_rate = str(workload.get("request_rate", "inf"))
        if concurrency < 1 or num_prompts < 1:
            raise ValueError("workload concurrency and num_prompts must be positive")
        name = "__".join(_slug(str(item["name"])) for item in (workload, pressure, controller))
        cell_dir = output_dir / name
        command = [
            python,
            str(RUNNER),
            "--output-dir",
            str(cell_dir),
            "--model",
            str(spec["model"]),
            "--deploy-config",
            str(_resolve_path(spec_path, spec["deploy_config"])),
            "--repeats",
            str(repeats),
            "--concurrency",
            str(concurrency),
            "--num-prompts",
            str(num_prompts),
            "--request-rate",
            request_rate,
        ]
        if "request_profile" in spec:
            command.extend(("--request-profile", str(spec["request_profile"])))
        if "dataset_path" in spec:
            command.extend(("--dataset-path", str(_resolve_path(spec_path, spec["dataset_path"]))))
        if "server_command" in spec:
            command.extend(("--server-command-json", _json_argv(spec["server_command"], "server_command")))
        if "benchmark_command" in spec:
            command.extend(
                ("--benchmark-command-json", _json_argv(spec["benchmark_command"], "benchmark_command"))
            )
        if "metric_map" in spec:
            if not isinstance(spec["metric_map"], dict):
                raise ValueError("metric_map must be a mapping")
            command.extend(("--metric-map-json", json.dumps(spec["metric_map"])))
        arms = spec.get("arms")
        if arms is not None:
            if not isinstance(arms, list) or not all(isinstance(arm, str) for arm in arms):
                raise ValueError("arms must be a list of names")
            command.extend(("--arms", *arms))
        stage_ids = spec.get("dynamic_stage_ids")
        if stage_ids is not None:
            if not isinstance(stage_ids, list) or not all(isinstance(stage, int) for stage in stage_ids):
                raise ValueError("dynamic_stage_ids must be a list of integers")
            command.extend(("--dynamic-stage-ids", *(str(stage) for stage in stage_ids)))
        for extra in spec.get("server_extra_args", []):
            command.extend(("--server-extra-arg", str(extra)))

        _append_options(command, common_options, context="runner_options")
        for entry, label in ((workload, "workload"), (pressure, "pressure profile"), (controller, "controller")):
            options = entry.get("options", {})
            if not isinstance(options, dict):
                raise ValueError(f"{label} options must be a mapping")
            _append_options(command, options, context=f"{label} {entry['name']}")

        cells.append(
            MatrixCell(
                name=name,
                workload=str(workload["name"]),
                pressure_profile=str(pressure["name"]),
                controller=str(controller["name"]),
                concurrency=concurrency,
                num_prompts=num_prompts,
                request_rate=request_rate,
                output_dir=str(cell_dir),
                command=tuple(command),
            )
        )
    return cells


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def aggregate_matrix(output_dir: Path, cells: list[MatrixCell], results: dict[str, int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        analysis_path = Path(cell.output_dir) / "analysis.json"
        summary_path = Path(cell.output_dir) / "summary.json"
        rows.append(
            asdict(cell)
            | {
                "command": list(cell.command),
                "exit_code": results.get(cell.name),
                "analysis": json.loads(analysis_path.read_text()) if analysis_path.exists() else None,
                "summary_path": str(summary_path) if summary_path.exists() else None,
            }
        )
    payload = {"generated_at": datetime.now(UTC).isoformat(), "cells": rows}
    write_json(output_dir / "matrix_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate and write the expanded plan only")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args.spec.resolve(), args.output_dir.resolve())
    write_json(args.output_dir / "matrix_plan.json", {"cells": [asdict(cell) for cell in cells]})
    if args.dry_run:
        print(f"Validated {len(cells)} experiment cells; plan: {args.output_dir / 'matrix_plan.json'}")
        return 0

    spec = _load_mapping(args.spec.resolve())
    child_env = os.environ.copy()
    child_env.update({key: str(value) for key, value in spec.get("env", {}).items()})
    results: dict[str, int] = {}
    for cell in cells:
        print(f"[{cell.name}] starting", flush=True)
        completed = subprocess.run(cell.command, cwd=REPO, env=child_env, check=False)
        results[cell.name] = completed.returncode
        aggregate_matrix(args.output_dir, cells, results)
        if completed.returncode and not args.continue_on_error:
            return completed.returncode
    aggregate_matrix(args.output_dir, cells, results)
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
