# Dynamic HBM experiment

## Milestone 0: reproducible characterization

The Milestone-0 wrapper freezes the source/environment manifest, delegates the
actual runs to the existing matrix runner, and converts existing case artifacts
into paper-ready CSV tables. It does not change the server, coordinator,
scheduler, or worker hot path.

```bash
.venv/bin/python benchmarks/hbm_admission/run_milestone0_characterization.py \
  --spec benchmarks/hbm_admission/specs/qwen3_tts.yaml \
  --output-dir benchmarks/results/milestone0_qwen3_tts \
  --dry-run

# Inspect matrix_plan.json and experiment_spec.yaml, then execute:
.venv/bin/python benchmarks/hbm_admission/run_milestone0_characterization.py \
  --spec benchmarks/hbm_admission/specs/qwen3_tts.yaml \
  --output-dir benchmarks/results/milestone0_qwen3_tts
```

The wrapper produces `manifest.json`, per-run `runs.csv`, per-arm
`aggregates.csv`, normalized `comparisons.csv`, `milestone0_report.json`, and a
short `MILESTONE0_SUMMARY.md`. Existing artifacts can be reanalyzed without
starting a server or GPU workload by adding `--summarize-only`.

For the paper, retain at least three load levels, two pressure shapes, three
repetitions, all A/B/C/D arms, and one E fixed-cap baseline. The example Qwen3
TTS matrix already satisfies these structural checks. Burst and mixed-size
claims still require a real burst-capable load generator and heterogeneous
dataset; the report calls this out instead of inferring them from labels.

## Model-independent matrix runner

For the full workload/pressure/controller matrix, copy the example spec and
change the model-specific fields:

```bash
cp benchmarks/hbm_admission/specs/qwen3_tts.yaml /tmp/my_model.yaml
python benchmarks/hbm_admission/run_dynamic_hbm_matrix.py \
  --spec /tmp/my_model.yaml \
  --output-dir benchmarks/results/my_model_dynamic_hbm \
  --dry-run
```

Remove `--dry-run` after inspecting `matrix_plan.json`. The matrix runner is
model-independent: `server_command` and `benchmark_command` are argv arrays,
and `metric_map` maps arbitrary benchmark JSON fields to normalized throughput
and latency names. It expands every `workloads × pressure_profiles ×
controllers` combination and runs the A/B/C/D arms (plus an optional fixed-cap
arm) in each cell. `matrix_summary.json` links the results from every cell.

Use `env` in the spec for model-specific environment variables. Commands are
executed directly without a shell. Relative deploy and dataset paths are
resolved relative to the spec file, so specs can be checked into a different
directory without depending on the caller's working directory.

`run_dynamic_hbm_experiment.py` runs four controlled arms and saves every
input and output needed to reproduce the comparison:

| Arm | Dynamic HBM | External pressure |
| --- | --- | --- |
| A | off | no |
| B | on | no |
| C | off | yes |
| D | on | yes |

The default workload is `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` on one GPU.
On the current A100 80 GB host, run the three-repeat experiment with:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_experiment.py \
  --output-dir benchmarks/results/dynamic_hbm_qwen3_tts \
  --repeats 3 \
  --concurrency 32 \
  --num-prompts 200
```

Start with one short arm when validating a new environment:

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_experiment.py \
  --output-dir benchmarks/results/dynamic_hbm_smoke \
  --arms D_dynamic_on_pressure \
  --repeats 1 --concurrency 8 --num-prompts 20 \
  --pressure-baseline-seconds 3 \
  --pressure-high-seconds 5 \
  --pressure-critical-seconds 5 \
  --pressure-recovery-seconds 5 \
  --pressure-post-release-seconds 10
```

Each case contains its generated deploy YAML, exact benchmark command, server
and client logs, 200 ms GPU telemetry, pressure-generator JSONL, raw benchmark
JSON, and status JSON. `summary.json` aggregates successful repetitions. A
failed case is retained and makes the runner exit nonzero. Interrupted runs are
resumable: completed case directories are skipped.

## Other models

Supply a model-specific deploy config and a JSON argv template. Placeholders
are expanded without a shell: `{host}`, `{port}`, `{model}`, `{concurrency}`,
`{num_prompts}`, `{result_json}`, and `{repo}`.

```bash
.venv/bin/python benchmarks/hbm_admission/run_dynamic_hbm_experiment.py \
  --output-dir benchmarks/results/my_model \
  --model organization/model \
  --deploy-config /path/to/deploy.yaml \
  --request-profile custom \
  --benchmark-command-json '["python", "/path/to/load.py", "--url", "http://{host}:{port}", "--model", "{model}", "--concurrency", "{concurrency}", "--requests", "{num_prompts}", "--output", "{result_json}"]'
```

The custom load generator must write a JSON object to `{result_json}`. The
four-arm safety data and scheduler events are always collected; standard
throughput/latency aggregation additionally recognizes `request_throughput`,
`p99_e2el_ms`, `p99_audio_ttfp_ms`, and `median_audio_rtf` when present.

For multi-stage models, dynamic control is applied to all stages by default.
Use `--dynamic-stage-ids 0 2` to select only specific stage IDs. Watermarks and
the pressure sidecar are CLI parameters so they can be calibrated for GPUs of
different capacities.
