# Milestone 1 真实 GPU Pressure 实验报告

## 结论

Milestone 1 的 reactive safety 闭环在 NVIDIA RTX A6000 上按设计工作：

- smoke 与正式 A/B/C/D/E 共 6 个 case 全部完成并通过验收；
- 动态压力 case 中两个共享 GPU stage 同步降低 admission cap；
- critical pressure 下 cap 降到 0，只阻止新 admission，不驱逐 running request；
- 120/120 个正式请求成功，服务端 OOM 和 traceback 均为 0；
- 压力释放后 cap 按稳定样本逐步恢复；
- 无压力时 B/A 吞吐比为 0.9996，本轮观测的监控开销约为 0.04%。

实验同时表明当前 safety-first 策略存在明显性能代价。D 相对 C 的吞吐比为 0.300，P99
延迟比为 12.401；这符合 Milestone 1 作为预测系统兜底层的定位，并构成后续 predictive
control 优化的动机。

## 环境与配置

- 日期：2026-08-28 UTC
- GPU：NVIDIA RTX A6000，49140 MiB
- 模型：Qwen3-TTS-12Hz-0.6B-CustomVoice
- Stage：Talker 与 Code2Wav，共享物理 GPU 0
- 正式 workload：并发 16，120 requests，4 warmups
- Controller 水位：low/high/critical = 0.72/0.82/0.90
- Guard：1024 MiB
- Pressure sidecar：128 MiB/chunk，始终预留 2048 MiB
- Pressure 曲线：8 s baseline → 0.84 保持 12 s → 0.92 保持 10 s → 释放至
  0.78 保持 12 s → 全释放观察 20 s
- 正式重复次数：每个 arm 1 次

## 正式结果

| Arm | 场景 | 吞吐 (req/s) | P99 E2E (ms) | GPU 峰值压力 | 最小 cap | OOM |
|---|---|---:|---:|---:|---:|---:|
| A | Dynamic off，无 pressure | 7.449 | 2829.3 | 0.381 | - | 0 |
| B | Dynamic on，无 pressure | 7.446 | 2735.5 | 0.381 | - | 0 |
| C | Dynamic off，有 pressure | 7.500 | 2657.4 | 0.921 | - | 0 |
| D | Dynamic on，有 pressure | 2.248 | 32954.7 | 0.921 | 0 | 0 |
| E | 固定 cap=4，有 pressure | 4.033 | 4321.8 | 0.923 | - | 0 |

聚合比值：

- B/A throughput：0.999586
- B/A P99 latency：0.966866
- D/C throughput：0.299695
- D/C P99 latency：12.401
- D/E throughput：0.557367

## Controller 行为

正式 D case 中，Stage 0 与 Stage 1 收到一致的 shared-device decision：

```text
16 → 8 → 4 → 2 → 0 → 2 → 4
```

- sidecar 达到 high target 后，guard 后的中央压力约为 0.8624；
- sidecar 达到 critical target 后，guard 后的中央压力约为 0.9415；
- 两个 stage 在 critical pressure 下同步进入 cap=0；
- pressure 释放后，controller 经过 recovery reports 和 stable-headroom samples 后恢复；
- 全过程没有 server OOM 或 traceback。

## 限制

1. 每个正式 arm 仅运行一次，因此数值用于功能性真实 GPU 验证，不能直接作为统计显著性结论。
2. Sidecar 始终预留 2048 MiB，C 是相同 pressure 下的无控制对照，而不是故意制造不可恢复 OOM；
   因此本轮没有证明 reactive controller 相比 C 显著减少 OOM。
3. 本轮没有注入 coordinator disconnect、stale report 或 missing-rank report；这些路径目前由 CPU
   单元/集成测试覆盖，仍需要独立真实 GPU failure-injection 实验。
4. Sidecar 主要制造 HBM allocation pressure，不模拟额外 GPU compute contention。

## 复现

```bash
.venv/bin/python benchmarks/hbm_admission/run_milestone1_gpu_pressure.py \
  --output-dir benchmarks/results/milestone1_gpu_pressure_a6000 \
  --phase all
```

实验脚本会保存 per-case deploy、server log、200 ms GPU telemetry、pressure JSONL、client log、
benchmark JSON、status JSON，以及聚合 summary、analysis、manifest 和 Markdown 报告。
