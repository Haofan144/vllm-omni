# H3: High watermark -> multiplicative cap decrease

**Hypothesis.** At/above high watermark (below critical) the cap is multiplied by scale_down_ratio each sample, immediately, and identically for every co-resident stage.

**Expectation.** pure ladder 16->8->4->2->1; shared stages identical; no decrease just below the line.

**Verdict.** MATCHES EXPECTATION

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | pure BudgetAllocator ladder is 16->8->4->2->1 (x0.5, floor min_num_seqs=1) | PASS | observed=[16, 8, 4, 2, 1, 1, 1] expected=[16, 8, 4, 2, 1, 1, 1] |
| 2 | first high-pressure sample decreases immediately (no stable wait) | PASS | step1 cap=8 state=high_pressure reason=high_pressure |
| 3 | shared-device: both stages follow the same multiplicative ladder | PASS | stage0=[8, 4, 2, 1, 1] stage1=[8, 4, 2, 1, 1] |
| 4 | pressure just below high watermark does NOT decrease the cap | PASS | caps=[16, 16, 16] states=['normal', 'normal', 'normal'] |

## Observations

```json
{
  "pure_ladder": [
    16,
    8,
    4,
    2,
    1,
    1,
    1
  ],
  "pure_states": [
    "normal",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure"
  ],
  "pure_reasons": [
    "awaiting_stable_headroom",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure",
    "high_pressure"
  ],
  "shared_caps_stage0": [
    8,
    4,
    2,
    1,
    1
  ],
  "shared_caps_stage1": [
    8,
    4,
    2,
    1,
    1
  ],
  "below_high_caps": [
    16,
    16,
    16
  ],
  "below_high_states": [
    "normal",
    "normal",
    "normal"
  ]
}
```

## Analysis

MATCHES EXPECTATION. In allocator.py the high-pressure branch sets cap = max(min_num_seqs, floor(current * scale_down_ratio)) on every sample where high_watermark <= pressure < critical_watermark, with no stable-sample gate, so the response is a geometric decay 16->8->4->2->1 that begins on the very first crossing sample (state=high_pressure, reason=high_pressure). The shared-device test shows the coordinator recomputes `shared_hbm_pressure` as the max guarded pressure over the affected group and feeds it to every co-resident allocator via pressure_override, so stage 1 tracks the identical ladder even though its own rank report stayed calm. The boundary case confirms the comparison is a true >= high_watermark test: one epsilon below leaves the cap untouched (state=normal, reason=within_hysteresis). Note the ladder floors at min_num_seqs (1 here); reaching 0 requires the critical branch (H4).
