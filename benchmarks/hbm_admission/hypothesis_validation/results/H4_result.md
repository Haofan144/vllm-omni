# H4: Critical watermark -> cap 0, admission halted

**Hypothesis.** At/above critical watermark the safety cap goes straight to critical_admission_cap (0) in one step, on every co-resident stage, and the scheduler admits nothing new; a KV-only critical does the same.

**Expectation.** one-step 16->0; shared fan-out to 0; scheduler clamps inconsistent wire cap; KV path works.

**Verdict.** MATCHES EXPECTATION

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | critical report drives cap 16 -> 0 in ONE step | PASS | ('critical', 0, 'critical', 'critical_pressure') |
| 2 | critical fans out to every co-resident stage as cap 0 | PASS | [(0, 'shared_device_critical_pressure', 'critical', 0), (1, 'shared_device_critical_pressure', 'critical', 0)] |
| 3 | shared critical decision reason is shared_device_critical_pressure | PASS | reasons=['shared_device_critical_pressure', 'shared_device_critical_pressure'] |
| 4 | scheduler clamps CRITICAL wire cap=4 down to 0 and blocks admission | PASS | {'applied': True, 'safety_cap': 0, 'effective_cap': 0, 'effective_tokens': 1, 'allows_new_admission': False, 'state': 'critical'} |
| 5 | KV pressure >= critical also drives cap 0 (source=kv) | PASS | ('kv_critical', 0, 'critical', 'critical_pressure', 'kv') |

## Observations

```json
{
  "pure": [
    [
      "idle",
      16,
      "normal",
      "awaiting_stable_headroom"
    ],
    [
      "critical",
      0,
      "critical",
      "critical_pressure"
    ]
  ],
  "shared": [
    [
      0,
      "shared_device_critical_pressure",
      "critical",
      0
    ],
    [
      1,
      "shared_device_critical_pressure",
      "critical",
      0
    ]
  ],
  "scheduler_clamp": {
    "applied": true,
    "safety_cap": 0,
    "effective_cap": 0,
    "effective_tokens": 1,
    "allows_new_admission": false,
    "state": "critical"
  },
  "kv_critical": [
    [
      "idle",
      16,
      "normal",
      "awaiting_stable_headroom",
      "physical_hbm"
    ],
    [
      "kv_critical",
      0,
      "critical",
      "critical_pressure",
      "kv"
    ]
  ]
}
```

## Analysis

MATCHES EXPECTATION. allocator.py handles critical before the multiplicative branch and sets cap = critical_admission_cap directly, so 16 collapses to 0 on the first sample at or above critical_watermark, independent of the current cap. The coordinator propagates this to the whole shared-device group (reason prefixed shared_device_) using the group's max guarded pressure. apply_stage_budget_decision in omni_scheduler_mixin.py has an extra belt-and-braces clamp: in the CRITICAL state it takes min(configured, critical_admission_cap, max(0, wire_cap)), so a malformed wire decision that pairs state=critical with a nonzero cap still lands at 0 and _dynamic_hbm_allows_new_admission() returns False. Because pressure = max(guarded_physical, kv), a KV-pool exhaustion alone trips the same path with pressure_source=kv. New admission is therefore halted from every angle the design intends; H5 covers that running requests are not killed.
