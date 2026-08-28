# H6: Gradual recovery after pressure release

**Hypothesis.** After pressure returns below low_watermark, the controller holds for recovery_complete_samples then increases additively by scale_up_step per scale_up_stable_samples, reaching NORMAL only at the configured cap; a high sample mid-recovery re-arms the decrease.

**Expectation.** no instant jump; RECOVERING hold >= 3 samples; additive +1 steps; NORMAL only at configured; re-arm works.

**Verdict.** MATCHES EXPECTATION

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | cap is 0 immediately after critical | PASS | caps[:3]=[16, 0, 0] |
| 2 | recovery does NOT jump straight to configured cap on the first low sample | PASS | first low sample cap=0 state=recovering reason=recovering_complete_reports |
| 3 | controller spends >= recovery_complete_samples in RECOVERING hold before expanding | PASS | first cap increase at low-sample index 7 (recovery_complete_samples=3) |
| 4 | expansion is additive (+scale_up_step), not multiplicative | PASS | cap at first increase = 1 (scale_up_step=1) |
| 5 | state returns to NORMAL only once cap reaches configured | PASS | reached configured at index 82, state there=normal |
| 6 | a high sample during recovery re-arms the decrease and leaves RECOVERING | PASS | pre=(0,recovering) post=(0,high_pressure) |

## Observations

```json
{
  "caps_rle": [
    [
      16,
      1
    ],
    [
      0,
      8
    ],
    [
      1,
      5
    ],
    [
      2,
      5
    ],
    [
      3,
      5
    ],
    [
      4,
      5
    ],
    [
      5,
      5
    ],
    [
      6,
      5
    ],
    [
      7,
      5
    ],
    [
      8,
      5
    ],
    [
      9,
      5
    ],
    [
      10,
      5
    ],
    [
      11,
      5
    ],
    [
      12,
      5
    ],
    [
      13,
      5
    ],
    [
      14,
      5
    ],
    [
      15,
      5
    ],
    [
      16,
      11
    ]
  ],
  "states_rle": [
    [
      "normal",
      1
    ],
    [
      "critical",
      1
    ],
    [
      "recovering",
      82
    ],
    [
      "normal",
      11
    ]
  ],
  "n_low_samples": 93,
  "caps_head": [
    16,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    3
  ],
  "rearm_caps": [
    16,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "rearm_states": [
    "normal",
    "critical",
    "recovering",
    "recovering",
    "recovering",
    "recovering",
    "recovering",
    "recovering",
    "high_pressure"
  ]
}
```

## Analysis

MATCHES EXPECTATION. The low-pressure branch in allocator.py has two gates. First, if the previous state was any of CRITICAL/HIGH_PRESSURE/INCOMPLETE/STALE/DISCONNECTED/RECOVERING and fewer than recovery_complete_samples confirming samples have been seen, it only holds the current cap (reason=recovering_complete_reports, state=RECOVERING) and increments a counter - no expansion. Only after that does it count scale_up_stable_samples consecutive low samples and then add scale_up_step (=1). So from cap 0 the trajectory is a hold of 3 samples, then +1 every 5 samples (in this run the first +1 lands at low-sample index 7 = 3 completion-hold samples plus the stable-headroom counter reaching scale_up_stable_samples), and the state flips to NORMAL exactly when the cap regains the configured value (before that it stays RECOVERING). This is asymmetric by design: fast multiplicative decrease, slow additive increase, matching AIMD. The re-arm case shows recovery is not sticky - a single sample back above high_watermark immediately re-enters the high_pressure branch and multiplies the cap down again, so a flapping pressure signal cannot ratchet the cap up.
