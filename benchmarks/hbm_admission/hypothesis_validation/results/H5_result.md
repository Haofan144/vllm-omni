# H5: cap=0 halts admission without evicting running requests

**Hypothesis.** A zero/low safety cap blocks new admission while retaining the running set and its token budget, and admission resumes automatically once running drops below the cap.

**Expectation.** 6 running survive cap 0; drain re-enables admission; recovery restores admission.

**Verdict.** MATCHES EXPECTATION

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | cap=0 keeps all 6 running (execution budget == running count) | PASS | {'effective_cap': 0, 'max_running_execution_budget': 6, 'effective_tokens': 1536, 'allows_new_admission': False} |
| 2 | cap=0 still grants a nonzero scheduled-token budget for in-flight work | PASS | effective_tokens=1536 |
| 3 | cap=0 blocks NEW admission | PASS | allows_new_admission=False |
| 4 | admission resumes when running drops below effective cap (no new decision) | PASS | {'blocked_when_running_2_cap_2': True, 'allowed_when_running_1_cap_2': True} |
| 5 | after NORMAL decision restores cap, admission is allowed again | PASS | {'allows_when_critical': False, 'allows_after_normal_restore': True, 'effective_cap_after': 16} |

## Observations

```json
{
  "critical_with_running": {
    "effective_cap": 0,
    "max_running_execution_budget": 6,
    "effective_tokens": 1536,
    "allows_new_admission": false
  },
  "drain_unblocks": {
    "blocked_when_running_2_cap_2": true,
    "allowed_when_running_1_cap_2": true
  },
  "recovery": {
    "allows_when_critical": false,
    "allows_after_normal_restore": true,
    "effective_cap_after": 16
  }
}
```

## Analysis

MATCHES EXPECTATION. _recompute_effective_dynamic_hbm_budget in omni_scheduler_mixin.py separates two quantities: the ADMISSION cap (_effective_max_num_seqs, which the safety controller can pull to 0) and the EXECUTION budget (_dynamic_max_num_running_reqs / scheduling_slots = max(effective_cap, occupied_slots)). Because occupied_slots = len(running) + waiting-for-streaming-input, a cap of 0 while 6 requests run yields an execution budget of 6 and a token budget scaled by min(1, scheduling_slots/configured), never zero - so the scheduler keeps stepping the in-flight requests to completion and evicts nothing. _dynamic_hbm_allows_new_admission() is a pure `len(running) < effective_cap` test with no side effects, so as soon as a request finishes and running drops below the cap, admission re-enables on the next scheduler tick with no new coordinator message. Restoring the cap to its configured value (recovery, H6) also re-enables admission immediately. This is the design's 'reduce a cap, do not evict' invariant, verified against the real scheduler mixin.
