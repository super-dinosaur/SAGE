# DataTree report

λ for headline numbers: 0.05

## Summary
```json
{
  "n_scenarios": 3,
  "scenario_types": {
    "B_key/stale/cross": 1,
    "C_key/missing": 1,
    "A_self": 1
  },
  "n_decision_points": 8,
  "nodes_by_depth": {
    "0": 6,
    "1": 2
  },
  "nodes_by_scenario_type": {
    "B_key": 4,
    "C_key": 2,
    "A_self": 2
  },
  "siblings_complete_frac": 1.0,
  "n_rollouts": 96,
  "need_info_frac": 0.125,
  "sample_prob_values": [
    0.2,
    0.8,
    1.0
  ],
  "behavior_policy_unforced": {
    "success": 0.6667,
    "cost": 0.0433,
    "asks_per_episode": 0.3333
  },
  "api_stats": {
    "api_calls": 839,
    "cache_hits": 8,
    "errors": 0,
    "by_role": {
      "A": 742,
      "C": 46,
      "pruner": 24,
      "B": 27
    }
  }
}
```

## Headroom at λ=0.05 (IPW-weighted, complete decision points)

| policy | value | success | cost | n |
|---|---|---|---|---|
| oracle | 0.8373 | 0.8454 | 0.1621 | 8 |
| never_ask | 0.2657 | 0.268 | 0.0471 | 8 |
| always_B | 0.4435 | 0.4536 | 0.2019 | 8 |
| always_C | 0.7637 | 0.7732 | 0.1891 | 8 |
| always_ask_best | 0.7843 | 0.7938 | 0.1899 | 8 |

Headroom: oracle − never_ask = **+0.5716**, oracle − best fixed ask = **+0.0736**. If both are small, revise the task design before scaling up (spec §13.4).

## Label distribution vs λ (SFT, unbalanced)

| λ | NO_COMM | ASK_B | ASK_C | high-conf |
|---|---|---|---|---|
| 0.0 | 3 | 2 | 3 | 1 |
| 0.05 | 3 | 1 | 4 | 2 |
| 0.1 | 3 | 1 | 4 | 2 |
| 0.2 | 3 | 1 | 4 | 2 |
| 0.5 | 3 | 1 | 4 | 2 |
| 1.0 | 3 | 1 | 4 | 2 |
| 2.0 | 3 | 1 | 4 | 2 |

## λ sweep (success–cost)

| λ | policy | value | success | cost |
|---|---|---|---|---|
| 0.0 | oracle | 0.8454 | 0.8454 | 0.1713 |
| 0.0 | never_ask | 0.268 | 0.268 | 0.0471 |
| 0.0 | always_B | 0.4536 | 0.4536 | 0.2019 |
| 0.0 | always_C | 0.7732 | 0.7732 | 0.1891 |
| 0.0 | always_ask_best | 0.7938 | 0.7938 | 0.2179 |
| 0.05 | oracle | 0.8373 | 0.8454 | 0.1621 |
| 0.05 | never_ask | 0.2657 | 0.268 | 0.0471 |
| 0.05 | always_B | 0.4435 | 0.4536 | 0.2019 |
| 0.05 | always_C | 0.7637 | 0.7732 | 0.1891 |
| 0.05 | always_ask_best | 0.7843 | 0.7938 | 0.1899 |
| 0.1 | oracle | 0.8292 | 0.8454 | 0.1621 |
| 0.1 | never_ask | 0.2633 | 0.268 | 0.0471 |
| 0.1 | always_B | 0.4334 | 0.4536 | 0.2019 |
| 0.1 | always_C | 0.7543 | 0.7732 | 0.1891 |
| 0.1 | always_ask_best | 0.7748 | 0.7938 | 0.1899 |
| 0.2 | oracle | 0.8129 | 0.8454 | 0.1621 |
| 0.2 | never_ask | 0.2586 | 0.268 | 0.0471 |
| 0.2 | always_B | 0.4132 | 0.4536 | 0.2019 |
| 0.2 | always_C | 0.7354 | 0.7732 | 0.1891 |
| 0.2 | always_ask_best | 0.7558 | 0.7938 | 0.1899 |
| 0.5 | oracle | 0.7643 | 0.8454 | 0.1621 |
| 0.5 | never_ask | 0.2445 | 0.268 | 0.0471 |
| 0.5 | always_B | 0.3527 | 0.4536 | 0.2019 |
| 0.5 | always_C | 0.6787 | 0.7732 | 0.1891 |
| 0.5 | always_ask_best | 0.6989 | 0.7938 | 0.1899 |
| 1.0 | oracle | 0.6833 | 0.8454 | 0.1621 |
| 1.0 | never_ask | 0.2209 | 0.268 | 0.0471 |
| 1.0 | always_B | 0.2518 | 0.4536 | 0.2019 |
| 1.0 | always_C | 0.5841 | 0.7732 | 0.1891 |
| 1.0 | always_ask_best | 0.6039 | 0.7938 | 0.1899 |
| 2.0 | oracle | 0.5212 | 0.8454 | 0.1621 |
| 2.0 | never_ask | 0.1738 | 0.268 | 0.0471 |
| 2.0 | always_B | 0.0499 | 0.4536 | 0.2019 |
| 2.0 | always_C | 0.3951 | 0.7732 | 0.1891 |
| 2.0 | always_ask_best | 0.414 | 0.7938 | 0.1899 |

## Leakage check

nodes checked: 8, leaks: **0**, teammate-reply warnings (unsupported values, i.e. hallucination): 0
