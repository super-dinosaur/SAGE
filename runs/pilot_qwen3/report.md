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
  "n_decision_points": 7,
  "nodes_by_depth": {
    "0": 6,
    "1": 1
  },
  "nodes_by_scenario_type": {
    "B_key": 3,
    "C_key": 2,
    "A_self": 2
  },
  "siblings_complete_frac": 1.0,
  "n_rollouts": 84,
  "need_info_frac": 0.2857,
  "sample_prob_values": [
    0.2,
    0.8,
    1.0
  ],
  "behavior_policy_unforced": {
    "success": 0.3333,
    "cost": 0.13,
    "asks_per_episode": 0.6667
  },
  "api_stats": {
    "api_calls": 515,
    "cache_hits": 0,
    "errors": 0,
    "by_role": {
      "A": 420,
      "C": 46,
      "pruner": 21,
      "B": 28
    }
  }
}
```

## Headroom at λ=0.05 (IPW-weighted, complete decision points)

| policy | value | success | cost | n |
|---|---|---|---|---|
| oracle | 0.5423 | 0.5584 | 0.3219 | 7 |
| never_ask | 0.1096 | 0.1169 | 0.1462 | 7 |
| always_B | 0.5389 | 0.5584 | 0.3903 | 7 |
| always_C | 0.0989 | 0.1169 | 0.3597 | 7 |
| always_ask_best | 0.5403 | 0.5584 | 0.3621 | 7 |

Headroom: oracle − never_ask = **+0.4327**, oracle − best fixed ask = **+0.0034**. If both are small, revise the task design before scaling up (spec §13.4).

## Label distribution vs λ (SFT, unbalanced)

| λ | NO_COMM | ASK_B | ASK_C | high-conf |
|---|---|---|---|---|
| 0.0 | 4 | 3 | 0 | 3 |
| 0.05 | 4 | 3 | 0 | 5 |
| 0.1 | 4 | 3 | 0 | 5 |
| 0.2 | 4 | 3 | 0 | 5 |
| 0.5 | 4 | 3 | 0 | 5 |
| 1.0 | 4 | 3 | 0 | 3 |
| 2.0 | 4 | 3 | 0 | 3 |

## λ sweep (success–cost)

| λ | policy | value | success | cost |
|---|---|---|---|---|
| 0.0 | oracle | 0.5584 | 0.5584 | 0.3219 |
| 0.0 | never_ask | 0.1169 | 0.1169 | 0.1462 |
| 0.0 | always_B | 0.5584 | 0.5584 | 0.3903 |
| 0.0 | always_C | 0.1169 | 0.1169 | 0.3597 |
| 0.0 | always_ask_best | 0.5584 | 0.5584 | 0.3903 |
| 0.05 | oracle | 0.5423 | 0.5584 | 0.3219 |
| 0.05 | never_ask | 0.1096 | 0.1169 | 0.1462 |
| 0.05 | always_B | 0.5389 | 0.5584 | 0.3903 |
| 0.05 | always_C | 0.0989 | 0.1169 | 0.3597 |
| 0.05 | always_ask_best | 0.5403 | 0.5584 | 0.3621 |
| 0.1 | oracle | 0.5262 | 0.5584 | 0.3219 |
| 0.1 | never_ask | 0.1023 | 0.1169 | 0.1462 |
| 0.1 | always_B | 0.5194 | 0.5584 | 0.3903 |
| 0.1 | always_C | 0.0809 | 0.1169 | 0.3597 |
| 0.1 | always_ask_best | 0.5222 | 0.5584 | 0.3621 |
| 0.2 | oracle | 0.4941 | 0.5584 | 0.3219 |
| 0.2 | never_ask | 0.0876 | 0.1169 | 0.1462 |
| 0.2 | always_B | 0.4804 | 0.5584 | 0.3903 |
| 0.2 | always_C | 0.0449 | 0.1169 | 0.3597 |
| 0.2 | always_ask_best | 0.486 | 0.5584 | 0.3621 |
| 0.5 | oracle | 0.3975 | 0.5584 | 0.3219 |
| 0.5 | never_ask | 0.0438 | 0.1169 | 0.1462 |
| 0.5 | always_B | 0.3633 | 0.5584 | 0.3903 |
| 0.5 | always_C | -0.063 | 0.1169 | 0.3597 |
| 0.5 | always_ask_best | 0.3774 | 0.5584 | 0.3621 |
| 1.0 | oracle | 0.2365 | 0.5584 | 0.3219 |
| 1.0 | never_ask | -0.0294 | 0.1169 | 0.1462 |
| 1.0 | always_B | 0.1681 | 0.5584 | 0.3903 |
| 1.0 | always_C | -0.2429 | 0.1169 | 0.3597 |
| 1.0 | always_ask_best | 0.1964 | 0.5584 | 0.3621 |
| 2.0 | oracle | -0.0855 | 0.5584 | 0.3219 |
| 2.0 | never_ask | -0.1756 | 0.1169 | 0.1462 |
| 2.0 | always_B | -0.2222 | 0.5584 | 0.3903 |
| 2.0 | always_C | -0.6026 | 0.1169 | 0.3597 |
| 2.0 | always_ask_best | -0.1657 | 0.5584 | 0.3621 |

## Leakage check

nodes checked: 7, leaks: **0**, teammate-reply warnings (unsupported values, i.e. hallucination): 0
