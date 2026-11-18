# Three-Agent Communication-Decision Dataset — DataTree Design (v1, improved)

This document is the improved version of the original implementation plan (kept outside this repository). It records what was
implemented in `commtree/`, which open questions were settled (and how), and what changed relative to
the original plan. Section numbers in brackets (e.g. [§7]) point to the original spec.

---

## 1. Goal (unchanged)

Agents A, B, C cooperate on a task. A is the executor; B and C each hold private information. At
decision points A chooses **NO_COMM / ASK_B / ASK_C**. Objective: maximise `R_task − λ·C_comm`.
The dataset warm-starts the policy with SFT (optionally DPO) and then feeds PPO; the DataTree is both the
data collector and the advantage estimator.

## 2. Pipeline

```text
ScenarioSampler ──► scenarios.jsonl (ground truth, roles, private info)
        │
        ▼
layer −1  A's unforced behaviour run per scenario; every step is a candidate decision point,
          kept with prob p (p_root at step 0, p_high if A "needs info", else p_low)  ──► depth-0 nodes
        │
        ▼
layer d   for every node: [LLMPruner batch] → force a ∈ {NO_COMM, ASK_B, ASK_C}, K rollouts each
          (all rollouts of the layer advance in lock-step; each step = one concurrent batch)
          → [score pruner] → expanded children give the decision points sampled on their k=0
          rollout as depth d+1 nodes (until depth n)
        │
        ▼
tree.json (raw R_task / C_comm / n_steps per rollout)  +  trajectories.jsonl (event logs)
        │
        ▼   export with any λ, μ — no model calls
decisions / SFT / DPO / RL  +  report (headroom, λ sweep, leakage)  +  viewer.html
```

Invariants kept from the spec: only the first action is forced; B/C answer dynamically from their own
records; every branch has an isolated copy of the environment and agent memories and its own seed; A's
question text is regenerated in every rollout, so `Q(ASK_B)` is "the value of asking B *with A's own
asking skill*".

## 3. Settled open questions [§14]

| # | Question | Decision |
|---|---|---|
| 1 | How to choose decision points [§7] | Option 2 (every step is a candidate) with weighted sampling. "A judges it lacks info" is read from **A's own decision distribution**: `P_A(ASK_B)+P_A(ASK_C) ≥ 0.5` (fallback: the sampled label is an ASK). This costs no extra call, because the same decide call drives the rollout. Initial state uses `p_root = 1`. **All** sampled points on a trajectory become nodes, so a node's inclusion probability is exactly its `sample_prob` (not "first-hit" probability). |
| 2 | MVP task [§13.1] | **Distributed-fact QA** (1- or 2-hop questions over a synthetic organisation). Facts split between A's notes (local `LOOKUP` tool, free), B (People Ops) and C (Project Office). Programmatic verifier; see §5. |
| 3 | `scenario` / `sub` naming | Adopted. `scenario` = one scenario dict or a list (a forest, batched together). `sub` = a previous raw tree used for **resume**: scenarios whose config hash matches are reused verbatim. |
| 4 | Other cost terms | Raw data also stores `n_steps` per rollout. An optional step cost `μ` is applied at export: `R = R_task − λ·C_comm − μ·n_steps` (default μ = 0). Latency/turn costs therefore need no re-generation either. |
| 5 | Continuation policy for SFT data | Role A **must be the model that will be trained** (labels must reflect its own ability). Stronger models may be used only for B, C and the pruner. The version label is stored as `behavior_policy` on every node. |

## 4. Interface [§3]

```python
from commtree import DataTree, Policies, ThresholdPruner
from commtree.config import build_router, build_task

dt = DataTree(
    scenario=scenarios,           # dict or list of dicts
    sub=None,                     # previous tree.json (dict or path) -> resume
    desc="Communication Decision",
    n=3, m=3, K=8, max_steps=12,
    api=router,                   # Router: role -> backend (A, B, C, pruner)
    policy=Policies(...),         # continuation label, ExplorationPolicy, ScenarioSampler
    pruner=ThresholdPruner(thr=0.2, lam=0.05),
    keep_pruned_scores=True,
    seed=0, max_api_calls=None,
    task=build_task(cfg),
    k_init=None,                  # adaptive K (see §8)
)
dt.build()
dt.get_data("json")               # raw tree
dt.get_data("sft", lam=0.05)      # also "decisions", "dpo", "rl"; optional mu=...
dt.save("runs/x")
```

Everything is also reachable from a YAML/JSON config and the CLI (`python -m commtree build|export|report|check`).

## 5. MVP task: distributed-fact QA

* **World**: 8 people, 5 projects; relations `based_in, manager_of, phone_ext` (B's domain) and
  `lead_of, review_day, budget_code` (C's domain). Questions are 1-hop (“Who manages Mia Chen?”) or
  2-hop (“In which city is the lead of Project Orion based?”).
* **A's tools**: `LOOKUP <words>` on its private notes (free, not communication), `ANSWER <text>`
  (terminal). Communication: `ASK_B/ASK_C <question>`.
* **Scenario types** (exact proportions via `mix`, default 0.30 / 0.35 / 0.35 — more B/C-key items as
  required by [§11]):
  * `A_self` — every needed fact is in A's notes;
  * `B_key` / `C_key` — one needed fact is absent from A's notes and held by that teammate.
* **Two difficulty knobs** (improvements over the spec):
  * `stale_prob` — instead of missing, A's note is **outdated** (`[as of 2019] …`). A believes it knows
    the answer; these are exactly the "A thinks info is sufficient but should ask" states the spec feared
    would be missing from the data.
  * `cross_owner_prob` — the fact sits with the teammate whose description does *not* suggest it, so
    routing cannot be solved from role descriptions alone.
* **Verifier**: normalised exact match; an answer is wrong if it also names another candidate of the same
  kind ("Lisbon or Oslo" is rejected). No LLM judging [§8].

## 6. Backends [§4]

`Router` maps roles A/B/C/pruner to backends and sends a mixed batch concurrently per backend.

| backend | use |
|---|---|
| `MockBackend` + `CommunicationMock` | seeded simulators; A is an explicit stochastic policy (exact log-probs); B/C are deterministic record tools. They read only what their role may see. |
| `OpenAICompatBackend` | any OpenAI-compatible endpoint (SiliconFlow, vLLM, …), stdlib only, retries, `seed`, logprobs. |
| `CallableBackend` | wrap any Python function — e.g. the model under training for PPO data collection. |

**Improvements**

* **Two-stage A**: a *decide* call (prompt = `policy_input`, output = one label) followed by an *act* call
  (writes the LOOKUP/ANSWER/QUESTION text). The decision is a clean 3-way classification target and the
  same `decision_messages()` function is used for data generation, training and inference [§9].
* **Action log-probs** come from the decide call's logprobs (top-k walk over label tokens, renormalised
  over the 3 labels; exact for the mock). The node's distribution is reused for all siblings, so
  `action_logprob` costs nothing extra. If the provider has no logprobs the field is `null` and
  `need_info` falls back to the sampled label.
* **Response cache** (sqlite, keyed by backend fingerprint + messages + seed): repeated runs with real APIs
  are bit-identical, an interrupted build resumes for free, and `max_api_calls` counts only uncached calls
  (raises `BudgetExceeded`).

## 7. Pruning [§6]

The rule "pruning decides expansion, `keep_pruned_scores` decides whether scores are kept" is implemented
as specified, with one clarification:

| pruner | when | keep mode (default) | drop mode |
|---|---|---|---|
| `LLMPruner(explore_eps)` | **before** rollouts, one batch per layer | all children rolled out; pruned ones not expanded | pruned children get **no rollouts** and are not attached; listed in `node.dropped`; `siblings_complete=false` |
| `Threshold(thr, lam)`, `Hybrid(thr, lam, z)`, `NoBranch` | **after** rollouts (they need scores) | as above | can only save deeper expansion; pruned children are removed after scoring |

Every child records `prune_decision`, `prune_reason`, and (LLM) `explore_override`. The λ used by a
score-based pruner or by adaptive K is stored in the tree meta as `lam_gen`.

## 8. Rewards, statistics and class imbalance [§8, §10, §11]

* Raw data: per rollout `R_task ∈ {0,1}`, `C_comm = (question tokens + reply tokens)/100`, `n_steps`,
  `n_asks`, `seed`, `action_logprob`, `trajectory_id`; per action mean/std. Tokenizer is configurable and
  recorded (`regex` default, `tiktoken`, `hf:<model>`).
* **Q and its SE are computed from per-rollout returns** `q_i = R_i − λC_i − μS_i`, so the covariance of
  R and C is handled correctly (the spec's mean/std pairs alone would not allow that).
* **SFT**: label = argmax Q (ties broken towards NO_COMM, then B). `high` confidence iff gap to the
  runner-up > `z·SE` (z = 2); low-confidence rows are kept with weight ×0.3 (`--low-conf mark`) or dropped.
  NO_COMM is down-sampled to ≤ `ratio × mean(other class sizes)` (default 2:1, high-confidence kept first);
  weights ∝ 1/√class-size, normalised to mean 1; optional ×1/sample_prob (`--ipw`).
* **DPO**: chosen = best; rejected = actions significantly worse (gap > z·SE); ≤ 2 pairs per state.
* **RL**: `A(s,a) = Q(s,a) − mean_{b≠a} Q(s,b)` (leave-one-out over present siblings), per-rollout
  returns, `action_logprob`, `importance_weight = 1/sample_prob`, `is_contrastive` (Q spread ≥ 0.2).
  `rl.stratified_batches` builds batches with 30–50 % contrastive states; rewards are never modified.
  `rl.CommEnv` provides `reset/step` with the same reward for online PPO.
* **Adaptive K** (`k_init`): run `k_init` rollouts per action; if NO_COMM is already significantly best
  (under `lam_gen`) stop (`early_stopped: true`), otherwise top up to K.
* **λ scale**: one ask costs ≈ 0.2–0.4 (20–40 tokens). With λ = 0.05 the penalty is ≈ 0.015 per ask, which
  rarely flips a label. The default report sweep is therefore `λ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1, 2}`.

## 9. Output schema (one decision point) [§9]

`exports/decisions.jsonl` follows the spec exactly and adds `depth`, `parent`, `need_info`, and per-action
`status / prune_decision / prune_reason / n_steps_mean`:

```json
{"state_id": "ep_0003_t0.ASK_C_t2", "scenario_id": "ep_0003", "scenario_type": "C_key", "depth": 1,
 "parent": {"state_id": "ep_0003_t0", "action": "ASK_C"},
 "policy_input": "[system]\n...\n[user]\nTask question: ...", "sample_prob": 0.8, "need_info": true,
 "behavior_policy": "A-v0", "siblings_complete": true,
 "actions": {"NO_COMM": {"n": 8, "R_task_mean": 0.25, "R_task_std": 0.43, "C_comm_mean": 0.31, "C_comm_std": 0.12,
                         "n_steps_mean": 4.1, "status": "prune", "prune_decision": "PRUNE", "prune_reason": "...",
                         "rollouts": [{"seed": 11, "R_task": 0, "C_comm": 0.27, "n_steps": 4, "action_logprob": -1.2,
                                       "trajectory_id": "ep_0003_t0.ASK_C_t2.NO_COMM.k0"}]},
             "ASK_B": {"...": "..."}, "ASK_C": {"...": "..."}}}
```

`policy_input` contains only what A saw. Full event logs are in `trajectories.jsonl`, joined by
`trajectory_id`.

## 10. Acceptance criteria [§13] — all automated (`tests/test_acceptance.py`, 24 tests)

| criterion | test |
|---|---|
| same seed twice → identical tree and trajectories (also through the HTTP backend via the cache) | `test_same_seed_identical`, `test_http_backend_with_cache_replay` |
| no private info / answers in `policy_input` (channel-aware checker + injected-leak negative test) | `test_no_leakage`, `test_leak_detector_catches_injected_secret` |
| keep mode → all three actions have stats at every node | `test_keep_mode_all_actions_have_stats` |
| re-export with a new λ makes zero model calls | `test_reexport_with_new_lambda_makes_no_calls` |
| plus: pruner modes, explore_eps, adaptive K, LOO advantages, SFT balancing, DPO pairs, verifier, scenario mix, CommEnv, resume, budget | remaining tests |

The leakage checker splits A's prompt into the static part (must contain no private value) and the history
channels: values in LOOKUP results must come from A's notes; values in B/C replies should come from that
teammate's records (otherwise reported as a hallucination warning, which matters for LLM responders).

## 10b. Train / val / test split [§11]

`python -m commtree split --run <dir>` splits by **scenario** (decision points of one scenario share
facts, so splitting them across sets would leak answers), stratified by scenario type × variant ×
cross-owner, deterministic given `--seed`. Train/val exports are balanced; the test split keeps the
true distribution (no down-sampling, low-confidence rows kept) and is meant to be 1/sample_prob weighted.
Evaluation of trained policies is owned by the training side and is not part of this package.

## 11. Code layout [§12]

```
commtree/
  task.py                    Task interface (make_instance, initial_state, observe, allowed_actions, step,
                             is_terminal, evaluate, prompt functions, leakage_terms)
  tasks/communication.py     MVP task: scenario generator, environment, prompts, verifier
  tasks/communication_mock.py  seeded simulators for A/B/C/pruner
  backends.py                Request/Response, Mock/OpenAI-compatible/Callable backends, Router, cache, budget
  policies.py                ExplorationPolicy, ScenarioSampler, Policies
  pruners.py                 NoBranch, Threshold, Hybrid, LLMPruner
  datatree.py                DataTree (lock-step layer batching, branching, stats, save/resume)
  export.py                  decisions / sft / dpo / rl exporters
  report.py                  summary, headroom, λ sweep, leakage check, viewer.html
  rl.py                      CommEnv (reset/step), stratified_batches
  split.py                   scenario-level, type-stratified train/val/test split + per-split exports
  config.py, __main__.py     YAML/JSON config + CLI
configs/mock.yaml, configs/siliconflow.yaml
tests/test_acceptance.py
```

## 12. Mock-run reference numbers (30 scenarios, K=8, n=3; `configs/mock.yaml`)

152 decision points (56/53/43 at depth 0/1/2), 3 648 rollouts, 0 leaks, all nodes complete.
At λ = 0.05: oracle value 0.899, never-ask 0.736, best fixed ask 0.888 → headroom +0.16 vs never-ask,
+0.03 vs always-ask. Label mix (unbalanced) ≈ 53 % NO_COMM / 26 % ASK_B / 20 % ASK_C.
These numbers describe the *simulator*; the real headroom must be re-measured on the API pilot before
scaling up [§13.4].

## 13. Known limitations / next steps

* API log-probs are an approximation (first divergent label token from top-k); providers without logprobs
  yield `action_logprob = null` (PPO then needs the trainer to recompute old log-probs).
* For depth ≥ 1 nodes, `sample_prob` is the inclusion probability *given* the k = 0 rollout of the parent
  branch, and `max_children_per_branch` caps the count.
* The legacy word-guessing (interrupt) task was not in this folder; it can be re-attached as a regression
  test by implementing `commtree.task.Task`.
* Next: run the SiliconFlow pilot (3–5 scenarios, K = 4), check headroom/leakage/labels on real models,
  then scale; consider a real tool-based debugging task once the pipeline is stable [§13.1].
