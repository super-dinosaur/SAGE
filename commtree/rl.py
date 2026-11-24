"""RL helpers (spec §5, §11): a reset/step environment and stratified batching.

``CommEnv`` lets PPO roll out the policy being trained: the policy chooses the decision label
at every step; A's message text and B/C's replies come from the router's role backends
(point role A at the trained model with ``CallableBackend`` to keep text generation on-policy).
The reward is unchanged from data generation: terminal ``R_task − λ·C_comm − μ·n_steps``.
"""
from __future__ import annotations

import random

from .backends import Request, Router
from .utils import ACTIONS, DEFAULT_MU, derive_seed


class CommEnv:
    def __init__(self, task, router: Router, lam: float = 0.05, mu: float = DEFAULT_MU):
        self.task, self.router, self.lam, self.mu = task, router, lam, mu
        self.state = None
        self._seed = 0
        self._t = 0

    def _obs(self):
        o = self.task.observe(self.state, "A")
        msgs = self.task.decision_messages(o)
        return {"policy_messages": msgs, "policy_input": self.task.render_prompt(msgs), "allowed": self.task.allowed_actions(self.state)}

    def reset(self, scenario: dict, seed: int = 0) -> dict:
        self.state = self.task.initial_state(scenario)
        self._seed, self._t = seed, 0
        return self._obs()

    def step(self, decision: str):
        assert decision in ACTIONS, decision
        task, st = self.task, self.state
        obs = task.observe(st, "A")
        r = self.router.batch([Request("A", "act", task.action_messages(obs, decision), derive_seed(self._seed, self._t, "act"),
                                       meta={"obs": obs, "decision": decision}, max_tokens=96)])[0]
        action = task.parse_action(r.text, decision)
        if action["type"] == "ask":
            to = action["to"]
            rr = self.router.batch([Request(to, "respond", task.responder_messages(st, to, action["content"]),
                                            derive_seed(self._seed, self._t, "respond"),
                                            meta={"obs": task.observe(st, to), "question": action["content"]}, max_tokens=160)])[0]
            action["reply"] = rr.text
        task.step(st, action)
        self._t += 1
        done = task.is_terminal(st)
        info = {"action": action}
        reward = 0.0
        if done:
            ev = task.evaluate(st)
            reward = ev["R_task"] - self.lam * ev["C_comm"] - self.mu * ev["n_steps"]
            info.update(ev)
        return self._obs(), reward, done, info


def stratified_batches(rl_rows: list[dict], batch_size: int = 32, contrast_frac: tuple[float, float] = (0.3, 0.5),
                       seed: int = 0, drop_incomplete: bool = False):
    """Yield batches where 30-50% of states are contrastive (clear Q differences between actions).

    Rewards are untouched; use row["importance_weight"] (= 1/sample_prob) as the loss weight.
    """
    rows = [r for r in rl_rows if r["siblings_complete"] or not drop_incomplete]
    rng = random.Random(derive_seed(seed, "batches"))
    con = [r for r in rows if r["is_contrastive"]]
    flat = [r for r in rows if not r["is_contrastive"]]
    rng.shuffle(con)
    rng.shuffle(flat)
    lo, hi = contrast_frac
    while con or flat:
        target = rng.uniform(lo, hi)
        n_con = min(len(con), max(1, round(target * batch_size))) if con else 0
        n_flat = min(len(flat), batch_size - n_con)
        if n_flat < batch_size - n_con:  # flat pool exhausted: fill with contrastive
            n_con = min(len(con), batch_size - n_flat)
        batch = con[:n_con] + flat[:n_flat]
        con, flat = con[n_con:], flat[n_flat:]
        if not batch:
            break
        rng.shuffle(batch)
        yield batch
