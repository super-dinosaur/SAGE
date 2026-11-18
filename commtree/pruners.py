"""Pruners (spec §6).

Two families:
* pre-rollout  (``needs_stats = False``): decide before any rollout -> ``LLMPruner``.
  In ``keep_pruned_scores=False`` mode pruned children are never rolled out.
* post-rollout (``needs_stats = True``): decide from the child's rollout statistics ->
  ``NoBranchPruner``, ``ThresholdPruner``, ``HybridPruner``. They cannot save rollouts, only
  deeper expansion; in non-keep mode their pruned children are simply dropped from the tree.

``decide`` only says whether a child is *expanded deeper*; whether its scores are kept is the
DataTree's ``keep_pruned_scores`` switch.
"""
from __future__ import annotations

import random
import re

from .utils import ACTIONS, mean, stderr


def q_values(rollouts: list[dict], lam: float, mu: float = 0.0) -> list[float]:
    return [r["R_task"] - lam * r["C_comm"] - mu * r.get("n_steps", 0) for r in rollouts]


class Pruner:
    name = "base"
    needs_stats = True

    def config(self) -> dict:
        return {"type": self.name}

    def decide_post(self, node: dict, action: str) -> tuple[bool, str]:
        return True, ""


class NoBranchPruner(Pruner):
    """Never expands below the first layer (flat dataset of independent decision points)."""
    name = "nobranch"

    def decide_post(self, node, action):
        return False, "nobranch"


class ThresholdPruner(Pruner):
    """Expand a child iff its Q is within ``thr`` of the best sibling Q (under ``lam``)."""
    name = "threshold"

    def __init__(self, thr: float = 0.2, lam: float = 0.05):
        self.thr, self.lam = thr, lam

    def config(self):
        return {"type": self.name, "thr": self.thr, "lam": self.lam}

    def _qs(self, node):
        return {a: mean(q_values(c["rollouts"], self.lam)) for a, c in node["children"].items() if c.get("rollouts")}

    def decide_post(self, node, action):
        qs = self._qs(node)
        best = max(qs.values())
        ok = qs[action] >= best - self.thr
        return ok, f"Q={qs[action]:.3f} vs best={best:.3f} (thr={self.thr}, lam={self.lam})"


class HybridPruner(ThresholdPruner):
    """Threshold OR statistically indistinguishable from the best (keeps uncertain branches)."""
    name = "hybrid"

    def __init__(self, thr: float = 0.2, lam: float = 0.05, z: float = 1.0):
        super().__init__(thr, lam)
        self.z = z

    def config(self):
        return {**super().config(), "z": self.z}

    def decide_post(self, node, action):
        ok, why = super().decide_post(node, action)
        if ok:
            return ok, why
        qs = {a: q_values(c["rollouts"], self.lam) for a, c in node["children"].items() if c.get("rollouts")}
        best_a = max(qs, key=lambda a: mean(qs[a]))
        gap = mean(qs[best_a]) - mean(qs[action])
        se = (stderr(qs[best_a]) ** 2 + stderr(qs[action]) ** 2) ** 0.5
        if gap <= self.z * se:
            return True, f"{why}; kept: gap {gap:.3f} <= {self.z}*SE {se:.3f}"
        return False, why


class LLMPruner(Pruner):
    """Asks the ``pruner`` role model (before rollouts) whether a child is worth expanding.

    The request only shows A's visible prompt and the candidate action. With probability
    ``explore_eps`` a child the model wants to prune is kept anyway (``explore_override``).
    """
    name = "llm"
    needs_stats = False

    def __init__(self, explore_eps: float = 0.1):
        self.explore_eps = explore_eps

    def config(self):
        return {"type": self.name, "explore_eps": self.explore_eps}

    @staticmethod
    def messages(policy_input: str, action: str) -> list[dict]:
        return [
            {"role": "system", "content": "You help build a dataset of communication decisions for agent A. "
             "Given the exact prompt A saw and a candidate first action, judge whether exploring the states that "
             "follow this action is informative. Reply 'EXPAND: <reason>' or 'PRUNE: <reason>' in one line."},
            {"role": "user", "content": f"A's prompt:\n<<<\n{policy_input}\n>>>\n\nCandidate first action: {action}"},
        ]

    @staticmethod
    def parse(text: str) -> tuple[str, str]:
        m = re.search(r"\b(EXPAND|PRUNE)\b\s*[:\-]?\s*(.*)", text or "", re.I | re.S)
        if not m:
            return "EXPAND", f"unparseable pruner reply; default EXPAND ({(text or '')[:80]!r})"
        return m.group(1).upper(), m.group(2).strip().splitlines()[0][:300] if m.group(2).strip() else ""

    def apply_eps(self, decision: str, seed: int) -> tuple[bool, bool]:
        """Returns (keep_expanding, explore_override)."""
        if decision == "EXPAND":
            return True, False
        if random.Random(seed).random() < self.explore_eps:
            return True, True
        return False, False


def build_pruner(cfg: dict | None) -> Pruner:
    cfg = dict(cfg or {"type": "threshold"})
    t = cfg.pop("type", "threshold")
    return {"nobranch": NoBranchPruner, "threshold": ThresholdPruner, "hybrid": HybridPruner, "llm": LLMPruner}[t](**cfg)


__all__ = ["ACTIONS", "Pruner", "NoBranchPruner", "ThresholdPruner", "HybridPruner", "LLMPruner", "build_pruner", "q_values"]
