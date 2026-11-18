"""Exporters (spec §9-§11). Pure functions of the raw tree: changing λ / μ never calls a model.

Formats
  decisions : one record per decision point in the §9 schema (λ-independent)
  sft       : chat rows, label = argmax_a Q(s,a); 'high' confidence iff gap to runner-up > z·SE
  dpo       : (chosen, rejected) pairs per decision point, rejected significantly worse, ≤ max_pairs
  rl        : per-state Q, leave-one-out advantage, per-rollout returns, action_logprob, 1/sample_prob
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from .datatree import iter_nodes
from .pruners import q_values
from .utils import ACTIONS, derive_seed, mean, stderr, write_json, write_jsonl


# ------------------------------------------------------------------ helpers
def action_q(node: dict, lam: float, mu: float = 0.0) -> dict[str, dict]:
    out = {}
    for a, c in node["children"].items():
        ro = c.get("rollouts") or []
        if not ro:
            continue
        qs = q_values(ro, lam, mu)
        out[a] = {"Q": mean(qs), "se": stderr(qs), "n": len(qs), "qs": qs}
    return out


def _ranked(q: dict) -> list[str]:
    # deterministic tie-break: cheaper/no-communication first, then B, then C
    return sorted(q, key=lambda a: (-q[a]["Q"], ACTIONS.index(a)))


def _gap_se(q, a, b):
    se = math.sqrt(q[a]["se"] ** 2 + q[b]["se"] ** 2)
    return q[a]["Q"] - q[b]["Q"], se


def _significant(gap, se, z):
    return gap > z * se if math.isfinite(se) else False


# ------------------------------------------------------------------ decisions (§9)
def export_decisions(raw: dict, **_) -> list[dict]:
    rows = []
    for node in iter_nodes(raw):
        acts = {}
        for a, c in node["children"].items():
            if not c.get("rollouts"):
                continue
            acts[a] = {k: c[k] for k in ("n", "R_task_mean", "R_task_std", "C_comm_mean", "C_comm_std", "n_steps_mean")}
            acts[a].update(status=c.get("status"), prune_decision=c.get("prune_decision"), prune_reason=c.get("prune_reason"))
            acts[a]["rollouts"] = [{k: r[k] for k in ("seed", "R_task", "C_comm", "n_steps", "action_logprob", "trajectory_id")}
                                   for r in c["rollouts"]]
        rows.append({
            "state_id": node["state_id"], "scenario_id": node["scenario_id"], "scenario_type": node["scenario_type"],
            "depth": node["depth"], "parent": node["parent"], "policy_input": node["policy_input"],
            "sample_prob": node["sample_prob"], "need_info": node["need_info"],
            "behavior_policy": node["behavior_policy"], "siblings_complete": node.get("siblings_complete", False),
            "actions": acts,
        })
    return rows


# ------------------------------------------------------------------ SFT
def export_sft(raw: dict, lam: float, mu: float = 0.0, z: float = 2.0, low_conf: str = "mark",
               low_conf_weight: float = 0.3, balance: bool = True, nocomm_ratio: float = 2.0,
               ipw: bool = False, seed: int = 0, **_) -> list[dict]:
    rows = []
    for node in iter_nodes(raw):
        if not node.get("siblings_complete"):
            continue  # labels need all three actions (spec §6)
        q = action_q(node, lam, mu)
        rk = _ranked(q)
        best, second = rk[0], rk[1]
        gap, se = _gap_se(q, best, second)
        conf = "high" if _significant(gap, se, z) else "low"
        if conf == "low" and low_conf == "drop":
            continue
        rows.append({
            "state_id": node["state_id"], "scenario_type": node["scenario_type"],
            "messages": node["policy_messages"] + [{"role": "assistant", "content": best}],
            "label": best, "confidence": conf, "gap": round(gap, 6), "gap_se": round(se, 6) if math.isfinite(se) else None,
            "q": {a: round(v["Q"], 6) for a, v in q.items()}, "sample_prob": node["sample_prob"], "lam": lam, "mu": mu,
        })
    if balance and rows:
        rows = _balance(rows, nocomm_ratio, seed)
    # weights ∝ 1/sqrt(class count), normalised to mean 1, × low-confidence weight, × optional IPW
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    w = [(1 / math.sqrt(counts[r["label"]])) * (low_conf_weight if r["confidence"] == "low" else 1.0)
         * (1 / r["sample_prob"] if ipw else 1.0) for r in rows]
    z_ = mean(w) or 1.0
    for r, wi in zip(rows, w):
        r["weight"] = round(wi / z_, 6)
    return rows


def _balance(rows, ratio, seed):
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    others = [len(v) for k, v in by.items() if k != "NO_COMM"]
    if "NO_COMM" not in by or not others:
        return rows
    cap = max(1, round(ratio * mean(others)))
    nc = by["NO_COMM"]
    if len(nc) > cap:
        rng = random.Random(derive_seed(seed, "balance"))
        hi = [r for r in nc if r["confidence"] == "high"]
        lo = [r for r in nc if r["confidence"] != "high"]
        rng.shuffle(hi)
        rng.shuffle(lo)
        keep = {id(r) for r in (hi + lo)[:cap]}  # prefer high-confidence NO_COMM rows
        rows = [r for r in rows if r["label"] != "NO_COMM" or id(r) in keep]
    return rows


# ------------------------------------------------------------------ DPO
def export_dpo(raw: dict, lam: float, mu: float = 0.0, z: float = 2.0, max_pairs: int = 2, ipw: bool = False, **_) -> list[dict]:
    rows = []
    for node in iter_nodes(raw):
        q = action_q(node, lam, mu)
        if len(q) < 2:
            continue
        rk = _ranked(q)
        best = rk[0]
        cands = []
        for a in rk[1:]:
            gap, se = _gap_se(q, best, a)
            if _significant(gap, se, z):
                cands.append((gap, se, a))
        for gap, se, a in sorted(cands, reverse=True)[:max_pairs]:
            rows.append({
                "state_id": node["state_id"], "scenario_type": node["scenario_type"], "prompt": node["policy_messages"],
                "chosen": best, "rejected": a, "margin": round(gap, 6), "margin_se": round(se, 6),
                "siblings_complete": node.get("siblings_complete", False),
                "weight": round(1 / node["sample_prob"], 6) if ipw else 1.0, "lam": lam, "mu": mu,
            })
    return rows


# ------------------------------------------------------------------ RL
def export_rl(raw: dict, lam: float, mu: float = 0.0, contrast_threshold: float = 0.2, **_) -> list[dict]:
    rows = []
    for node in iter_nodes(raw):
        q = action_q(node, lam, mu)
        if not q:
            continue
        acts = {}
        for a, v in q.items():
            others = [q[b]["Q"] for b in q if b != a]
            adv = v["Q"] - mean(others) if others else 0.0
            c = node["children"][a]
            acts[a] = {
                "n": v["n"], "Q": round(v["Q"], 6), "Q_se": round(v["se"], 6) if math.isfinite(v["se"]) else None,
                "advantage": round(adv, 6),
                "action_logprob": None if not node["action_logprobs"] else round(node["action_logprobs"][a], 6),
                "returns": [{"seed": r["seed"], "return": round(qq, 6), "R_task": r["R_task"], "C_comm": r["C_comm"],
                             "n_steps": r["n_steps"], "trajectory_id": r["trajectory_id"]}
                            for r, qq in zip(c["rollouts"], v["qs"])],
            }
        spread = max(v["Q"] for v in q.values()) - min(v["Q"] for v in q.values())
        rows.append({
            "state_id": node["state_id"], "scenario_id": node["scenario_id"], "scenario_type": node["scenario_type"],
            "policy_input": node["policy_input"], "policy_messages": node["policy_messages"],
            "sample_prob": node["sample_prob"], "importance_weight": round(1 / node["sample_prob"], 6),
            "behavior_policy": node["behavior_policy"], "siblings_complete": node.get("siblings_complete", False),
            "lam": lam, "mu": mu, "q_spread": round(spread, 6), "is_contrastive": spread >= contrast_threshold,
            "actions": acts,
        })
    return rows


EXPORTERS = {"decisions": export_decisions, "sft": export_sft, "dpo": export_dpo, "rl": export_rl}


def export(raw: dict, fmt: str, lam: float = 0.05, mu: float = 0.0, **kw) -> list[dict]:
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown format {fmt!r}; choose from {list(EXPORTERS)}")
    return EXPORTERS[fmt](raw, lam=lam, mu=mu, **kw)


def export_all(raw: dict, out_dir: str, lam: float = 0.05, mu: float = 0.0, formats=("decisions", "sft", "dpo", "rl"), **kw) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    counts = {}
    tag = f"lam{lam:g}" + (f"_mu{mu:g}" if mu else "")
    for fmt in formats:
        rows = export(raw, fmt, lam=lam, mu=mu, **kw)
        name = "decisions.jsonl" if fmt == "decisions" else f"{fmt}_{tag}.jsonl"
        counts[fmt] = write_jsonl(p / name, rows)
    write_json(p / f"export_{tag}.json", {"lam": lam, "mu": mu, "options": kw, "counts": counts})
    return counts
