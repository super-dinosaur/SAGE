"""Scenario-level train / val / test splits (spec §11).

Splitting is done by *scenario*, never by decision point: decision points of one scenario share
the same facts, so splitting them across sets would leak answers into training.

* train / val : balanced SFT (NO_COMM down-sampling + class weights), DPO, RL
* test        : the true distribution — no balancing, every decision point kept (low-confidence
                labels marked, not dropped); evaluate with 1/sample_prob weights.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

from .export import export
from .utils import derive_seed, read_json, read_jsonl, write_json, write_jsonl

SPLITS = ("train", "val", "test")


def assign_splits(scenarios: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, str]:
    """Stratified by scenario type (incl. stale / cross-owner variant). Returns scenario_id -> split."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be three non-negative numbers")
    tot = sum(ratios)
    ratios = [r / tot for r in ratios]
    groups: dict[str, list[str]] = {}
    for s in scenarios:
        key = f"{s['scenario_type']}/{s.get('variant')}/{bool(s.get('cross_owner'))}"
        groups.setdefault(key, []).append(s["scenario_id"])
    # interleave groups in a seeded order so every split gets a proportional mix
    order: list[str] = []
    rng = random.Random(derive_seed(seed, "split"))
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        groups[key] = ids
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                order.append(groups[key].pop())
    n = len(order)
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    # never leave a split with positive ratio empty when there are enough scenarios
    for i in (2, 1):
        if ratios[i] > 0 and counts[i] == 0 and counts[0] > 1:
            counts[i] += 1
            counts[0] -= 1
    # take test and val from the interleaved order first (most balanced), train gets the rest
    out, pos = {}, 0
    for i in (2, 1, 0):
        for sid in order[pos: pos + counts[i]]:
            out[sid] = SPLITS[i]
        pos += counts[i]
    return out


def filter_raw(raw: dict, ids: set[str], split: str) -> dict:
    r = {"meta": copy.deepcopy(raw["meta"]), "trees": [t for t in raw["trees"] if t["scenario_id"] in ids]}
    r["meta"]["split"] = split
    r["meta"]["n_scenarios"] = len(r["trees"])
    return r


def write_splits(run_dir: str, out_dir: str | None = None, ratios=(0.8, 0.1, 0.1), seed: int = 0,
                 lams=(0.05,), mu: float = 0.0, **export_kw) -> dict:
    run = Path(run_dir)
    out = Path(out_dir) if out_dir else run / "splits"
    raw = read_json(run / "tree.json")
    scenarios = read_jsonl(run / "scenarios.jsonl")
    traj_path = run / "trajectories.jsonl"
    trajs = read_jsonl(traj_path) if traj_path.exists() else []
    assign = assign_splits(scenarios, ratios, seed)
    manifest = {"ratios": list(ratios), "seed": seed, "lams": list(lams), "mu": mu, "splits": {}}
    for split in SPLITS:
        ids = {sid for sid, s in assign.items() if s == split}
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        sc = [s for s in scenarios if s["scenario_id"] in ids]
        r = filter_raw(raw, ids, split)
        write_jsonl(d / "scenarios.jsonl", sc)
        write_json(d / "tree.json", r)
        write_jsonl(d / "trajectories.jsonl", (t for t in trajs if t["scenario_id"] in ids))
        files = {"decisions.jsonl": write_jsonl(d / "decisions.jsonl", export(r, "decisions"))}
        for lam in lams:
            tag = f"lam{lam:g}" + (f"_mu{mu:g}" if mu else "")
            if split == "test":  # true distribution: no balancing, keep low-confidence rows
                kw = {**export_kw, "balance": False, "low_conf": "mark"}
            else:
                kw = export_kw
            for fmt in ("sft", "dpo", "rl"):
                files[f"{fmt}_{tag}.jsonl"] = write_jsonl(d / f"{fmt}_{tag}.jsonl", export(r, fmt, lam=lam, mu=mu, **kw))
        types: dict[str, int] = {}
        for s in sc:
            types[s["scenario_type"]] = types.get(s["scenario_type"], 0) + 1
        manifest["splits"][split] = {"n_scenarios": len(sc), "scenario_types": types,
                                     "scenario_ids": sorted(ids), "files": files}
    write_json(out / "split.json", manifest)
    return manifest
