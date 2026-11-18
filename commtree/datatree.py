"""DataTree: branch A's communication decisions and estimate their value by rollouts (spec §2-§3, §6).

Build loop (layer-synchronous batching):

    layer -1  behaviour run of A on every scenario  -> sampled decision points = depth-0 nodes
    layer d   for every node in the layer:
                (LLM pruner requests, one batch)
                force each first action a in {NO_COMM, ASK_B, ASK_C}, K rollouts each
                (all rollouts of the layer advance in lock-step: every step's requests of
                 every rollout form one batch sent concurrently to the role backends)
                (score-based pruners)
              expanded children contribute the decision points sampled on their k=0 rollout
              as the nodes of layer d+1   (until depth n)

Raw stats only (R_task, C_comm, n_steps per rollout). λ is applied at export time.
"""
from __future__ import annotations

import math
import random
from typing import Any, Generator

from .backends import Request, Response, Router, parse_label
from .policies import Policies
from .pruners import LLMPruner, Pruner, ThresholdPruner, q_values
from .utils import ACTIONS, derive_seed, mean, pstd, read_json, stable_hash, stderr

SCHEMA_VERSION = "commtree/1"


class DataTree:
    def __init__(self, scenario: dict | list[dict], sub: dict | str | None = None, desc: str = "Communication Decision",
                 n: int = 3, m: int = 3, api: Router | None = None, policy: Policies | None = None, K: int = 8,
                 max_steps: int = 20, pruner: Pruner | None = None, keep_pruned_scores: bool = True, seed: int = 0,
                 max_api_calls: int | None = None, task=None, k_init: int | None = None, lam_gen: float = 0.05,
                 early_stop_z: float = 2.0):
        if task is None:
            from .tasks.communication import CommunicationTask
            task = CommunicationTask(max_steps=max_steps)
        if api is None:
            raise ValueError("api (a Router) is required; see commtree.config.build_router")
        if not 1 <= m <= 3:
            raise ValueError("m must be in 1..3")
        self.scenarios = scenario if isinstance(scenario, list) else [scenario]
        self.sub = read_json(sub) if isinstance(sub, str) else sub
        self.desc, self.n, self.m, self.K = desc, n, m, K
        self.api, self.policy, self.task = api, policy or Policies(), task
        self.task.max_steps = max_steps
        self.max_steps = max_steps
        self.pruner = pruner or ThresholdPruner(thr=0.2, lam=lam_gen)
        self.keep_pruned_scores = keep_pruned_scores
        self.seed = seed
        if max_api_calls is not None:
            api.max_api_calls = max_api_calls
        self.k_init = k_init if (k_init and k_init < K) else None
        self.lam_gen = getattr(self.pruner, "lam", lam_gen)
        self.early_stop_z = early_stop_z
        self.trees: list[dict] = []
        self.trajectories: dict[str, dict] = {}
        self._built = False

    # ================================================================ config / meta
    def config(self) -> dict:
        ex = self.policy.exploration
        return {
            "schema": SCHEMA_VERSION, "desc": self.desc, "task": self.task.name, "n": self.n, "m": self.m, "K": self.K,
            "k_init": self.k_init, "max_steps": self.max_steps, "seed": self.seed,
            "keep_pruned_scores": self.keep_pruned_scores, "pruner": self.pruner.config(),
            "lam_gen": self.lam_gen, "early_stop_z": self.early_stop_z,
            "behavior_policy": self.policy.continuation,
            "exploration": {k: getattr(ex, k) for k in ex.__dataclass_fields__},
            "tokenizer": getattr(self.task, "tokenizer", None),
            "backends": {r: b.fingerprint() for r, b in self.api.roles.items()},
        }

    # ================================================================ rollout generator
    def _episode(self, state: dict, forced: str | None, rseed: int, record: bool, is_behavior: bool
                 ) -> Generator[Request, Response, dict]:
        task, ex = self.task, self.policy.exploration
        events: list[dict] = []
        snaps: list[dict] = []
        sample_rng_seed = derive_seed(rseed, "sample")
        t = 0
        while not task.is_terminal(state):
            obs = task.observe(state, "A")
            dmsgs = task.decision_messages(obs)
            dist = None
            if t == 0 and forced is not None:
                label = forced
            else:
                resp = yield Request("A", "decide", dmsgs, derive_seed(rseed, t, "decide"), meta={"obs": obs}, max_tokens=8)
                dist = resp.label_logprobs
                label = parse_label(resp.text)
                if label is None:
                    label = max(dist, key=dist.get) if dist else "NO_COMM"
                    events.append({"t": t, "type": "parse_error", "raw": resp.text[:200]})
                # ---- candidate decision point (spec §7)
                if dist:
                    p_ask = math.exp(dist["ASK_B"]) + math.exp(dist["ASK_C"])
                    need_info = p_ask >= ex.need_info_threshold
                else:
                    need_info = label != "NO_COMM"
                if is_behavior and state["step"] == 0:
                    p = ex.p_root
                else:
                    p = ex.p_high if need_info else ex.p_low
                u = random.Random(derive_seed(sample_rng_seed, t)).random()
                if record and u < p:
                    snaps.append({"state": task.clone_state(state), "messages": dmsgs, "dist": dist,
                                  "need_info": need_info, "sample_prob": p, "step": state["step"]})
            amsgs = task.action_messages(obs, label)
            resp = yield Request("A", "act", amsgs, derive_seed(rseed, t, "act"), meta={"obs": obs, "decision": label}, max_tokens=96)
            action = task.parse_action(resp.text, label)
            ev = {"t": t, "step": state["step"], "decision": label, "forced": t == 0 and forced is not None,
                  "raw": resp.text[:500], "action": dict(action)}
            if action["type"] == "ask":
                to = action["to"]
                rmsgs = task.responder_messages(state, to, action["content"])
                r2 = yield Request(to, "respond", rmsgs, derive_seed(rseed, t, "respond"),
                                   meta={"obs": task.observe(state, to), "question": action["content"]}, max_tokens=160)
                action["reply"] = r2.text
                ev["reply"] = r2.text
            task.step(state, action)
            events.append(ev)
            t += 1
        result = task.evaluate(state)
        return {"result": result, "events": events, "snaps": snaps}

    def _drive(self, gens: list[Generator]) -> list[dict]:
        """Advance all generators in lock-step; each round's requests form one batch."""
        results: list[Any] = [None] * len(gens)
        pending: dict[int, Request] = {}

        def advance(i, resp):
            try:
                pending[i] = gens[i].send(resp) if resp is not None else next(gens[i])
            except StopIteration as e:
                results[i] = e.value

        for i in range(len(gens)):
            advance(i, None)
        while pending:
            idx = sorted(pending)
            reqs = [pending.pop(i) for i in idx]
            resps = self.api.batch(reqs)
            for i, r in zip(idx, resps):
                advance(i, r)
        return results

    # ================================================================ build
    def build(self) -> "DataTree":
        reuse = {}
        if self.sub:
            if self.sub.get("meta", {}).get("config_hash") == stable_hash(self.config()):
                reuse = {t["scenario_id"]: t for t in self.sub.get("trees", [])}
        todo = [sc for sc in self.scenarios if sc["scenario_id"] not in reuse]

        # ---- layer -1: behaviour runs
        gens = [self._episode(self.task.initial_state(sc), None, derive_seed(self.seed, sc["scenario_id"], "behavior"),
                              record=True, is_behavior=True) for sc in todo]
        outs = self._drive(gens)
        new_trees, layer = {}, []
        ex = self.policy.exploration
        for sc, out in zip(todo, outs):
            tid = f"{sc['scenario_id']}.behavior"
            self.trajectories[tid] = self._traj_record(tid, sc, None, None, None, out)
            snaps = out["snaps"][: ex.max_roots_per_scenario] if ex.max_roots_per_scenario else out["snaps"]
            roots = [self._make_node(sc, s, f"{sc['scenario_id']}_t{s['step']}", 0, None) for s in snaps]
            new_trees[sc["scenario_id"]] = {"scenario_id": sc["scenario_id"], "scenario_type": sc["scenario_type"],
                                            "behavior_trajectory_id": tid, "behavior_result": out["result"], "roots": roots}
            layer += roots

        for depth in range(self.n):
            if not layer:
                break
            layer = self._process_layer(layer, depth)

        self.trees = [reuse.get(sc["scenario_id"]) or new_trees[sc["scenario_id"]] for sc in self.scenarios]
        self._built = True
        return self

    def _make_node(self, sc, snap, node_id, depth, parent):
        dist = snap["dist"]
        node = {
            "state_id": node_id, "scenario_id": sc["scenario_id"], "scenario_type": sc["scenario_type"],
            "depth": depth, "step": snap["step"], "parent": parent,
            "policy_input": self.task.render_prompt(snap["messages"]), "policy_messages": snap["messages"],
            "sample_prob": snap["sample_prob"], "need_info": snap["need_info"],
            "behavior_policy": self.policy.continuation,
            "action_logprobs": dist, "children": {}, "dropped": [],
        }
        node["_state"] = snap["state"]  # in-memory only
        return node

    def _process_layer(self, layer: list[dict], depth: int) -> list[dict]:
        pr = self.pruner
        # ---- which actions to try
        plan: list[tuple[dict, str]] = []
        for node in layer:
            acts = list(ACTIONS)
            if self.m < 3:
                lp = node["action_logprobs"] or {}
                acts = sorted(acts, key=lambda a: -lp.get(a, 0.0))[: self.m]
            for a in acts:
                node["children"][a] = {"action": a, "status": None, "prune_decision": None, "prune_reason": None}
                plan.append((node, a))

        # ---- pre-rollout pruner (LLM), batched
        if isinstance(pr, LLMPruner):
            reqs = []
            for node, a in plan:
                dist = node["action_logprobs"]
                a_dist = {k: math.exp(v) for k, v in dist.items()} if dist else {k: 1 / 3 for k in ACTIONS}
                reqs.append(Request("pruner", "prune", LLMPruner.messages(node["policy_input"], a),
                                    derive_seed(self.seed, node["state_id"], a, "prune"),
                                    meta={"a_dist": a_dist, "action": a}, max_tokens=64, temperature=0.0))
            resps = self.api.batch(reqs)
            for (node, a), r in zip(plan, resps):
                d, why = LLMPruner.parse(r.text)
                keep, override = pr.apply_eps(d, derive_seed(self.seed, node["state_id"], a, "eps"))
                c = node["children"][a]
                c.update(prune_decision=d, prune_reason=why, explore_override=override,
                         status="expand" if keep else "prune")
            if not self.keep_pruned_scores:
                kept = []
                for node, a in plan:
                    c = node["children"][a]
                    if c["status"] == "prune":
                        node["dropped"].append({"action": a, "prune_decision": c["prune_decision"], "prune_reason": c["prune_reason"]})
                        del node["children"][a]
                    else:
                        kept.append((node, a))
                plan = kept

        # ---- rollouts (optionally adaptive: k_init first, top up uncertain nodes)
        first_k = self.k_init or self.K
        self._rollouts(plan, 0, first_k, depth)
        if self.k_init:
            topup = []
            for node in {id(n): n for n, _ in plan}.values():
                if self._nocomm_clearly_best(node):
                    node["early_stopped"] = True
                else:
                    topup += [(node, a) for a in node["children"]]
            self._rollouts(topup, first_k, self.K, depth)

        # ---- post-rollout pruners + stats
        nodes = list({id(n): n for n, _ in plan}.values())
        for node in nodes:
            for a, c in list(node["children"].items()):
                c.update(self._stats(c["rollouts"]))
                if pr.needs_stats:
                    keep, why = pr.decide_post(node, a)
                    c.update(prune_decision="EXPAND" if keep else "PRUNE", prune_reason=why, status="expand" if keep else "prune")
                    if not keep and not self.keep_pruned_scores:
                        node["dropped"].append({"action": a, "prune_decision": "PRUNE", "prune_reason": why})
                        del node["children"][a]
            node["siblings_complete"] = all(a in node["children"] and node["children"][a].get("n", 0) > 0 for a in ACTIONS)

        # ---- next layer
        nxt = []
        for node in nodes:
            for a, c in node["children"].items():
                c["subtree"] = []
                snaps = c.pop("_snaps", [])
                if c["status"] != "expand" or depth + 1 >= self.n:
                    c["expanded"] = False
                    continue
                c["expanded"] = True
                sc = node["_state"]["_sc"]
                for s in snaps[: self.policy.exploration.max_children_per_branch]:
                    child = self._make_node(sc, s, f"{node['state_id']}.{a}_t{s['step']}", depth + 1,
                                            {"state_id": node["state_id"], "action": a})
                    c["subtree"].append(child)
                    nxt.append(child)
        for node in nodes:
            node.pop("_state", None)
        return nxt

    def _rollouts(self, plan, k0, k1, depth):
        if not plan or k1 <= k0:
            return
        jobs, gens = [], []
        for node, a in plan:
            c = node["children"][a]
            c.setdefault("rollouts", [])
            for k in range(k0, k1):
                rseed = derive_seed(self.seed, node["state_id"], a, k)
                st = self.task.clone_state(node["_state"])
                gens.append(self._episode(st, a, rseed, record=(k == 0 and depth + 1 < self.n), is_behavior=False))
                jobs.append((node, a, k, rseed))
        outs = self._drive(gens)
        for (node, a, k, rseed), out in zip(jobs, outs):
            c = node["children"][a]
            tid = f"{node['state_id']}.{a}.k{k}"
            sc = node["_state"]["_sc"]
            self.trajectories[tid] = self._traj_record(tid, sc, node["state_id"], a, k, out, rseed)
            lp = node["action_logprobs"]
            res = out["result"]
            c["rollouts"].append({"k": k, "seed": rseed, "R_task": res["R_task"], "C_comm": res["C_comm"],
                                  "n_steps": res["n_steps"], "n_asks": res["n_asks"],
                                  "action_logprob": None if lp is None else round(lp[a], 6), "trajectory_id": tid})
            if k == 0:
                c["_snaps"] = out["snaps"]

    @staticmethod
    def _stats(rollouts):
        R = [r["R_task"] for r in rollouts]
        C = [r["C_comm"] for r in rollouts]
        S = [r["n_steps"] for r in rollouts]
        return {"n": len(rollouts), "R_task_mean": round(mean(R), 6), "R_task_std": round(pstd(R), 6),
                "C_comm_mean": round(mean(C), 6), "C_comm_std": round(pstd(C), 6),
                "n_steps_mean": round(mean(S), 6)}

    def _nocomm_clearly_best(self, node) -> bool:
        ch = node["children"]
        if "NO_COMM" not in ch or not ch["NO_COMM"].get("rollouts"):
            return False
        q0 = q_values(ch["NO_COMM"]["rollouts"], self.lam_gen)
        for a, c in ch.items():
            if a == "NO_COMM" or not c.get("rollouts"):
                continue
            qa = q_values(c["rollouts"], self.lam_gen)
            se = math.sqrt(stderr(q0) ** 2 + stderr(qa) ** 2)
            gap = mean(q0) - mean(qa)
            if not (gap > self.early_stop_z * se) and not (se == 0 and gap > 0):
                return False
        return True

    @staticmethod
    def _traj_record(tid, sc, node_id, action, k, out, seed=None):
        return {"trajectory_id": tid, "scenario_id": sc["scenario_id"], "node_id": node_id, "forced_action": action,
                "k": k, "seed": seed, "result": out["result"], "events": out["events"]}

    # ================================================================ output
    def get_data(self, fmt: str = "json", lam: float = 0.05, mu: float = 0.0, **kw):
        if not self._built:
            raise RuntimeError("call build() first")
        raw = self.raw()
        if fmt == "json":
            return raw
        from . import export
        return export.export(raw, fmt, lam=lam, mu=mu, **kw)

    def raw(self) -> dict:
        cfg = self.config()
        return {"meta": {"config": cfg, "config_hash": stable_hash(cfg), "n_scenarios": len(self.scenarios)},
                "trees": self.trees}

    def save(self, out_dir: str) -> dict:
        from pathlib import Path

        from .utils import write_json, write_jsonl
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        raw = self.raw()
        raw["meta"]["api_stats"] = self.api.stats()
        write_json(p / "tree.json", raw)
        write_jsonl(p / "trajectories.jsonl", (self.trajectories[k] for k in sorted(self.trajectories)))
        write_jsonl(p / "scenarios.jsonl", self.scenarios)
        return {"tree": str(p / "tree.json"), "trajectories": str(p / "trajectories.jsonl"), "scenarios": str(p / "scenarios.jsonl")}


def iter_nodes(raw_or_trees):
    """Depth-first iterator over every decision node of a raw tree dict (or list of trees)."""
    trees = raw_or_trees["trees"] if isinstance(raw_or_trees, dict) else raw_or_trees
    stack = [n for t in reversed(trees) for n in reversed(t["roots"])]
    while stack:
        node = stack.pop()
        yield node
        for a in reversed(list(node["children"])):
            stack.extend(reversed(node["children"][a].get("subtree", [])))
