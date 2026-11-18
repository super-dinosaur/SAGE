"""Seeded simulators that stand in for LLMs on the distributed-fact QA task.

They read only what the corresponding role is allowed to see (A: its observation; B/C: their
own records + the question), so a mock run exercises exactly the same information flow as a
real run. A's policy is an explicit stochastic policy, so its action log-probs are exact.
"""
from __future__ import annotations

import math
import random

from .communication import FACT_TEMPLATES, parse_facts, parse_question

REL_KEYWORDS = {
    "based_in": ["based", "city"],
    "manager_of": ["manages", "manager"],
    "phone_ext": ["phone", "extension"],
    "lead_of": ["leads", "lead"],
    "review_day": ["review"],
    "budget_code": ["budget"],
}
QUESTION_FORMS = {
    "based_in": "Where is {s} based?",
    "manager_of": "Who manages {s}?",
    "phone_ext": "What is {s}'s phone extension?",
    "lead_of": "Who leads {s}?",
    "review_day": "On which day is the {s} review?",
    "budget_code": "Which budget code does {s} use?",
}


class CommunicationMock:
    def __init__(self, p_notice_stale: float = 0.5, p_good_question: float = 0.85,
                 responder_verbose: float = 0.3, responder_noise: float = 0.0, prune_min_prob: float = 0.12):
        self.p_notice_stale = p_notice_stale
        self.p_good_question = p_good_question
        self.responder_verbose = responder_verbose
        self.responder_noise = responder_noise
        self.prune_min_prob = prune_min_prob

    # ------------------------------------------------------------------ A's beliefs
    @staticmethod
    def _beliefs(obs: dict) -> dict:
        chain = parse_question(obs["question"]) or []
        known: dict[tuple[str, str], tuple[str, bool]] = {}
        looked: set[str] = set()
        failed: dict[tuple[str, str], set[str]] = {}
        for h in obs["history"]:
            if h["type"] == "lookup":
                looked.add(h["query"].lower())
                for f in parse_facts(h["result"]):
                    known.setdefault((f["rel"], f["subj"]), (f["obj"], f["stale"]))
            elif h["type"] == "ask":
                fs = parse_facts(h["reply"])
                for f in fs:
                    known[(f["rel"], f["subj"])] = (f["obj"], False)  # fresh info overrides notes
                h["_facts"] = fs
        # resolve chain
        values: list[str | None] = []
        missing = None   # (rel, subj) first hop that is unknown
        stale_hop = None
        for i, hop in enumerate(chain):
            subj = hop["subj"] if hop["subj"] != "$0" else (values[0] if values else None)
            if subj is None:
                values.append(None)
                continue
            if (hop["rel"], subj) in known:
                v, st = known[(hop["rel"], subj)]
                values.append(v)
                if st and stale_hop is None:
                    stale_hop = (hop["rel"], subj)
            else:
                values.append(None)
                if missing is None:
                    missing = (hop["rel"], subj)
        for h in obs["history"]:
            if h["type"] == "ask":
                for target in (missing, stale_hop):
                    if target and target[1].lower() in h["question"].lower() and not any(
                            (f["rel"], f["subj"]) == target for f in h.get("_facts", [])):
                        failed.setdefault(target, set()).add(h["to"])
        return {"chain": chain, "values": values, "missing": missing, "stale_hop": stale_hop,
                "looked": looked, "failed": failed}

    @staticmethod
    def _preferred(obs: dict, rel: str) -> str:
        """Which teammate the role descriptions suggest for this relation."""
        kws = REL_KEYWORDS[rel]
        for t, desc in obs["teammates"].items():
            if any(k in desc.lower() for k in kws):
                return t
        return "B"

    def _ask_split(self, obs, rel, failed_set, p_ask):
        pref = self._preferred(obs, rel)
        other = "C" if pref == "B" else "B"
        if pref in failed_set and other in failed_set:
            return {f"ASK_{pref}": 0.05, f"ASK_{other}": 0.05}
        if pref in failed_set:
            return {f"ASK_{other}": p_ask * 0.85, f"ASK_{pref}": p_ask * 0.15}
        return {f"ASK_{pref}": p_ask * 0.8, f"ASK_{other}": p_ask * 0.2}

    def decide(self, obs: dict) -> dict[str, float]:
        b = self._beliefs(obs)
        if not b["chain"]:
            return {"NO_COMM": 0.5, "ASK_B": 0.25, "ASK_C": 0.25}
        if b["missing"] is not None:
            rel, subj = b["missing"]
            fs = b["failed"].get(b["missing"], set())
            looked = any(subj.lower() in q for q in b["looked"])
            if len(fs) == 2:
                p_ask = 0.1
            else:
                p_ask = 0.85 if looked else 0.4
            dist = {"NO_COMM": 1 - p_ask, **self._ask_split(obs, rel, fs, p_ask)}
        elif b["values"] and b["values"][-1] is None:  # blocked by an earlier hop
            dist = {"NO_COMM": 0.6, "ASK_B": 0.2, "ASK_C": 0.2}
        else:
            last_rel = b["chain"][-1]["rel"]
            sure = {"NO_COMM": 0.9, **self._ask_split(obs, last_rel, set(), 0.1)}
            if b["stale_hop"] is not None:
                rel = b["stale_hop"][0]
                fs = b["failed"].get(b["stale_hop"], set())
                suspect = {"NO_COMM": 0.25, **self._ask_split(obs, rel, fs, 0.75)}
                p = self.p_notice_stale
                keys = set(sure) | set(suspect)
                dist = {k: (1 - p) * sure.get(k, 0) + p * suspect.get(k, 0) for k in keys}
            else:
                dist = sure
        for a in ("NO_COMM", "ASK_B", "ASK_C"):
            dist.setdefault(a, 0.0)
        # small floor so every action has finite log-prob
        dist = {k: max(v, 0.01) for k, v in dist.items()}
        z = sum(dist.values())
        return {k: v / z for k, v in dist.items()}

    @staticmethod
    def logprobs(dist: dict[str, float]) -> dict[str, float]:
        return {k: math.log(v) for k, v in dist.items()}

    # ------------------------------------------------------------------ A's action text
    def act(self, obs: dict, decision: str, rng: random.Random) -> str:
        b = self._beliefs(obs)
        chain = b["chain"]
        if decision == "NO_COMM":
            if b["missing"] is not None:
                subj = b["missing"][1]
                if not any(subj.lower() in q for q in b["looked"]):
                    return f"LOOKUP: {subj}"
            if chain and b["values"] and b["values"][-1] is not None:
                return f"ANSWER: {b['values'][-1]}"
            if not b["looked"] and chain:
                return f"LOOKUP: {chain[0]['subj']}"
            return "ANSWER: I could not find it" if rng.random() < 0.5 else "ANSWER: unknown"
        # asking: target the missing hop, else the stale hop, else the last hop
        target = b["missing"] or b["stale_hop"]
        if target is None and chain:
            i = len(chain) - 1
            subj = chain[i]["subj"] if chain[i]["subj"] != "$0" else (b["values"][0] or chain[0]["subj"])
            target = (chain[i]["rel"], subj)
        if target is None or rng.random() > self.p_good_question:
            return "QUESTION: Do you know anything that could help me answer my task question?"
        q = QUESTION_FORMS[target[0]].format(s=target[1])
        if rng.random() < 0.4:
            q = "Hi, quick question for my task. " + q + " Thanks!"
        return f"QUESTION: {q}"

    # ------------------------------------------------------------------ B / C
    def respond(self, obs: dict, question: str, rng: random.Random) -> str:
        ql = question.lower()
        facts = [f for line in obs["db"] for f in parse_facts(line)]
        subj_hits = [f for f in facts if f["subj"].lower() in ql]
        if not subj_hits or rng.random() < self.responder_noise:
            return "Sorry, I don't have anything on that. Could you be more specific?"
        rel_hits = [f for f in subj_hits if any(k in ql for k in REL_KEYWORDS[f["rel"]])]
        chosen = rel_hits or subj_hits[:2]
        if rel_hits and rng.random() < self.responder_verbose:
            extra = [f for f in subj_hits if f not in rel_hits][:1]
            chosen = chosen + extra
        return "Sure. " + " ".join(FACT_TEMPLATES[f["rel"]].format(s=f["subj"], o=f["obj"]) for f in chosen)

    # ------------------------------------------------------------------ pruner
    def prune(self, a_dist: dict[str, float], action: str) -> tuple[str, str]:
        p = a_dist.get(action, 0.0)
        if p < self.prune_min_prob:
            return "PRUNE", f"A itself gives {action} only p={p:.2f}; unlikely to be on-policy"
        return "EXPAND", f"{action} is plausible under A (p={p:.2f})"
