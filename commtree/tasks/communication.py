"""MVP task: distributed-fact QA with three agents (spec §13.1).

A (executor) must answer a 1- or 2-hop question about a small synthetic organisation.
Facts are split between A's private notes (searchable with a local LOOKUP tool, which is
*not* communication), B (People Ops) and C (Project Office). Scenario types:

* ``A_self``  - every fact A needs is in its own notes; asking only adds cost.
* ``B_key``   - a needed fact is missing from (or *stale* in) A's notes and only B has it.
* ``C_key``   - same, held by C.

Two knobs make the decision non-trivial:
* ``stale_prob``: A's note exists but is outdated (marked "[as of 2019]"), so A *thinks* it
  knows. This produces exactly the "A believes it has enough info but should ask" states
  that spec §7 worries about.
* ``cross_owner_prob``: the missing fact sits with the teammate whose role description does
  *not* suggest it, so routing cannot be solved from descriptions alone.

Everything is programmatically verifiable (no LLM judge).
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any

from ..task import Task
from ..utils import ASK_TARGET, TokenCounter

# --------------------------------------------------------------------------- world vocabulary
PEOPLE = [
    "Dana Kim", "Ravi Patel", "Mia Chen", "Omar Haddad", "Lena Novak", "Tom Becker", "Aiko Sato",
    "Lucas Silva", "Nora Jensen", "Ivan Petrov", "Sara Cohen", "Ben Okafor", "Elif Yilmaz",
    "Marco Rossi", "Hana Park", "Yusuf Demir", "Clara Weber", "Leo Martin", "Zara Ahmed", "Kofi Mensah",
    "Ingrid Berg", "Pablo Ruiz", "Maya Singh", "Felix Wagner",
]
PROJECTS = [
    "Project Orion", "Project Lynx", "Project Atlas", "Project Nova", "Project Falcon", "Project Ember",
    "Project Delta", "Project Vega", "Project Iris", "Project Comet", "Project Basalt", "Project Quill",
]
CITIES = [
    "Lisbon", "Oslo", "Nairobi", "Toronto", "Osaka", "Bogota", "Krakow", "Austin", "Dublin", "Seoul",
    "Porto", "Lyon", "Denver", "Hanoi", "Accra", "Perth",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# relation -> (domain owner, subject kind, value kind)
RELATIONS = {
    "based_in": ("B", "person", "city"),
    "manager_of": ("B", "person", "person"),
    "phone_ext": ("B", "person", "ext"),
    "lead_of": ("C", "project", "person"),
    "review_day": ("C", "project", "weekday"),
    "budget_code": ("C", "project", "code"),
}

FACT_TEMPLATES = {
    "based_in": "{s} is based in {o}.",
    "manager_of": "{s} is managed by {o}.",
    "phone_ext": "{s}'s phone extension is {o}.",
    "lead_of": "{s} is led by {o}.",
    "review_day": "The {s} review is on {o}.",
    "budget_code": "{s} uses budget code {o}.",
}
_NAME = r"([A-Z][a-z]+ [A-Z][a-z]+)"
_PROJ = r"(Project [A-Z][a-z]+)"
FACT_PATTERNS = {
    "based_in": re.compile(_NAME + r" is based in ([A-Z][a-z]+)\."),
    "manager_of": re.compile(_NAME + r" is managed by " + _NAME + r"\."),
    "phone_ext": re.compile(_NAME + r"'s phone extension is (\d{4})\."),
    "lead_of": re.compile(_PROJ + r" is led by " + _NAME + r"\."),
    "review_day": re.compile(r"The " + _PROJ + r" review is on ([A-Z][a-z]+day)\."),
    "budget_code": re.compile(_PROJ + r" uses budget code (BC-\d{4})\."),
}
STALE_MARK = "[as of 2019]"

# question templates: (template, chain)
QUESTIONS_2HOP = {
    "based_in": "In which city is the lead of {P} based?",
    "manager_of": "Who manages the lead of {P}?",
    "phone_ext": "What is the phone extension of the lead of {P}?",
}
QUESTIONS_1HOP = {
    "based_in": "In which city is {X} based?",
    "manager_of": "Who manages {X}?",
    "phone_ext": "What is {X}'s phone extension?",
    "lead_of": "Who leads {P}?",
    "review_day": "On which day is the {P} review?",
    "budget_code": "Which budget code does {P} use?",
}

TEAMMATES = {
    "B": "People Ops - knows where each employee is based, who manages whom, and phone extensions.",
    "C": "Project Office - knows who leads each project, project review days and budget codes.",
}

SYSTEM_A = (
    "You are agent A, the executor in a three-agent team. You must answer the task question correctly. "
    "You have private notes that you can search with LOOKUP (free): it returns the note lines that mention the "
    "person or project names you search for, so search one name at a time and follow multi-step questions step by "
    "step. You may also ask teammate B or C, but every word sent or received in a conversation costs the team. "
    "Ask only when it is worth it. When you answer, give only the answer value (e.g. a city, a name, a code)."
)


def render_fact(rel: str, s: str, o: str, stale: bool = False) -> str:
    text = FACT_TEMPLATES[rel].format(s=s, o=o)
    return f"{STALE_MARK} {text}" if stale else text


def parse_facts(text: str) -> list[dict]:
    """Parse templated fact sentences out of free text (used by the mock agents and checks)."""
    out = []
    for line in re.split(r"(?<=\.)\s+|\n", text):
        stale = STALE_MARK in line
        for rel, pat in FACT_PATTERNS.items():
            for m in pat.finditer(line):
                out.append({"rel": rel, "subj": m.group(1), "obj": m.group(2), "stale": stale})
    return out


def parse_question(q: str) -> list[dict] | None:
    """Recover the hop chain from a question (A is allowed to understand its own question)."""
    for rel, t in QUESTIONS_2HOP.items():
        m = re.fullmatch(re.escape(t).replace(re.escape("{P}"), _PROJ), q)
        if m:
            return [{"rel": "lead_of", "subj": m.group(1)}, {"rel": rel, "subj": "$0"}]
    for rel, t in QUESTIONS_1HOP.items():
        pat = re.escape(t).replace(re.escape("{P}"), _PROJ).replace(re.escape("{X}"), _NAME)
        m = re.fullmatch(pat, q)
        if m:
            return [{"rel": rel, "subj": m.group(1)}]
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", s.lower())).strip()


class CommunicationTask(Task):
    name = "distributed_fact_qa"

    def __init__(self, max_steps: int = 12, tokenizer: str = "regex", stale_prob: float = 0.3,
                 cross_owner_prob: float = 0.15, two_hop_prob: float = 0.6, n_distractors: int = 5):
        self.max_steps = max_steps
        self.count_tokens = TokenCounter(tokenizer)
        self.tokenizer = tokenizer
        self.stale_prob = stale_prob
        self.cross_owner_prob = cross_owner_prob
        self.two_hop_prob = two_hop_prob
        self.n_distractors = n_distractors

    # ======================================================================== scenarios
    def make_instance(self, idx: int, rng_seed: int, scenario_type: str | None = None) -> dict:
        rng = random.Random(rng_seed)
        scenario_type = scenario_type or rng.choice(["A_self", "B_key", "C_key"])
        people = rng.sample(PEOPLE, 8)
        projects = rng.sample(PROJECTS, 5)
        facts: dict[tuple[str, str], str] = {}
        for p in people:
            facts[("based_in", p)] = rng.choice(CITIES)
            facts[("manager_of", p)] = rng.choice([x for x in people if x != p])
            facts[("phone_ext", p)] = f"{rng.randint(1000, 9999)}"
        for pr in projects:
            facts[("lead_of", pr)] = rng.choice(people)
            facts[("review_day", pr)] = rng.choice(WEEKDAYS)
            facts[("budget_code", pr)] = f"BC-{rng.randint(1000, 9999)}"

        # --- choose question & chain
        two_hop = rng.random() < self.two_hop_prob
        if scenario_type == "C_key" and not two_hop:
            rel = rng.choice(["lead_of", "review_day", "budget_code"])
        elif scenario_type == "B_key" and not two_hop:
            rel = rng.choice(["based_in", "manager_of", "phone_ext"])
        else:
            rel = rng.choice(list(QUESTIONS_2HOP) if two_hop else list(QUESTIONS_1HOP))
        if two_hop:
            P = rng.choice(projects)
            lead = facts[("lead_of", P)]
            question = QUESTIONS_2HOP[rel].format(P=P)
            hops = [("lead_of", P), (rel, lead)]
        else:
            subj = rng.choice(projects if RELATIONS[rel][1] == "project" else people)
            question = QUESTIONS_1HOP[rel].format(P=subj, X=subj)
            hops = [(rel, subj)]
        answer = facts[hops[-1]]

        # --- which hop is the key one (missing/stale for A)
        key_hop = None
        intended_owner = None
        if scenario_type != "A_self":
            want = scenario_type[0]  # "B" or "C"
            # cross-owner: the fact lives with the teammate whose description does NOT suggest it
            cross = rng.random() < self.cross_owner_prob
            domain_wanted = ("C" if want == "B" else "B") if cross else want
            cands = [h for h in hops if RELATIONS[h[0]][0] == domain_wanted]
            if not cands:  # template has no hop of that domain -> no cross possible
                cross, domain_wanted = False, want
                cands = [h for h in hops if RELATIONS[h[0]][0] == want]
            key_hop = rng.choice(cands)
            intended_owner = want
        else:
            cross = False
        variant = None
        if key_hop is not None:
            variant = "stale" if rng.random() < self.stale_prob else "missing"

        # --- A's notes: chain facts (except key) + distractors
        a_notes: list[str] = []
        for h in hops:
            if h == key_hop:
                if variant == "stale":
                    true = facts[h]
                    pool = [v for v in self._value_pool(h[0], people) if v != true and v != h[1]]
                    if pool:
                        old = rng.choice(pool)
                    else:  # extensions / budget codes
                        old = re.sub(r"\d{4}", lambda _: str(rng.randint(1000, 9999)), true)
                        while old == true:
                            old = re.sub(r"\d{4}", lambda _: str(rng.randint(1000, 9999)), true)
                    a_notes.append(render_fact(h[0], h[1], old, stale=True))
                continue
            a_notes.append(render_fact(h[0], h[1], facts[h]))
        chain_keys = set(hops)
        others = [k for k in facts if k not in chain_keys]
        # keep distractors from revealing the key fact (same subject+relation)
        for k in rng.sample(others, min(self.n_distractors, len(others))):
            a_notes.append(render_fact(k[0], k[1], facts[k]))
        rng.shuffle(a_notes)

        # --- teammates' databases
        b_db = [render_fact(r, s, o) for (r, s), o in facts.items() if RELATIONS[r][0] == "B"]
        c_db = [render_fact(r, s, o) for (r, s), o in facts.items() if RELATIONS[r][0] == "C"]
        if key_hop is not None and cross:
            line = render_fact(key_hop[0], key_hop[1], facts[key_hop])
            src, dst = (b_db, c_db) if RELATIONS[key_hop[0]][0] == "B" else (c_db, b_db)
            src.remove(line)
            dst.append(line)

        return {
            "scenario_id": f"ep_{idx:04d}",
            "scenario_type": scenario_type,
            "variant": variant,
            "cross_owner": cross,
            "question": question,
            "chain": [{"rel": r, "subj": s if i == 0 else "$0"} for i, (r, s) in enumerate(hops)],
            "answer": answer,
            "answer_kind": RELATIONS[hops[-1][0]][2],
            "key_fact": None if key_hop is None else render_fact(key_hop[0], key_hop[1], facts[key_hop]),
            "key_holder": intended_owner,
            "teammates": dict(TEAMMATES),
            "a_notes": a_notes,
            "b_db": b_db,
            "c_db": c_db,
            "entities": people + projects,
            "value_pool": self._value_pool(hops[-1][0], people),
            "intermediate_values": [facts[h] for h in hops[:-1]],
        }

    @staticmethod
    def _value_pool(rel: str, people: list[str]) -> list[str]:
        kind = RELATIONS[rel][2]
        if kind == "city":
            return list(CITIES)
        if kind == "person":
            return list(PEOPLE)
        if kind == "weekday":
            return list(WEEKDAYS)
        return []  # codes/extensions: exact string only

    # ======================================================================== environment
    def initial_state(self, scenario: dict) -> dict:
        return {
            "_sc": scenario,  # shared, read-only reference (not deep-copied)
            "step": 0,
            "max_steps": self.max_steps,
            "history": [],
            "memory": {"B": [], "C": []},
            "done": False,
            "final_answer": None,
            "comm_tokens": 0,
            "n_asks": {"B": 0, "C": 0},
        }

    @staticmethod
    def clone_state(state: dict) -> dict:
        """Isolated copy for branching: everything mutable is copied, the scenario is shared."""
        sc = state["_sc"]
        rest = {k: v for k, v in state.items() if k != "_sc"}
        new = copy.deepcopy(rest)
        new["_sc"] = sc
        return new

    def observe(self, state: dict, role: str) -> dict:
        sc = state["_sc"]
        if role == "A":
            return {
                "question": sc["question"],
                "teammates": sc["teammates"],
                "history": copy.deepcopy(state["history"]),
                "step": state["step"],
                "max_steps": state["max_steps"],
            }
        db = sc["b_db"] if role == "B" else sc["c_db"]
        return {"role": role, "description": sc["teammates"][role], "db": list(db),
                "memory": copy.deepcopy(state["memory"][role])}

    def allowed_actions(self, state: dict) -> list[str]:
        return [] if self.is_terminal(state) else ["NO_COMM", "ASK_B", "ASK_C"]

    def lookup(self, state: dict, query: str) -> str:
        sc = state["_sc"]
        q = query.lower()
        ents = [e for e in sc["entities"] if e.lower() in q]
        if ents:
            hits = [n for n in sc["a_notes"] if any(e in n for e in ents)]
        else:
            words = {w for w in re.findall(r"[a-z]{4,}", q)} - {"project", "what", "which", "where", "about", "does"}
            hits = [n for n in sc["a_notes"] if words & set(re.findall(r"[a-z]{4,}", n.lower()))]
        return "\n".join(hits[:4]) if hits else "No matching notes."

    def step(self, state: dict, action: dict) -> dict:
        s = state  # mutate in place; callers clone before branching
        t = action["type"]
        if t == "lookup":
            s["history"].append({"type": "lookup", "query": action["content"], "result": self.lookup(s, action["content"])})
        elif t == "ask":
            to = action["to"]
            tok = self.count_tokens(action["content"]) + self.count_tokens(action["reply"])
            s["history"].append({"type": "ask", "to": to, "question": action["content"], "reply": action["reply"], "tokens": tok})
            s["memory"][to].append({"question": action["content"], "reply": action["reply"]})
            s["comm_tokens"] += tok
            s["n_asks"][to] += 1
        elif t == "answer":
            s["history"].append({"type": "answer", "text": action["content"]})
            s["final_answer"] = action["content"]
            s["done"] = True
        else:
            s["history"].append({"type": "invalid", "raw": action.get("content", "")[:200]})
        s["step"] += 1
        if s["step"] >= s["max_steps"]:
            s["done"] = True
        return s

    def is_terminal(self, state: dict) -> bool:
        return bool(state["done"])

    def verify(self, scenario: dict, answer: str | None) -> int:
        if not answer:
            return 0
        a = f" {_norm(answer)} "
        gold = _norm(scenario["answer"])
        if f" {gold} " not in a:
            return 0
        # reject "shotgun" answers that also name other candidates of the same kind
        allowed = {_norm(x) for x in scenario.get("intermediate_values", [])}
        allowed |= {_norm(e) for e in scenario.get("entities", []) if e in scenario["question"]}
        for v in scenario["value_pool"]:
            nv = _norm(v)
            if nv != gold and nv not in allowed and f" {nv} " in a:
                return 0
        return 1

    def evaluate(self, state: dict) -> dict[str, Any]:
        return {
            "R_task": self.verify(state["_sc"], state["final_answer"]),
            "C_comm": round(state["comm_tokens"] / 100.0, 4),
            "comm_tokens": state["comm_tokens"],
            "n_steps": state["step"],
            "n_asks": dict(state["n_asks"]),
            "final_answer": state["final_answer"],
        }

    # ======================================================================== prompts
    @staticmethod
    def _render_context(obs: dict) -> str:
        lines = [f"Task question: {obs['question']}", "", "Teammates:"]
        for k, v in obs["teammates"].items():
            lines.append(f"- {k}: {v}")
        lines += ["", f"Step {obs['step'] + 1} of {obs['max_steps']}.", "History of your actions and observations:"]
        if not obs["history"]:
            lines.append("(none yet)")
        for i, h in enumerate(obs["history"], 1):
            if h["type"] == "lookup":
                lines.append(f"{i}. LOOKUP: {h['query']}\n   result: {h['result']}")
            elif h["type"] == "ask":
                lines.append(f"{i}. ASK_{h['to']}: {h['question']}\n   {h['to']} replied: {h['reply']}")
            elif h["type"] == "answer":
                lines.append(f"{i}. ANSWER: {h['text']}")
            else:
                lines.append(f"{i}. (invalid action: {h['raw']})")
        return "\n".join(lines)

    def decision_messages(self, obs: dict) -> list[dict]:
        user = (
            self._render_context(obs)
            + "\n\nDecide your next move. Reply with exactly one label:\n"
            "NO_COMM - act on your own now (LOOKUP your notes or give the final ANSWER)\n"
            "ASK_B - send a question to B\n"
            "ASK_C - send a question to C\n"
            "Label:"
        )
        return [{"role": "system", "content": SYSTEM_A}, {"role": "user", "content": user}]

    def action_messages(self, obs: dict, decision: str) -> list[dict]:
        if decision == "NO_COMM":
            instr = ("You decided not to communicate now. Output exactly one line that starts with LOOKUP: or ANSWER: "
                     "(do not repeat the label NO_COMM):\n"
                     "LOOKUP: <one person or project name>\nor\nANSWER: <only the answer value>")
        else:
            to = ASK_TARGET[decision]
            instr = (f"You decided to ask {to}. Output exactly one line:\nQUESTION: <a short, specific question for {to}>")
        user = self._render_context(obs) + f"\n\nYour decision: {decision}.\n" + instr
        return [{"role": "system", "content": SYSTEM_A}, {"role": "user", "content": user}]

    def responder_messages(self, state: dict, role: str, question: str) -> list[dict]:
        o = self.observe(state, role)
        mem = "\n".join(f"A: {m['question']}\nYou: {m['reply']}" for m in o["memory"]) or "(no earlier messages)"
        system = (f"You are agent {role} ({o['description']}). Answer agent A truthfully and concisely using only "
                  "your records. If your records do not contain the answer, say you don't know. Never invent facts.")
        user = ("Your private records:\n" + "\n".join(o["db"]) + f"\n\nEarlier conversation with A:\n{mem}\n\n"
                f"A asks: {question}\nYour reply:")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def render_prompt(messages: list[dict]) -> str:
        return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)

    def parse_action(self, text: str, decision: str) -> dict:
        text = (text or "").strip()
        if decision == "NO_COMM":
            m = re.search(r"\b(ANSWER|LOOKUP)\s*:\s*(.+)", text, re.I)
            if not m:
                return {"type": "invalid", "content": text}
            return {"type": m.group(1).lower(), "content": m.group(2).strip().splitlines()[0]}
        m = re.search(r"QUESTION\s*:\s*(.+)", text, re.I | re.S)
        q = (m.group(1) if m else text).strip()
        q = re.sub(r"^(ASK_[BC]\s*:\s*)", "", q).strip()
        if not q:
            return {"type": "invalid", "content": text}
        return {"type": "ask", "to": ASK_TARGET[decision], "content": q[:500]}

    # ======================================================================== checks
    def leakage_terms(self, scenario: dict) -> list[str]:
        vals = {scenario["answer"]}
        for line in scenario["b_db"] + scenario["c_db"] + scenario["a_notes"]:
            for f in parse_facts(line):
                vals.add(f["obj"])
        if scenario.get("key_fact"):
            vals.add(scenario["key_fact"])
        return sorted(v for v in vals if v not in scenario["question"])
