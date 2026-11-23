"""Acceptance + regression tests (spec §13). Run:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commtree import BudgetExceeded, iter_nodes  # noqa: E402
from commtree.backends import label_logprobs_from_tokens  # noqa: E402
from commtree.config import build_datatree, build_router, build_task  # noqa: E402
from commtree.export import export  # noqa: E402
from commtree.report import check_leakage, write_report  # noqa: E402
from commtree.rl import CommEnv, stratified_batches  # noqa: E402
from commtree.utils import ACTIONS  # noqa: E402

BASE = {
    "run": {"seed": 0, "n": 2, "m": 3, "K": 4, "max_steps": 10, "keep_pruned_scores": True, "lam_gen": 0.05},
    "exploration": {"p_high": 0.8, "p_low": 0.2, "p_root": 1.0, "max_roots_per_scenario": 2},
    "scenarios": {"n": 8, "seed": 0},
    "pruner": {"type": "threshold", "thr": 0.2, "lam": 0.05},
    "backends": {"default": {"type": "mock"}},
}


def cfg(**over):
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict) and k not in ("pruner", "backends"):
            c[k].update(v)
        else:
            c[k] = v
    return c


def build(c):
    dt = build_datatree(c)
    dt.build()
    return dt


class TestAcceptance(unittest.TestCase):
    def test_same_seed_identical(self):
        a, b = build(cfg()), build(cfg())
        self.assertEqual(json.dumps(a.raw(), sort_keys=True), json.dumps(b.raw(), sort_keys=True))
        self.assertEqual(json.dumps(a.trajectories, sort_keys=True), json.dumps(b.trajectories, sort_keys=True))
        c = build(cfg(run={"seed": 1}))
        self.assertNotEqual(json.dumps(a.raw(), sort_keys=True), json.dumps(c.raw(), sort_keys=True))

    def test_no_leakage(self):
        dt = build(cfg(scenarios={"n": 20, "seed": 3}))
        rep = check_leakage(dt.raw(), dt.scenarios, dt.task)
        self.assertGreater(rep["nodes_checked"], 0)
        self.assertEqual(rep["n_leaks"], 0, rep["leaks"][:3])
        for n in iter_nodes(dt.raw()):
            sc = next(s for s in dt.scenarios if s["scenario_id"] == n["scenario_id"])
            if sc.get("key_fact") and n["depth"] == 0 and n["step"] == 0:
                self.assertNotIn(sc["key_fact"], n["policy_input"])

    def test_leak_detector_catches_injected_secret(self):
        dt = build(cfg(scenarios={"n": 3}))
        raw = copy.deepcopy(dt.raw())
        node = next(iter_nodes(raw))
        sc = next(s for s in dt.scenarios if s["scenario_id"] == node["scenario_id"])
        node["policy_input"] = node["policy_input"].replace("Teammates:", f"Hint: {sc['answer']}\nTeammates:")
        self.assertGreater(check_leakage(raw, dt.scenarios)["n_leaks"], 0)

    def test_keep_mode_all_actions_have_stats(self):
        dt = build(cfg())
        nodes = list(iter_nodes(dt.raw()))
        self.assertGreater(len(nodes), 0)
        self.assertTrue(any(n["depth"] > 0 for n in nodes), "expected deeper layers")
        for n in nodes:
            self.assertTrue(n["siblings_complete"])
            for a in ACTIONS:
                self.assertEqual(n["children"][a]["n"], 4)
                self.assertEqual(len(n["children"][a]["rollouts"]), 4)

    def test_reexport_with_new_lambda_makes_no_calls(self):
        dt = build(cfg())
        calls = dt.api.calls
        outs = {}
        for lam in (0.0, 0.05, 1.0, 5.0):
            for fmt in ("decisions", "sft", "dpo", "rl"):
                outs[(lam, fmt)] = dt.get_data(fmt, lam=lam)
        self.assertEqual(dt.api.calls, calls)
        # λ really changes Q
        q0 = outs[(0.0, "rl")][0]["actions"]["ASK_B"]["Q"]
        q5 = outs[(5.0, "rl")][0]["actions"]["ASK_B"]["Q"]
        self.assertLess(q5, q0)


class TestPruning(unittest.TestCase):
    def test_llm_pruner_keep_mode(self):
        dt = build(cfg(pruner={"type": "llm", "explore_eps": 0.0}))
        nodes = list(iter_nodes(dt.raw()))
        decisions = [c["prune_decision"] for n in nodes for c in n["children"].values()]
        self.assertIn("PRUNE", decisions)
        self.assertIn("EXPAND", decisions)
        for n in nodes:
            self.assertTrue(n["siblings_complete"])
            for c in n["children"].values():
                self.assertTrue(c["prune_reason"])
                if c["status"] == "prune":
                    self.assertFalse(c["subtree"])
        self.assertGreater(dt.api.calls_by_role.get("pruner", 0), 0)

    def test_llm_pruner_drop_mode(self):
        dt = build(cfg(pruner={"type": "llm", "explore_eps": 0.0}, run={"keep_pruned_scores": False}))
        nodes = list(iter_nodes(dt.raw()))
        dropped = [n for n in nodes if n["dropped"]]
        self.assertTrue(dropped)
        for n in dropped:
            self.assertFalse(n["siblings_complete"])
            for d in n["dropped"]:
                self.assertNotIn(d["action"], n["children"])
        # SFT skips incomplete states, decisions export flags them
        dec = {r["state_id"]: r for r in dt.get_data("decisions")}
        self.assertFalse(dec[dropped[0]["state_id"]]["siblings_complete"])
        self.assertNotIn(dropped[0]["state_id"], {r["state_id"] for r in dt.get_data("sft")})

    def test_explore_eps_overrides(self):
        dt = build(cfg(pruner={"type": "llm", "explore_eps": 1.0}))
        cs = [c for n in iter_nodes(dt.raw()) for c in n["children"].values() if c["prune_decision"] == "PRUNE"]
        self.assertTrue(cs)
        self.assertTrue(all(c["explore_override"] and c["status"] == "expand" for c in cs))

    def test_nobranch_and_hybrid(self):
        dt = build(cfg(pruner={"type": "nobranch"}))
        self.assertTrue(all(n["depth"] == 0 for n in iter_nodes(dt.raw())))
        dt = build(cfg(pruner={"type": "hybrid", "thr": 0.1, "lam": 0.05, "z": 1.0}))
        self.assertTrue(list(iter_nodes(dt.raw())))

    def test_adaptive_k(self):
        dt = build(cfg(run={"K": 8, "k_init": 3}, scenarios={"n": 12}))
        nodes = list(iter_nodes(dt.raw()))
        early = [n for n in nodes if n.get("early_stopped")]
        full = [n for n in nodes if not n.get("early_stopped")]
        for n in early:
            self.assertTrue(all(c["n"] == 3 for c in n["children"].values()))
        for n in full:
            self.assertTrue(all(c["n"] == 8 for c in n["children"].values()))
        self.assertTrue(full)


class TestExportsAndMisc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dt = build(cfg(scenarios={"n": 20, "seed": 5}, run={"K": 6}))

    def test_rl_advantage_loo(self):
        for r in self.dt.get_data("rl", lam=0.1):
            if r["siblings_complete"]:
                self.assertAlmostEqual(sum(v["advantage"] for v in r["actions"].values()), 0.0, places=5)
                for a, v in r["actions"].items():
                    self.assertIsNotNone(v["action_logprob"])
                    self.assertEqual(len(v["returns"]), v["n"])
                self.assertAlmostEqual(r["importance_weight"], 1 / r["sample_prob"], places=5)

    def test_sft_balance_and_weights(self):
        rows = self.dt.get_data("sft", lam=0.05, nocomm_ratio=1.0)
        cnt = {a: sum(r["label"] == a for r in rows) for a in ACTIONS}
        others = [cnt[a] for a in ("ASK_B", "ASK_C") if cnt[a]]
        if others:
            self.assertLessEqual(cnt["NO_COMM"], max(1, round(1.0 * sum(others) / len(others))))
        self.assertAlmostEqual(sum(r["weight"] for r in rows) / len(rows), 1.0, places=4)
        hard = self.dt.get_data("sft", lam=0.05, low_conf="drop")
        self.assertTrue(all(r["confidence"] == "high" for r in hard))
        self.assertEqual(rows[0]["messages"][-1]["role"], "assistant")

    def test_dpo_pairs(self):
        rows = self.dt.get_data("dpo", lam=0.05)
        per = {}
        for r in rows:
            self.assertNotEqual(r["chosen"], r["rejected"])
            self.assertGreater(r["margin"], 0)
            per[r["state_id"]] = per.get(r["state_id"], 0) + 1
        self.assertTrue(all(v <= 2 for v in per.values()))

    def test_batches(self):
        rows = self.dt.get_data("rl", lam=0.05)
        bs = list(stratified_batches(rows, batch_size=16))
        self.assertEqual(sum(len(b) for b in bs), len(rows))

    def test_report_writes(self):
        with tempfile.TemporaryDirectory() as d:
            r = write_report(self.dt.raw(), self.dt.scenarios, d)
            self.assertTrue((Path(d) / "viewer.html").exists())
            self.assertEqual(r["leakage"]["n_leaks"], 0)

    def test_verifier(self):
        task = build_task(BASE)
        sc = self.dt.scenarios[0]
        self.assertEqual(task.verify(sc, sc["answer"]), 1)
        self.assertEqual(task.verify(sc, f"The answer is {sc['answer']}."), 1)
        self.assertEqual(task.verify(sc, None), 0)
        if sc["value_pool"]:
            other = next(v for v in sc["value_pool"] if v != sc["answer"])
            self.assertEqual(task.verify(sc, f"{sc['answer']} or {other}"), 0)

    def test_verifier_allows_intermediate_entities(self):
        task = build_task(BASE)
        for i in range(40):
            sc = task.make_instance(i, 1000 + i)
            if sc["intermediate_values"] and sc["answer_kind"] == "person":
                mid = sc["intermediate_values"][0]
                if mid != sc["answer"]:
                    self.assertEqual(task.verify(sc, f"{sc['answer']} manages {mid}."), 1)
                    return
        self.skipTest("no 2-hop person question sampled")

    def test_scenario_mix(self):
        types = [s["scenario_type"] for s in self.dt.scenarios]
        self.assertEqual(types.count("A_self"), 6)  # 0.3 * 20
        self.assertEqual(types.count("B_key") + types.count("C_key"), 14)

    def test_commenv(self):
        c = cfg()
        env = CommEnv(build_task(c), build_router(c), lam=0.05)
        obs = env.reset(self.dt.scenarios[0], seed=1)
        self.assertIn("Label:", obs["policy_input"])
        done, steps = False, 0
        for d in ["NO_COMM", "ASK_B", "ASK_C"] + ["NO_COMM"] * 10:
            obs, rew, done, info = env.step(d)
            steps += 1
            if done:
                break
        self.assertTrue(done)
        self.assertIn("R_task", info)

    def test_resume_from_sub(self):
        raw = self.dt.raw()
        dt2 = build_datatree(cfg(scenarios={"n": 20, "seed": 5}, run={"K": 6}), sub=raw)
        dt2.build()
        self.assertEqual(dt2.api.calls, 0)
        self.assertEqual(json.dumps(dt2.raw()["trees"], sort_keys=True), json.dumps(raw["trees"], sort_keys=True))

    def test_split_by_scenario(self):
        from commtree.split import assign_splits, write_splits
        from commtree.utils import read_json, read_jsonl
        a = assign_splits(self.dt.scenarios, (0.6, 0.2, 0.2), seed=0)
        self.assertEqual(set(a), {s["scenario_id"] for s in self.dt.scenarios})
        self.assertEqual(a, assign_splits(self.dt.scenarios, (0.6, 0.2, 0.2), seed=0))  # deterministic
        cnt = {k: list(a.values()).count(k) for k in ("train", "val", "test")}
        self.assertEqual(cnt, {"train": 12, "val": 4, "test": 4})
        with tempfile.TemporaryDirectory() as d:
            self.dt.save(d)
            m = write_splits(d, ratios=(0.6, 0.2, 0.2), lams=(0.05, 0.5))
            seen = {}
            for split in ("train", "val", "test"):
                sd = Path(d) / "splits" / split
                ids = {s["scenario_id"] for s in read_jsonl(sd / "scenarios.jsonl")}
                for fname in ("decisions.jsonl", "sft_lam0.05_mu0.01.jsonl", "rl_lam0.5_mu0.01.jsonl"):
                    for r in read_jsonl(sd / fname):
                        sid = r.get("scenario_id") or r["state_id"].split("_t")[0]
                        self.assertIn(sid, ids, f"{fname} in {split} contains a foreign scenario")
                for t in read_json(sd / "tree.json")["trees"]:
                    self.assertNotIn(t["scenario_id"], seen)
                    seen[t["scenario_id"]] = split
                self.assertEqual(m["splits"][split]["n_scenarios"], len(ids))
            # test split keeps the true distribution: every complete decision point gets an SFT row
            sd = Path(d) / "splits" / "test"
            n_complete = sum(1 for r in read_jsonl(sd / "decisions.jsonl") if r["siblings_complete"])
            self.assertEqual(len(read_jsonl(sd / "sft_lam0.05_mu0.01.jsonl")), n_complete)
            # scenario types are spread across splits
            self.assertGreaterEqual(len(m["splits"]["test"]["scenario_types"]), 2)

    def test_budget(self):
        with self.assertRaises(BudgetExceeded):
            build(cfg(run={"max_api_calls": 50}))


# ---------------------------------------------------------------- OpenAI-compatible path (fake local server)
class _FakeLLM(BaseHTTPRequestHandler):
    hits = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        _FakeLLM.hits += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]
        lp = None
        if "Label:" in user:
            text = "ASK_C" if "(none yet)" in user else "NO_COMM"
            first = "ASK" if text.startswith("ASK") else "NO"
            lp = {"content": [
                {"token": first, "logprob": -0.2, "top_logprobs": [{"token": first, "logprob": -0.2},
                                                                   {"token": "NO" if first == "ASK" else "ASK", "logprob": -1.8}]},
                {"token": text[len(first):], "logprob": -0.1,
                 "top_logprobs": [{"token": text[len(first):], "logprob": -0.1}, {"token": "_B", "logprob": -2.5}]}]}
        elif "Your decision: NO_COMM" in user:
            text = "ANSWER: Lisbon"
        elif "Your decision: ASK" in user:
            text = "QUESTION: Who leads the project in my question?"
        elif "Candidate first action" in user:
            text = "EXPAND: looks informative"
        else:
            text = "I don't know."
        out = {"choices": [{"message": {"role": "assistant", "content": text}, "logprobs": lp}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestOpenAICompat(unittest.TestCase):
    def test_label_logprobs(self):
        toks = [{"token": "ASK", "logprob": -0.2, "top_logprobs": [{"token": "ASK", "logprob": -0.2}, {"token": "NO", "logprob": -1.8}]},
                {"token": "_C", "logprob": -0.1, "top_logprobs": [{"token": "_C", "logprob": -0.1}, {"token": "_B", "logprob": -2.5}]}]
        lp = label_logprobs_from_tokens(toks)
        self.assertAlmostEqual(sum(math.exp(v) for v in lp.values()), 1.0, places=6)
        self.assertGreater(lp["ASK_C"], lp["NO_COMM"])
        self.assertGreater(lp["NO_COMM"], lp["ASK_B"])

    def test_http_backend_with_cache_replay(self):
        srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        os.environ["FAKE_LLM_KEY"] = "test"
        try:
            with tempfile.TemporaryDirectory() as d:
                spec = {"type": "openai_compat", "model": "fake", "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                        "api_key_env": "FAKE_LLM_KEY", "max_concurrency": 4, "max_retries": 0}
                c = cfg(backends={"default": spec}, scenarios={"n": 2}, run={"K": 2, "n": 1}, pruner={"type": "llm"},
                        cache=str(Path(d) / "c.sqlite"))
                a = build(c)
                self.assertGreater(a.api.calls, 0)
                self.assertEqual(a.api.errors, 0)
                n0 = next(iter_nodes(a.raw()))
                self.assertIsNotNone(n0["action_logprobs"])
                hits = _FakeLLM.hits
                b = build(c)  # identical run is served from the cache
                self.assertEqual(b.api.calls, 0)
                self.assertEqual(_FakeLLM.hits, hits)
                self.assertEqual(json.dumps(a.raw()["trees"], sort_keys=True), json.dumps(b.raw()["trees"], sort_keys=True))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
