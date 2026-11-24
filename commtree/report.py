"""Reports and sanity checks (spec §13): headroom, λ sweep, class balance, leakage, tree viewer."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .datatree import iter_nodes
from .export import action_q, export_sft
from .tasks.communication import CommunicationTask
from .utils import ACTIONS, DEFAULT_MU, mean

DEFAULT_LAMS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # C_comm is in units of 100 tokens


# ------------------------------------------------------------------ leakage (§9, §13 acceptance)
def check_leakage(raw: dict, scenarios: list[dict], task: CommunicationTask | None = None) -> dict:
    """A's prompt may only contain private values that arrived through a legitimate channel.

    * static part (task text, teammate descriptions, instructions): no private value at all;
    * LOOKUP results: values must come from A's own notes;
    * replies from B / C: values should come from that teammate's records (else: hallucination warning).
    """
    task = task or CommunicationTask()
    by_id = {s["scenario_id"]: s for s in scenarios}
    leaks, warnings, checked = [], [], 0
    for node in iter_nodes(raw):
        sc = by_id[node["scenario_id"]]
        terms = task.leakage_terms(sc)
        text = node["policy_input"]
        head, _, rest = text.partition("History of your actions and observations:")
        hist, _, tail = rest.partition("Decide your next move.")
        static = head + tail
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", static):
                leaks.append({"state_id": node["state_id"], "term": t, "where": "static"})
        channel, buf = None, []

        def flush():
            if channel is None:
                return
            body = "\n".join(buf)
            src = {"notes": "\n".join(sc["a_notes"]), "B": "\n".join(sc["b_db"]), "C": "\n".join(sc["c_db"])}[channel]
            for t in terms:
                if re.search(rf"\b{re.escape(t)}\b", body) and t not in src:
                    (leaks if channel == "notes" else warnings).append(
                        {"state_id": node["state_id"], "term": t, "where": f"{channel} channel"})

        for line in hist.splitlines():
            if re.match(r"^\d+\. ", line):
                flush(); channel, buf = None, []
            m = re.match(r"^\s+result: (.*)", line)
            m2 = re.match(r"^\s+([BC]) replied: (.*)", line)
            if m:
                flush(); channel, buf = "notes", [m.group(1)]
            elif m2:
                flush(); channel, buf = m2.group(1), [m2.group(2)]
            elif channel is not None and not re.match(r"^\d+\. ", line):
                buf.append(line)
        flush()
        checked += 1
    return {"nodes_checked": checked, "leaks": leaks, "n_leaks": len(leaks),
            "reply_warnings": warnings[:50], "n_reply_warnings": len(warnings)}


# ------------------------------------------------------------------ policy values
def policy_values(raw: dict, lam: float, mu: float = DEFAULT_MU, ipw: bool = True) -> dict:
    """Value of fixed first-action policies over complete decision points (headroom, §13.4)."""
    pols = {"oracle": None, "never_ask": "NO_COMM", "always_B": "ASK_B", "always_C": "ASK_C", "always_ask_best": "ASKBEST"}
    acc = {p: {"V": [], "succ": [], "cost": [], "w": []} for p in pols}
    for node in iter_nodes(raw):
        if not node.get("siblings_complete"):
            continue
        q = action_q(node, lam, mu)
        c = node["children"]
        w = 1 / node["sample_prob"] if ipw else 1.0
        for p, a in pols.items():
            if a is None:
                a = max(q, key=lambda x: (q[x]["Q"], -ACTIONS.index(x)))
            elif a == "ASKBEST":
                a = max(("ASK_B", "ASK_C"), key=lambda x: q[x]["Q"])
            acc[p]["V"].append(q[a]["Q"]); acc[p]["succ"].append(c[a]["R_task_mean"])
            acc[p]["cost"].append(c[a]["C_comm_mean"]); acc[p]["w"].append(w)
    out = {}
    for p, d in acc.items():
        W = sum(d["w"]) or 1.0
        wm = lambda xs: sum(x * w for x, w in zip(xs, d["w"])) / W  # noqa: E731
        out[p] = {"value": round(wm(d["V"]), 4), "success": round(wm(d["succ"]), 4), "cost": round(wm(d["cost"]), 4),
                  "n_states": len(d["V"])}
    return out


def lambda_sweep(raw: dict, lams=DEFAULT_LAMS, mu: float = DEFAULT_MU) -> list[dict]:
    rows = []
    for lam in lams:
        pv = policy_values(raw, lam, mu)
        sft = export_sft(raw, lam=lam, mu=mu, balance=False)
        dist = {a: sum(1 for r in sft if r["label"] == a) for a in ACTIONS}
        hi = sum(1 for r in sft if r["confidence"] == "high")
        for p, v in pv.items():
            rows.append({"lam": lam, "policy": p, **v})
        rows.append({"lam": lam, "policy": "_labels", **{f"n_{a}": dist[a] for a in ACTIONS}, "n_high_conf": hi})
    return rows


# ------------------------------------------------------------------ summary
def summarize(raw: dict, scenarios: list[dict], trajectories: list[dict] | None = None, lam: float = 0.05) -> dict:
    nodes = list(iter_nodes(raw))
    by_depth, by_type = {}, {}
    for n in nodes:
        by_depth[n["depth"]] = by_depth.get(n["depth"], 0) + 1
        by_type[n["scenario_type"]] = by_type.get(n["scenario_type"], 0) + 1
    n_roll = sum(len(c.get("rollouts", [])) for n in nodes for c in n["children"].values())
    beh = [t["behavior_result"] for t in raw["trees"]]
    sc_types = {}
    for s in scenarios:
        key = s["scenario_type"] + (f"/{s['variant']}" if s.get("variant") else "") + ("/cross" if s.get("cross_owner") else "")
        sc_types[key] = sc_types.get(key, 0) + 1
    sp = [n["sample_prob"] for n in nodes]
    return {
        "n_scenarios": len(scenarios), "scenario_types": sc_types, "n_decision_points": len(nodes),
        "nodes_by_depth": by_depth, "nodes_by_scenario_type": by_type,
        "siblings_complete_frac": round(mean([1.0 if n.get("siblings_complete") else 0.0 for n in nodes]), 4) if nodes else 0,
        "n_rollouts": n_roll, "need_info_frac": round(mean([1.0 if n["need_info"] else 0.0 for n in nodes]), 4) if nodes else 0,
        "sample_prob_values": sorted(set(sp)),
        "behavior_policy_unforced": {"success": round(mean([b["R_task"] for b in beh]), 4),
                                     "cost": round(mean([b["C_comm"] for b in beh]), 4),
                                     "asks_per_episode": round(mean([sum(b["n_asks"].values()) for b in beh]), 4)},
        "api_stats": raw.get("meta", {}).get("api_stats"),
    }


def write_report(raw: dict, scenarios: list[dict], out_dir: str, lam: float = 0.05, lams=DEFAULT_LAMS,
                 mu: float = DEFAULT_MU) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    summ = summarize(raw, scenarios, lam=lam)
    leak = check_leakage(raw, scenarios)
    sweep = lambda_sweep(raw, lams, mu)
    head = policy_values(raw, lam, mu)
    # csv
    cols = ["lam", "policy", "value", "success", "cost", "n_states"]
    with open(p / "lambda_sweep.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in sweep:
            if r["policy"] != "_labels":
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    lab_rows = [r for r in sweep if r["policy"] == "_labels"]
    md = ["# DataTree report", "", f"λ for headline numbers: {lam}   μ (step cost): {mu}", "", "## Summary", "```json",
          json.dumps(summ, indent=2, ensure_ascii=False), "```", "", f"## Headroom at λ={lam} (IPW-weighted, complete decision points)",
          "", "| policy | value | success | cost | n |", "|---|---|---|---|---|"]
    for k, v in head.items():
        md.append(f"| {k} | {v['value']} | {v['success']} | {v['cost']} | {v['n_states']} |")
    o, nv = head["oracle"]["value"], head["never_ask"]["value"]
    ab = max(head["always_B"]["value"], head["always_C"]["value"])
    md += ["", f"Headroom: oracle − never_ask = **{o - nv:+.4f}**, oracle − best fixed ask = **{o - ab:+.4f}**. "
           "If both are small, revise the task design before scaling up (spec §13.4).", "",
           "## Label distribution vs λ (SFT, unbalanced)", "", "| λ | NO_COMM | ASK_B | ASK_C | high-conf |", "|---|---|---|---|---|"]
    for r in lab_rows:
        md.append(f"| {r['lam']} | {r['n_NO_COMM']} | {r['n_ASK_B']} | {r['n_ASK_C']} | {r['n_high_conf']} |")
    md += ["", "## λ sweep (success–cost)", "", "| λ | policy | value | success | cost |", "|---|---|---|---|---|"]
    for r in sweep:
        if r["policy"] != "_labels":
            md.append(f"| {r['lam']} | {r['policy']} | {r['value']} | {r['success']} | {r['cost']} |")
    md += ["", "## Leakage check", "", f"nodes checked: {leak['nodes_checked']}, leaks: **{leak['n_leaks']}**, "
           f"teammate-reply warnings (unsupported values, i.e. hallucination): {leak['n_reply_warnings']}"]
    if leak["leaks"]:
        md += ["", "```json", json.dumps(leak["leaks"][:20], indent=2), "```"]
    (p / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (p / "report.json").write_text(json.dumps({"summary": summ, "headroom": head, "leakage": leak, "sweep": sweep}, indent=2,
                                              ensure_ascii=False), encoding="utf-8")
    write_viewer(raw, p / "viewer.html", lam=lam, mu=mu)
    return {"summary": summ, "headroom": head, "leakage": leak}


# ------------------------------------------------------------------ viewer
def write_viewer(raw: dict, path, lam: float = 0.05, max_nodes: int = 400, mu: float = DEFAULT_MU) -> None:
    esc = html.escape
    count = [0]

    def node_html(n):
        count[0] += 1
        if count[0] > max_nodes:
            return ""
        q = action_q(n, lam, mu)
        rows = []
        for a, c in n["children"].items():
            qa = q.get(a, {}).get("Q")
            sub = "".join(node_html(s) for s in c.get("subtree", []))
            lp = (n.get("action_logprobs") or {}).get(a)
            rows.append(
                f"<details class='act'><summary><b>{a}</b> · n={c.get('n', 0)} · R={c.get('R_task_mean', '–')} · "
                f"C={c.get('C_comm_mean', '–')} · Q={'–' if qa is None else f'{qa:.3f}'} · "
                f"logp={'–' if lp is None else f'{lp:.2f}'} · <i>{c.get('status')}</i></summary>"
                f"<div class='why'>{esc(str(c.get('prune_reason') or ''))}</div>{sub}</details>")
        return (f"<details class='node'><summary>{esc(n['state_id'])} · depth {n['depth']} · p={n['sample_prob']} · "
                f"need_info={n['need_info']} · complete={n.get('siblings_complete')}</summary>"
                f"<pre>{esc(n['policy_input'])}</pre>{''.join(rows)}</details>")

    body = []
    for t in raw["trees"]:
        b = t["behavior_result"]
        body.append(f"<section><h2>{esc(t['scenario_id'])} <small>{esc(t['scenario_type'])} · behaviour: "
                    f"R={b['R_task']} C={b['C_comm']}</small></h2>{''.join(node_html(r) for r in t['roots'])}</section>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataTree Viewer</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--bd:#e7e5e4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--bd:#44403c}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}}
section{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px 12px;margin:10px 0}}
h2{{font-size:15px}} small{{color:var(--mut);font-weight:400}} details{{margin:4px 0 4px 12px}}
pre{{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px;overflow-x:auto}}
.why{{color:var(--mut);font-size:12px;margin-left:12px}}
</style></head><body><h1>DataTree Viewer</h1><p>Q shown at λ={lam}, μ={mu}. Click to expand.</p>{''.join(body)}</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
