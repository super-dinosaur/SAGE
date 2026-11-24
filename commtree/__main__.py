"""Command line interface.

  python -m commtree gen-scenarios --config configs/mock.yaml --out runs/scenarios.jsonl
  python -m commtree build   --config configs/mock.yaml --out runs/mock
  python -m commtree export  --run runs/mock --lam 0.05 [--mu 0.01] [--low-conf drop] [--ipw]
  python -m commtree report  --run runs/mock --lam 0.05
  python -m commtree check   --run runs/mock
  python -m commtree split   --run runs/mock --ratios 0.8 0.1 0.1 --lam 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import BudgetExceeded
from .config import build_datatree, build_policies, build_task
from .export import export_all
from .report import DEFAULT_LAMS, check_leakage, write_report
from .utils import DEFAULT_MU, load_config, read_json, read_jsonl, write_jsonl


def cmd_gen(a):
    cfg = load_config(a.config)
    if a.n:
        cfg.setdefault("scenarios", {})["n"] = a.n
    task, pol = build_task(cfg), build_policies(cfg)
    rows = pol.sampler.sample(task, cfg.get("scenarios", {}).get("n", 20))
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} scenarios -> {a.out}")


def cmd_build(a):
    cfg = load_config(a.config)
    if a.n_scenarios:
        cfg.setdefault("scenarios", {})["n"] = a.n_scenarios
        cfg["scenarios"].pop("file", None)
    if a.scenarios:
        cfg.setdefault("scenarios", {})["file"] = a.scenarios
    sub = a.resume if a.resume else None
    dt = build_datatree(cfg, sub=sub)
    t0 = time.time()
    try:
        dt.build()
    except BudgetExceeded as e:
        print(f"[budget] {e}. Re-run with a larger max_api_calls: cached responses are replayed for free.", file=sys.stderr)
        sys.exit(2)
    paths = dt.save(a.out)
    st = dt.api.stats()
    print(f"built {len(dt.scenarios)} scenarios in {time.time() - t0:.1f}s; api={st}")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    if not a.no_export:
        lams = a.lams or [cfg.get("export", {}).get("lam", 0.05)]
        for lam in lams:
            c = export_all(dt.raw(), Path(a.out) / "exports", lam=lam, mu=a.mu)
            print(f"  exports (λ={lam}): {c}")
        write_report(read_json(paths["tree"]), dt.scenarios, a.out, lam=lams[0], mu=a.mu)
        print(f"  report: {Path(a.out) / 'report.md'}  viewer: {Path(a.out) / 'viewer.html'}")


def cmd_export(a):
    raw = read_json(Path(a.run) / "tree.json")
    kw = {"z": a.z, "low_conf": a.low_conf, "balance": not a.no_balance, "nocomm_ratio": a.nocomm_ratio, "ipw": a.ipw,
          "max_pairs": a.max_pairs}
    for lam in a.lam:
        c = export_all(raw, a.out or Path(a.run) / "exports", lam=lam, mu=a.mu, **kw)
        print(f"λ={lam} μ={a.mu}: {c}")


def cmd_report(a):
    raw = read_json(Path(a.run) / "tree.json")
    sc = read_jsonl(Path(a.run) / "scenarios.jsonl")
    r = write_report(raw, sc, a.run, lam=a.lam, lams=a.lams or DEFAULT_LAMS, mu=a.mu)
    print(json.dumps(r["headroom"], indent=2))
    print(f"leaks: {r['leakage']['n_leaks']}  -> {Path(a.run) / 'report.md'}")


def cmd_check(a):
    raw = read_json(Path(a.run) / "tree.json")
    sc = read_jsonl(Path(a.run) / "scenarios.jsonl")
    leak = check_leakage(raw, sc)
    from .datatree import iter_nodes
    nodes = list(iter_nodes(raw))
    keep = raw["meta"]["config"]["keep_pruned_scores"]
    incomplete = [n["state_id"] for n in nodes if not n.get("siblings_complete")]
    ok = leak["n_leaks"] == 0 and (not keep or not incomplete)
    print(json.dumps({"nodes": len(nodes), "leaks": leak["n_leaks"], "keep_pruned_scores": keep,
                      "incomplete_nodes": len(incomplete), "ok": ok}, indent=2))
    sys.exit(0 if ok else 1)


def cmd_split(a):
    from .split import write_splits
    kw = {"z": a.z, "low_conf": a.low_conf, "balance": not a.no_balance, "nocomm_ratio": a.nocomm_ratio, "ipw": a.ipw,
          "max_pairs": a.max_pairs}
    m = write_splits(a.run, a.out, tuple(a.ratios), a.seed, tuple(a.lam), a.mu, **kw)
    out = a.out or str(Path(a.run) / "splits")
    for name, info in m["splits"].items():
        print(f"{name:5s}: {info['n_scenarios']} scenarios {info['scenario_types']}  "
              f"decisions={info['files']['decisions.jsonl']}")
    print(f"-> {out}/{{train,val,test}}/   manifest: {out}/split.json")


def main(argv=None):
    p = argparse.ArgumentParser(prog="commtree", description="Three-agent communication-decision dataset builder")
    s = p.add_subparsers(dest="cmd", required=True)

    g = s.add_parser("gen-scenarios"); g.add_argument("--config", required=True); g.add_argument("--out", required=True)
    g.add_argument("--n", type=int); g.set_defaults(f=cmd_gen)

    b = s.add_parser("build"); b.add_argument("--config", required=True); b.add_argument("--out", required=True)
    b.add_argument("--scenarios", help="jsonl of scenarios (overrides the config)")
    b.add_argument("--n-scenarios", type=int); b.add_argument("--resume", help="previous tree.json to reuse")
    b.add_argument("--lams", type=float, nargs="*"); b.add_argument("--mu", type=float, default=DEFAULT_MU)
    b.add_argument("--no-export", action="store_true"); b.set_defaults(f=cmd_build)

    e = s.add_parser("export"); e.add_argument("--run", required=True); e.add_argument("--out")
    e.add_argument("--lam", type=float, nargs="+", default=[0.05]); e.add_argument("--mu", type=float, default=DEFAULT_MU)
    e.add_argument("--z", type=float, default=2.0); e.add_argument("--low-conf", choices=["mark", "drop"], default="mark")
    e.add_argument("--no-balance", action="store_true"); e.add_argument("--nocomm-ratio", type=float, default=2.0)
    e.add_argument("--ipw", action="store_true"); e.add_argument("--max-pairs", type=int, default=2)
    e.set_defaults(f=cmd_export)

    r = s.add_parser("report"); r.add_argument("--run", required=True); r.add_argument("--lam", type=float, default=0.05)
    r.add_argument("--mu", type=float, default=DEFAULT_MU)
    r.add_argument("--lams", type=float, nargs="*"); r.set_defaults(f=cmd_report)

    sp = s.add_parser("split", help="scenario-level train/val/test split + per-split exports")
    sp.add_argument("--run", required=True); sp.add_argument("--out")
    sp.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1]); sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--lam", type=float, nargs="+", default=[0.05]); sp.add_argument("--mu", type=float, default=DEFAULT_MU)
    sp.add_argument("--z", type=float, default=2.0); sp.add_argument("--low-conf", choices=["mark", "drop"], default="mark")
    sp.add_argument("--no-balance", action="store_true"); sp.add_argument("--nocomm-ratio", type=float, default=2.0)
    sp.add_argument("--ipw", action="store_true"); sp.add_argument("--max-pairs", type=int, default=2)
    sp.set_defaults(f=cmd_split)

    c = s.add_parser("check"); c.add_argument("--run", required=True); c.set_defaults(f=cmd_check)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
