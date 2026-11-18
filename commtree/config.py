"""Build task / backends / policies / DataTree from a YAML or JSON config (see configs/)."""
from __future__ import annotations

from pathlib import Path

from .backends import MockBackend, OpenAICompatBackend, ResponseCache, Router
from .datatree import DataTree
from .policies import ExplorationPolicy, Policies, ScenarioSampler
from .pruners import build_pruner
from .tasks.communication import CommunicationTask
from .tasks.communication_mock import CommunicationMock
from .utils import load_config, read_jsonl


def build_task(cfg: dict) -> CommunicationTask:
    run, sc = cfg.get("run", {}), cfg.get("scenarios", {})
    return CommunicationTask(max_steps=run.get("max_steps", 12), tokenizer=cfg.get("cost", {}).get("tokenizer", "regex"),
                             stale_prob=sc.get("stale_prob", 0.3), cross_owner_prob=sc.get("cross_owner_prob", 0.15),
                             two_hop_prob=sc.get("two_hop_prob", 0.6), n_distractors=sc.get("n_distractors", 5))


def _make_backend(spec: dict, default: dict, mock_cfg: dict, name: str):
    spec = {**default, **spec} if spec.get("type") is None else dict(spec)
    t = spec.pop("type", "mock")
    if t == "mock":
        return MockBackend(CommunicationMock(**mock_cfg), name=spec.get("name", "mock"))
    if t == "openai_compat":
        return OpenAICompatBackend(**spec)
    raise ValueError(f"unknown backend type {t!r} for role {name}")


def build_router(cfg: dict, root: Path | None = None) -> Router:
    b = cfg.get("backends", {"default": {"type": "mock"}})
    default = b.get("default", {"type": "mock"})
    mock_cfg = cfg.get("mock", {})
    roles = {}
    for role in ("A", "B", "C", "pruner"):
        spec = b.get("roles", {}).get(role, "default")
        spec = default if spec == "default" or spec is None else spec
        roles[role] = _make_backend(dict(spec), default, mock_cfg, role)
    cache_path = cfg.get("cache")
    cache = ResponseCache(str((root or Path(".")) / cache_path)) if cache_path else None
    return Router(roles, cache=cache, max_api_calls=cfg.get("run", {}).get("max_api_calls"),
                  cache_mock=cfg.get("cache_mock", False))


def build_policies(cfg: dict) -> Policies:
    run, sc = cfg.get("run", {}), cfg.get("scenarios", {})
    ex = ExplorationPolicy(**cfg.get("exploration", {}))
    return Policies(continuation=run.get("behavior_policy", "A-v0"), exploration=ex,
                    sampler=ScenarioSampler(mix=sc.get("mix", {"A_self": 0.3, "B_key": 0.35, "C_key": 0.35}),
                                            seed=sc.get("seed", 0)))


def load_scenarios(cfg: dict, task, policies: Policies, root: Path | None = None) -> list[dict]:
    sc = cfg.get("scenarios", {})
    if sc.get("file"):
        return read_jsonl((root or Path(".")) / sc["file"])
    return policies.sampler.sample(task, sc.get("n", 20))


def build_datatree(cfg: dict, scenarios: list[dict] | None = None, root: Path | None = None, router: Router | None = None,
                   sub=None) -> DataTree:
    run = cfg.get("run", {})
    task = build_task(cfg)
    policies = build_policies(cfg)
    router = router or build_router(cfg, root)
    scenarios = scenarios if scenarios is not None else load_scenarios(cfg, task, policies, root)
    return DataTree(
        scenario=scenarios, sub=sub, desc=run.get("desc", "Communication Decision"), n=run.get("n", 3), m=run.get("m", 3),
        api=router, policy=policies, K=run.get("K", 8), max_steps=run.get("max_steps", 12),
        pruner=build_pruner(cfg.get("pruner")), keep_pruned_scores=run.get("keep_pruned_scores", True),
        seed=run.get("seed", 0), max_api_calls=run.get("max_api_calls"), task=task, k_init=run.get("k_init"),
        lam_gen=run.get("lam_gen", 0.05),
    )


def from_file(path: str, **kw) -> DataTree:
    cfg = load_config(path)
    return build_datatree(cfg, **kw)
