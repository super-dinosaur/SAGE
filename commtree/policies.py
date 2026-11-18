"""The three policies of spec §5."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .utils import derive_seed


@dataclass
class ExplorationPolicy:
    """Where to branch (spec §7, option 2 with weighted sampling).

    Every A step is a candidate decision point. A candidate where A itself judges that it
    lacks information (P_A(ASK_B)+P_A(ASK_C) >= ``need_info_threshold``, or the sampled label
    is an ASK when log-probs are unavailable) is selected with ``p_high``, others with
    ``p_low``. The initial state of each scenario is selected with ``p_root``. The selection
    probability is stored as ``sample_prob`` so 1/p re-weighting recovers the true distribution.
    """
    p_high: float = 0.8
    p_low: float = 0.2
    p_root: float = 1.0
    need_info_threshold: float = 0.5
    max_roots_per_scenario: int | None = 3
    max_children_per_branch: int = 1


@dataclass
class ScenarioSampler:
    """What tasks to generate: exact class proportions (largest-remainder rounding)."""
    mix: dict = field(default_factory=lambda: {"A_self": 0.3, "B_key": 0.35, "C_key": 0.35})
    seed: int = 0

    def sample(self, task, n: int, start_idx: int = 0) -> list[dict]:
        tot = sum(self.mix.values())
        raw = {k: n * v / tot for k, v in self.mix.items()}
        counts = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
            counts[k] += 1
        types = [k for k, c in counts.items() for _ in range(c)]
        random.Random(derive_seed(self.seed, "mix")).shuffle(types)
        return [task.make_instance(start_idx + i, derive_seed(self.seed, "scenario", start_idx + i), t)
                for i, t in enumerate(types)]


@dataclass
class Policies:
    """``continuation`` is the version label of role A's backend; all siblings of a decision
    point are rolled out with this same A (it is a single backend for the whole build). For RL,
    point role A at the model being trained (CallableBackend) and bump this label."""
    continuation: str = "A-v0"
    exploration: ExplorationPolicy = field(default_factory=ExplorationPolicy)
    sampler: ScenarioSampler = field(default_factory=ScenarioSampler)
