"""Abstract Task interface (spec §12.1).

A Task owns scenario generation, the environment state machine, the prompt functions and
the programmatic verifier. DataTree only talks to this interface, so new tasks (e.g. the
legacy word-guessing "interrupt" task, or a real tool-based debugging task) can be plugged
in without touching the tree builder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    name: str = "task"

    # ---- scenarios -------------------------------------------------------------
    @abstractmethod
    def make_instance(self, idx: int, rng_seed: int, scenario_type: str | None = None) -> dict:
        """Create one scenario dict (ground truth + private info of every role)."""

    # ---- environment -----------------------------------------------------------
    @abstractmethod
    def initial_state(self, scenario: dict) -> dict:
        """Return a JSON-serialisable, deep-copyable environment state."""

    @abstractmethod
    def observe(self, state: dict, role: str) -> dict:
        """What ``role`` may see. For role A this must contain no private info of B/C."""

    @abstractmethod
    def allowed_actions(self, state: dict) -> list[str]:
        ...

    @abstractmethod
    def step(self, state: dict, action: dict) -> dict:
        """Apply an executed action (already contains any reply) and return the new state."""

    @abstractmethod
    def is_terminal(self, state: dict) -> bool:
        ...

    @abstractmethod
    def evaluate(self, state: dict) -> dict[str, Any]:
        """Programmatic verifier: returns at least {"R_task": 0|1, "C_comm": float, "n_steps": int}."""

    # ---- prompts (shared by data generation, training and inference) -----------
    @abstractmethod
    def decision_messages(self, obs: dict) -> list[dict]:
        ...

    @abstractmethod
    def action_messages(self, obs: dict, decision: str) -> list[dict]:
        ...

    @abstractmethod
    def responder_messages(self, state: dict, role: str, question: str) -> list[dict]:
        ...

    @abstractmethod
    def parse_action(self, text: str, decision: str) -> dict:
        ...

    @abstractmethod
    def leakage_terms(self, scenario: dict) -> list[str]:
        """Strings that must never appear in A's prompt unless A has observed them."""
