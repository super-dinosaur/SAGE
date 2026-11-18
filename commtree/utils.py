"""Small shared helpers: deterministic seeding, statistics, token counting, IO."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

ACTIONS = ("NO_COMM", "ASK_B", "ASK_C")
ASK_TARGET = {"ASK_B": "B", "ASK_C": "C"}


# --------------------------------------------------------------------------- seeds
def derive_seed(*parts: Any) -> int:
    """Stable 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:n]


# --------------------------------------------------------------------------- stats
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstd(xs: list[float]) -> float:
    """Population standard deviation (ddof=0), as stored in the raw tree."""
    if not xs:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def stderr(xs: list[float]) -> float:
    """Standard error of the mean (ddof=1). Returns inf for n<2 so tiny samples are never 'significant'."""
    n = len(xs)
    if n < 2:
        return float("inf")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


# --------------------------------------------------------------------------- tokens
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter:
    """Counts tokens for communication cost.

    ``regex`` (default, dependency free) splits on words/punctuation, which is close to
    BPE counts for English. ``tiktoken`` or ``hf:<model>`` use a real tokenizer if installed.
    Whatever is chosen is recorded in the tree meta so costs stay comparable.
    """

    def __init__(self, kind: str = "regex"):
        self.kind = kind
        self._enc = None
        if kind == "tiktoken":
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding("cl100k_base")
        elif kind.startswith("hf:"):
            from transformers import AutoTokenizer  # type: ignore

            self._enc = AutoTokenizer.from_pretrained(kind[3:])

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is None:
            return len(_TOKEN_RE.findall(text))
        return len(self._enc.encode(text))


# --------------------------------------------------------------------------- io
def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs (pip install pyyaml), or use a .json config") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)
