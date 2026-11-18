"""Role-based model backends (spec §4) with batching, response cache and an API budget.

Every model call is a :class:`Request` carrying its own seed. Backends must be pure functions of
the request (mock: by construction; real APIs: via the on-disk cache), which is what makes two
runs with the same seed identical and makes resuming an interrupted build free.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .utils import ACTIONS, stable_hash


@dataclass
class Request:
    role: str                 # "A" | "B" | "C" | "pruner"
    kind: str                 # "decide" | "act" | "respond" | "prune"
    messages: list[dict]
    seed: int
    meta: dict = field(default_factory=dict)  # structured view for mock / tool backends (same info as messages)
    max_tokens: int = 128
    temperature: float | None = None


@dataclass
class Response:
    text: str
    label_logprobs: dict[str, float] | None = None
    cached: bool = False
    error: str | None = None


class BudgetExceeded(RuntimeError):
    pass


class Backend:
    name = "backend"

    def fingerprint(self) -> str:
        return self.name

    def batch(self, reqs: list[Request]) -> list[Response]:  # pragma: no cover - interface
        raise NotImplementedError


# =============================================================================== mock
class MockBackend(Backend):
    """Seeded simulator backend; ``sim`` is a task-specific simulator (e.g. CommunicationMock)."""

    def __init__(self, sim, name: str = "mock"):
        self.sim = sim
        self.name = name

    def _one(self, r: Request) -> Response:
        rng = random.Random(r.seed)
        m = r.meta
        if r.kind == "decide":
            dist = self.sim.decide(m["obs"])
            labels = list(ACTIONS)
            label = rng.choices(labels, weights=[dist[a] for a in labels])[0]
            return Response(text=label, label_logprobs={a: math.log(dist[a]) for a in labels})
        if r.kind == "act":
            return Response(text=self.sim.act(m["obs"], m["decision"], rng))
        if r.kind == "respond":
            return Response(text=self.sim.respond(m["obs"], m["question"], rng))
        if r.kind == "prune":
            d, why = self.sim.prune(m["a_dist"], m["action"])
            return Response(text=f"{d}: {why}")
        raise ValueError(r.kind)

    def batch(self, reqs):
        return [self._one(r) for r in reqs]


# =============================================================================== callable
class CallableBackend(Backend):
    """Wrap any python function ``fn(Request) -> str | Response``.

    Use this to plug the model being trained (local HF / vLLM) in as role A, which turns
    DataTree into an on-policy data collector for PPO (spec §5).
    """

    def __init__(self, fn: Callable[[Request], Any], name: str = "callable", batch_fn=None):
        self.fn, self.batch_fn, self.name = fn, batch_fn, name

    def batch(self, reqs):
        outs = self.batch_fn(reqs) if self.batch_fn else [self.fn(r) for r in reqs]
        return [o if isinstance(o, Response) else Response(text=str(o)) for o in outs]


# =============================================================================== OpenAI-compatible
LABEL_RE = re.compile(r"NO[_ ]?COMM|ASK[_ ]?B|ASK[_ ]?C", re.I)


def parse_label(text: str) -> str | None:
    m = LABEL_RE.search(text or "")
    if not m:
        return None
    s = re.sub(r"[_ ]", "", m.group(0).upper())
    return {"NOCOMM": "NO_COMM", "ASKB": "ASK_B", "ASKC": "ASK_C"}[s]


def label_logprobs_from_tokens(tokens: list[dict], labels=ACTIONS, floor: float = -20.0) -> dict[str, float] | None:
    """Approximate log P(label) from chat-completion ``logprobs.content``.

    Walks the generated tokens; at the first position where a label diverges from the
    generated text, it takes that label's continuation token from ``top_logprobs`` and assumes
    the remaining tokens have probability ~1. Result is renormalised over the 3 labels.
    """
    if not tokens:
        return None
    out = {}
    for L in labels:
        pos, total, found = 0, 0.0, False
        for tok in tokens:
            cands = [(tok["token"], tok["logprob"])] + [(t["token"], t["logprob"]) for t in tok.get("top_logprobs", [])]
            rest = L[pos:]
            hit = None
            for i, (t, lp) in enumerate(cands):
                tt = t.lstrip() if pos == 0 else t
                if tt and (rest.startswith(tt) or tt.startswith(rest)):
                    hit = (i, tt, lp)
                    break
            if hit is None:
                if pos == 0 and not tok["token"].strip():  # leading whitespace token
                    total += tok["logprob"]
                    continue
                break
            i, tt, lp = hit
            total += lp
            pos += len(tt)
            if pos >= len(L) or i != 0:  # complete, or diverged from the sampled path
                found = True
                break
        out[L] = total if found else floor
    z = math.log(sum(math.exp(v) for v in out.values()))
    return {k: v - z for k, v in out.items()}


class OpenAICompatBackend(Backend):
    def __init__(self, model: str, base_url: str = "https://api.siliconflow.cn/v1", api_key_env: str = "SILICONFLOW_API_KEY",
                 temperature: float = 0.7, max_concurrency: int = 8, timeout: float = 60, max_retries: int = 4,
                 logprobs: bool = True, top_logprobs: int = 10, extra: dict | None = None, name: str | None = None, **_):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env
        self.temperature, self.max_concurrency, self.timeout, self.max_retries = temperature, max_concurrency, timeout, max_retries
        self.logprobs, self.top_logprobs, self.extra = logprobs, top_logprobs, extra or {}
        self.name = name or f"openai_compat:{model}"

    def fingerprint(self):
        return f"{self.name}|{self.base_url}|{self.temperature}|{json.dumps(self.extra, sort_keys=True)}"

    def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError(f"environment variable {self.api_key_env} is not set")
        delay, attempt = 2.0, 0
        while True:
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code == 400 and ("logprobs" in payload or "top_logprobs" in payload):
                    # provider/model does not support logprobs: drop them for good and retry (no retry consumed)
                    self.logprobs = False
                    payload = {k: v for k, v in payload.items() if k not in ("logprobs", "top_logprobs")}
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(delay); delay *= 2; continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(delay); delay *= 2; continue
                raise RuntimeError(str(e)) from e

    def _one(self, r: Request) -> Response:
        rng = random.Random(r.seed)
        m = r.meta
        if r.kind == "decide":
            dist = self.sim.decide(m["obs"])
            labels = list(ACTIONS)
            label = rng.choices(labels, weights=[dist[a] for a in labels])[0]
            return Response(text=label, label_logprobs={a: math.log(dist[a]) for a in labels})
        if r.kind == "act":
            return Response(text=self.sim.act(m["obs"], m["decision"], rng))
        if r.kind == "respond":
            return Response(text=self.sim.respond(m["obs"], m["question"], rng))
        if r.kind == "prune":
            d, why = self.sim.prune(m["a_dist"], m["action"])
            return Response(text=f"{d}: {why}")
        raise ValueError(r.kind)

    def batch(self, reqs):
        return [self._one(r) for r in reqs]


# =============================================================================== callable
class CallableBackend(Backend):
    """Wrap any python function ``fn(Request) -> str | Response``.

    Use this to plug the model being trained (local HF / vLLM) in as role A, which turns
    DataTree into an on-policy data collector for PPO (spec §5).
    """

    def __init__(self, fn: Callable[[Request], Any], name: str = "callable", batch_fn=None):
        self.fn, self.batch_fn, self.name = fn, batch_fn, name

    def batch(self, reqs):
        outs = self.batch_fn(reqs) if self.batch_fn else [self.fn(r) for r in reqs]
        return [o if isinstance(o, Response) else Response(text=str(o)) for o in outs]


# =============================================================================== OpenAI-compatible
LABEL_RE = re.compile(r"NO[_ ]?COMM|ASK[_ ]?B|ASK[_ ]?C", re.I)


def parse_label(text: str) -> str | None:
    m = LABEL_RE.search(text or "")
    if not m:
        return None
    s = re.sub(r"[_ ]", "", m.group(0).upper())
    return {"NOCOMM": "NO_COMM", "ASKB": "ASK_B", "ASKC": "ASK_C"}[s]


def label_logprobs_from_tokens(tokens: list[dict], labels=ACTIONS, floor: float = -20.0) -> dict[str, float] | None:
    """Approximate log P(label) from chat-completion ``logprobs.content``.

    Walks the generated tokens; at the first position where a label diverges from the
    generated text, it takes that label's continuation token from ``top_logprobs`` and assumes
    the remaining tokens have probability ~1. Result is renormalised over the 3 labels.
    """
    if not tokens:
        return None
    out = {}
    for L in labels:
        pos, total, found = 0, 0.0, False
        for tok in tokens:
            cands = [(tok["token"], tok["logprob"])] + [(t["token"], t["logprob"]) for t in tok.get("top_logprobs", [])]
            rest = L[pos:]
            hit = None
            for i, (t, lp) in enumerate(cands):
                tt = t.lstrip() if pos == 0 else t
                if tt and (rest.startswith(tt) or tt.startswith(rest)):
                    hit = (i, tt, lp)
                    break
            if hit is None:
                if pos == 0 and not tok["token"].strip():  # leading whitespace token
                    total += tok["logprob"]
                    continue
                break
            i, tt, lp = hit
            total += lp
            pos += len(tt)
            if pos >= len(L) or i != 0:  # complete, or diverged from the sampled path
                found = True
                break
        out[L] = total if found else floor
    z = math.log(sum(math.exp(v) for v in out.values()))
    return {k: v - z for k, v in out.items()}


class OpenAICompatBackend(Backend):
    def __init__(self, model: str, base_url: str = "https://api.siliconflow.cn/v1", api_key_env: str = "SILICONFLOW_API_KEY",
                 temperature: float = 0.7, max_concurrency: int = 8, timeout: float = 60, max_retries: int = 4,
                 logprobs: bool = True, top_logprobs: int = 10, extra: dict | None = None, name: str | None = None, **_):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env
        self.temperature, self.max_concurrency, self.timeout, self.max_retries = temperature, max_concurrency, timeout, max_retries
        self.logprobs, self.top_logprobs, self.extra = logprobs, top_logprobs, extra or {}
        self.name = name or f"openai_compat:{model}"

    def fingerprint(self):
        return f"{self.name}|{self.base_url}|{self.temperature}|{json.dumps(self.extra, sort_keys=True)}"

    def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError(f"environment variable {self.api_key_env} is not set")
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay); delay *= 2; continue
                if e.code == 400 and payload.get("logprobs"):
                    # provider does not support logprobs: retry without them
                    payload = {k: v for k, v in payload.items() if k not in ("logprobs", "top_logprobs")}
                    data = json.dumps(payload).encode()
                    req = urllib.request.Request(req.full_url, data=data, method="POST", headers=dict(req.header_items()))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay); delay *= 2; continue
                raise RuntimeError(str(e)) from e
        raise RuntimeError("unreachable")

    def _one(self, r: Request) -> Response:
        payload = {"model": self.model, "messages": r.messages, "max_tokens": r.max_tokens,
                   "temperature": self.temperature if r.temperature is None else r.temperature,
                   "seed": r.seed, **self.extra}
        if r.kind == "decide" and self.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        try:
            out = self._post(payload)
        except RuntimeError as e:
            return Response(text="", error=str(e))
        ch = out["choices"][0]
        text = ch["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # Qwen3 thinking traces
        lps = None
        if r.kind == "decide":
            content = ((ch.get("logprobs") or {}).get("content")) or []
            lps = label_logprobs_from_tokens(content)
        return Response(text=text, label_logprobs=lps)

    def batch(self, reqs):
        if not reqs:
            return []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(self._one, reqs))


# =============================================================================== router
class ResponseCache:
    """sqlite cache: request fingerprint -> response. Makes real-API runs replayable/resumable."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            row = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(self, items):
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO cache VALUES (?,?)", [(k, json.dumps(v)) for k, v in items])
            self.conn.commit()


class Router:
    """Dispatches a mixed batch of requests to per-role backends, concurrently per backend."""

    def __init__(self, roles: dict[str, Backend], cache: ResponseCache | None = None, max_api_calls: int | None = None,
                 cache_mock: bool = False):
        self.roles = roles
        self.cache = cache
        self.cache_mock = cache_mock
        self.max_api_calls = max_api_calls
        self.calls = 0          # real (non-cached) backend calls
        self.cache_hits = 0
        self.calls_by_role: dict[str, int] = {}
        self.errors = 0

    def backend(self, role: str) -> Backend:
        return self.roles[role]

    def _key(self, r: Request) -> str:
        b = self.roles[r.role]
        return stable_hash([b.fingerprint(), r.kind, r.messages, r.seed, r.max_tokens, r.temperature], 32)

    def batch(self, reqs: list[Request]) -> list[Response]:
        out: list[Response | None] = [None] * len(reqs)
        todo: dict[int, list[int]] = {}  # backend id -> request indices
        keys = {}
        for i, r in enumerate(reqs):
            b = self.roles[r.role]
            use_cache = self.cache is not None and (self.cache_mock or not isinstance(b, MockBackend))
            if use_cache:
                keys[i] = self._key(r)
                hit = self.cache.get(keys[i])
                if hit is not None:
                    out[i] = Response(**hit, cached=True)
                    self.cache_hits += 1
                    continue
            todo.setdefault(id(b), []).append(i)
        n_new = sum(len(v) for v in todo.values())
        if self.max_api_calls is not None and self.calls + n_new > self.max_api_calls:
            raise BudgetExceeded(f"max_api_calls={self.max_api_calls} would be exceeded ({self.calls}+{n_new})")

        def run(idx):
            b = self.roles[reqs[idx[0]].role]
            return idx, b.batch([reqs[i] for i in idx])

        groups = list(todo.values())
        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
            results = list(ex.map(run, groups))
        to_cache = []
        for idx, resps in results:
            for i, resp in zip(idx, resps):
                out[i] = resp
                self.calls += 1
                self.calls_by_role[reqs[i].role] = self.calls_by_role.get(reqs[i].role, 0) + 1
                if resp.error:
                    self.errors += 1
                elif i in keys:
                    to_cache.append((keys[i], {"text": resp.text, "label_logprobs": resp.label_logprobs}))
        if to_cache and self.cache is not None:
            self.cache.put_many(to_cache)
        return out  # type: ignore[return-value]

    def stats(self) -> dict:
        return {"api_calls": self.calls, "cache_hits": self.cache_hits, "errors": self.errors, "by_role": dict(self.calls_by_role)}
