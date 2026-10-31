"""Model router behind one interface: response cache, retries with backoff, fallback to a second provider, and
per-call cost and latency telemetry.

Providers return {"content": [blocks], "usage": {"input", "output"}, "model"}. Errors are split into retryable
(rate limits, overload, 5xx, timeouts) and fatal (bad request, auth). The cache key is a hash of the full request,
so a re-run of an eval suite with unchanged prompts costs nothing; delete the cache file to force fresh calls.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

# US$ per million tokens (input, output). Check current pricing before relying on the cost figures.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-5": (3.0, 15.0)}


class Retryable(Exception):
    pass


class Fatal(Exception):
    pass


class AllProvidersFailed(Exception):
    pass


def price(model: str, usage: dict) -> float:
    key = next((k for k in PRICES if model.startswith(k)), None)
    if key is None:
        return 0.0
    pin, pout = PRICES[key]
    return (usage["input"] * pin + usage["output"] * pout) / 1e6


class AnthropicProvider:
    def __init__(self, model: str, client=None, timeout: float = 60.0):
        import anthropic
        self.model = model
        self.client = client or anthropic.Anthropic(timeout=timeout, max_retries=0)  # the router owns retries
        self.optional = {"temperature": 0, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
        self._anthropic = anthropic

    def complete(self, req: dict) -> dict:
        a = self._anthropic
        kwargs = {"model": self.model, "system": req["system"], "messages": req["messages"], "tools": req["tools"],
                  "max_tokens": req.get("max_tokens", 700), **self.optional}
        try:
            try:
                r = self.client.messages.create(**kwargs)
            except TypeError as e:
                # SDK versions differ in which optional arguments they accept: drop the rejected one and remember
                bad = next((k for k in list(self.optional) if f"'{k}'" in str(e)), None)
                if bad is None:
                    raise Fatal(str(e)) from e
                self.optional.pop(bad)
                return self.complete(req)
        except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError, a.InternalServerError) as e:
            raise Retryable(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code in (408, 409, 429, 529) or e.status_code >= 500:
                raise Retryable(str(e)) from e
            raise Fatal(str(e)) from e
        content = []
        for b in r.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return {"content": content, "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
                "model": r.model, "stop_reason": r.stop_reason}


class SimProvider:
    """For the offline router study: a provider with a latency distribution and an error rate that can spike."""

    def __init__(self, name: str, latency: tuple, error_rate, cost_per_call: float, clock, seed: int = 0):
        self.model, self.latency, self.error_rate, self.cost = name, latency, error_rate, cost_per_call
        self.clock, self.rng = clock, random.Random(seed)

    def complete(self, req: dict) -> dict:
        lat = self.rng.lognormvariate(*self.latency)  # seconds
        self.clock.advance(lat)
        if self.rng.random() < self.error_rate(self.clock.now):
            raise Retryable("overloaded")
        return {"content": [{"type": "text", "text": "ok"}], "usage": {"input": 0, "output": 0},
                "model": self.model, "sim_cost": self.cost}


class Clock:
    def __init__(self, virtual: bool = False):
        self.virtual, self._t = virtual, 0.0

    @property
    def now(self):
        return self._t if self.virtual else time.monotonic()

    def advance(self, dt):
        if self.virtual:
            self._t += dt

    def sleep(self, dt):
        if self.virtual:
            self._t += dt
        else:
            time.sleep(dt)


class Router:
    def __init__(self, providers: list, retries: int = 2, backoff: float = 1.0, cache_path: str | Path | None = None,
                 telemetry=None, clock: Clock | None = None, use_cache: bool = True, breaker: bool = False,
                 breaker_window: float = 60.0, breaker_cooloff: float = 30.0):
        self.providers, self.retries, self.backoff = providers, retries, backoff
        # circuit breaker: after >=5 failures making up >50% of a provider's calls in the last minute, skip it
        # for 30 s instead of spending retries and backoff on it
        self.breaker, self.window, self.cooloff = breaker, breaker_window, breaker_cooloff
        self.history = {id(p): [] for p in providers}
        self.open_until = {id(p): -1.0 for p in providers}
        self.tele, self.clock = telemetry, clock or Clock()
        self.use_cache = use_cache
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["response"]
        self.calls = []  # per-call log: model, latency, tokens, cost, cached, attempts

    def complete(self, req: dict) -> dict:
        key = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
        if self.use_cache and key in self.cache:
            resp = dict(self.cache[key])
            resp["cost"] = 0.0
            self._record(resp["model"], 0.0, resp["usage"], 0.0, True, 0)
            return resp
        t0 = self.clock.now
        attempts = 0
        for n, p in enumerate(self.providers):
            if self.breaker and n < len(self.providers) - 1 and self.clock.now < self.open_until[id(p)]:
                continue  # breaker open: go straight to the next provider
            for i in range(self.retries + 1):
                attempts += 1
                try:
                    resp = p.complete(req)
                    self._health(p, True)
                except Retryable:
                    self._health(p, False)
                    if self.tele:
                        self.tele.inc("model_errors", model=p.model)
                    if i < self.retries:
                        self.clock.sleep(self.backoff * (2 ** i))
                    continue
                except Fatal:
                    break
                cost = resp.pop("sim_cost", None)
                resp["cost"] = cost if cost is not None else price(resp["model"], resp["usage"])
                if p is not self.providers[0] and self.tele:
                    self.tele.inc("model_fallbacks", model=p.model)
                self._record(resp["model"], self.clock.now - t0, resp["usage"], resp["cost"], False, attempts)
                if self.use_cache:
                    self.cache[key] = {k: v for k, v in resp.items() if k != "cost"}
                    if self.cache_path:
                        with self.cache_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": key, "response": self.cache[key]}) + "\n")
                return resp
        self._record("none", self.clock.now - t0, {"input": 0, "output": 0}, 0.0, False, attempts, failed=True)
        raise AllProvidersFailed(f"all providers failed after {attempts} attempts")

    def _health(self, p, ok: bool):
        if not self.breaker:
            return
        now = self.clock.now
        h = [x for x in self.history[id(p)] if now - x[0] <= self.window] + [(now, ok)]
        self.history[id(p)] = h
        fails = sum(not x[1] for x in h)
        if fails >= 5 and fails / len(h) > 0.5:
            self.open_until[id(p)] = now + self.cooloff
            self.history[id(p)] = []

    def _record(self, model, latency, usage, cost, cached, attempts, failed=False):
        self.calls.append({"model": model, "latency": latency, "input": usage["input"], "output": usage["output"],
                           "cost": cost, "cached": cached, "attempts": attempts, "failed": failed})
        if self.tele:
            self.tele.inc("model_calls", model=model, cached=str(cached).lower())
            self.tele.inc("model_cost_usd", cost, model=model)
            self.tele.inc("model_tokens", usage["input"] + usage["output"], model=model)
            if not cached:
                self.tele.observe("model_latency_seconds", latency, model=model)
