"""Every offline number in the README -> results/study.json (about a minute).

    python -m agentlab.study
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evals import crash_suite, review, run_suite, summarise
from .gate import DRILL_KINDS, promotion, regression_gate
from .planners import ScriptedPlanner
from .router import Clock, Router, SimProvider
from .runtime import Config, naive
from .scenarios import KINDS, build_suite
from .world import build

RES = Path("results")
SEEDS = (0, 1, 2)


def matrix(base, suite):
    out = {}
    for rt_name, cfg in (("governed", Config()), ("no controls", naive())):
        for pl_name, rate in (("planner without mistakes", 0.0), ("planner with mistakes (30%)", 0.3)):
            seeds = (0,) if rate == 0 else SEEDS
            res = []
            for sd in seeds:
                res += run_suite(base, suite, lambda sd=sd, rate=rate: ScriptedPlanner(rate, seed=sd), cfg)
            out[f"{rt_name} | {pl_name}"] = {"summary": summarise(res), "results": res}
    return out


def crash_study(base, suite):
    drill = [x for x in suite if x["kind"] in DRILL_KINDS]
    cfgs = {  # business preconditions off to isolate the durability mechanism; the last row is everything on
        "neither": Config(preconditions=False, idempotency=False, write_ahead=False),
        "idempotency key only": Config(preconditions=False, write_ahead=False),
        "write-ahead record only": Config(preconditions=False, idempotency=False),
        "write-ahead + idempotency key": Config(preconditions=False),
        "full runtime": Config(),
    }
    out = {}
    for name, cfg in cfgs.items():
        r = crash_suite(base, drill, lambda: ScriptedPlanner(0.0, seed=0, resume_divergence=0.5), cfg)
        r.pop("rows")
        out[name] = r
    return out


def router_study(n: int = 3000, seed: int = 0):
    """Simulated providers: a primary with a 400-second partial outage, and a slower, pricier fallback."""
    outage = (1000, 1400)
    policies = {"primary only, no retry": (0, False, False), "primary with 2 retries": (2, False, False),
                "retries, then fallback model": (2, True, False),
                "+ circuit breaker": (2, True, True)}
    out = {}
    for name, (retries, fb, brk) in policies.items():
        clock = Clock(virtual=True)
        p = SimProvider("primary", (np.log(1.2), 0.35), lambda t: 0.7 if outage[0] <= t % 3000 < outage[1] else 0.02,
                        0.002, clock, seed)
        f = SimProvider("fallback", (np.log(2.2), 0.35), lambda t: 0.02, 0.006, clock, seed + 1)
        r = Router([p, f] if fb else [p], retries=retries, backoff=0.5, clock=clock, use_cache=False,
                   breaker=brk)
        ok = 0
        for i in range(n):
            clock._t = float(i)  # one request per second
            try:
                r.complete({"i": i})
                ok += 1
            except Exception:
                pass
        lat = np.array([c["latency"] for c in r.calls if not c["failed"]])
        out[name] = {"requests": n, "success": ok / n, "p50_s": float(np.percentile(lat, 50)),
                     "p95_s": float(np.percentile(lat, 95)),
                     "cost_per_1k": 1000 * sum(c["cost"] for c in r.calls) / n,
                     "fallback_share": sum(c["model"] == "fallback" for c in r.calls) / n}
    return out


def main():
    RES.mkdir(exist_ok=True)
    base = build(RES / "base.db")
    suite = build_suite(base, per_kind=5)
    (RES / "scenarios.json").write_text(json.dumps(suite, indent=1))
    print(f"{len(suite)} scenarios across {len(KINDS)} kinds")
    m = matrix(base, suite)
    for k, v in m.items():
        s = v["summary"]
        print(f"  {k:<55} success {s['success']:.1%}  unsafe runs {s['runs_with_unsafe_outcome']}")
    crash = crash_study(base, suite)
    for k, v in crash.items():
        print(f"  crash: {k:<32} {v['duplicate_effects']}/{v['cases']} duplicated  success {v['success']:.0%}")
    gate = regression_gate(base, suite)
    print(f"  gate caught {gate['caught']}/{gate['regressions']}, false alarm on unchanged: {gate['false_alarm_on_unchanged']}")
    router = router_study()
    for k, v in router.items():
        print(f"  router: {k:<30} success {v['success']:.1%}  p95 {v['p95_s']:.1f}s  ${v['cost_per_1k']:.2f}/1k")
    promo = {k: promotion(m[k]["results"]) for k in ("governed | planner without mistakes",
                                                      "governed | planner with mistakes (30%)")}
    (RES / "review_offline.txt").write_text(review(m["governed | planner with mistakes (30%)"]["results"], 8))
    out = {"scenarios": len(suite), "kinds": {k: v[2] for k, v in KINDS.items()},
           "matrix": {k: v["summary"] for k, v in m.items()}, "crash": crash, "gate": gate, "router": router,
           "promotion": promo}
    (RES / "study.json").write_text(json.dumps(out, indent=2, default=float))
    print("-> results/study.json, results/scenarios.json, results/review_offline.txt")


if __name__ == "__main__":
    main()
