"""Run the scenario library against a planner and a runtime configuration; score; summarise; review trajectories."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from .runtime import Runtime, SimulatedCrash, Telemetry
from .scenarios import approver_decision, check, hooks_for
from .world import connect


def clone(base: sqlite3.Connection, path: str | None = None) -> sqlite3.Connection:
    """A private copy of the base database for one scenario (in memory, or on disk for crash tests)."""
    dst = sqlite3.connect(path or ":memory:", isolation_level=None, check_same_thread=False)
    base.backup(dst)
    dst.row_factory = sqlite3.Row
    return dst


def settle(rt: Runtime, sc: dict, run_id: str, max_rounds: int = 5) -> None:
    """Play the human approver until the run stops asking."""
    for _ in range(max_rounds):
        run = rt.con.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
        if run["status"] != "awaiting_approval":
            return
        p = rt.con.execute("SELECT * FROM proposals WHERE run_id=? AND status='pending'", (run_id,)).fetchone()
        decision, note = approver_decision(sc, p)
        rt.decide(run_id, p["id"], decision, f"{sc['tenant']}-owner", note)


def run_scenario(base, sc, planner, config, telemetry=None) -> dict:
    con = clone(base)
    rt = Runtime(con, planner, config, telemetry or Telemetry(), hooks=hooks_for(sc),
                 faults={k: list(v) for k, v in sc.get("faults", {}).items()})
    try:
        run_id = rt.start(sc["user_id"], sc["capability"], sc["request"],
                          {"kind": sc["kind"], "intent": sc["intent"], "scenario_id": sc["id"]})
        settle(rt, sc, run_id)
    except Exception as e:  # a planner or provider failure is a failed run, not a crashed eval
        return {"scenario": sc["id"], "kind": sc["kind"], "category": sc["category"], "capability": sc["capability"],
                "status": "error", "success": False, "unsafe": [], "blocked": 0, "approvals": 0,
                "approvals_rejected": 0, "steps": 0, "final": f"{type(e).__name__}: {e}", "usage": None, "trace": []}
    out = check(sc, con, run_id)
    out["trace"] = trace(con, run_id)
    con.close()
    return out


def trace(con, run_id) -> list[dict]:
    rows = []
    for s in con.execute("SELECT * FROM steps WHERE run_id=? ORDER BY step", (run_id,)):
        a = json.loads(s["args"])
        r = json.loads(s["result"]) if s["result"] else None
        rows.append({"step": s["step"], "tool": a.get("tool"), "args": a.get("args"),
                     "outcome": (r.get("error") if isinstance(r, dict) and "error" in r else "ok") if r else s["status"]})
    for p in con.execute("SELECT step, tool, reason, status FROM proposals WHERE run_id=?", (run_id,)):
        rows.append({"step": p["step"], "tool": p["tool"], "approval": p["status"], "reason": p["reason"]})
    return sorted(rows, key=lambda x: x["step"])


def run_suite(base, suite, planner_factory, config, telemetry=None) -> list[dict]:
    return [run_scenario(base, sc, planner_factory(), config, telemetry) for sc in suite]


def summarise(results: list[dict]) -> dict:
    n = len(results)
    by = defaultdict(list)
    for r in results:
        by[r["category"]].append(r)
    unsafe = defaultdict(int)
    for r in results:
        for u in r["unsafe"]:
            unsafe[u] += 1
    return {"runs": n, "success": sum(r["success"] for r in results) / n,
            "runs_with_unsafe_outcome": sum(bool(r["unsafe"]) for r in results),
            "unsafe_by_type": dict(unsafe),
            "blocked_attempts": sum(r["blocked"] for r in results),
            "approvals_per_100_runs": 100 * sum(r["approvals"] for r in results) / n,
            "approvals_rejected": sum(r["approvals_rejected"] for r in results),
            "errors": sum(r["status"] == "error" for r in results),
            "mean_steps": sum(r["steps"] for r in results) / n,
            "by_category": {c: {"runs": len(v), "success": sum(x["success"] for x in v) / len(v),
                                "unsafe": sum(bool(x["unsafe"]) for x in v)} for c, v in sorted(by.items())}}


# ------------------------------------------------------------------------------------------- crash recovery
def crash_case(base, sc, planner_factory, config, point: str, workdir: str) -> dict:
    """Run a scenario, kill the 'process' at the first write, start a new runtime on the same database, resume."""
    path = str(Path(workdir) / f"{sc['id']}-{point}.db")
    con = clone(base, path)
    con.execute("PRAGMA journal_mode=WAL")
    rt = Runtime(con, planner_factory(), config, hooks=hooks_for(sc), crash_at=None)
    # find the step of the first write by a dry run on a throwaway copy
    probe = clone(base)
    prt = Runtime(probe, planner_factory(), config, hooks=hooks_for(sc))
    pid = prt.start(sc["user_id"], sc["capability"], sc["request"], {"kind": sc["kind"], "intent": sc["intent"],
                                                                     "scenario_id": sc["id"]})
    settle(prt, sc, pid)
    w = probe.execute("SELECT MIN(step) FROM audit WHERE run_id=? AND outcome='executed'", (pid,)).fetchone()[0]
    probe.close()
    if w is None:
        con.close()
        return {"scenario": sc["id"], "point": point, "applicable": False}
    rt.crash_at = (w, point)
    run_id = None
    crashed = False
    try:
        run_id = rt.start(sc["user_id"], sc["capability"], sc["request"],
                          {"kind": sc["kind"], "intent": sc["intent"], "scenario_id": sc["id"]})
        settle(rt, sc, run_id)
    except SimulatedCrash:
        crashed = True
    con.close()
    con2 = connect(path)  # the new process
    if run_id is None:
        run_id = con2.execute("SELECT id FROM runs ORDER BY created DESC LIMIT 1").fetchone()["id"]
    rt2 = Runtime(con2, planner_factory(), config, hooks={})
    status = con2.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()["status"]
    if status == "awaiting_approval":
        settle(rt2, sc, run_id)
    else:
        rt2.drive(run_id)
        settle(rt2, sc, run_id)
    res = check(sc, con2, run_id)
    effects = con2.execute("SELECT COUNT(*) FROM audit WHERE run_id=? AND outcome='executed'", (run_id,)).fetchone()[0]
    con2.close()
    return {"scenario": sc["id"], "kind": sc["kind"], "point": point, "applicable": True, "crashed": crashed,
            "success": res["success"], "duplicate": ("duplicate_side_effect" in res["unsafe"]
                                                     or "over_refund" in res["unsafe"]),
            "writes": effects, "unsafe": res["unsafe"]}


def crash_suite(base, suite, planner_factory, config, points=("before_effect", "after_effect")) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        rows = [crash_case(base, sc, planner_factory, config, p, d) for sc in suite for p in points]
    rows = [r for r in rows if r["applicable"]]
    return {"cases": len(rows), "crashed": sum(r["crashed"] for r in rows),
            "duplicate_effects": sum(r["duplicate"] for r in rows),
            "lost_or_failed": sum(not r["success"] and not r["duplicate"] for r in rows),
            "success": sum(r["success"] for r in rows) / max(len(rows), 1), "rows": rows}


def review(results: list[dict], limit: int = 5) -> str:
    """Trajectory review: the failed or unsafe runs, step by step."""
    bad = [r for r in results if r["unsafe"] or not r["success"]]
    lines = [f"{len(bad)} of {len(results)} runs need review"]
    for r in bad[:limit]:
        lines.append(f"\n== {r['scenario']} ({r['category']}) status={r['status']} unsafe={r['unsafe']}")
        for s in r.get("trace", []):
            if "approval" in s:
                lines.append(f"  {s['step']:>2} approval {s['approval']}: {s['tool']} ({s['reason']})")
            else:
                lines.append(f"  {s['step']:>2} {s['tool']} {json.dumps(s['args'])[:90]} -> {s['outcome']}")
        lines.append(f"  final: {r['final'][:160]}")
    return "\n".join(lines)
