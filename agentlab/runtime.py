"""The agent runtime: deterministic controls around a non-deterministic planner.

One run = one request handled by one capability for one user. The loop asks the planner for the next action, then:

  capability check  ->  the tool must be in the capability's tool set
  schema validation ->  arguments must match the tool's pydantic schema
  tenant scope      ->  reads and writes only see the user's tenant (other tenants' rows look "not found")
  permission check  ->  the user's role must hold the write's scope
  preconditions     ->  version check (stale reads), business invariants
  approval gate     ->  writes that need a person become proposals; the run pauses until decided
  durable execution ->  write-ahead step record + idempotency key stored with the effect in one transaction,
                        so a crash or retry neither loses nor repeats a side effect
  audit             ->  every attempted write, allowed or not, with the prior state for rollback

Every control can be switched off in `Config`. The evals use that to show what each one prevents.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields

from pydantic import ValidationError

from .capabilities import CAPABILITIES
from .tools import (CONTROL, READS, ROLE_SCOPES, WRITES, InvalidArgs, PermissionDenied, ToolError,
                    ToolNotAvailable, TransientError)

log = logging.getLogger("agentlab")
PAUSED = object()


@dataclass
class Config:
    tenant_scope: bool = True
    capability_tools: bool = True
    permissions: bool = True
    preconditions: bool = True
    approvals: bool = True
    idempotency: bool = True
    write_ahead: bool = True
    mark_untrusted: bool = True
    max_steps: int = 14
    max_retries: int = 3
    autonomy: dict = field(default_factory=dict)  # capability -> autonomy level override


CONTROLS = [f.name for f in fields(Config) if f.type in ("bool", bool)]


def governed(**changes) -> Config:
    return Config(**changes)


def naive() -> Config:
    return Config(**{c: False for c in CONTROLS})


class SimulatedCrash(Exception):
    """Raised at a chosen step and point to emulate the worker process dying."""


class Telemetry:
    def __init__(self):
        self.counters = Counter()
        self.samples = defaultdict(list)

    def inc(self, name: str, n: float = 1, **labels):
        self.counters[(name, tuple(sorted(labels.items())))] += n

    def observe(self, name: str, value: float, **labels):
        self.samples[(name, tuple(sorted(labels.items())))].append(value)

    def get(self, name: str, **labels) -> float:
        want = tuple(sorted(labels.items()))
        return sum(v for (n, l), v in self.counters.items() if n == name and set(want) <= set(l))

    def prometheus(self) -> str:
        out = []
        for (n, labels), v in sorted(self.counters.items()):
            lab = ",".join(f'{k}="{val}"' for k, val in labels)
            out.append(f"agentlab_{n}{{{lab}}} {v}")
        for (n, labels), vals in sorted(self.samples.items()):
            lab = ",".join(f'{k}="{val}"' for k, val in labels)
            sep = "," if lab else ""
            out.append(f"agentlab_{n}_count{{{lab}}} {len(vals)}")
            out.append(f"agentlab_{n}_sum{{{lab}}} {sum(vals):.6f}")
            for q in (0.5, 0.95):
                s = sorted(vals)
                out.append(f'agentlab_{n}{{{lab}{sep}quantile="{q}"}} {s[min(len(s) - 1, int(q * len(s)))]:.6f}')
        return "\n".join(out) + "\n"


@dataclass
class Ctx:
    con: object
    config: Config
    tenant: str
    actor: dict
    run_id: str
    step: int
    idempotency_key: str | None = None
    approved_by: str | None = None


class Runtime:
    def __init__(self, con, planner, config: Config | None = None, telemetry: Telemetry | None = None,
                 hooks: dict | None = None, faults: dict | None = None, crash_at: tuple | None = None):
        self.con, self.planner = con, planner
        self.config = config or Config()
        self.tele = telemetry or Telemetry()
        self.hooks = hooks if hooks is not None else {}      # "tool:id" -> fn(con), fired once after that read
        self.faults = faults if faults is not None else {}   # tool -> list of faults, consumed per attempt
        self.crash_at = crash_at                              # (step, "before_effect" | "after_effect")
        self.instance = uuid.uuid4().hex[:8]                  # a "process"; a new instance resuming = recovery

    # ------------------------------------------------------------------ public API
    def start(self, user_id: str, capability: str, request: str, planner_args: dict | None = None,
              client_request_id: str | None = None) -> str:
        if client_request_id:
            r = self.con.execute("SELECT id FROM runs WHERE client_request_id=?", (client_request_id,)).fetchone()
            if r:
                self.tele.inc("duplicate_requests")
                return r["id"]
        user = self._user(user_id)
        if capability not in CAPABILITIES:
            raise ValueError(f"unknown capability {capability}")
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        state = {"step": 0, "driver": self.instance,
                 "planner": self.planner.init_state(user, capability, request, planner_args or {})}
        now = int(time.time())
        self.con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (run_id, client_request_id, user["tenant_id"], user_id, capability, request, "running",
                          json.dumps(state), now, now))
        self.tele.inc("runs_started", capability=capability)
        self._log("run_started", run_id=run_id, tenant=user["tenant_id"], capability=capability)
        self.drive(run_id)
        return run_id

    def drive(self, run_id: str) -> str:
        """Advance a run until it finishes or pauses. Safe to call again after a crash: it resumes from the last
        checkpoint and reconciles any write that was in flight."""
        while True:
            run = self._run(run_id)
            if run["status"] != "running":
                return run["status"]
            state = json.loads(run["state"])
            if state.get("driver") != self.instance:  # picked up by a new process: recovery
                state["driver"] = self.instance
                if hasattr(self.planner, "on_resume"):
                    self.planner.on_resume(state["planner"])
                self._checkpoint(run_id, state)
            step = state["step"]
            ctx = self._ctx(run, step)
            if step >= self.config.max_steps:
                self._finish(run_id, state, "failed", "step limit reached")
                return "failed"
            inflight = self.con.execute("SELECT * FROM steps WHERE run_id=? AND step=? AND status='executing'",
                                        (run_id, step)).fetchone()
            if inflight:  # crashed mid-write: finish that exact call instead of asking the planner again
                action = json.loads(inflight["args"])
                self.tele.inc("writes_reconciled")
                result = self._execute_write(ctx, action, reconciling=True)
                self._complete(run_id, state, action, result)
                continue
            t0 = time.perf_counter()
            action = self.planner.next(state["planner"])
            self.tele.observe("plan_seconds", time.perf_counter() - t0)
            out = self._dispatch(run, state, ctx, action)
            if out is PAUSED:
                return "awaiting_approval"

    def decide(self, run_id: str, proposal_id: int, decision: str, approver_id: str, note: str = "") -> str:
        run = self._run(run_id)
        approver = self._user(approver_id)
        if approver["tenant_id"] != run["tenant_id"] or "approve" not in ROLE_SCOPES[approver["role"]]:
            raise PermissionError(f"{approver_id} cannot approve actions for this run")
        prop = self.con.execute("SELECT * FROM proposals WHERE id=? AND run_id=?", (proposal_id, run_id)).fetchone()
        if prop is None or prop["status"] != "pending":
            return run["status"]
        state = json.loads(run["state"])
        action = state.pop("pending")["action"]
        ctx = self._ctx(run, prop["step"])
        ctx.approved_by = approver_id
        name = action["tool"]
        schema_, check = WRITES[name][0], WRITES[name][1]
        if decision == "approve":
            try:  # re-check now: the world may have changed while the proposal waited
                check(ctx, schema_.model_validate(action["args"]))
                result = self._execute_write(ctx, action)
            except ToolError as e:
                result = e.as_result()
                self._audit(ctx, name, action["args"], "blocked_at_approval", e.message)
        else:
            result = {"error": "ApprovalRejected", "message": note or "rejected by approver"}
        self.con.execute("UPDATE proposals SET status=?, decided_by=?, note=? WHERE id=?",
                         ("approved" if decision == "approve" else "rejected", approver_id, note, proposal_id))
        self.tele.inc("approvals_decided", decision=decision)
        self._log("approval_decided", run_id=run_id, proposal=proposal_id, decision=decision)
        self.con.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
        self._complete(run_id, state, action, result)
        return self.drive(run_id)

    def rollback(self, audit_id: int, user_id: str) -> dict:
        user = self._user(user_id)
        a = self.con.execute("SELECT * FROM audit WHERE id=?", (audit_id,)).fetchone()
        if a is None or a["tenant_id"] != user["tenant_id"]:
            raise LookupError("audit entry not found")
        if "rollback" not in ROLE_SCOPES[user["role"]]:
            raise PermissionError("role cannot roll back actions")
        if a["outcome"] != "executed" or a["rolled_back"]:
            raise ValueError("nothing to roll back")
        if not WRITES[a["tool"]][4]:
            raise ValueError(f"{a['tool']} is irreversible")
        prior = json.loads(a["prior"])
        cur = self.con.execute(f"SELECT version FROM {prior['table']} WHERE id=?", (prior["id"],)).fetchone()
        if cur["version"] != prior["version_after"]:
            raise ValueError("record changed since this action; roll back by hand")
        sets = ", ".join(f"{k}=?" for k in prior["set"])
        self.con.execute("BEGIN IMMEDIATE")
        self.con.execute(f"UPDATE {prior['table']} SET {sets}, version=version+1 WHERE id=?",
                         (*prior["set"].values(), prior["id"]))
        self.con.execute("UPDATE audit SET rolled_back=1 WHERE id=?", (audit_id,))
        self.con.execute("INSERT INTO audit(run_id, step, tenant_id, actor, tool, args, outcome, detail) VALUES (?,?,?,?,?,?,?,?)",
                         (a["run_id"], a["step"], a["tenant_id"], user_id, a["tool"], a["args"], "rolled_back",
                          f"restored {prior['set']}"))
        self.con.execute("COMMIT")
        self.tele.inc("rollbacks")
        return {"rolled_back": audit_id, "restored": prior["set"]}

    # ------------------------------------------------------------------ the loop
    def _dispatch(self, run, state, ctx, action):
        run_id, name, args = run["id"], action.get("tool"), action.get("args") or {}
        cap = CAPABILITIES[run["capability"]]
        self.tele.inc("tool_calls", tool=name)
        if name == "final":
            self._finish(run_id, state, "completed", action.get("message", ""))
            return None
        if self.config.capability_tools and name not in cap["tools"]:
            result = ToolNotAvailable(f"{name} is not available in the {run['capability']} capability").as_result()
            if name in WRITES:
                self._audit(ctx, name, args, "blocked", result["message"])
            self.tele.inc("blocked", reason="capability")
        elif name == "escalate":
            self._complete(run_id, state, action, {"escalated": True})
            self._finish(run_id, json.loads(self._run(run_id)["state"]), "escalated", args.get("reason", ""))
            return None
        elif name in READS:
            schema_, fn, _ = READS[name]
            try:
                result = fn(ctx, schema_.model_validate(args))
            except ValidationError as e:
                result = InvalidArgs(_short(e)).as_result()
            except ToolError as e:
                result = e.as_result()
            hook = self.hooks.pop(f"{name}:{next(iter(args.values()), '')}", None)
            if hook:  # a concurrent actor changes the record right after the agent read it
                hook(self.con)
        elif name in WRITES:
            result = self._write(run, state, ctx, action)
            if result is PAUSED:
                return PAUSED
        else:
            result = ToolNotAvailable(f"unknown tool {name}").as_result()
        self._complete(run_id, state, action, result)
        return None

    def _write(self, run, state, ctx, action):
        name, args = action["tool"], action.get("args") or {}
        schema_, check, _apply, scope, _rev, _ = WRITES[name]
        try:
            a = schema_.model_validate(args)
        except ValidationError as e:
            return InvalidArgs(_short(e)).as_result()
        if self.config.permissions and scope not in ROLE_SCOPES[ctx.actor["role"]]:
            err = PermissionDenied(f"role '{ctx.actor['role']}' lacks the '{scope}' permission")
            self._audit(ctx, name, args, "denied", err.message)
            self.tele.inc("blocked", reason="permission")
            return err.as_result()
        try:
            reason = check(ctx, a)
        except ToolError as e:
            self._audit(ctx, name, args, "blocked", f"{e.code}: {e.message}")
            self.tele.inc("blocked", reason=e.code)
            return e.as_result()
        level = self.config.autonomy.get(run["capability"], CAPABILITIES[run["capability"]]["autonomy"])
        if self.config.approvals and (reason or level == "propose_only"):
            reason = reason or "capability is propose-only"
            cur = self.con.execute("INSERT INTO proposals(run_id, step, tool, args, reason, status) VALUES (?,?,?,?,?,?)",
                                   (run["id"], ctx.step, name, json.dumps(args), reason, "pending"))
            self._audit(ctx, name, args, "proposed", reason)
            state["pending"] = {"proposal_id": cur.lastrowid, "action": action}
            self.con.execute("UPDATE runs SET status='awaiting_approval', state=?, updated=? WHERE id=?",
                             (json.dumps(state), int(time.time()), run["id"]))
            self.tele.inc("approvals_requested", tool=name)
            self._log("approval_requested", run_id=run["id"], tool=name, reason=reason)
            return PAUSED
        return self._execute_write(ctx, action)

    def _execute_write(self, ctx, action, reconciling: bool = False):
        name, args = action["tool"], action.get("args") or {}
        schema_, _check, apply, *_ = WRITES[name]
        a = schema_.model_validate(args)
        key = f"{ctx.run_id}:{ctx.step}" if self.config.idempotency else None
        ctx.idempotency_key = key
        if self.config.write_ahead and not reconciling:
            self.con.execute("INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?)",
                             (ctx.run_id, ctx.step, "write", name, json.dumps(action), "executing", None))
        for attempt in range(self.config.max_retries + 1):
            if key:
                done = self.con.execute("SELECT result FROM effects WHERE idempotency_key=?", (key,)).fetchone()
                if done:
                    self.tele.inc("duplicate_effects_suppressed", tool=name)
                    return json.loads(done["result"])
            fault = self.faults.get(name, []).pop(0) if self.faults.get(name) else None
            if fault == "unavailable":
                self.tele.inc("retries", tool=name)
                continue
            self._maybe_crash(ctx.step, "before_effect")
            self.con.execute("BEGIN IMMEDIATE")
            try:
                result, prior = apply(ctx, a)
                if key:
                    self.con.execute("INSERT INTO effects VALUES (?,?,?,?)", (key, ctx.run_id, name, json.dumps(result)))
                self._audit(ctx, name, args, "executed", json.dumps(result), prior=prior)
                self.con.execute("COMMIT")
            except ToolError as e:
                self.con.execute("ROLLBACK")
                return e.as_result()
            except Exception:
                self.con.execute("ROLLBACK")
                raise
            self.tele.inc("writes_executed", tool=name)
            self._log("write_executed", run_id=ctx.run_id, step=ctx.step, tool=name, key=key)
            self._maybe_crash(ctx.step, "after_effect")
            if fault == "timeout_after_commit":  # the effect happened but the response was lost: retry
                self.tele.inc("retries", tool=name)
                continue
            return result
        return TransientError(f"{name} failed after {self.config.max_retries + 1} attempts").as_result()

    def _complete(self, run_id, state, action, result):
        self.planner.observe(state["planner"], action, result)
        self.con.execute("INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?)",
                         (run_id, state["step"], "write" if action.get("tool") in WRITES else "call",
                          action.get("tool"), json.dumps(action), "done", json.dumps(result, default=str)))
        state["step"] += 1
        self._checkpoint(run_id, state)

    # ------------------------------------------------------------------ helpers
    def _checkpoint(self, run_id, state):
        self.con.execute("UPDATE runs SET state=?, updated=? WHERE id=?", (json.dumps(state), int(time.time()), run_id))

    def _finish(self, run_id, state, status, message):
        state["final"] = message
        self.con.execute("UPDATE runs SET status=?, state=?, updated=? WHERE id=?",
                         (status, json.dumps(state), int(time.time()), run_id))
        self.tele.inc("runs_finished", status=status)
        self._log("run_finished", run_id=run_id, status=status)

    def _audit(self, ctx, tool, args, outcome, detail, prior=None):
        self.con.execute("INSERT INTO audit(run_id, step, tenant_id, actor, tool, args, outcome, detail, prior, idempotency_key) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (ctx.run_id, ctx.step, ctx.tenant, ctx.actor["id"], tool, json.dumps(args), outcome, detail,
                          json.dumps(prior) if prior else None, ctx.idempotency_key))

    def _maybe_crash(self, step, point):
        if self.crash_at == (step, point):
            self.crash_at = None
            raise SimulatedCrash(f"crash at step {step} {point}")

    def _ctx(self, run, step) -> Ctx:
        return Ctx(self.con, self.config, run["tenant_id"], self._user(run["user_id"]), run["id"], step)

    def _run(self, run_id):
        r = self.con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            raise LookupError(run_id)
        return r

    def _user(self, user_id):
        u = self.con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if u is None:
            raise LookupError(f"unknown user {user_id}")
        return dict(u)

    def _log(self, event, **kw):
        if log.isEnabledFor(logging.INFO):
            log.info(json.dumps({"event": event, **kw}, default=str))


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:3])
