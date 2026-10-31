"""HTTP API for the runtime: runs execute on a background worker; approvals and rollbacks are endpoints.

    uvicorn agentlab.service:default_app --factory      # Claude planner (needs ANTHROPIC_API_KEY)

The caller is identified by the X-User header (a stand-in for real authentication). The tenant always comes from
the user record, never from the request body, so a caller cannot ask for another tenant's data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from .runtime import Config, Runtime, Telemetry
from .world import build, connect


class RunRequest(BaseModel):
    capability: str
    request: str
    client_request_id: str | None = None
    planner_args: dict | None = None  # only used by the offline scripted planner (tests, demos)


class Decision(BaseModel):
    decision: str  # approve | reject
    note: str = ""


def create_app(con, planner, config: Config | None = None) -> FastAPI:
    tele = Telemetry()
    lock = asyncio.Lock()  # one writer at a time on the SQLite connection

    async def worker(q: asyncio.Queue):
        while True:
            fn = await q.get()
            try:
                async with lock:
                    await asyncio.to_thread(fn)
            except Exception as e:  # noqa: BLE001  a failed run must not kill the worker
                logging.getLogger("agentlab").error(json.dumps({"event": "worker_error", "error": str(e)}))
                tele.inc("worker_errors")
            finally:
                q.task_done()

    @asynccontextmanager
    async def lifespan(app):
        app.state.queue = asyncio.Queue()
        task = asyncio.create_task(worker(app.state.queue))
        yield
        task.cancel()

    app = FastAPI(title="Agent runtime", lifespan=lifespan)
    rt = Runtime(con, planner, config or Config(), tele)
    app.state.runtime = rt

    def user(uid):
        u = con.execute("SELECT * FROM users WHERE id=?", (uid or "",)).fetchone()
        if u is None:
            raise HTTPException(401, "unknown user")
        return dict(u)

    def own_run(run_id, u):
        r = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None or r["tenant_id"] != u["tenant_id"]:
            raise HTTPException(404, "run not found")
        return r

    @app.post("/runs", status_code=202)
    async def create_run(body: RunRequest, x_user: str = Header(None)):
        u = user(x_user)
        if body.client_request_id:
            r = con.execute("SELECT id, status FROM runs WHERE client_request_id=?", (body.client_request_id,)).fetchone()
            if r:
                return {"run_id": r["id"], "status": r["status"], "duplicate": True}
        holder = {}

        def go():
            holder["id"] = rt.start(u["id"], body.capability, body.request, body.planner_args, body.client_request_id)
        await app.state.queue.put(go)
        await app.state.queue.join()  # demo simplicity: return once the run has started and settled or paused
        r = con.execute("SELECT status FROM runs WHERE id=?", (holder.get("id"),)).fetchone()
        if r is None:
            raise HTTPException(500, "run failed to start")
        return {"run_id": holder["id"], "status": r["status"]}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, x_user: str = Header(None)):
        r = own_run(run_id, user(x_user))
        state = json.loads(r["state"])
        pending = con.execute("SELECT id, tool, args, reason FROM proposals WHERE run_id=? AND status='pending'",
                              (run_id,)).fetchall()
        return {"run_id": run_id, "status": r["status"], "capability": r["capability"], "final": state.get("final"),
                "pending_approvals": [dict(p) | {"args": json.loads(p["args"])} for p in pending]}

    @app.get("/runs/{run_id}/trace")
    def get_trace(run_id: str, x_user: str = Header(None)):
        own_run(run_id, user(x_user))
        rows = con.execute("SELECT step, tool, args, status, result FROM steps WHERE run_id=? ORDER BY step", (run_id,))
        return [{"step": s["step"], "tool": s["tool"], "status": s["status"],
                 "args": json.loads(s["args"]).get("args"), "result": json.loads(s["result"]) if s["result"] else None}
                for s in rows]

    @app.post("/runs/{run_id}/proposals/{proposal_id}")
    async def decide(run_id: str, proposal_id: int, body: Decision, x_user: str = Header(None)):
        u = user(x_user)
        own_run(run_id, u)
        if body.decision not in ("approve", "reject"):
            raise HTTPException(422, "decision must be approve or reject")
        err = {}

        def go():
            try:
                rt.decide(run_id, proposal_id, body.decision, u["id"], body.note)
            except PermissionError as e:
                err["e"] = str(e)
        await app.state.queue.put(go)
        await app.state.queue.join()
        if err:
            raise HTTPException(403, err["e"])
        return {"run_id": run_id, "status": con.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0]}

    @app.get("/audit")
    def audit(x_user: str = Header(None), run_id: str | None = None):
        u = user(x_user)
        q, p = "SELECT id, run_id, step, actor, tool, args, outcome, detail, rolled_back FROM audit WHERE tenant_id=?", [u["tenant_id"]]
        if run_id:
            q, p = q + " AND run_id=?", p + [run_id]
        return [dict(r) for r in con.execute(q + " ORDER BY id DESC LIMIT 200", p)]

    @app.post("/audit/{audit_id}/rollback")
    def rollback(audit_id: int, x_user: str = Header(None)):
        try:
            return rt.rollback(audit_id, user(x_user)["id"])
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/metrics")
    def metrics():
        return Response(tele.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    return app


def default_app() -> FastAPI:
    from .planners import ClaudePlanner
    from .router import AnthropicProvider, Router
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(os.environ.get("AGENTLAB_DB", "results/service.db"))
    path.parent.mkdir(exist_ok=True)
    con = connect(path) if path.exists() else build(path)
    router = Router([AnthropicProvider("claude-haiku-4-5"), AnthropicProvider("claude-sonnet-4-5")],
                    cache_path="results/llm_cache.jsonl", breaker=True)
    return create_app(con, ClaudePlanner(router))
