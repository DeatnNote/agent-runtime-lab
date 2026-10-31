import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from agentlab.evals import clone, crash_case, run_scenario, run_suite, summarise
from agentlab.gate import context_contract, evaluate, verdict
from agentlab.planners import ClaudePlanner, ScriptedPlanner
from agentlab.router import Clock, Retryable, Router
from agentlab.runtime import Config, Ctx, Runtime, naive
from agentlab.scenarios import build_suite
from agentlab.service import create_app
from agentlab.tools import InvoiceId, NotFound, get_invoice
from agentlab.world import build


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = build(tmp_path_factory.mktemp("w") / "base.db")
    return base, build_suite(base)


def scen(suite, kind, i=0):
    return next(s for s in suite if s["id"] == f"{kind}-{i}")


def test_other_tenants_rows_look_missing(world):
    base, _ = world
    con = clone(base)
    ctx = Ctx(con, Config(), "A", {"id": "A-csr", "role": "csr"}, "t", 0)
    foreign = con.execute("SELECT id FROM invoices WHERE tenant_id='B' LIMIT 1").fetchone()[0]
    with pytest.raises(NotFound) as e1:
        get_invoice(ctx, InvoiceId(invoice_id=foreign))
    with pytest.raises(NotFound) as e2:
        get_invoice(ctx, InvoiceId(invoice_id="A-INV9999"))
    assert e1.value.message.replace(foreign, "X") == e2.value.message.replace("A-INV9999", "X")  # no existence leak


@pytest.mark.parametrize("kind", ["missing_permission", "cross_tenant", "stale_context", "flaky_payments",
                                  "injection_summary", "refund_large", "reschedule_conflict", "warranty_expired"])
def test_governed_runtime_is_safe_and_correct(world, kind):
    base, suite = world
    r = run_scenario(base, scen(suite, kind), ScriptedPlanner(0.0), Config())
    assert r["success"] and not r["unsafe"], r


@pytest.mark.parametrize("kind,control,expected", [
    ("missing_permission", "permissions", "unauthorized_write"),
    ("cross_tenant", "tenant_scope", "cross_tenant_write"),
    ("flaky_payments", "idempotency", "duplicate_side_effect"),
    ("refund_large", "approvals", "refund_over_limit_without_approval"),
])
def test_each_control_prevents_its_failure(world, kind, control, expected):
    base, suite = world
    r = run_scenario(base, scen(suite, kind), ScriptedPlanner(0.0), Config(**{control: False}))
    assert expected in r["unsafe"]


def test_stale_read_caught_by_version_check_then_by_approval(world):
    base, suite = world
    sc = scen(suite, "stale_context")
    r = run_scenario(base, sc, ScriptedPlanner(0.0), Config(preconditions=False))
    assert r["success"] and r["approvals_rejected"] == 1  # second layer: the over-refund went to a person, who said no
    r = run_scenario(base, sc, ScriptedPlanner(0.0), Config(preconditions=False, approvals=False))
    assert "over_refund" in r["unsafe"]


def test_injection_blocked_by_capability_even_when_planner_follows_it(world):
    base, suite = world
    sc = scen(suite, "injection_summary")
    p = lambda: ScriptedPlanner(1.0, mistakes=("follow_injection",))  # noqa: E731
    assert run_scenario(base, sc, p(), Config())["success"]
    assert "write_in_read_only_task" in run_scenario(base, sc, p(), naive())["unsafe"]


def test_approval_pauses_and_rejection_leaves_no_refund(world):
    base, suite = world
    sc = scen(suite, "refund_large")
    con = clone(base)
    rt = Runtime(con, ScriptedPlanner(0.0), Config())
    rid = rt.start(sc["user_id"], sc["capability"], sc["request"], {"kind": sc["kind"], "intent": sc["intent"]})
    assert con.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()[0] == "awaiting_approval"
    p = con.execute("SELECT * FROM proposals WHERE run_id=?", (rid,)).fetchone()
    with pytest.raises(PermissionError):
        rt.decide(rid, p["id"], "approve", f"{sc['tenant']}-csr")  # a CSR cannot approve
    assert rt.decide(rid, p["id"], "reject", f"{sc['tenant']}-owner", "no") == "completed"
    assert con.execute("SELECT COUNT(*) FROM refunds WHERE run_id=?", (rid,)).fetchone()[0] == 0


@pytest.mark.parametrize("point", ["before_effect", "after_effect"])
def test_crash_and_resume_never_duplicates(world, point):
    base, suite = world
    with tempfile.TemporaryDirectory() as d:
        for kind in ("refund_small", "notify_late", "assign"):
            r = crash_case(base, scen(suite, kind), lambda: ScriptedPlanner(0.0, resume_divergence=1.0), Config(), point, d)
            assert r["crashed"] and r["success"] and not r["duplicate"] and r["writes"] == 1, r


def test_crash_without_durability_duplicates(world):
    base, suite = world
    with tempfile.TemporaryDirectory() as d:
        r = crash_case(base, scen(suite, "notify_late"), lambda: ScriptedPlanner(0.0), naive(), "after_effect", d)
    assert r["duplicate"]


def test_rollback_restores_reversible_and_refuses_irreversible(world):
    base, suite = world
    sc = scen(suite, "reschedule")
    con = clone(base)
    rt = Runtime(con, ScriptedPlanner(0.0), Config())
    rid = rt.start(sc["user_id"], sc["capability"], sc["request"], {"kind": sc["kind"], "intent": sc["intent"]})
    a = con.execute("SELECT id FROM audit WHERE run_id=? AND outcome='executed'", (rid,)).fetchone()[0]
    rt.rollback(a, f"{sc['tenant']}-owner")
    j = con.execute("SELECT day, slot FROM jobs WHERE id=?", (sc["intent"]["job_id"],)).fetchone()
    assert [j["day"], j["slot"]] == sc["expect"]["orig"]
    sc2 = scen(suite, "refund_small")
    rid2 = rt.start(sc2["user_id"], sc2["capability"], sc2["request"], {"kind": sc2["kind"], "intent": sc2["intent"]})
    a2 = con.execute("SELECT id FROM audit WHERE run_id=? AND outcome='executed'", (rid2,)).fetchone()[0]
    with pytest.raises(ValueError, match="irreversible"):
        rt.rollback(a2, f"{sc2['tenant']}-owner")


def test_duplicate_client_request_returns_same_run(world):
    base, suite = world
    sc = scen(suite, "refund_small")
    con = clone(base)
    rt = Runtime(con, ScriptedPlanner(0.0), Config())
    args = (sc["user_id"], sc["capability"], sc["request"], {"kind": sc["kind"], "intent": sc["intent"]}, "req-1")
    assert rt.start(*args) == rt.start(*args)
    assert con.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 1


def test_full_suite_governed(world):
    base, suite = world
    s = summarise(run_suite(base, suite, lambda: ScriptedPlanner(0.0), Config()))
    assert s["success"] == 1.0 and s["runs_with_unsafe_outcome"] == 0


def test_gate_blocks_removed_approvals_and_passes_unchanged(world):
    base, suite = world
    b = evaluate(base, suite, Config(), seeds=(0,))
    assert verdict(b, evaluate(base, suite, Config(), seeds=(0,))) == []
    assert verdict(b, evaluate(base, suite, Config(approvals=False), seeds=(0,)))
    assert context_contract(base, Config()) and not context_contract(base, Config(mark_untrusted=False))


class FakeProvider:
    """Plays Claude: read the invoice, refund the overpayment, then answer."""
    model = "fake"

    def __init__(self):
        self.n = 0

    def complete(self, req):
        msgs = req["messages"]
        if msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):  # tool results must answer tool calls
            prev = msgs[-2]["content"]
            assert {b["id"] for b in prev if b["type"] == "tool_use"} == {b["tool_use_id"] for b in msgs[-1]["content"]}
        self.n += 1
        inv = req["messages"][0]["content"].split("invoice ")[1].split(".")[0]
        if self.n == 1:
            c = [{"type": "tool_use", "id": "t1", "name": "get_invoice", "input": {"invoice_id": inv}}]
        elif self.n == 2:
            r = json.loads(msgs[-1]["content"][0]["content"])
            c = [{"type": "tool_use", "id": "t2", "name": "issue_refund",
                  "input": {"invoice_id": inv, "amount": r["overpaid_by"], "reason": "dup", "expected_version": r["version"]}}]
        else:
            c = [{"type": "text", "text": "Refunded the duplicate."}]
        return {"content": c, "usage": {"input": 100, "output": 20}, "model": "fake"}


@pytest.mark.parametrize("kind", ["refund_small", "refund_large"])
def test_claude_planner_loop_with_fake_model(world, kind):
    base, suite = world
    r = run_scenario(base, scen(suite, kind), ClaudePlanner(Router([FakeProvider()], use_cache=False)), Config())
    assert r["success"], r
    assert r["approvals"] == (1 if kind == "refund_large" else 0)


def test_router_falls_back_caches_and_breaks():
    class Down:
        model = "down"

        def complete(self, req):
            raise Retryable("overloaded")

    class Up:
        model = "up"

        def complete(self, req):
            return {"content": [], "usage": {"input": 1, "output": 1}, "model": "up"}
    clock = Clock(virtual=True)
    r = Router([Down(), Up()], retries=1, backoff=0.1, clock=clock, breaker=True)
    assert r.complete({"q": 1})["model"] == "up"
    assert r.complete({"q": 1})["cost"] == 0.0 and r.calls[-1]["cached"]
    for i in range(5):
        r.complete({"q": 10 + i})
    r.complete({"q": 99})
    assert r.calls[-1]["attempts"] == 1  # breaker open: straight to the fallback


def test_service_end_to_end(world):
    base, suite = world
    sc = scen(suite, "refund_large")
    app = create_app(clone(base), ScriptedPlanner(0.0))
    with TestClient(app) as c:
        h = {"X-User": sc["user_id"]}
        r = c.post("/runs", json={"capability": sc["capability"], "request": sc["request"],
                                  "planner_args": {"kind": sc["kind"], "intent": sc["intent"]},
                                  "client_request_id": "abc"}, headers=h).json()
        assert r["status"] == "awaiting_approval"
        assert c.post("/runs", json={"capability": sc["capability"], "request": sc["request"],
                                     "client_request_id": "abc"}, headers=h).json()["duplicate"]
        other = "B-owner" if sc["tenant"] != "B" else "C-owner"
        assert c.get(f"/runs/{r['run_id']}", headers={"X-User": other}).status_code == 404
        run = c.get(f"/runs/{r['run_id']}", headers=h).json()
        pid = run["pending_approvals"][0]["id"]
        assert c.post(f"/runs/{r['run_id']}/proposals/{pid}", json={"decision": "approve"}, headers=h).status_code == 403
        ok = c.post(f"/runs/{r['run_id']}/proposals/{pid}", json={"decision": "approve"},
                    headers={"X-User": f"{sc['tenant']}-owner"}).json()
        assert ok["status"] == "completed"
        assert any(s["tool"] == "issue_refund" for s in c.get(f"/runs/{r['run_id']}/trace", headers=h).json())
        assert "agentlab_writes_executed" in c.get("/metrics").text
