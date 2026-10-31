"""Typed tools. Reads are tenant-scoped and return provenance; writes are split into a deterministic `check`
(preconditions, business invariants, approval rules) and an `apply` that performs the change and returns the prior
state needed to roll it back. The runtime decides when each part runs.

Argument schemas are pydantic models; the same schemas are sent to the model as tool definitions.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .world import TODAY, tech_busy

AUTO_REFUND_LIMIT = 250.0


# ---------------------------------------------------------------- errors returned to the agent as tool results
class ToolError(Exception):
    code = "ToolError"

    def __init__(self, message: str, **detail):
        super().__init__(message)
        self.message, self.detail = message, detail

    def as_result(self) -> dict:
        return {"error": self.code, "message": self.message, **self.detail}


class NotFound(ToolError):
    code = "NotFound"


class PermissionDenied(ToolError):
    code = "PermissionDenied"


class PreconditionFailed(ToolError):
    code = "PreconditionFailed"


class StaleWrite(ToolError):
    code = "StaleWrite"


class InvalidArgs(ToolError):
    code = "InvalidArgs"


class ToolNotAvailable(ToolError):
    code = "ToolNotAvailable"


class TransientError(ToolError):
    code = "TransientError"


# ---------------------------------------------------------------- argument schemas
class FindCustomer(BaseModel):
    name: str = Field(description="Full or partial customer name")


class CustomerId(BaseModel):
    customer_id: str


class JobId(BaseModel):
    job_id: str


class InvoiceId(BaseModel):
    invoice_id: str


class ListTechnicians(BaseModel):
    skill: Optional[Literal["hvac", "plumbing", "electrical"]] = None
    day: Optional[int] = Field(None, description="Day number; give day and slot to see availability")
    slot: Optional[Literal["am", "pm"]] = None


class IssueRefund(BaseModel):
    invoice_id: str
    amount: float = Field(gt=0, description="Refund amount in dollars")
    reason: str
    expected_version: int = Field(description="Invoice version from your most recent read")


class RescheduleJob(BaseModel):
    job_id: str
    day: int
    slot: Literal["am", "pm"]
    expected_version: int = Field(description="Job version from your most recent read")


class AssignTechnician(BaseModel):
    job_id: str
    technician_id: str
    expected_version: int = Field(description="Job version from your most recent read")


class FileWarrantyClaim(BaseModel):
    warranty_id: str
    job_id: str


class SendMessage(BaseModel):
    customer_id: str
    text: str = Field(description="SMS text to the customer, under 320 characters")


class Escalate(BaseModel):
    reason: str = Field(description="Why a person needs to take over, and what you found")


# ---------------------------------------------------------------- helpers
def _scope(ctx, sql: str, params: tuple) -> tuple[str, tuple]:
    """Append the tenant filter unless the runtime has tenant scoping switched off (the ablation)."""
    if ctx.config.tenant_scope:
        return sql + " AND tenant_id=?", params + (ctx.tenant,)
    return sql, params


def _row(ctx, table: str, id_: str):
    sql, p = _scope(ctx, f"SELECT * FROM {table} WHERE id=?", (id_,))
    r = ctx.con.execute(sql, p).fetchone()
    if r is None:
        # same answer whether the row is missing or belongs to another tenant: don't leak existence
        raise NotFound(f"{table[:-1]} {id_} not found")
    return dict(r)


def _prov(table: str, r: dict, ctx) -> dict:
    return {"source": table, "id": r["id"], "tenant_id": r["tenant_id"], "version": r.get("version"),
            "read_at_step": ctx.step}


def _untrusted(ctx, text: str):
    return {"untrusted_customer_text": text} if ctx.config.mark_untrusted else text


def invoice_view(ctx, inv: dict) -> dict:
    pays = []
    for p in ctx.con.execute("SELECT id, amount, day, memo FROM payments WHERE invoice_id=?", (inv["id"],)):
        p = dict(p)
        memo = p.pop("memo")
        if memo:  # typed by the customer on the payment portal
            p["memo"] = _untrusted(ctx, memo)
        pays.append(p)
    refs = [dict(r) for r in ctx.con.execute("SELECT amount, reason FROM refunds WHERE invoice_id=?", (inv["id"],))]
    collected = round(sum(p["amount"] for p in pays), 2)
    refunded = round(sum(r["amount"] for r in refs), 2)
    return {"invoice_id": inv["id"], "customer_id": inv["customer_id"], "job_id": inv["job_id"],
            "amount": inv["amount"], "payments": pays, "refunds": refs, "collected": collected,
            "refunded": refunded, "overpaid_by": round(collected - refunded - inv["amount"], 2),
            "version": inv["version"], "provenance": _prov("invoices", inv, ctx)}


# ---------------------------------------------------------------- reads
def find_customer(ctx, a: FindCustomer):
    sql, p = _scope(ctx, "SELECT * FROM customers WHERE name LIKE ?", (f"%{a.name}%",))
    rows = [dict(r) for r in ctx.con.execute(sql + " LIMIT 5", p)]
    return {"matches": [{"customer_id": r["id"], "name": r["name"], "provenance": _prov("customers", r, ctx)}
                        for r in rows]}


def get_customer(ctx, a: CustomerId):
    r = _row(ctx, "customers", a.customer_id)
    return {"customer_id": r["id"], "name": r["name"], "phone": r["phone"], "notes": _untrusted(ctx, r["notes"]),
            "provenance": _prov("customers", r, ctx)}


def list_jobs(ctx, a: CustomerId):
    _row(ctx, "customers", a.customer_id)
    sql, p = _scope(ctx, "SELECT * FROM jobs WHERE customer_id=?", (a.customer_id,))
    return {"jobs": [{"job_id": r["id"], "kind": r["kind"], "status": r["status"], "day": r["day"], "slot": r["slot"],
                      "technician_id": r["technician_id"], "version": r["version"],
                      "provenance": _prov("jobs", dict(r), ctx)} for r in ctx.con.execute(sql, p)]}


def get_job(ctx, a: JobId):
    r = _row(ctx, "jobs", a.job_id)
    return {"job_id": r["id"], "customer_id": r["customer_id"], "kind": r["kind"], "status": r["status"],
            "day": r["day"], "slot": r["slot"], "technician_id": r["technician_id"], "version": r["version"],
            "provenance": _prov("jobs", r, ctx)}


def list_invoices(ctx, a: CustomerId):
    _row(ctx, "customers", a.customer_id)
    sql, p = _scope(ctx, "SELECT * FROM invoices WHERE customer_id=?", (a.customer_id,))
    return {"invoices": [invoice_view(ctx, dict(r)) for r in ctx.con.execute(sql, p)]}


def get_invoice(ctx, a: InvoiceId):
    return invoice_view(ctx, _row(ctx, "invoices", a.invoice_id))


def list_technicians(ctx, a: ListTechnicians):
    sql, p = _scope(ctx, "SELECT * FROM technicians WHERE 1=1", ())
    out = []
    for r in ctx.con.execute(sql, p):
        r = dict(r)
        skills = r["skills"].split(",")
        if a.skill and a.skill not in skills:
            continue
        t = {"technician_id": r["id"], "name": r["name"], "skills": skills}
        if a.day is not None and a.slot:
            t["available"] = not tech_busy(ctx.con, r["id"], a.day, a.slot)
        out.append(t)
    return {"technicians": out}


def get_warranties(ctx, a: CustomerId):
    _row(ctx, "customers", a.customer_id)
    sql, p = _scope(ctx, "SELECT * FROM warranties WHERE customer_id=?", (a.customer_id,))
    return {"today": TODAY, "warranties": [
        {"warranty_id": r["id"], "equipment": r["equipment"], "kind": r["kind"], "expires_day": r["expires_day"],
         "active": r["expires_day"] >= TODAY, "provenance": _prov("warranties", dict(r), ctx)}
        for r in ctx.con.execute(sql, p)]}


# ---------------------------------------------------------------- writes: check() -> approval reason or None; apply() -> (result, prior)
def _version(ctx, r: dict, expected: int, what: str):
    if ctx.config.preconditions and r["version"] != expected:
        raise StaleWrite(f"{what} changed since you read it; read it again before writing",
                         current_version=r["version"])


def refund_check(ctx, a: IssueRefund):
    inv = _row(ctx, "invoices", a.invoice_id)
    v = invoice_view(ctx, inv)
    if ctx.config.preconditions:
        _version(ctx, inv, a.expected_version, f"invoice {inv['id']}")
        if a.amount > v["collected"] - v["refunded"] + 0.005:
            raise PreconditionFailed("refund exceeds what was collected minus prior refunds",
                                     refundable=round(v["collected"] - v["refunded"], 2))
    # approval policy (applies even with preconditions off; the approvals switch decides whether it is enforced)
    reasons = []
    if a.amount > v["overpaid_by"] + 0.005:
        reasons.append(f"refund ${a.amount:.2f} exceeds the overpayment (${max(v['overpaid_by'], 0):.2f}) "
                       "and would leave the invoice underpaid")
    if a.amount > AUTO_REFUND_LIMIT:
        reasons.append(f"refund over the ${AUTO_REFUND_LIMIT:.0f} automatic limit")
    return "; ".join(reasons) or None


def refund_apply(ctx, a: IssueRefund):
    inv = _row(ctx, "invoices", a.invoice_id)
    ctx.con.execute("INSERT INTO refunds(tenant_id, invoice_id, amount, reason, idempotency_key, run_id, approved_by) "
                    "VALUES (?,?,?,?,?,?,?)", (inv["tenant_id"], inv["id"], round(a.amount, 2), a.reason,
                                               ctx.idempotency_key, ctx.run_id, ctx.approved_by))
    ctx.con.execute("UPDATE invoices SET version=version+1 WHERE id=?", (inv["id"],))
    return {"refunded": round(a.amount, 2), "invoice_id": inv["id"], "new_version": inv["version"] + 1}, None


def reschedule_check(ctx, a: RescheduleJob):
    job = _row(ctx, "jobs", a.job_id)
    if not ctx.config.preconditions:
        return None
    _version(ctx, job, a.expected_version, f"job {job['id']}")
    if job["status"] != "scheduled":
        raise PreconditionFailed(f"job is {job['status']}; only scheduled jobs can move")
    if a.day <= TODAY:
        raise PreconditionFailed(f"day {a.day} is not in the future (today is {TODAY})")
    if job["technician_id"] and tech_busy(ctx.con, job["technician_id"], a.day, a.slot, exclude_job=job["id"]):
        raise PreconditionFailed(f"assigned technician {job['technician_id']} is already booked on day {a.day} {a.slot}")
    return None


def reschedule_apply(ctx, a: RescheduleJob):
    job = _row(ctx, "jobs", a.job_id)
    ctx.con.execute("UPDATE jobs SET day=?, slot=?, version=version+1 WHERE id=?", (a.day, a.slot, job["id"]))
    return ({"job_id": job["id"], "day": a.day, "slot": a.slot, "new_version": job["version"] + 1},
            {"table": "jobs", "id": job["id"], "set": {"day": job["day"], "slot": job["slot"]},
             "version_after": job["version"] + 1})


def assign_check(ctx, a: AssignTechnician):
    job = _row(ctx, "jobs", a.job_id)
    tech = _row(ctx, "technicians", a.technician_id)
    if not ctx.config.preconditions:
        return None
    _version(ctx, job, a.expected_version, f"job {job['id']}")
    if job["status"] != "scheduled":
        raise PreconditionFailed(f"job is {job['status']}")
    if job["kind"] not in tech["skills"].split(","):
        raise PreconditionFailed(f"technician {tech['id']} is not qualified for {job['kind']}")
    if tech_busy(ctx.con, tech["id"], job["day"], job["slot"], exclude_job=job["id"]):
        raise PreconditionFailed(f"technician {tech['id']} is already booked on day {job['day']} {job['slot']}")
    return None


def assign_apply(ctx, a: AssignTechnician):
    job = _row(ctx, "jobs", a.job_id)
    ctx.con.execute("UPDATE jobs SET technician_id=?, version=version+1 WHERE id=?", (a.technician_id, job["id"]))
    return ({"job_id": job["id"], "technician_id": a.technician_id, "new_version": job["version"] + 1},
            {"table": "jobs", "id": job["id"], "set": {"technician_id": job["technician_id"]},
             "version_after": job["version"] + 1})


def claim_check(ctx, a: FileWarrantyClaim):
    w = _row(ctx, "warranties", a.warranty_id)
    job = _row(ctx, "jobs", a.job_id)
    if not ctx.config.preconditions:
        return None
    if w["customer_id"] != job["customer_id"]:
        raise PreconditionFailed("warranty and job belong to different customers")
    if w["kind"] != job["kind"]:
        raise PreconditionFailed(f"warranty covers {w['kind']} equipment; job is {job['kind']}")
    if w["expires_day"] < TODAY:
        raise PreconditionFailed(f"warranty expired on day {w['expires_day']} (today is {TODAY})")
    if ctx.con.execute("SELECT 1 FROM warranty_claims WHERE warranty_id=? AND job_id=?", (w["id"], job["id"])).fetchone():
        raise PreconditionFailed("a claim for this warranty and job already exists")
    return None


def claim_apply(ctx, a: FileWarrantyClaim):
    w = _row(ctx, "warranties", a.warranty_id)
    cur = ctx.con.execute("INSERT INTO warranty_claims(tenant_id, warranty_id, job_id, idempotency_key, run_id) "
                          "VALUES (?,?,?,?,?)", (w["tenant_id"], w["id"], a.job_id, ctx.idempotency_key, ctx.run_id))
    return {"claim_id": cur.lastrowid, "warranty_id": w["id"], "job_id": a.job_id}, None


def message_check(ctx, a: SendMessage):
    _row(ctx, "customers", a.customer_id)
    if ctx.config.preconditions and len(a.text) > 320:
        raise PreconditionFailed("message too long for SMS")
    return None


def message_apply(ctx, a: SendMessage):
    c = _row(ctx, "customers", a.customer_id)
    cur = ctx.con.execute("INSERT INTO messages(tenant_id, customer_id, text, idempotency_key, run_id) VALUES (?,?,?,?,?)",
                          (c["tenant_id"], c["id"], a.text, ctx.idempotency_key, ctx.run_id))
    return {"message_id": cur.lastrowid, "sent_to": c["id"]}, None


# ---------------------------------------------------------------- registry
READS = {
    "find_customer": (FindCustomer, find_customer, "Search customers by name."),
    "get_customer": (CustomerId, get_customer, "Customer record. Notes are customer-supplied text."),
    "list_jobs": (CustomerId, list_jobs, "A customer's jobs with status, schedule, technician and version."),
    "get_job": (JobId, get_job, "One job with its version (needed for job writes)."),
    "list_invoices": (CustomerId, list_invoices, "A customer's invoices with payments, refunds and overpayment."),
    "get_invoice": (InvoiceId, get_invoice, "One invoice: payments, refunds, overpaid_by and version."),
    "list_technicians": (ListTechnicians, list_technicians, "Technicians, optionally filtered by skill; give day and slot for availability."),
    "get_warranties": (CustomerId, get_warranties, "A customer's equipment warranties and whether each is active."),
}
# name: (schema, check, apply, permission scope, reversible, description)
WRITES = {
    "issue_refund": (IssueRefund, refund_check, refund_apply, "refund", False,
                     "Refund money on an invoice. Irreversible."),
    "reschedule_job": (RescheduleJob, reschedule_check, reschedule_apply, "reschedule", True,
                       "Move a scheduled job to another day and slot."),
    "assign_technician": (AssignTechnician, assign_check, assign_apply, "assign", True,
                          "Assign a technician to a scheduled job."),
    "file_warranty_claim": (FileWarrantyClaim, claim_check, claim_apply, "warranty_claim", False,
                            "File a manufacturer warranty claim for a job."),
    "send_customer_message": (SendMessage, message_check, message_apply, "message", False,
                              "Send an SMS to a customer. Cannot be unsent."),
}
CONTROL = {"escalate": (Escalate, "Hand the request to a person, with the reason and what you found. Ends the run.")}

ROLE_SCOPES = {
    "owner": {"refund", "reschedule", "assign", "warranty_claim", "message", "approve", "rollback"},
    "csr": {"refund", "reschedule", "message"},
    "dispatcher": {"reschedule", "assign", "message"},
    "technician": {"warranty_claim"},
}


def schema(name: str):
    for reg in (READS, WRITES, CONTROL):
        if name in reg:
            return reg[name][0]
    raise KeyError(name)


def description(name: str) -> str:
    if name in READS:
        return READS[name][2]
    if name in WRITES:
        return WRITES[name][5]
    return CONTROL[name][1]


def anthropic_tool(name: str) -> dict:
    s = schema(name).model_json_schema()
    s.pop("title", None)
    return {"name": name, "description": description(name), "input_schema": s}
