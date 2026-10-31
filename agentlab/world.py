"""A simulated system of record for three field-service businesses (tenants): customers, technicians, jobs,
invoices, payments, refunds, warranties. SQLite stands in for the production database; the schema keeps a
`tenant_id` and a `version` on every mutable row (optimistic concurrency).

The agent never owns this data: the database is authoritative, and every change goes through a typed tool that
writes an audit row. Days are integers; TODAY is day 100.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

TODAY = 100
SKILLS = ["hvac", "plumbing", "electrical"]
TENANTS = {"A": "Northside Heating & Air", "B": "Bluewater Plumbing", "C": "Brightline Home Services"}
FIRST = ["Dana", "Luis", "Priya", "Marcus", "Elena", "Tom", "Aisha", "Grace", "Omar", "Nina", "Ravi", "Chloe",
         "Sam", "Ivy", "Jorge", "Mei", "Peter", "Zoe", "Kofi", "Hana", "Ben", "Lena", "Arjun", "Rosa"]
LAST = ["Lopez", "Nguyen", "Patel", "Brooks", "Kim", "Reyes", "Shah", "Miller", "Okafor", "Novak", "Garcia",
        "Chen", "Haddad", "Ward", "Silva", "Ito", "Morgan", "Diaz", "Fischer", "Rossi"]
EQUIPMENT = {"hvac": ["furnace", "heat pump", "AC condenser"], "plumbing": ["water heater", "sump pump"],
             "electrical": ["panel", "generator"]}

SCHEMA = """
CREATE TABLE tenants(id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE users(id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, role TEXT, technician_id TEXT);
CREATE TABLE customers(id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, phone TEXT, notes TEXT, version INT DEFAULT 1);
CREATE TABLE technicians(id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT, skills TEXT);
CREATE TABLE jobs(id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT, kind TEXT, status TEXT, day INT, slot TEXT,
                  technician_id TEXT, version INT DEFAULT 1);
CREATE TABLE invoices(id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT, job_id TEXT, amount REAL, version INT DEFAULT 1);
CREATE TABLE payments(id TEXT PRIMARY KEY, tenant_id TEXT, invoice_id TEXT, amount REAL, day INT, memo TEXT DEFAULT '');
CREATE TABLE refunds(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, invoice_id TEXT, amount REAL, reason TEXT,
                     idempotency_key TEXT, run_id TEXT, approved_by TEXT);
CREATE TABLE warranties(id TEXT PRIMARY KEY, tenant_id TEXT, customer_id TEXT, equipment TEXT, kind TEXT, expires_day INT);
CREATE TABLE warranty_claims(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, warranty_id TEXT, job_id TEXT,
                             idempotency_key TEXT, run_id TEXT);
CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, customer_id TEXT, text TEXT,
                      idempotency_key TEXT, run_id TEXT);
-- every attempted write, allowed or not, with the prior state needed to roll it back
CREATE TABLE audit(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INT, tenant_id TEXT, actor TEXT, tool TEXT,
                   args TEXT, outcome TEXT, detail TEXT, prior TEXT, idempotency_key TEXT, rolled_back INT DEFAULT 0);
-- idempotency ledger: key -> stored result, written in the same transaction as the side effect
CREATE TABLE effects(idempotency_key TEXT PRIMARY KEY, run_id TEXT, tool TEXT, result TEXT);
-- durable run state: checkpoints and pending approvals
CREATE TABLE runs(id TEXT PRIMARY KEY, client_request_id TEXT UNIQUE, tenant_id TEXT, user_id TEXT, capability TEXT,
                  request TEXT, status TEXT, state TEXT, created INT, updated INT);
CREATE TABLE steps(run_id TEXT, step INT, kind TEXT, tool TEXT, args TEXT, status TEXT, result TEXT,
                   PRIMARY KEY(run_id, step));
CREATE TABLE proposals(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INT, tool TEXT, args TEXT, reason TEXT,
                       status TEXT, decided_by TEXT, note TEXT);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def build(path: str | Path, seed: int = 0, customers_per_tenant: int = 70) -> sqlite3.Connection:
    path = Path(path)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    con = connect(path)
    con.executescript(SCHEMA)
    rng = random.Random(seed)
    for t, name in TENANTS.items():
        con.execute("INSERT INTO tenants VALUES (?,?)", (t, name))
        techs = []
        for i in range(6):
            sk = sorted(rng.sample(SKILLS, rng.choice([1, 1, 2])))
            tid = f"{t}-T{i:02d}"
            techs.append((tid, sk))
            con.execute("INSERT INTO technicians VALUES (?,?,?,?)",
                        (tid, t, f"{rng.choice(FIRST)} {rng.choice(LAST)}", ",".join(sk)))
        for s in SKILLS:  # every skill covered by at least two technicians
            have = [x for x in techs if s in x[1]]
            for tid, sk in techs[:max(0, 2 - len(have))]:
                if s not in sk:
                    sk.append(s)
                    con.execute("UPDATE technicians SET skills=? WHERE id=?", (",".join(sorted(sk)), tid))
        for role in ("owner", "csr", "dispatcher", "technician"):
            con.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                        (f"{t}-{role}", t, f"{role.title()} ({name})", role, f"{t}-T00" if role == "technician" else None))
        names = set()
        j = inv = pay = w = 0
        for c in range(customers_per_tenant):
            while True:
                n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if n not in names:
                    names.add(n)
                    break
            cid = f"{t}-C{c:03d}"
            con.execute("INSERT INTO customers(id,tenant_id,name,phone,notes) VALUES (?,?,?,?,?)",
                        (cid, t, n, f"555-{rng.randint(1000, 9999)}", rng.choice(
                            ["Prefers morning visits.", "Has a dog; call before arriving.", "Gate code on file.", ""])))
            for _ in range(rng.randint(1, 3)):
                kind = rng.choice(SKILLS)
                done = rng.random() < 0.6
                day = rng.randint(60, 99) if done else rng.randint(101, 112)
                slot = rng.choice(["am", "pm"])
                cand = [x[0] for x in techs if kind in x[1]]
                if not done:  # never double-book in the starting data
                    cand = [x for x in cand if not tech_busy(con, x, day, slot)]
                tech = rng.choice(cand) if cand and (done or rng.random() < 0.8) else None
                jid = f"{t}-J{j:04d}"
                j += 1
                con.execute("INSERT INTO jobs(id,tenant_id,customer_id,kind,status,day,slot,technician_id) VALUES (?,?,?,?,?,?,?,?)",
                            (jid, t, cid, kind, "completed" if done else "scheduled", day, slot, tech))
                if done:
                    amt = round(rng.choice([rng.uniform(90, 240), rng.uniform(260, 900)]), 2)
                    iid = f"{t}-INV{inv:04d}"
                    inv += 1
                    con.execute("INSERT INTO invoices(id,tenant_id,customer_id,job_id,amount) VALUES (?,?,?,?,?)",
                                (iid, t, cid, jid, amt))
                    con.execute("INSERT INTO payments(id,tenant_id,invoice_id,amount,day) VALUES (?,?,?,?,?)",
                                (f"{t}-P{pay:05d}", t, iid, amt, day))
                    pay += 1
            if rng.random() < 0.5:
                kind = rng.choice(SKILLS)
                con.execute("INSERT INTO warranties VALUES (?,?,?,?,?,?)",
                            (f"{t}-W{w:03d}", t, cid, rng.choice(EQUIPMENT[kind]), kind,
                             TODAY + rng.randint(-200, 400)))
                w += 1
    return con


def add_payment(con, tenant: str, invoice_id: str, amount: float, day: int = TODAY - 1, memo: str = "") -> str:
    n = con.execute("SELECT COUNT(*) FROM payments WHERE tenant_id=?", (tenant,)).fetchone()[0]
    pid = f"{tenant}-P9{n:04d}"
    con.execute("INSERT INTO payments VALUES (?,?,?,?,?,?)", (pid, tenant, invoice_id, amount, day, memo))
    return pid


def tech_busy(con, tech_id: str, day: int, slot: str, exclude_job: str | None = None) -> bool:
    r = con.execute("SELECT COUNT(*) FROM jobs WHERE technician_id=? AND day=? AND slot=? AND status='scheduled' AND id!=?",
                    (tech_id, day, slot, exclude_job or "")).fetchone()[0]
    return r > 0
