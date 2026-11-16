"""Deterministic demo data for the structured side (customers, shipments, invoices, claims).

Same seed -> same rows, so eval gold answers and test assertions stay stable.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .store import Store

CITIES = ["Chicago", "Dallas", "Atlanta", "Seattle", "Denver", "Newark", "Memphis", "Phoenix", "Toronto", "Monterrey"]
REGIONS = ["midwest", "south", "west", "northeast", "international"]
NAMES_A = ["Halden", "Corvid", "Bluepeak", "Ostrava", "Lumen", "Tidewater", "Granite", "Juniper", "Kestrel", "Marlow"]
NAMES_B = ["Robotics", "Foods", "Medical", "Outfitters", "Textiles", "Instruments", "Supply", "Labs"]
TRANSIT = {"express": 2, "standard": 4, "economy": 8}
RATE = {"express": 9.5, "standard": 5.0, "economy": 3.2}      # USD per kg


def _story(t: str) -> tuple[list, list, list, list]:
    """Hand-written records that the eval scenarios refer to by id."""
    d = date
    customers = [
        ("CUS-9001", t, "Halden Robotics", "enterprise", "midwest"),
        ("CUS-9002", t, "Corvid Outfitters", "standard", "west"),
        ("CUS-9003", t, "Lumen Medical", "priority", "northeast"),
    ]
    # id, tenant, customer, origin, destination, service, status, kg, shipped, promised, delivered
    shipments = [
        ("SHP-90001", t, "CUS-9001", "Chicago", "Denver", "express", "delivered", 18.0, d(2026, 10, 4), d(2026, 10, 6), d(2026, 10, 8)),
        ("SHP-90002", t, "CUS-9003", "Newark", "Atlanta", "express", "delivered", 44.0, d(2026, 10, 1), d(2026, 10, 3), d(2026, 10, 7)),
        ("SHP-90003", t, "CUS-9002", "Seattle", "Phoenix", "standard", "damaged", 12.0, d(2026, 10, 5), d(2026, 10, 9), d(2026, 10, 9)),
        ("SHP-90004", t, "CUS-9002", "Dallas", "Memphis", "economy", "delivered", 30.0, d(2026, 9, 20), d(2026, 9, 28), d(2026, 10, 3)),
        ("SHP-90005", t, "CUS-9003", "Atlanta", "Chicago", "standard", "delivered", 9.0, d(2026, 10, 2), d(2026, 10, 6), d(2026, 10, 8)),
        ("SHP-90006", t, "CUS-9001", "Chicago", "Toronto", "express", "delivered", 210.0, d(2026, 10, 6), d(2026, 10, 8), d(2026, 10, 11)),
        ("SHP-90007", t, "CUS-9002", "Denver", "Seattle", "standard", "damaged", 20.0, d(2026, 10, 1), d(2026, 10, 5), d(2026, 10, 5)),
        ("SHP-90008", t, "CUS-9003", "Memphis", "Newark", "standard", "lost", 40.0, d(2026, 9, 22), d(2026, 9, 26), None),
    ]
    invoices = [
        ("INV-90001", t, "SHP-90001", 640.00, "open", d(2026, 10, 4)),
        ("INV-90002", t, "SHP-90002", 1240.00, "paid", d(2026, 10, 1)),
        ("INV-90003", t, "SHP-90003", 480.00, "paid", d(2026, 10, 5)),
        ("INV-90004", t, "SHP-90004", 210.00, "paid", d(2026, 9, 20)),
        ("INV-90005", t, "SHP-90005", 95.00, "open", d(2026, 10, 2)),
        ("INV-90006", t, "SHP-90006", 2900.00, "open", d(2026, 10, 6)),
        ("INV-90007", t, "SHP-90007", 330.00, "paid", d(2026, 10, 1)),
        ("INV-90008", t, "SHP-90008", 700.00, "paid", d(2026, 9, 22)),
    ]
    claims = [
        ("CLM-90071", t, "SHP-90007", "damage", 120.00, "denied", d(2026, 10, 6)),
        ("CLM-90072", t, "SHP-90007", "damage", 150.00, "open", d(2026, 10, 9)),
    ]
    return customers, shipments, invoices, claims


def generate(tenant: str, n_customers: int, n_shipments: int, base: int, seed: int, story: bool):
    rng = random.Random(seed)
    customers, shipments, invoices, claims = [], [], [], []
    for i in range(n_customers):
        name = f"{NAMES_A[i % len(NAMES_A)]} {NAMES_B[(i * 3 + i // len(NAMES_A)) % len(NAMES_B)]} {i + 1:02d}"
        tier = rng.choices(["standard", "priority", "enterprise"], [6, 3, 1])[0]
        customers.append((f"CUS-{base + i + 1:04d}", tenant, name, tier, rng.choice(REGIONS)))

    start = date(2026, 1, 5)
    for i in range(n_shipments):
        sid = f"SHP-{base * 10 + 10001 + i}"
        cust = rng.choice(customers)[0]
        origin, dest = rng.sample(CITIES, 2)
        service = rng.choices(["express", "standard", "economy"], [3, 5, 2])[0]
        kg = round(rng.uniform(1.5, 180.0), 1)
        shipped = start + timedelta(days=rng.randrange(0, 290))
        promised = shipped + timedelta(days=TRANSIT[service])
        roll = rng.random()
        if roll < 0.70:
            status, delivered = "delivered", promised - timedelta(days=rng.choice([0, 0, 1]))
        elif roll < 0.86:
            status, delivered = "delivered", promised + timedelta(days=rng.choice([1, 1, 2, 3, 4, 6]))
        elif roll < 0.91:
            status, delivered = "damaged", promised + timedelta(days=rng.choice([0, 0, 1]))
        elif roll < 0.94:
            status, delivered = "lost", None
        elif roll < 0.97:
            status, delivered = "delayed", None
        else:
            status, delivered = "in_transit", None
        shipments.append((sid, tenant, cust, origin, dest, service, status, kg, shipped, promised, delivered))

        amount = round(max(28.0, kg * RATE[service]) * rng.uniform(0.9, 1.25), 2)
        inv_status = rng.choices(["paid", "open", "overdue", "credited"], [70, 18, 9, 3])[0]
        invoices.append((f"INV-{base * 10 + 20001 + i}", tenant, sid, amount, inv_status, shipped))

        if status in ("damaged", "lost") or (delivered and delivered > promised and rng.random() < 0.25):
            ctype = {"damaged": "damage", "lost": "loss"}.get(status, "delay")
            cap = min(kg * 25, amount) if ctype != "delay" else amount * 0.25
            claims.append((f"CLM-{base * 10 + 30001 + len(claims)}", tenant, sid, ctype, round(cap * rng.uniform(0.4, 1.0), 2),
                           rng.choices(["open", "approved", "denied", "paid"], [3, 2, 2, 3])[0],
                           (delivered or promised) + timedelta(days=rng.randrange(1, 12))))

    if story:
        for dst, src in zip((customers, shipments, invoices, claims), _story(tenant)):
            dst.extend(src)
    return customers, shipments, invoices, claims


def seed_tenant(store: Store, tenant: str, n_customers: int, n_shipments: int, base: int, seed: int, story: bool = False):
    customers, shipments, invoices, claims = generate(tenant, n_customers, n_shipments, base, seed, story)
    with store.tenant(tenant) as conn:
        for table in ("claims", "invoices", "shipments", "customers"):
            conn.execute(f"DELETE FROM {table}")          # RLS limits this to the tenant's own rows
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO customers VALUES (%s,%s,%s,%s,%s)", customers)
            cur.executemany("INSERT INTO shipments VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", shipments)
            cur.executemany("INSERT INTO invoices VALUES (%s,%s,%s,%s,%s,%s)", invoices)
            cur.executemany("INSERT INTO claims VALUES (%s,%s,%s,%s,%s,%s,%s)", claims)
    return {"customers": len(customers), "shipments": len(shipments), "invoices": len(invoices), "claims": len(claims)}


def seed_demo(store: Store) -> dict:
    """Two tenants so isolation is always exercised: 'meridian' (main) and 'acme' (decoy)."""
    return {
        "meridian": seed_tenant(store, "meridian", 40, 600, base=0, seed=7, story=True),
        "acme": seed_tenant(store, "acme", 6, 40, base=5000, seed=11),
    }
