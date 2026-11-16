"""Evaluation harness and CI gate.

    python -m evals.run_eval offline          # retrieval + guard + ingest checks, no model needed
    python -m evals.run_eval e2e              # full answers and text-to-SQL (needs QUARRY_PROVIDER=gemini)
    python -m evals.run_eval all --gate       # exit 1 if any metric is under evals/thresholds.json

Results are printed as a table and written to evals/out/<suite>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from quarry.config import Settings
from quarry.pipeline import Quarry
from quarry.sqlguard import SqlRejected, guard

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "evals" / "out"
TENANT = "meridian"


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / "eval" / name).read_text().splitlines() if line.strip()]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


# ------------------------------------------------------------------ suites --
def eval_retrieval(q: Quarry) -> dict:
    """recall@k and MRR@10 for each retrieval mode: the ablation that justifies hybrid."""
    cases = _load("retrieval.jsonl")
    out: dict = {"cases": len(cases)}
    for mode in ("lexical", "dense", "hybrid"):
        ranks, lat = [], []
        for c in cases:
            t0 = time.perf_counter()
            hits = q.retriever.search(TENANT, c["q"], 10, mode)  # type: ignore[arg-type]
            lat.append((time.perf_counter() - t0) * 1000)
            rank = next((i for i, h in enumerate(hits, 1)
                         if h.source_uri == c["source"] and c["section"] in h.heading_path), None)
            ranks.append(rank)
        n = len(cases)
        out[mode] = {
            "recall@1": round(sum(1 for r in ranks if r and r <= 1) / n, 3),
            "recall@3": round(sum(1 for r in ranks if r and r <= 3) / n, 3),
            "recall@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
            "p50_ms": round(statistics.median(lat), 1),
            "p95_ms": round(_pct(lat, 0.95), 1),
        }
    return out


def eval_guard(q: Quarry) -> dict:
    """Every attack must be rejected before reaching the database; every gold query must pass."""
    tables = set(q.settings.sql_tables)
    attacks, gold = _load("sql_attacks.jsonl"), _load("sql.jsonl")
    leaked = []
    for a in attacks:
        try:
            guard(a["sql"], tables)
            leaked.append(a["sql"])
        except SqlRejected:
            pass
    wrongly_blocked = []
    for g in gold:
        try:
            guard(g["gold"], tables)
        except SqlRejected as e:
            wrongly_blocked.append(f"{g['gold']} -> {e}")
    return {
        "attacks": len(attacks), "attacks_blocked": len(attacks) - len(leaked), "leaked": leaked,
        "benign": len(gold), "benign_allowed": len(gold) - len(wrongly_blocked), "wrongly_blocked": wrongly_blocked,
        "block_rate": round((len(attacks) - len(leaked)) / len(attacks), 3),
    }


def eval_ingest(q: Quarry) -> dict:
    """Idempotency: a no-op re-ingest embeds nothing; a one-section edit embeds only that section."""
    docs = DATA / "docs"
    q.ingestor.ingest_dir(TENANT, docs)                      # make sure the corpus is present
    again = q.ingestor.ingest_dir(TENANT, docs)
    path = docs / "surcharges.md"
    original = path.read_text()
    edited = original.replace("costs USD 75", "costs USD 80")
    try:
        r = q.ingestor.ingest_text(TENANT, "surcharges.md", "Surcharges and Accessorial Fees", edited)
    finally:
        q.ingestor.ingest_text(TENANT, "surcharges.md", "Surcharges and Accessorial Fees", original)
    return {
        "documents": len(again),
        "reingest_chunks_embedded": sum(x.chunks_embedded for x in again),
        "reingest_unchanged": sum(1 for x in again if x.status == "unchanged"),
        "edit_chunks_total": r.chunks_total,
        "edit_chunks_embedded": r.chunks_embedded,
        "edit_chunks_reused": r.chunks_reused,
    }


def _rows_equal(a: list[list], b: list[list]) -> bool:
    norm = lambda rows: sorted(tuple(sorted(str(round(v, 2)) if isinstance(v, float) else str(v) for v in r)) for r in rows)
    return norm(a) == norm(b)


def eval_e2e(q: Quarry) -> dict:
    """End-to-end quality, latency and cost with the configured model."""
    if q.llm is None or q.structured is None:
        raise SystemExit("e2e needs a model: set QUARRY_PROVIDER=gemini and credentials")
    docs, sqls = _load("retrieval.jsonl"), _load("sql.jsonl")
    lat, cost, tokens = [], [], []
    correct = grounded = cited_right = 0
    failures = []
    for c in docs:
        r = q.query(TENANT, c["q"])
        lat.append(r.timings_ms["total"]); cost.append(r.usage.cost_usd)
        tokens.append(r.usage.input_tokens + r.usage.output_tokens)
        ok = all(m.lower() in r.answer.lower() for m in c["must"])
        correct += ok
        grounded += r.grounded and not r.abstained
        cited_right += any(ci.get("source") == c["source"] for ci in r.citations)
        if not ok:
            failures.append({"q": c["q"], "answer": r.answer[:300]})
    sql_ok = 0
    for c in sqls:
        _, _, gold_rows = q.structured.run_sql(TENANT, c["gold"])
        t0 = time.perf_counter()
        try:
            a = q.structured.ask(TENANT, c["q"])
            good = _rows_equal(a.rows, gold_rows)
            cost.append(a.usage.cost_usd)
        except Exception as e:                                  # noqa: BLE001 - recorded as a failure
            good, a = False, None
            failures.append({"q": c["q"], "error": str(e)[:200]})
        lat.append((time.perf_counter() - t0) * 1000)
        sql_ok += good
        if a is not None and not good:
            failures.append({"q": c["q"], "sql": a.sql})
    n = len(docs)
    return {
        "model": q.settings.chat_model,
        "doc_questions": n,
        "answer_accuracy": round(correct / n, 3),
        "grounded_rate": round(grounded / n, 3),
        "citation_precision": round(cited_right / n, 3),
        "sql_questions": len(sqls),
        "sql_execution_accuracy": round(sql_ok / len(sqls), 3),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(_pct(lat, 0.95), 1),
        "avg_cost_usd_per_request": round(sum(cost) / len(cost), 6),
        "avg_tokens_per_doc_request": round(sum(tokens) / n),
        "failures": failures,
    }


SUITES = {"retrieval": eval_retrieval, "guard": eval_guard, "ingest": eval_ingest, "e2e": eval_e2e}
GROUPS = {"offline": ["ingest", "retrieval", "guard"], "all": ["ingest", "retrieval", "guard", "e2e"]}


def _gate(results: dict, provider: str) -> list[str]:
    """Structural checks always gate; quality thresholds only apply with a real model."""
    spec = json.loads((ROOT / "evals" / "thresholds.json").read_text())
    thresholds = {**spec["always"], **(spec["gemini"] if provider == "gemini" else {})}
    problems = []
    for dotted, minimum in thresholds.items():
        node = results
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            continue                                            # suite not run
        if node < minimum:
            problems.append(f"{dotted} = {node} < {minimum}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=[*SUITES, *GROUPS])
    ap.add_argument("--gate", action="store_true", help="fail if a metric is below thresholds.json")
    args = ap.parse_args()

    settings = Settings.from_env()
    q = Quarry.from_settings(settings)
    results = {"provider": {"name": settings.provider, "embed_model": settings.embed_model if settings.provider == "gemini" else "hash-768 (offline baseline)"}}
    try:
        for name in GROUPS.get(args.suite, [args.suite]):
            results[name] = SUITES[name](q)
    finally:
        q.store.close()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.suite}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"} for k, v in results.items()}, indent=2))
    if args.gate:
        problems = _gate(results, settings.provider)
        for p in problems:
            print("GATE FAIL:", p, file=sys.stderr)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
