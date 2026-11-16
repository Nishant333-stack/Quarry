"""The query path: route -> retrieve and/or SQL -> grounded answer.

Each stage is timed and every model call is metered, so a response carries its
own latency breakdown, token counts and cost.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .answer import ABSTAIN, Answer, Route, generate_answer, route
from .config import Settings
from .ingest import Ingestor
from .llm import LLM, Embedder, GeminiEmbedder, GeminiLLM, HashEmbedder, Usage
from .retrieval import Retriever
from .store import Hit, Store
from .structured import StructuredError, StructuredQA


@dataclass
class QueryResult:
    answer: str
    route: Route
    grounded: bool
    abstained: bool
    citations: list[dict]
    sql: str | None
    sql_rows: int | None
    timings_ms: dict[str, float]
    usage: Usage
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "route": self.route,
            "grounded": self.grounded,
            "abstained": self.abstained,
            "citations": self.citations,
            "sql": self.sql,
            "sql_rows": self.sql_rows,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "usage": self.usage.as_dict(),
            "notes": self.notes,
        }


class Quarry:
    """Wires storage, models and the query/ingest paths together."""

    def __init__(self, settings: Settings, store: Store, embedder: Embedder, llm: LLM | None):
        self.settings, self.store, self.embedder, self.llm = settings, store, embedder, llm
        self.ingestor = Ingestor(store, embedder, settings.chunk_tokens, settings.chunk_overlap)
        self.retriever = Retriever(store, embedder, settings.candidates)
        self.structured = (
            StructuredQA(store, llm, settings.sql_tables, settings.sql_max_rows, settings.sql_timeout_ms)
            if llm is not None else None
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Quarry":
        store = Store(settings.database_url)
        if settings.provider == "gemini":
            embedder: Embedder = GeminiEmbedder(settings.embed_model, settings.embed_dim, settings.embed_batch)
            llm: LLM | None = GeminiLLM(settings.chat_model)
        else:
            embedder, llm = HashEmbedder(settings.embed_dim), None
        return cls(settings, store, embedder, llm)

    def query(self, tenant: str, question: str, k: int | None = None) -> QueryResult:
        k = k or self.settings.top_k
        timings: dict[str, float] = {}
        usage = Usage()
        notes: list[str] = []

        @contextmanager
        def stage(name: str):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                timings[name] = (time.perf_counter() - t0) * 1000

        t_start = time.perf_counter()
        with stage("route"):
            chosen, u = route(self.llm, question)
            usage.add(u)

        hits: list[Hit] = []
        if chosen in ("docs", "both"):
            with stage("retrieve"):
                hits = self.retriever.search(tenant, question, k)

        sql_answer = None
        if chosen in ("sql", "both") and self.structured is not None:
            with stage("sql"):
                try:
                    sql_answer = self.structured.ask(tenant, question)
                    usage.add(sql_answer.usage)
                except StructuredError as e:
                    notes.append(f"structured lookup failed: {e}")
        if sql_answer is None and not hits:
            # Degrade gracefully: fall back to documents rather than failing the request.
            with stage("retrieve"):
                hits = self.retriever.search(tenant, question, k)

        if self.llm is None:
            notes.append("no model configured: returning retrieved passages only")
            ans = Answer(hits[0].text if hits else ABSTAIN, ["D1"] if hits else [], abstained=not hits)
        else:
            with stage("generate"):
                ans = generate_answer(self.llm, question, hits, sql_answer.as_evidence() if sql_answer else None)
                usage.add(ans.usage)

        timings["total"] = (time.perf_counter() - t_start) * 1000
        citations = []
        for label in ans.citations:
            if label == "S1":
                citations.append({"label": "S1", "source": "database", "sql": sql_answer.sql if sql_answer else None})
            else:
                h = hits[int(label[1:]) - 1]
                citations.append({"label": label, "source": h.source_uri, "section": h.heading_path,
                                  "chunk_id": h.chunk_id})
        if ans.invalid_citations:
            notes.append(f"removed citations to evidence that was never supplied: {ans.invalid_citations}")
        return QueryResult(
            answer=ans.text, route=chosen, grounded=ans.grounded, abstained=ans.abstained,
            citations=citations, sql=sql_answer.sql if sql_answer else None,
            sql_rows=len(sql_answer.rows) if sql_answer else None,
            timings_ms=timings, usage=usage, notes=notes,
        )
