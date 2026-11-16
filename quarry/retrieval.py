"""Hybrid retrieval: dense (pgvector HNSW) + lexical (Postgres FTS), fused with RRF.

Dense search finds paraphrases; lexical search finds exact identifiers
("INV-20417", "HS code 8471") that embeddings blur. Reciprocal Rank Fusion
combines them using ranks only, so the two incomparable score scales never
need calibrating.
"""
from __future__ import annotations

from typing import Literal

from .llm import Embedder
from .store import Hit, Store

Mode = Literal["hybrid", "dense", "lexical"]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, candidates: int = 24):
        self.store, self.embedder, self.candidates = store, embedder, candidates

    def search(self, tenant: str, query: str, k: int = 6, mode: Mode = "hybrid") -> list[Hit]:
        qvec = self.embedder.embed([query], kind="query")[0] if mode != "lexical" else None
        with self.store.tenant(tenant) as conn:
            dense = self.store.dense_search(conn, qvec, self.candidates) if qvec is not None else []
            lexical = self.store.lexical_search(conn, query, self.candidates) if mode != "dense" else []

        if mode == "dense":
            return dense[:k]
        if mode == "lexical":
            return lexical[:k]

        by_id = {h.chunk_id: h for h in (*lexical, *dense)}
        fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h in lexical]])
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        out = []
        for chunk_id, score in ranked:
            hit = by_id[chunk_id]
            hit.score = score
            out.append(hit)
        return out
