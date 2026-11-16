"""Idempotent, incremental ingestion.

Re-running ingestion is always safe:
  * unchanged document  -> skipped before chunking (0 embedding calls)
  * edited document     -> only chunks whose content hash is new get embedded;
                           untouched chunks keep their stored vector
  * everything for one document commits in a single transaction, so readers
    never see a half-replaced document.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_document
from .llm import Embedder
from .loaders import load_path
from .store import Store


@dataclass
class IngestResult:
    source_uri: str
    status: str            # "created" | "updated" | "unchanged"
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Ingestor:
    def __init__(self, store: Store, embedder: Embedder, max_tokens: int = 320, overlap_tokens: int = 40):
        self.store, self.embedder = store, embedder
        self.max_tokens, self.overlap_tokens = max_tokens, overlap_tokens

    def ingest_text(self, tenant: str, source_uri: str, title: str, text: str) -> IngestResult:
        doc_hash = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()

        with self.store.tenant(tenant) as conn:
            existing = self.store.get_document(conn, source_uri)
            if existing and existing["content_hash"] == doc_hash:
                return IngestResult(source_uri, "unchanged")

        chunks = chunk_document(title, text, self.max_tokens, self.overlap_tokens)

        with self.store.tenant(tenant) as conn:
            # The upsert takes the document's row lock until commit, so two concurrent
            # ingests of the same file serialise instead of interleaving chunks.
            doc_id = self.store.upsert_document(conn, tenant, source_uri, title, doc_hash)

            stored: dict[str, list[int]] = defaultdict(list)
            for row in self.store.chunk_hashes(conn, doc_id):
                stored[row["content_hash"]].append(row["id"])

            keep: list[tuple[int, int]] = []      # (chunk_id, new_ordinal)
            fresh = []
            for ch in chunks:
                ids = stored.get(ch.content_hash)
                if ids:
                    keep.append((ids.pop(), ch.ordinal))
                else:
                    fresh.append(ch)
            stale = [i for ids in stored.values() for i in ids]

            # Embed only what changed. Done inside the transaction on purpose: if the
            # provider fails we roll back and the previous version stays fully served.
            vectors = self.embedder.embed([c.text for c in fresh], kind="document") if fresh else []

            if stale:
                conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (stale,))
            for chunk_id, ordinal in keep:
                conn.execute("UPDATE chunks SET ordinal = %s WHERE id = %s", (ordinal, chunk_id))
            for ch, vec in zip(fresh, vectors):
                self.store.insert_chunk(conn, tenant, doc_id, ch, vec)

        return IngestResult(
            source_uri,
            "updated" if existing else "created",
            chunks_total=len(chunks),
            chunks_embedded=len(fresh),
            chunks_reused=len(keep),
            chunks_deleted=len(stale),
        )

    def ingest_path(self, tenant: str, path: Path, root: Path | None = None) -> IngestResult:
        title, text = load_path(path)
        uri = str(path.relative_to(root)) if root else path.name
        return self.ingest_text(tenant, uri, title, text)

    def ingest_dir(self, tenant: str, root: Path) -> list[IngestResult]:
        exts = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
        return [self.ingest_path(tenant, p, root) for p in files]
