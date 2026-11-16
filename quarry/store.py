"""Postgres + pgvector storage.

One database holds vectors, full-text and the business tables, so a single
transaction-scoped setting (``app.tenant_id``) plus row-level security isolates
tenants across all three retrieval paths.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or our the to "
    "was we what when where which who why will with you your".split()
)


@dataclass
class Hit:
    chunk_id: int
    text: str
    heading_path: str
    title: str
    source_uri: str
    score: float


def vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def init_schema(admin_dsn: str, dim: int) -> None:
    """Create extension, roles, tables, indexes and RLS policies (idempotent)."""
    sql = SCHEMA.read_text().replace("{dim}", str(int(dim)))
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(sql)


class Store:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8):
        self.pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, kwargs={"row_factory": dict_row}, open=True
        )

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def tenant(self, tenant_id: str) -> Iterator[psycopg.Connection]:
        """A transaction in which RLS only exposes ``tenant_id``'s rows."""
        with self.pool.connection() as conn:
            with conn.transaction():
                # is_local=true: the setting dies with the transaction, so a pooled
                # connection can never leak one tenant's scope into the next request.
                conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
                yield conn

    # ------------------------------------------------------------ documents --
    @staticmethod
    def get_document(conn, source_uri: str) -> dict | None:
        return conn.execute(
            "SELECT id, content_hash FROM documents WHERE source_uri = %s", (source_uri,)
        ).fetchone()

    @staticmethod
    def upsert_document(conn, tenant: str, source_uri: str, title: str, content_hash: str) -> int:
        row = conn.execute(
            """INSERT INTO documents (tenant_id, source_uri, title, content_hash)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (tenant_id, source_uri)
               DO UPDATE SET title = EXCLUDED.title, content_hash = EXCLUDED.content_hash, updated_at = now()
               RETURNING id""",
            (tenant, source_uri, title, content_hash),
        ).fetchone()
        return row["id"]

    @staticmethod
    def chunk_hashes(conn, document_id: int) -> list[dict]:
        return conn.execute(
            "SELECT id, content_hash FROM chunks WHERE document_id = %s ORDER BY ordinal", (document_id,)
        ).fetchall()

    @staticmethod
    def insert_chunk(conn, tenant: str, document_id: int, chunk, embedding: Sequence[float]) -> None:
        conn.execute(
            """INSERT INTO chunks (tenant_id, document_id, ordinal, heading_path, text, content_hash, token_count, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)""",
            (tenant, document_id, chunk.ordinal, chunk.heading_path, chunk.text, chunk.content_hash,
             chunk.token_count, vec_literal(embedding)),
        )

    @staticmethod
    def delete_document(conn, source_uri: str) -> bool:
        return conn.execute("DELETE FROM documents WHERE source_uri = %s", (source_uri,)).rowcount > 0

    @staticmethod
    def list_documents(conn) -> list[dict]:
        return conn.execute(
            """SELECT d.source_uri, d.title, d.updated_at, count(c.id) AS chunks
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY d.source_uri"""
        ).fetchall()

    # --------------------------------------------------------------- search --
    _SELECT = """SELECT c.id AS chunk_id, c.text, c.heading_path, d.title, d.source_uri, {score} AS score
                 FROM chunks c JOIN documents d ON d.id = c.document_id"""

    def dense_search(self, conn, query_vec: Sequence[float], n: int) -> list[Hit]:
        v = vec_literal(query_vec)
        # RLS filters after the ANN scan, so widen the scan to keep recall for small tenants.
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(100, n * 4)),))
        rows = conn.execute(
            self._SELECT.format(score="1 - (c.embedding <=> %s::vector)")
            + " ORDER BY c.embedding <=> %s::vector LIMIT %s",
            (v, v, n),
        ).fetchall()
        return [Hit(**r) for r in rows]

    def lexical_search(self, conn, query: str, n: int) -> list[Hit]:
        terms = [w for w in dict.fromkeys(m.lower() for m in _WORD.findall(query)) if w not in _STOP]
        if not terms:
            return []
        # OR the terms: natural-language questions rarely contain *every* word of the passage.
        tsq = " | ".join(terms)
        rows = conn.execute(
            self._SELECT.format(score="ts_rank_cd(c.tsv, q, 32)")
            + ", to_tsquery('english', %s) q WHERE c.tsv @@ q ORDER BY score DESC, c.id LIMIT %s",
            (tsq, n),
        ).fetchall()
        return [Hit(**{**r, "score": float(r["score"])}) for r in rows]
