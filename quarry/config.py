"""Runtime configuration, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _keys_from_env(raw: str) -> dict[str, str]:
    """Parse ``tenant:sha256hex,tenant2:sha256hex`` into {sha256hex: tenant}."""
    out: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        tenant, _, digest = pair.partition(":")
        if tenant and len(digest) == 64:
            out[digest.lower()] = tenant
    return out


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://quarry_app:quarry_app@localhost:5432/quarry"
    provider: str = "offline"            # "gemini" | "offline"
    chat_model: str = "gemini-3.8-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    embed_batch: int = 16
    chunk_tokens: int = 320
    chunk_overlap: int = 40
    top_k: int = 6
    candidates: int = 24                 # per-retriever candidates before fusion
    sql_tables: tuple[str, ...] = ("customers", "shipments", "invoices", "claims")
    sql_max_rows: int = 200
    sql_timeout_ms: int = 3000
    rate_limit_rpm: int = 120
    api_keys: dict[str, str] = field(default_factory=dict)   # sha256(key) -> tenant

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ
        d = cls()
        return cls(
            database_url=e.get("QUARRY_DATABASE_URL", d.database_url),
            provider=e.get("QUARRY_PROVIDER", d.provider),
            chat_model=e.get("QUARRY_CHAT_MODEL", d.chat_model),
            embed_model=e.get("QUARRY_EMBED_MODEL", d.embed_model),
            embed_dim=int(e.get("QUARRY_EMBED_DIM", d.embed_dim)),
            embed_batch=int(e.get("QUARRY_EMBED_BATCH", d.embed_batch)),
            chunk_tokens=int(e.get("QUARRY_CHUNK_TOKENS", d.chunk_tokens)),
            chunk_overlap=int(e.get("QUARRY_CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(e.get("QUARRY_TOP_K", d.top_k)),
            candidates=int(e.get("QUARRY_CANDIDATES", d.candidates)),
            sql_max_rows=int(e.get("QUARRY_SQL_MAX_ROWS", d.sql_max_rows)),
            sql_timeout_ms=int(e.get("QUARRY_SQL_TIMEOUT_MS", d.sql_timeout_ms)),
            rate_limit_rpm=int(e.get("QUARRY_RATE_LIMIT_RPM", d.rate_limit_rpm)),
            api_keys=_keys_from_env(e.get("QUARRY_API_KEYS", "")),
        )
