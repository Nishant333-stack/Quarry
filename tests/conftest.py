"""Tests run against a real Postgres + pgvector (RLS and HNSW cannot be faked).

Set QUARRY_TEST_ADMIN_DSN to an owner connection for a scratch server, e.g.
    docker compose up -d db
    export QUARRY_TEST_ADMIN_DSN=postgresql://postgres:postgres@localhost:5432/postgres
"""
import os
from pathlib import Path

import psycopg
import pytest

from quarry.config import Settings
from quarry.llm import HashEmbedder
from quarry.pipeline import Quarry
from quarry.seed import seed_demo
from quarry.store import Store, init_schema

ADMIN = os.environ.get("QUARRY_TEST_ADMIN_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
DB = "quarry_pytest"
DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def _swap_db(dsn: str, db: str, user: str | None = None) -> str:
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    info["dbname"] = db
    if user:
        info["user"], info["password"] = user, user
    return psycopg.conninfo.make_conninfo(**info)


@pytest.fixture(scope="session")
def dsns():
    try:
        with psycopg.connect(ADMIN, autocommit=True, connect_timeout=3) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
            conn.execute(f"CREATE DATABASE {DB}")
    except psycopg.OperationalError as e:
        pytest.skip(f"no test Postgres available: {e}")
    admin = _swap_db(ADMIN, DB)
    init_schema(admin, 768)
    return {"admin": admin, "app": _swap_db(ADMIN, DB, "quarry_app")}


@pytest.fixture(scope="session")
def store(dsns):
    s = Store(dsns["app"])
    seed_demo(s)
    yield s
    s.close()


@pytest.fixture()
def embedder():
    return HashEmbedder(768)


def make_quarry(store, llm=None, **overrides) -> Quarry:
    settings = Settings(provider="offline", **overrides)
    return Quarry(settings, store, HashEmbedder(768), llm)


@pytest.fixture(scope="session")
def corpus(store):
    """The demo corpus ingested once for tenant 'meridian'."""
    q = make_quarry(store)
    q.ingestor.ingest_dir("meridian", DOCS)
    return q
