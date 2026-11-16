"""HTTP API. Run with: uvicorn quarry.api:create_app --factory"""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import Quarry
from .security import TokenBucket, redact, tenant_for_key

log = logging.getLogger("quarry")


class DocumentIn(BaseModel):
    source_uri: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=2_000_000)


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(quarry: Quarry | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (quarry.settings if quarry else Settings.from_env())
    quarry = quarry or Quarry.from_settings(settings)
    bucket = TokenBucket(settings.rate_limit_rpm)
    app = FastAPI(title="Quarry", version="0.1.0")

    def tenant(authorization: str | None = Header(default=None)) -> str:
        """Authenticate and return the caller's tenant. The tenant never comes from the request body."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        key = authorization.removeprefix("Bearer ").strip()
        t = tenant_for_key(key, settings.api_keys)
        if t is None:
            raise HTTPException(401, "invalid API key")
        if not bucket.allow(t):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "5"})
        return t

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(json.dumps({"rid": rid, "path": request.url.path, "status": response.status_code,
                             "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        return response

    @app.get("/healthz")
    def healthz():
        with quarry.store.pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "provider": settings.provider}

    @app.post("/v1/documents")
    def put_document(doc: DocumentIn, t: str = Depends(tenant)):
        return quarry.ingestor.ingest_text(t, doc.source_uri, doc.title, doc.text).as_dict()

    @app.get("/v1/documents")
    def list_documents(t: str = Depends(tenant)):
        with quarry.store.tenant(t) as conn:
            rows = quarry.store.list_documents(conn)
        return {"documents": [{**r, "updated_at": r["updated_at"].isoformat()} for r in rows]}

    @app.delete("/v1/documents")
    def delete_document(source_uri: str, t: str = Depends(tenant)):
        with quarry.store.tenant(t) as conn:
            if not quarry.store.delete_document(conn, source_uri):
                raise HTTPException(404, "no such document")
        return {"deleted": source_uri}

    @app.post("/v1/search")
    def search(q: QueryIn, mode: str = "hybrid", t: str = Depends(tenant)):
        if mode not in ("hybrid", "dense", "lexical"):
            raise HTTPException(422, "mode must be hybrid, dense or lexical")
        hits = quarry.retriever.search(t, q.question, q.top_k or settings.top_k, mode)  # type: ignore[arg-type]
        return {"hits": [h.__dict__ for h in hits]}

    @app.post("/v1/query")
    def query(q: QueryIn, t: str = Depends(tenant)):
        result = quarry.query(t, q.question, q.top_k)
        log.info(json.dumps({"tenant": t, "q": redact(q.question)[:200], "route": result.route,
                             "grounded": result.grounded, **result.usage.as_dict(),
                             "total_ms": round(result.timings_ms["total"], 1)}))
        return result.as_dict()

    return app
