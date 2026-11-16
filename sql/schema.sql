-- Quarry schema. Applied by `python -m quarry.admin init` (admin connection).
-- {dim} is substituted with QUARRY_EMBED_DIM.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- roles ----
-- quarry_app : what the API connects as. Not an owner, not a superuser, so
--              row-level security always applies to it.
-- quarry_ro  : SELECT-only role that model-written SQL runs under.
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'quarry_ro') THEN
    CREATE ROLE quarry_ro NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'quarry_app') THEN
    CREATE ROLE quarry_app LOGIN PASSWORD 'quarry_app';
  END IF;
END $$;
GRANT quarry_ro TO quarry_app;

-- ------------------------------------------------- unstructured: documents --
CREATE TABLE IF NOT EXISTS documents (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id     TEXT NOT NULL,
  source_uri    TEXT NOT NULL,
  title         TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, source_uri)
);

CREATE TABLE IF NOT EXISTS chunks (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id     TEXT NOT NULL,
  document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal       INT  NOT NULL,
  heading_path  TEXT NOT NULL,
  text          TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  token_count   INT  NOT NULL,
  embedding     vector({dim}) NOT NULL,
  tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin        ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx        ON chunks (tenant_id, document_id);

-- --------------------------------------------- structured: business tables --
CREATE TABLE IF NOT EXISTS customers (
  id         TEXT PRIMARY KEY,
  tenant_id  TEXT NOT NULL,
  name       TEXT NOT NULL,
  tier       TEXT NOT NULL CHECK (tier IN ('standard', 'priority', 'enterprise')),
  region     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shipments (
  id             TEXT PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  customer_id    TEXT NOT NULL REFERENCES customers(id),
  origin         TEXT NOT NULL,
  destination    TEXT NOT NULL,
  service_level  TEXT NOT NULL CHECK (service_level IN ('economy', 'standard', 'express')),
  status         TEXT NOT NULL CHECK (status IN ('in_transit', 'delivered', 'delayed', 'lost', 'damaged')),
  weight_kg      NUMERIC(8,1) NOT NULL,
  shipped_at     DATE NOT NULL,
  promised_at    DATE NOT NULL,
  delivered_at   DATE
);

CREATE TABLE IF NOT EXISTS invoices (
  id           TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL,
  shipment_id  TEXT NOT NULL REFERENCES shipments(id),
  amount_usd   NUMERIC(10,2) NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('open', 'paid', 'overdue', 'credited')),
  issued_at    DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
  id           TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL,
  shipment_id  TEXT NOT NULL REFERENCES shipments(id),
  type         TEXT NOT NULL CHECK (type IN ('damage', 'loss', 'delay')),
  amount_usd   NUMERIC(10,2) NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('open', 'approved', 'denied', 'paid')),
  filed_at     DATE NOT NULL
);

-- --------------------------------------------------- tenant isolation (RLS) --
-- Every table is filtered by the transaction-local setting app.tenant_id.
-- If it is unset the predicate is NULL and no rows are visible: fail closed.
DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY['documents','chunks','customers','shipments','invoices','claims'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I
         USING (tenant_id = current_setting(''app.tenant_id'', true))
         WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))', t);
  END LOOP;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks TO quarry_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON customers, shipments, invoices, claims TO quarry_app;
GRANT SELECT ON customers, shipments, invoices, claims TO quarry_ro;
