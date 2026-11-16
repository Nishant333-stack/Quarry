export QUARRY_ADMIN_DATABASE_URL ?= postgresql://postgres:postgres@localhost:5432/quarry
export QUARRY_DATABASE_URL ?= postgresql://quarry_app:quarry_app@localhost:5432/quarry
export QUARRY_TEST_ADMIN_DSN ?= postgresql://postgres:postgres@localhost:5432/postgres

.PHONY: db setup test eval eval-e2e serve
db:            ## start local Postgres + pgvector
	docker compose up -d --wait db
setup: db      ## schema, demo data, ingest the sample corpus
	python -m quarry.admin init
	python -m quarry.admin seed
	python -m quarry.admin ingest data/docs -t meridian
test:          ## unit + integration tests (needs `make db`)
	python -m pytest
eval:          ## offline checks: ingestion idempotency, retrieval ablation, SQL guard
	python -m evals.run_eval offline --gate
eval-e2e:      ## full quality/latency/cost run with Gemini (needs credentials)
	QUARRY_PROVIDER=gemini python -m evals.run_eval all --gate
serve:
	uvicorn quarry.api:create_app --factory --reload --port 8080
