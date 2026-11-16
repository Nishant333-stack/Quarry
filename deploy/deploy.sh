#!/usr/bin/env bash
# Deploy Quarry to Cloud Run with Cloud SQL (Postgres 16 + pgvector) and Vertex AI.
#
#   PROJECT_ID=my-proj REGION=asia-south1 ./deploy/deploy.sh
#
# Idempotent: safe to re-run. Requires gcloud authenticated as a project owner/editor.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-quarry}"
INSTANCE="${INSTANCE:-quarry-pg}"
DB="${DB:-quarry}"
SA="${SERVICE}-run@${PROJECT_ID}.iam.gserviceaccount.com"
CONN="${PROJECT_ID}:${REGION}:${INSTANCE}"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com \
  aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# --- Cloud SQL -------------------------------------------------------------
if ! gcloud sql instances describe "$INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances create "$INSTANCE" --database-version=POSTGRES_16 --edition=ENTERPRISE \
    --tier=db-custom-1-3840 --region="$REGION" --storage-size=10 --storage-auto-increase
fi
gcloud sql databases describe "$DB" --instance="$INSTANCE" >/dev/null 2>&1 || \
  gcloud sql databases create "$DB" --instance="$INSTANCE"

# --- Secrets: generated once, never printed -------------------------------
mksecret() {  # name
  gcloud secrets describe "$1" >/dev/null 2>&1 || \
    openssl rand -hex 24 | tr -d '\n' | gcloud secrets create "$1" --data-file=- --replication-policy=automatic
}
mksecret quarry-admin-password
mksecret quarry-app-password
mksecret quarry-api-key
ADMIN_PW="$(gcloud secrets versions access latest --secret=quarry-admin-password)"
APP_PW="$(gcloud secrets versions access latest --secret=quarry-app-password)"
gcloud sql users set-password postgres --instance="$INSTANCE" --password="$ADMIN_PW" >/dev/null

# --- Service account: least privilege --------------------------------------
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SERVICE}-run" --display-name="Quarry Cloud Run"
for role in roles/cloudsql.client roles/aiplatform.user roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${SA}" --role="$role" \
    --condition=None >/dev/null
done

# --- Schema, roles, RLS, demo data (through the Cloud SQL Auth Proxy) -------
if command -v cloud-sql-proxy >/dev/null; then
  cloud-sql-proxy --port 6543 "$CONN" & PROXY=$!
  trap 'kill $PROXY' EXIT
  sleep 4
  export QUARRY_ADMIN_DATABASE_URL="postgresql://postgres:${ADMIN_PW}@127.0.0.1:6543/${DB}"
  python -m quarry.admin init
  psql "$QUARRY_ADMIN_DATABASE_URL" -qc "ALTER ROLE quarry_app PASSWORD '${APP_PW}'"
  export QUARRY_DATABASE_URL="postgresql://quarry_app:${APP_PW}@127.0.0.1:6543/${DB}"
  export QUARRY_PROVIDER=gemini GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT="$PROJECT_ID" GOOGLE_CLOUD_LOCATION=global
  python -m quarry.admin seed
  python -m quarry.admin ingest data/docs -t meridian
else
  echo "cloud-sql-proxy not found: skipping schema/seed. Install it and re-run." >&2
fi

# The runtime DSN (unix socket) lives in Secret Manager, not in the service config.
printf 'postgresql://quarry_app:%s@/%s?host=/cloudsql/%s' "$APP_PW" "$DB" "$CONN" | \
  gcloud secrets versions add quarry-database-url --data-file=- 2>/dev/null || \
  printf 'postgresql://quarry_app:%s@/%s?host=/cloudsql/%s' "$APP_PW" "$DB" "$CONN" | \
  gcloud secrets create quarry-database-url --data-file=- --replication-policy=automatic

API_KEY="$(gcloud secrets versions access latest --secret=quarry-api-key)"
KEY_HASH="$(printf '%s' "$API_KEY" | sha256sum | cut -d' ' -f1)"

# --- Cloud Run ---------------------------------------------------------------
gcloud run deploy "$SERVICE" --source . --region="$REGION" --service-account="$SA" \
  --add-cloudsql-instances="$CONN" --allow-unauthenticated \
  --cpu=1 --memory=512Mi --concurrency=40 --min-instances=0 --max-instances=3 \
  --set-secrets="QUARRY_DATABASE_URL=quarry-database-url:latest" \
  --set-env-vars="QUARRY_PROVIDER=gemini,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,QUARRY_API_KEYS=meridian:${KEY_HASH}"

URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
echo
echo "Deployed: $URL"
echo "API key:  gcloud secrets versions access latest --secret=quarry-api-key"
echo "Try:      curl -s $URL/v1/query -H \"Authorization: Bearer \$KEY\" -H 'content-type: application/json' \\"
echo "               -d '{\"question\": \"How long do I have to file a damage claim?\"}'"
