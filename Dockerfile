FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY quarry ./quarry
COPY sql ./sql
RUN pip install .

# Run as a non-root user; Cloud Run injects $PORT.
RUN useradd --create-home --uid 10001 app
USER app
ENV PORT=8080
CMD exec uvicorn quarry.api:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1
