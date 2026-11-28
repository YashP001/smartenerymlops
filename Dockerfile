# Dockerfile for the Smart Home Energy Advisor MLOps stack.
#
# Two runnable targets share one base:
#   tools : monitoring worker + registry bootstrap (needs scipy/evidently/git)
#   api   : FastAPI inference service (DEFAULT / last stage, used by CI and
#           `docker build .`)
#
# docker compose selects targets explicitly (see docker-compose.yml).

FROM python:3.13-slim AS base

WORKDIR /app

# - PYTHONDONTWRITEBYTECODE: prevent .pyc generation
# - PYTHONUNBUFFERED: direct log streaming without buffering
# - PYTHONPATH: ensures root import resolution (api, src, models, scripts)
# - MLFLOW_DISABLE_AGENT_HINT: silence non-runtime MLflow telemetry hints
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1

# ─── tools: monitoring worker and registry bootstrap ─────────────────
FROM base AS tools

# git lets the training script record the code revision in MLflow lineage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt requirements-monitor.txt ./
RUN pip install --no-cache-dir -r requirements-monitor.txt

COPY src/ src/
COPY models/ models/
COPY scripts/ scripts/

# Run as an unprivileged user; volumes are chowned by compose at first start.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/monitoring-store /app/mlflow-store /app/reports/demo \
    && chown -R appuser:appuser /app
USER appuser

# ─── api: inference service (default target) ─────────────────────────
FROM base AS api

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY api/ api/
COPY src/ src/
COPY models/ models/

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/monitoring-store /app/mlflow-store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Bind 0.0.0.0 so published ports reach the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
