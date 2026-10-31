# Dockerfile for Smart Home Energy Advisor FastAPI Inference Service
# Member 4 deliverable: Containerization and application integration

FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Environment configuration:
# - PYTHONDONTWRITEBYTECODE: prevent .pyc generation
# - PYTHONUNBUFFERED: direct log streaming without buffering
# - PYTHONPATH: ensures root import resolution (api, src, models)
# - MLFLOW_DISABLE_AGENT_HINT: silence non-runtime MLflow telemetry hints
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1

# Install dedicated inference dependencies first for layer caching
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy only the necessary application source code and model artifacts
COPY api/ api/
COPY src/ src/
COPY models/ models/

# Expose FastAPI inference port
EXPOSE 8000

# Run Uvicorn server bound to 0.0.0.0 for external container routing
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
