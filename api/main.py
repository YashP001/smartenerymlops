"""
FastAPI inference service for Smart Home Energy Advisor.

Member 3 deliverable: Exposes the trained model through a REST
API with /health and /predict endpoints as specified in Phase 2
requirements (PDF §4, Member 3).

Run with:
    uvicorn api.main:app --reload
"""

import os
import sys

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.predict import load_model, predict
from src.features import FEATURE_COLUMNS
from src.monitoring.database import (
    init_db,
    log_prediction,
    log_feedback,
    PredictionNotFoundError,
    DuplicateFeedbackError,
    InvalidFeedbackValueError,
)
from src.monitoring.metrics import default_metrics

logger = logging.getLogger("api.monitoring")


def safe_log_prediction(*args, **kwargs):
    """Safely log predictions to the monitoring database without disrupting the response."""
    try:
        log_prediction(*args, **kwargs)
    except Exception as e:
        logger.error(
            "Failed to record prediction in monitoring database: %s",
            e,
            exc_info=True,
        )


# ─── Model lifecycle & Lifespan ──────────────────────────────────────

model = None


def get_active_model():
    """Retrieve the model from application state, loading once if uninitialized."""
    global model
    active = getattr(app.state, "model", None)
    if active is not None:
        return active
    if model is not None:
        app.state.model = model
        return model
    try:
        loaded = load_model()
        app.state.model = loaded
        model = loaded
        return loaded
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: initialize monitoring db and load model once."""
    global model
    try:
        init_db()
        app.state.database_initialized = True
        print("Monitoring persistence layer initialized.")
    except Exception as e:
        app.state.database_initialized = False
        print(f"Monitoring DB initialization notice: {e}")

    try:
        loaded = load_model()
        app.state.model = loaded
        model = loaded
        print("Model loaded successfully into application state.")
    except Exception as e:
        print(f"Startup Warning: Failed to load model ({e}).")
        app.state.model = None
        model = None
    yield
    app.state.model = None
    model = None
    app.state.database_initialized = False


# ─── FastAPI application ─────────────────────────────────────────────

app = FastAPI(
    title="Smart Home Energy Advisor",
    description=(
        "Predict next-hour electricity consumption from smart meter "
        "readings using a Gradient Boosting Regressor trained on "
        "CEEW Mathura 2021 data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    """Record HTTP request counts and durations with low-cardinality endpoint paths."""
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        # Exclude internal /metrics scrape requests from application HTTP traffic metrics
        if request.url.path != "/metrics":
            duration = time.perf_counter() - start
            route = request.scope.get("route")
            endpoint = route.path if route and hasattr(route, "path") else "unmatched"
            default_metrics.record_http(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
                duration_seconds=duration,
            )


# ─── Request / response schemas ──────────────────────────────────────

class PredictionInput(BaseModel):
    """Input features for energy consumption prediction."""

    energy_current_hour: float = Field(
        ...,
        ge=0,
        description="Energy consumption in the current hour (kWh)",
        examples=[0.05],
    )
    energy_previous_hour: float = Field(
        ...,
        ge=0,
        description="Energy consumption in the previous hour (kWh)",
        examples=[0.04],
    )
    energy_two_hours_ago: float = Field(
        ...,
        ge=0,
        description="Energy consumption two hours ago (kWh)",
        examples=[0.06],
    )
    target_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the day for prediction target (0-23)",
        examples=[14],
    )
    target_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday)",
        examples=[2],
    )
    target_is_weekend: int = Field(
        ...,
        ge=0,
        le=1,
        description="Weekend flag (0=weekday, 1=weekend)",
        examples=[0],
    )


class PredictionOutput(BaseModel):
    """Prediction response with model metadata and tracking identifier."""

    predicted_energy_kwh: float = Field(
        ...,
        description="Predicted energy consumption for the next hour (kWh)",
    )
    model: str = Field(
        default="GradientBoostingRegressor",
        description="Model used for prediction",
    )
    features_used: list = Field(
        default_factory=lambda: FEATURE_COLUMNS.copy(),
        description="Feature columns used by the model",
    )
    prediction_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the prediction (used for ground-truth feedback)",
    )


class FeedbackInput(BaseModel):
    """Payload for recording actual ground-truth energy consumption."""

    prediction_id: str = Field(
        ...,
        description="Unique prediction ID returned during inference",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    actual_energy_kwh: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Actual observed energy consumption for the target hour (kWh)",
        examples=[0.052],
    )


class ValueOnlyFeedbackInput(BaseModel):
    """Payload for recording actual energy when prediction_id is in the path URL."""

    actual_energy_kwh: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Actual observed energy consumption for the target hour (kWh)",
        examples=[0.052],
    )


class FeedbackResponse(BaseModel):
    """Response for successful feedback ingestion."""

    status: str = Field(default="success", description="Feedback ingestion status")
    prediction_id: str = Field(..., description="Prediction identifier")
    actual_energy_kwh: float = Field(..., description="Recorded actual energy (kWh)")
    feedback_timestamp: str = Field(..., description="UTC timestamp of feedback ingestion")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response indicating operational prediction capability."""

    status: str = Field(..., description="Service readiness status")
    model_loaded: bool = Field(..., description="Whether the model is ready for predictions")
    model_version: Optional[str] = Field(None, description="Active model version name")
    database_initialized: bool = Field(..., description="Whether SQLite monitoring store is initialized")


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root_homepage():
    """Render a clean, responsive landing page for the Smart Home Energy Advisor service."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Home Energy Advisor</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --success: #10b981;
      --border: #334155;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .container {
      max-width: 680px;
      width: 100%;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2.5rem;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(16, 185, 129, 0.12);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.875rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
    }
    .pulse {
      width: 8px;
      height: 8px;
      background: var(--success);
      border-radius: 50%;
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
      animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
      0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
      }
      70% {
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
      }
      100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
      }
    }
    h1 {
      font-size: 1.875rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 0.75rem;
      letter-spacing: -0.025em;
    }
    .description {
      color: var(--text-muted);
      font-size: 1.05rem;
      line-height: 1.6;
      margin-bottom: 2rem;
    }
    .links-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.875rem;
      margin-bottom: 2rem;
    }
    .card-link {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 1rem 0.75rem;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text-main);
      text-decoration: none;
      font-weight: 500;
      transition: all 0.2s ease-in-out;
    }
    .card-link:hover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
      color: var(--accent);
      transform: translateY(-2px);
    }
    .card-link .endpoint-path {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.95rem;
      font-weight: 600;
    }
    .card-link .endpoint-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }
    .footer {
      border-top: 1px solid var(--border);
      padding-top: 1.25rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.875rem;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .footer a {
      color: var(--accent);
      text-decoration: none;
    }
    .footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="badge">
      <span class="pulse"></span>
      API is Running &amp; Operational
    </div>
    <h1>Smart Home Energy Advisor</h1>
    <p class="description">
      An ML-powered application that predicts the next hour's energy consumption from residential smart-meter telemetry.
    </p>

    <div class="links-grid">
      <a href="/docs" class="card-link" id="link-docs">
        <span class="endpoint-path">/docs</span>
        <span class="endpoint-label">Swagger UI</span>
      </a>
      <a href="/health" class="card-link" id="link-health">
        <span class="endpoint-path">/health</span>
        <span class="endpoint-label">Liveness Probe</span>
      </a>
      <a href="/ready" class="card-link" id="link-ready">
        <span class="endpoint-path">/ready</span>
        <span class="endpoint-label">Readiness Probe</span>
      </a>
      <a href="/metrics" class="card-link" id="link-metrics">
        <span class="endpoint-path">/metrics</span>
        <span class="endpoint-label">Prometheus Metrics</span>
      </a>
    </div>

    <div class="footer">
      <span>AI-Assisted Smart Home Energy Advisor (v1.0.0)</span>
      <span>Model: GradientBoostingRegressor</span>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the service is healthy and the model is loaded."""
    active_model = get_active_model()
    return HealthResponse(
        status="healthy",
        model_loaded=active_model is not None,
    )


@app.get("/ready", response_model=ReadinessResponse)
def readiness_check():
    """Verify that the model is loaded and the service is ready to serve predictions.

    Returns HTTP 200 when the model is operational, or HTTP 503 when the model
    is unavailable and predictions cannot be served.
    """
    active_model = get_active_model()
    db_initialized = bool(getattr(app.state, "database_initialized", False))

    if active_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": False,
                "model_version": None,
                "database_initialized": db_initialized,
                "message": "Model is not loaded. Predictions unavailable.",
            },
        )

    if not db_initialized:
        model_name = getattr(active_model, "__class__", type(active_model)).__name__
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": True,
                "model_version": model_name,
                "database_initialized": False,
                "message": "Monitoring database is unavailable. Predictions and feedback cannot be persisted.",
            },
        )

    model_name = getattr(active_model, "__class__", type(active_model)).__name__
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_version=model_name,
        database_initialized=True,
    )


@app.get("/metrics")
def metrics_endpoint():
    """Expose Prometheus metrics for operational observability scraping."""
    payload = default_metrics.generate_payload()
    return Response(
        content=payload,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(
    input_data: PredictionInput,
    background_tasks: BackgroundTasks,
    response: Response,
):
    """Predict next-hour energy consumption.

    Accepts the six input features used by the trained Gradient
    Boosting Regressor, returns the predicted energy value, and asynchronously
    records the prediction for performance and drift monitoring.
    """
    active_model = get_active_model()
    if active_model is None:
        default_metrics.record_prediction(
            model_version="unknown",
            status="error",
        )
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run evaluate_model.py first.",
        )

    start_time = time.perf_counter()
    try:
        input_dict = input_data.model_dump()
        prediction = predict(active_model, input_dict)
    except ValueError as e:
        model_name = getattr(active_model, "__class__", type(active_model)).__name__
        default_metrics.record_prediction(
            model_version=model_name,
            status="error",
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        model_name = getattr(active_model, "__class__", type(active_model)).__name__
        default_metrics.record_prediction(
            model_version=model_name,
            status="error",
        )
        raise

    latency_s = time.perf_counter() - start_time
    latency_ms = round(latency_s * 1000, 2)
    prediction_id = str(uuid.uuid4())
    model_name = getattr(active_model, "__class__", type(active_model)).__name__

    default_metrics.record_prediction(
        model_version=model_name,
        status="success",
        latency_seconds=latency_s,
        predicted_energy=prediction,
    )

    # Asynchronously persist prediction without blocking client response
    background_tasks.add_task(
        safe_log_prediction,
        prediction_id=prediction_id,
        features=input_dict,
        predicted_energy_kwh=prediction,
        model_version=model_name,
        latency_ms=latency_ms,
    )

    response.headers["X-Prediction-ID"] = prediction_id

    return PredictionOutput(
        predicted_energy_kwh=prediction,
        model=model_name,
        prediction_id=prediction_id,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def record_feedback(feedback_data: FeedbackInput):
    """Attach actual energy consumption feedback to an existing prediction."""
    try:
        result = log_feedback(
            prediction_id=feedback_data.prediction_id,
            actual_energy_kwh=feedback_data.actual_energy_kwh,
        )
        return FeedbackResponse(
            status="success",
            prediction_id=result["prediction_id"],
            actual_energy_kwh=result["actual_energy_kwh"],
            feedback_timestamp=result["feedback_timestamp"],
        )
    except PredictionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DuplicateFeedbackError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except InvalidFeedbackValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predictions/{prediction_id}/feedback", response_model=FeedbackResponse)
def record_prediction_feedback(prediction_id: str, feedback_data: ValueOnlyFeedbackInput):
    """Attach actual energy consumption feedback using path parameter prediction_id."""
    try:
        result = log_feedback(
            prediction_id=prediction_id,
            actual_energy_kwh=feedback_data.actual_energy_kwh,
        )
        return FeedbackResponse(
            status="success",
            prediction_id=result["prediction_id"],
            actual_energy_kwh=result["actual_energy_kwh"],
            feedback_timestamp=result["feedback_timestamp"],
        )
    except PredictionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DuplicateFeedbackError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except InvalidFeedbackValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
