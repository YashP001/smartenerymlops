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
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
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
        print("Monitoring persistence layer initialized.")
    except Exception as e:
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


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the service is healthy and the model is loaded."""
    active_model = get_active_model()
    return HealthResponse(
        status="healthy",
        model_loaded=active_model is not None,
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
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run evaluate_model.py first.",
        )

    start_time = time.perf_counter()
    try:
        input_dict = input_data.model_dump()
        prediction = predict(active_model, input_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    prediction_id = str(uuid.uuid4())
    model_name = getattr(active_model, "__class__", type(active_model)).__name__

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
