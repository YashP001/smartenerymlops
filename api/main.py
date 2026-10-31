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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict
from src.features import FEATURE_COLUMNS

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
    """Manage application lifecycle: load model once at startup into app.state."""
    global model
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
    """Prediction response with model metadata."""

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
def make_prediction(input_data: PredictionInput):
    """Predict next-hour energy consumption.

    Accepts the six input features used by the trained Gradient
    Boosting Regressor and returns the predicted energy value.
    """
    active_model = get_active_model()
    if active_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run evaluate_model.py first.",
        )

    try:
        input_dict = input_data.model_dump()
        prediction = predict(active_model, input_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionOutput(
        predicted_energy_kwh=prediction,
    )
