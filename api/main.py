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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict
from src.features import FEATURE_COLUMNS

# ─── FastAPI application ─────────────────────────────────────────────

app = FastAPI(
    title="Smart Home Energy Advisor",
    description=(
        "Predict next-hour electricity consumption from smart meter "
        "readings using a Gradient Boosting Regressor trained on "
        "CEEW Mathura 2021 data."
    ),
    version="1.0.0",
)

# ─── Model lifecycle ─────────────────────────────────────────────────

model = None


@app.on_event("startup")
def startup_event():
    """Load the trained model when the application starts."""
    global model
    try:
        model = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("The /predict endpoint will return 503 until a model is available.")


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
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(input_data: PredictionInput):
    """Predict next-hour energy consumption.

    Accepts the six input features used by the trained Gradient
    Boosting Regressor and returns the predicted energy value.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run evaluate_model.py first.",
        )

    try:
        input_dict = input_data.model_dump()
        prediction = predict(model, input_dict)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionOutput(
        predicted_energy_kwh=prediction,
    )
