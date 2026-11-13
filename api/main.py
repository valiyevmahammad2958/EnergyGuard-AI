import json
from pathlib import Path
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI
from src.preprocessing import preprocess_features, FEATURE_COLUMNS
from src.model import Autoencoder
import pandas as pd
from src.feature_engineering import create_features
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder.pt"

RESULTS_PATH = PROJECT_ROOT / "models" / "autoencoder_results.json"

with open(RESULTS_PATH) as f:
    results = json.load(f)

THRESHOLD = results["threshold"]

# Create FastAPI application
app = FastAPI(
    title="EnergyGuard AI",
    description="Oil well anomaly detection API",
    version="1.0.0"
)

# Load the trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = Autoencoder(input_dim=checkpoint["input_dim"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to EnergyGuard AI",
        "status": "running"
    }

# Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EnergyGuard AI",
        "model_loaded": True,
        "input_features": checkpoint["input_dim"]
    }

# Define the expected input
class PredictionInput(BaseModel):
    features: list[float] = Field(min_length=13, max_length=13)


@app.post("/predict")
def predict(data: PredictionInput):

    # Convert input into a PyTorch tensor
    x = torch.tensor(
        [data.features],
        dtype=torch.float32
    )

    # Run the trained autoencoder
    with torch.no_grad():
        reconstructed = model(x)

        # Mean squared reconstruction error
        anomaly_score = torch.mean(
            (x - reconstructed) ** 2
        ).item()

    # Use the frozen validation threshold
    is_anomaly = anomaly_score > THRESHOLD

    return {
        "anomaly_score": anomaly_score,
        "threshold": THRESHOLD,
        "is_anomaly": is_anomaly
    }

from pydantic import ConfigDict


class EngineeredFeatureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oil_rate: float
    gas_rate: float
    water_rate: float
    tubing_pressure: float
    casing_pressure: float
    temperature: float
    choke_size: float
    oil_rate_diff: float
    pressure_diff: float
    gas_rate_diff: float
    pressure_rolling_mean: float
    pressure_rolling_std: float
    oil_rolling_mean: float


@app.post("/predict-features")
def predict_features(data: EngineeredFeatureInput):

    # Convert named input into a dictionary
    features = data.model_dump()

    # Apply the original training scaler
    scaled = preprocess_features(features)

    # Convert scaled features to a PyTorch tensor
    x = torch.from_numpy(scaled)

    # Run the trained autoencoder
    with torch.no_grad():
        reconstructed = model(x)
        anomaly_score = torch.mean(
            (x - reconstructed) ** 2
        ).item()

    return {
        "anomaly_score": anomaly_score,
        "threshold": THRESHOLD,
        "is_anomaly": anomaly_score > THRESHOLD
    }

from datetime import datetime
from pydantic import Field


class RawMeasurement(BaseModel):
    timestamp: datetime
    oil_rate: float
    gas_rate: float
    water_rate: float
    tubing_pressure: float
    casing_pressure: float
    temperature: float
    choke_size: float


class RawPredictionInput(BaseModel):
    well_id: str
    measurements: list[RawMeasurement] = Field(min_length=6)


@app.post("/predict-raw")
def predict_raw(data: RawPredictionInput):

    # Convert incoming measurements to a DataFrame
    records = [
        measurement.model_dump()
        for measurement in data.measurements
    ]

    df = pd.DataFrame(records)
    df["well_id"] = data.well_id
    # Calculate the six engineered features
    df = create_features(df)

    # Use the latest measurement from each well
    latest = (
        df.sort_values(["well_id", "timestamp"])
        .groupby("well_id")
        .tail(1)
    )

    predictions = []

    for _, row in latest.iterrows():

        # Apply the saved training scaler
        scaled = preprocess_features(row.to_dict())

        # Convert to a PyTorch tensor
        x = torch.from_numpy(scaled)

        # Run inference
        with torch.no_grad():
            reconstructed = model(x)

            score = torch.mean(
                (x - reconstructed) ** 2
            ).item()

        predictions.append({
            "well_id": row["well_id"],
            "timestamp": row["timestamp"].isoformat(),
            "anomaly_score": score,
            "threshold": THRESHOLD,
            "is_anomaly": score > THRESHOLD
        })

    return {"predictions": predictions}
