
from pathlib import Path
import json

import mlflow
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
# Store experiment information in a SQLite database
DB_PATH = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")



mlflow.set_experiment("EnergyGuard_AI")

# Load existing experiment results
with open(MODEL_DIR / "autoencoder_results.json") as f:
    results = json.load(f)

# Load saved model metadata without retraining
checkpoint = torch.load(
    MODEL_DIR / "autoencoder.pt",
    map_location="cpu",
    weights_only=True
)

# Verify saved metadata is consistent
assert checkpoint["input_dim"] == len(results["feature_columns"])
assert checkpoint["num_epochs"] == results["training_epochs"]
assert checkpoint["feature_columns"] == results["feature_columns"]

# Extract validation metrics automatically
metrics = {
    key.removeprefix("validation_"): value
    for key, value in results.items()
    if key.startswith("validation_")
}

# Extract parameters automatically
params = {
    "model_type": results["model"],
    "architecture": "-".join(
        map(str, results["architecture"])
    ),
    "training_epochs": results["training_epochs"],
    "input_features": checkpoint["input_dim"],
    "threshold": results["threshold"],
    "threshold_source": "validation"
}

# Log the existing experiment
with mlflow.start_run(
    run_name="autoencoder_day9_validation_import"
):

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.log_artifact(
        str(MODEL_DIR / "autoencoder.pt"),
        artifact_path="model"
    )

    mlflow.log_artifact(
        str(MODEL_DIR / "autoencoder_results.json"),
        artifact_path="results"
    )

    print("MLflow experiment logged successfully!")
    print("\nParameters:", params)
    print("\nMetrics:", metrics)
