
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import Autoencoder

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

THRESHOLD = 0.472979

# Load test data
test_df = pd.read_csv(
    DATA_DIR / "test_model.csv"
)

# Load preprocessing metadata
preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.joblib"
)

feature_columns = preprocessor["feature_columns"]

X_test = test_df[
    feature_columns
].to_numpy(dtype=np.float32)

y_test = test_df[
    "anomaly_label"
].to_numpy()

print("Test observations:", len(X_test))
print("Number of features:", X_test.shape[1])
print("Actual anomalies:", int(y_test.sum()))



# Load saved checkpoint
checkpoint = torch.load(
    MODEL_DIR / "autoencoder.pt",
    map_location="cpu",
    weights_only=True
)

assert checkpoint["feature_columns"] == feature_columns
assert checkpoint["input_dim"] == X_test.shape[1]

model = Autoencoder(
    input_dim=checkpoint["input_dim"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nSaved autoencoder loaded successfully!")


test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

with torch.no_grad():
    reconstructed = model(test_tensor)

test_errors = torch.mean(
    (test_tensor - reconstructed) ** 2,
    dim=1
).numpy()

assert len(test_errors) == len(y_test)
assert np.isfinite(test_errors).all()

print("\nReconstruction errors calculated!")
print(pd.Series(test_errors).describe())


test_predictions = (
    test_errors > THRESHOLD
).astype(int)

print("\nFrozen threshold:", THRESHOLD)
print("Predicted anomalous hours:", int(test_predictions.sum()))
print("Actual anomalous hours:", int(y_test.sum()))



# Calculate test metrics
precision = precision_score(
    y_test, test_predictions, zero_division=0
)

recall = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, test_errors
)

average_precision = average_precision_score(
    y_test, test_errors
)

print("\nAUTOENCODER TEST RESULTS")
print("------------------------")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")


comparison = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Average Precision"
    ],
    "Validation": [
        0.739,
        0.574,
        0.646,
        0.876,
        0.518
    ],
    "Test": [
        precision,
        recall,
        f1,
        roc_auc,
        average_precision
    ]
})

comparison["Difference"] = (
    comparison["Test"] - comparison["Validation"]
)

print("\nVALIDATION VS TEST")
print("------------------")
print(
    comparison.round(3).to_string(index=False)
)


cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
print("----------------")
print(cm)

print("\nTrue positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)


# Add scores and predictions to test data
test_df["reconstruction_error"] = test_errors
test_df["prediction"] = test_predictions


# Normal observations incorrectly flagged
false_positives = test_df[
    (test_df["anomaly_label"] == 0) &
    (test_df["prediction"] == 1)
].copy()

print("\nFALSE POSITIVE ANALYSIS")
print("-----------------------")

print("False positive hours:", len(false_positives))

print("\nFalse positives by well:")
print(
    false_positives["well_id"]
    .value_counts()
    .to_string()
)

print("\nReconstruction error statistics:")
print(
    false_positives["reconstruction_error"]
    .describe()
)


anomalies = test_df[
    test_df["anomaly_label"] == 1
].copy()

type_results = (
    anomalies.groupby("anomaly_type")
    .agg(
        total_hours=("prediction", "size"),
        detected_hours=("prediction", "sum")
    )
    .reset_index()
)

type_results["recall"] = (
    type_results["detected_hours"]
    / type_results["total_hours"]
)

print("\nRECALL BY ANOMALY TYPE")
print("----------------------")
print(type_results.to_string(index=False))



anomalies = test_df[
    test_df["anomaly_label"] == 1
].copy()

type_results = (
    anomalies.groupby("anomaly_type")
    .agg(
        total_hours=("prediction", "size"),
        detected_hours=("prediction", "sum")
    )
    .reset_index()
)

type_results["recall"] = (
    type_results["detected_hours"]
    / type_results["total_hours"]
)

print("\nRECALL BY ANOMALY TYPE")
print("----------------------")
print(type_results.to_string(index=False))


val_df = pd.read_csv(
    DATA_DIR / "val_model.csv"
)

X_val = val_df[
    feature_columns
].to_numpy(dtype=np.float32)

with torch.no_grad():
    val_reconstructed = model(
        torch.tensor(X_val)
    )

val_errors = torch.mean(
    (
        torch.tensor(X_val) -
        val_reconstructed
    ) ** 2,
    dim=1
).numpy()

val_normal_errors = val_errors[
    val_df["anomaly_label"].to_numpy() == 0
]

test_normal_errors = test_errors[
    y_test == 0
]

print("\nNORMAL RECONSTRUCTION ERRORS")
print("----------------------------")

print(
    "Validation median:",
    np.median(val_normal_errors)
)

print(
    "Test median:",
    np.median(test_normal_errors)
)

print(
    "Validation 95th percentile:",
    np.percentile(val_normal_errors, 95)
)

print(
    "Test 95th percentile:",
    np.percentile(test_normal_errors, 95)
)


import json

results = {
    "model": "PyTorchAutoencoder",
    "threshold": THRESHOLD,
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_roc_auc": float(roc_auc),
    "test_average_precision": float(average_precision),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_negatives": int(tn),
    "performance_by_anomaly_type": (
        type_results.to_dict(orient="records")
    )
}

with open(
    MODEL_DIR / "autoencoder_test_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

# Save predictions for later analysis
test_df.to_csv(
    DATA_DIR / "autoencoder_test_predictions.csv",
    index=False
)

print("\nFinal test evaluation saved!")
