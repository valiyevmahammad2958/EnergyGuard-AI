
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


# Load datasets
train_df = pd.read_csv(
    DATA_DIR / "train_model.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "val_model.csv"
)


# Load feature names
preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.joblib"
)

feature_columns = preprocessor["feature_columns"]


# Extract model inputs
X_train = train_df[feature_columns].to_numpy()
X_val = val_df[feature_columns].to_numpy()

# Validation labels for later evaluation
y_val = val_df["anomaly_label"].to_numpy()


print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Number of features:", len(feature_columns))



# Create Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train)

print("\nIsolation Forest training completed!")


# Calculate anomaly scores
train_scores = -model.score_samples(X_train)

val_scores = -model.score_samples(X_val)

print("\nTraining anomaly scores:")
print(pd.Series(train_scores).describe())

print("\nValidation anomaly scores:")
print(pd.Series(val_scores).describe())

# Verify scores
assert len(train_scores) == len(X_train)
assert len(val_scores) == len(X_val)

assert np.isfinite(train_scores).all()
assert np.isfinite(val_scores).all()

print("\nAll score checks passed!")



import matplotlib.pyplot as plt

normal_scores = val_scores[y_val == 0]
anomaly_scores = val_scores[y_val == 1]

plt.figure(figsize=(10, 5))

plt.hist(
    normal_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Normal"
)

plt.hist(
    anomaly_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Anomaly"
)

plt.xlabel("Isolation Forest Anomaly Score")
plt.ylabel("Density")
plt.title("Validation Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Threshold-independent metrics
roc_auc = roc_auc_score(
    y_val,
    val_scores
)

average_precision = average_precision_score(
    y_val,
    val_scores
)

print("\nIsolation Forest Ranking Performance")
print("------------------------------------")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")


# Candidate threshold percentiles
percentiles = np.arange(70, 100, 1)

thresholds = np.percentile(
    val_scores,
    percentiles
)

results = []

for threshold in thresholds:

    predictions = (
        val_scores > threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": int(predictions.sum())
    })

results_df = pd.DataFrame(results)

print("\nThreshold Optimization Results:")
print(results_df.round(4).to_string(index=False))



# Find the threshold with highest F1-score
best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

best_precision = float(best_row["precision"])
best_recall = float(best_row["recall"])
best_f1 = float(best_row["f1"])

print("\nBest Isolation Forest Threshold")
print("-------------------------------")
print(f"Threshold: {best_threshold:.4f}")
print(f"Precision: {best_precision:.3f}")
print(f"Recall:    {best_recall:.3f}")
print(f"F1-score:  {best_f1:.3f}")


# Predictions at the selected threshold
best_predictions = (
    val_scores > best_threshold
).astype(int)

# Confusion matrix
cm = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix")
print("----------------")
print(cm)

print(f"\nTrue negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")


comparison = pd.DataFrame({
    "Model": [
        "Statistical Z-score",
        "Isolation Forest"
    ],
    "Precision": [
        0.234,
        best_precision
    ],
    "Recall": [
        0.557,
        best_recall
    ],
    "F1": [
        0.330,
        best_f1
    ],
    "ROC-AUC": [
        0.792,
        roc_auc
    ],
    "Average Precision": [
        0.266,
        average_precision
    ]
})

print("\nModel Comparison")
print("----------------")
print(
    comparison.round(3).to_string(index=False)
)


import json

# Save trained Isolation Forest model
joblib.dump(
    model,
    MODEL_DIR / "isolation_forest.joblib"
)

# Save selected threshold and metrics
results_summary = {
    "model": "IsolationForest",
    "n_estimators": 100,
    "max_samples": 256,
    "threshold": best_threshold,
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "validation_roc_auc": float(roc_auc),
    "validation_average_precision": float(
        average_precision
    ),
    "feature_columns": feature_columns
}

with open(
    MODEL_DIR / "isolation_forest_results.json",
    "w"
) as f:
    json.dump(results_summary, f, indent=4)

print("\nModel and results saved successfully!")
