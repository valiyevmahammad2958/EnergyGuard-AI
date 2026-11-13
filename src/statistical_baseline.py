
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


# Load validation dataset
val_df = pd.read_csv(
    DATA_DIR / "val_model.csv"
)

# Load feature information
preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.joblib"
)

feature_columns = preprocessor["feature_columns"]


# Extract standardized features
X_val = val_df[feature_columns].to_numpy()

# Calculate anomaly score
anomaly_scores = np.max(
    np.abs(X_val),
    axis=1
)

print("Validation observations:", len(X_val))
print("Number of features:", X_val.shape[1])

print("\nAnomaly score statistics:")
print(pd.Series(anomaly_scores).describe())


# Initial threshold
threshold = 3.0

# Predict anomalies
predictions = (
    anomaly_scores > threshold
).astype(int)

print("\nPredicted anomalies:", predictions.sum())
print("Predicted normal:", len(predictions) - predictions.sum())


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# True labels
y_val = val_df["anomaly_label"].to_numpy()

# Calculate metrics
precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

f1 = f1_score(
    y_val, predictions, zero_division=0
)

cm = confusion_matrix(y_val, predictions)

print("\nBaseline Model Performance")
print("--------------------------")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

print("\nConfusion matrix:")
print(cm)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Anomaly"])
ax.set_yticklabels(["Normal", "Anomaly"])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Statistical Baseline Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black"
        )

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()



# Compare multiple thresholds
thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

results = []

for threshold in thresholds:

    predictions = (
        anomaly_scores > threshold
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
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": predictions.sum()
    })

results_df = pd.DataFrame(results)

print("\nThreshold comparison:")
print(results_df.round(3).to_string(index=False))


# Find threshold with highest validation F1
best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])
best_f1 = float(best_row["f1"])

print("\nBest threshold:", best_threshold)
print("Best validation F1:", round(best_f1, 3))


plt.figure(figsize=(10, 5))

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1"],
    marker="o",
    label="F1-score"
)

plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label=f"Best threshold = {best_threshold}"
)

plt.title("Threshold Optimization — Validation Data")
plt.xlabel("Anomaly Score Threshold")
plt.ylabel("Metric Value")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

roc_auc = roc_auc_score(
    y_val,
    anomaly_scores
)

pr_auc = average_precision_score(
    y_val,
    anomaly_scores
)

print("\nThreshold-independent metrics:")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Average Precision: {pr_auc:.3f}")


import json

best_predictions = (
    anomaly_scores > best_threshold
).astype(int)

baseline_results = {
    "model": "max_absolute_zscore",
    "threshold": best_threshold,
    "validation_precision": float(
        precision_score(
            y_val, best_predictions,
            zero_division=0
        )
    ),
    "validation_recall": float(
        recall_score(
            y_val, best_predictions,
            zero_division=0
        )
    ),
    "validation_f1": float(
        f1_score(
            y_val, best_predictions,
            zero_division=0
        )
    ),
    "validation_roc_auc": float(roc_auc),
    "validation_average_precision": float(pr_auc)
}

output_path = (
    MODEL_DIR / "statistical_baseline_results.json"
)

with open(output_path, "w") as f:
    json.dump(
        baseline_results,
        f,
        indent=4
    )

print("\nBaseline results saved to:", output_path)
