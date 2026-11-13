
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load datasets
train = pd.read_csv(DATA_DIR / "train_model.csv")
val = pd.read_csv(DATA_DIR / "val_model.csv")
test = pd.read_csv(DATA_DIR / "test_model.csv")

print("\nDATASET AUDIT")
print("=" * 50)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:
    print(f"\n{name}")
    print("Total rows:", len(df))
    print("Normal hours:", (df["anomaly_label"] == 0).sum())
    print("Anomalous hours:", (df["anomaly_label"] == 1).sum())
    print("Unique wells:", df["well_id"].nunique())
    print("First timestamp:", df["timestamp"].min())
    print("Last timestamp:", df["timestamp"].max())



from sklearn.metrics import confusion_matrix

# Load saved test predictions
predictions_path = (
    DATA_DIR / "autoencoder_test_predictions.csv"
)

predictions = pd.read_csv(predictions_path)

print("\nSAVED PREDICTIONS AUDIT")
print("=" * 50)

print("Prediction rows:", len(predictions))
print("Columns:", predictions.columns.tolist())

# Verify the predictions match the test dataset
assert len(predictions) == len(test)

assert (
    predictions["anomaly_label"].to_numpy()
    == test["anomaly_label"].to_numpy()
).all()

# Calculate exact confusion matrix
y_true = predictions["anomaly_label"]
y_pred = predictions["prediction"]

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

print("\nEXACT CONFUSION MATRIX")
print("=" * 50)

print("True positives (TP):", tp)
print("False positives (FP):", fp)
print("False negatives (FN):", fn)
print("True negatives (TN):", tn)

print("\nTotal observations:", tp + fp + fn + tn)
