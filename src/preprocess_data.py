
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import create_features


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load chronological splits
train = pd.read_csv(
    DATA_DIR / "train.csv",
    parse_dates=["timestamp"]
)

val = pd.read_csv(
    DATA_DIR / "val.csv",
    parse_dates=["timestamp"]
)

test = pd.read_csv(
    DATA_DIR / "test.csv",
    parse_dates=["timestamp"]
)


# 2. Add split identifiers
train["split"] = "train"
val["split"] = "val"
test["split"] = "test"


# 3. Combine datasets for historical context
combined = pd.concat(
    [train, val, test],
    ignore_index=True
)

# Validate chronological boundaries
assert train["timestamp"].max() < val["timestamp"].min()
assert val["timestamp"].max() < test["timestamp"].min()

assert not combined.duplicated(
    ["well_id", "timestamp"]
).any()


# 4. Generate causal time-series features
combined = create_features(combined)


# 5. Separate the datasets again
train_features = combined[
    combined["split"] == "train"
].copy()

val_features = combined[
    combined["split"] == "val"
].copy()

test_features = combined[
    combined["split"] == "test"
].copy()


# 6. Define model input features
RAW_FEATURES = [
    "oil_rate",
    "gas_rate",
    "water_rate",
    "tubing_pressure",
    "casing_pressure",
    "temperature",
    "choke_size",
]

ENGINEERED_FEATURES = [
    "oil_rate_diff",
    "pressure_diff",
    "gas_rate_diff",
    "pressure_rolling_mean",
    "pressure_rolling_std",
    "oil_rolling_mean",
]

FEATURE_COLUMNS = RAW_FEATURES + ENGINEERED_FEATURES


# 7. Fit scaler on training data only
scaler = StandardScaler()

scaler.fit(train_features[FEATURE_COLUMNS])


# 8. Transform all three splits
for dataset in [
    train_features,
    val_features,
    test_features,
]:
    dataset[FEATURE_COLUMNS] = scaler.transform(
        dataset[FEATURE_COLUMNS]
    )


# 9. Save processed datasets
train_features.to_csv(
    DATA_DIR / "train_model.csv",
    index=False
)

val_features.to_csv(
    DATA_DIR / "val_model.csv",
    index=False
)

test_features.to_csv(
    DATA_DIR / "test_model.csv",
    index=False
)


# 10. Save fitted scaler and feature names
joblib.dump(
    {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    },
    MODEL_DIR / "preprocessor.joblib"
)


# 11. Display results
print("Training:", train_features.shape)
print("Validation:", val_features.shape)
print("Testing:", test_features.shape)

print("\nNumber of model features:", len(FEATURE_COLUMNS))

print("\nSaved:")
print(DATA_DIR / "train_model.csv")
print(DATA_DIR / "val_model.csv")
print(DATA_DIR / "test_model.csv")
print(MODEL_DIR / "preprocessor.joblib")
