
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load processed datasets
train = pd.read_csv(DATA_DIR / "train_model.csv")
val = pd.read_csv(DATA_DIR / "val_model.csv")
test = pd.read_csv(DATA_DIR / "test_model.csv")

# Load saved preprocessing information
preprocessor = joblib.load(
    PROJECT_ROOT / "models" / "preprocessor.joblib"
)

scaler = preprocessor["scaler"]
features = preprocessor["feature_columns"]


# Check dataset sizes
assert len(train) == 122640
assert len(val) == 26280
assert len(test) == 26280

# Check model input dimensions
assert len(features) == 13

# Check for missing or infinite feature values
for name, dataset in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    X = dataset[features].to_numpy()

    assert np.isfinite(X).all(), (
        f"{name} contains invalid feature values"
    )

    print(f"{name}: {X.shape} — valid")


# Verify fitted scaler has 13 features
assert scaler.n_features_in_ == 13

# Verify training features are centered near zero
train_means = train[features].mean()

assert np.allclose(
    train_means.to_numpy(),
    0,
    atol=1e-6
)

# Verify metadata is not used as model input
for forbidden in [
    "anomaly_label",
    "anomaly_type",
    "well_id",
    "timestamp",
    "split",
]:
    assert forbidden not in features


print("\nAll preprocessing checks passed!")
