
import pandas as pd
from pathlib import Path

from feature_engineering import create_features

project_root = Path(__file__).resolve().parent.parent

# Load training dataset
train_df = pd.read_csv(
    project_root / "data/processed/train.csv",
    parse_dates=["timestamp"]
)

# Generate features
train_features = create_features(train_df)

print("Original shape:", train_df.shape)
print("New shape:", train_features.shape)

# Display one well
well_data = train_features[
    train_features["well_id"] == "WELL_001"
]

columns = [
    "timestamp",
    "tubing_pressure",
    "pressure_diff",
    "pressure_rolling_mean",
    "pressure_rolling_std",
]

print("\nFirst 10 observations:")
print(well_data[columns].head(10))

# Validation
assert train_features.shape[1] == train_df.shape[1] + 6
assert train_features.isna().sum().sum() == 0

print("\nFeature engineering checks passed!")
