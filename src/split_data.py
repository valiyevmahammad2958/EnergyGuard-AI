
import pandas as pd
from pathlib import Path

# Locate the project root
project_root = Path(__file__).resolve().parent.parent

# Load dataset
input_path = (
    project_root / "data" / "raw"
    / "wells_with_anomalies.csv"
)

df = pd.read_csv(
    input_path,
    parse_dates=["timestamp"]
)

# Sort chronologically
df = df.sort_values(
    ["timestamp", "well_id"]
).reset_index(drop=True)

# Get unique timestamps
timestamps = df["timestamp"].sort_values().unique()

# Define split boundaries
train_end_idx = int(len(timestamps) * 0.70)
val_end_idx = int(len(timestamps) * 0.85)

train_end_time = timestamps[train_end_idx]
val_end_time = timestamps[val_end_idx]

# Split by time
train_df = df[
    df["timestamp"] < train_end_time
].copy()

val_df = df[
    (df["timestamp"] >= train_end_time)
    & (df["timestamp"] < val_end_time)
].copy()

test_df = df[
    df["timestamp"] >= val_end_time
].copy()

# Save datasets
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

print("\nTime boundaries:")
print("Train:", train_df["timestamp"].min(),
      "to", train_df["timestamp"].max())
print("Val:", val_df["timestamp"].min(),
      "to", val_df["timestamp"].max())
print("Test:", test_df["timestamp"].min(),
      "to", test_df["timestamp"].max())



# Verify row counts
assert len(train_df) == 122640
assert len(val_df) == 26280
assert len(test_df) == 26280

# Verify chronological ordering
assert (
    train_df["timestamp"].max()
    < val_df["timestamp"].min()
)

assert (
    val_df["timestamp"].max()
    < test_df["timestamp"].min()
)

# Verify no observations were lost
assert (
    len(train_df) + len(val_df) + len(test_df)
    == len(df)
)

# Verify every split contains all 20 wells
for dataset in [train_df, val_df, test_df]:
    assert dataset["well_id"].nunique() == 20

print("\nAll chronological split checks passed!")
