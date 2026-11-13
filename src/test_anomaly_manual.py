
import pandas as pd

from anomaly_injection import inject_pressure_loss


# Load normal well dataset
df = pd.read_csv(
    "data/raw/wells_normal.csv",
    parse_dates=["timestamp"],
)

# Inject one pressure-loss event
df_anomaly = inject_pressure_loss(
    df=df,
    well_id="WELL_003",
    start_time="2025-03-10 08:00:00",
    duration_hours=12,
    severity=0.25,
)

# Check the results
print("Original anomaly count:")
print(df["anomaly_label"].sum())

print("\nModified anomaly count:")
print(df_anomaly["anomaly_label"].sum())

print("\nAnomaly types:")
print(df_anomaly["anomaly_type"].value_counts())

# Show affected observations
affected = df_anomaly[
    df_anomaly["anomaly_label"] == 1
]

print("\nAffected observations:")
print(
    affected[
        [
            "well_id",
            "timestamp",
            "tubing_pressure",
            "anomaly_label",
            "anomaly_type",
        ]
    ].to_string(index=False)
)

# Confirm original data was not modified
assert df["anomaly_label"].sum() == 0

# Confirm exactly 12 observations were affected
assert df_anomaly["anomaly_label"].sum() == 12

print("\nPressure-loss injection test passed!")
