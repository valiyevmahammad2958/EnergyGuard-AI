
import pandas as pd

from anomaly_injection import inject_production_disruption


# Load the normal dataset
df = pd.read_csv(
    "data/raw/wells_normal.csv",
    parse_dates=["timestamp"],
)

# Inject a 60% production disruption
df_anomaly = inject_production_disruption(
    df=df,
    well_id="WELL_005",
    start_time="2025-04-15 08:00:00",
    duration_hours=24,
    severity=0.60,
)

# Count anomalous observations
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
            "oil_rate",
            "gas_rate",
            "water_rate",
            "anomaly_type",
        ]
    ].head(10).to_string(index=False)
)

# Verify anomaly count
assert df_anomaly["anomaly_label"].sum() == 24

# Verify original data is unchanged
assert df["anomaly_label"].sum() == 0

# Verify production rates were reduced by 60%
mask = df_anomaly["anomaly_label"] == 1

for col in ["oil_rate", "gas_rate", "water_rate"]:

    original_values = df.loc[mask, col].to_numpy()
    modified_values = df_anomaly.loc[mask, col].to_numpy()

    assert (
        abs(modified_values - original_values * 0.40)
        < 1e-8
    ).all()

print("\nProduction disruption test passed!")
