
import numpy as np
import pandas as pd

from anomaly_injection import inject_sensor_spike

df = pd.read_csv(
    "data/raw/wells_normal.csv",
    parse_dates=["timestamp"],
)

df_spike = inject_sensor_spike(
    df,
    well_id="WELL_009",
    start_time="2025-06-15 10:00:00",
    duration_hours=1,
    sensor_column="tubing_pressure",
    severity=0.50,
)

# Select the affected observation
mask = df_spike["anomaly_label"] == 1

print(
    df_spike.loc[
        mask,
        ["well_id", "timestamp",
         "tubing_pressure", "anomaly_type"]
    ]
)

# Verify exactly one anomaly
assert mask.sum() == 1

# Verify the spike is 50% higher
assert np.allclose(
    df_spike.loc[mask, "tubing_pressure"],
    df.loc[mask, "tubing_pressure"] * 1.50,
)

# Verify original data is unchanged
assert df["anomaly_label"].sum() == 0

print("\nSensor spike test passed!")
