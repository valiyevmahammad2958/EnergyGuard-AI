
import pandas as pd

from anomaly_injection import inject_sensor_flatline

df = pd.read_csv(
    "data/raw/wells_normal.csv",
    parse_dates=["timestamp"],
)

df_faulty = inject_sensor_flatline(
    df,
    well_id="WELL_007",
    start_time="2025-05-10 08:00:00",
    duration_hours=12,
    sensor_column="tubing_pressure",
)

affected = df_faulty[
    df_faulty["anomaly_label"] == 1
]

print(
    affected[
        ["timestamp", "tubing_pressure", "anomaly_type"]
    ].to_string(index=False)
)

assert len(affected) == 12

assert affected["tubing_pressure"].nunique() == 1

assert df["anomaly_label"].sum() == 0

print("\nSensor flatline test passed!")
