
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data/raw/wells_with_anomalies.csv",
    parse_dates=["timestamp"],
)

events = pd.read_csv(
    ROOT / "data/raw/anomaly_events.csv",
    parse_dates=["start_time", "end_time"],
)

# Basic dataset checks
assert len(df) == 175200
assert df["well_id"].nunique() == 20
assert df["timestamp"].notna().all()
assert not df.duplicated(
    ["well_id", "timestamp"]
).any()

# Check anomaly target
assert df["anomaly_label"].sum() == 3504

# Check event duration totals
assert events["duration_hours"].sum() == 3504

# Check labels and anomaly categories
assert set(df["anomaly_label"].unique()) == {0, 1}

assert (
    df.loc[df["anomaly_label"] == 0, "anomaly_type"]
    == "normal"
).all()

assert (
    df.loc[df["anomaly_label"] == 1, "anomaly_type"]
    != "normal"
).all()

# Verify each event matches its hourly records
for event in events.itertuples(index=False):
    mask = (
        (df["well_id"] == event.well_id)
        & (df["timestamp"] >= event.start_time)
        & (df["timestamp"] < event.end_time)
    )

    selected = df.loc[mask]

    assert len(selected) == event.duration_hours
    assert (selected["anomaly_label"] == 1).all()
    assert (
        selected["anomaly_type"] == event.anomaly_type
    ).all()

# No missing values in ML feature columns
features = [
    "oil_rate",
    "gas_rate",
    "water_rate",
    "tubing_pressure",
    "casing_pressure",
    "temperature",
    "choke_size",
]

assert df[features].notna().all().all()
assert (df[features] >= 0).all().all()

print("All anomaly dataset validation checks passed!")
print("Total events:", len(events))
print("Anomaly percentage:",
      round(df["anomaly_label"].mean() * 100, 2))
