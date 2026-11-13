
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_injection import (
    inject_pressure_loss,
    inject_production_disruption,
    inject_sensor_flatline,
    inject_sensor_spike,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

INPUT_PATH = RAW_DIR / "wells_normal.csv"
OUTPUT_PATH = RAW_DIR / "wells_with_anomalies.csv"
EVENT_LOG_PATH = RAW_DIR / "anomaly_events.csv"


def generate_anomaly_dataset(
    df,
    anomaly_fraction=0.02,
    seed=42,
):
    """Generate non-overlapping anomaly events."""

    rng = np.random.default_rng(seed)

    df = df.copy()
    df = df.sort_values(
        ["well_id", "timestamp"]
    ).reset_index(drop=True)

    target = round(len(df) * anomaly_fraction)
    injected = 0
    event_log = []
    event_id = 1

    # Our planned chronological split boundaries.
    # All three intervals are half-open: [start, end).
    boundaries = [
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-09-13 12:00", tz="UTC"),
        pd.Timestamp("2025-11-07 06:00", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
    ]

    anomaly_types = [
        "pressure_loss",
        "production_disruption",
        "sensor_flatline",
        "sensor_spike",
    ]

    wells = sorted(df["well_id"].unique())

    attempts = 0
    max_attempts = 100000

    while injected < target:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to reach anomaly target "
                "without overlapping events."
            )

        remaining = target - injected

        anomaly_type = rng.choice(anomaly_types)
        well_id = rng.choice(wells)

        # Select duration and severity
        if anomaly_type == "pressure_loss":
            duration = int(rng.integers(6, 49))
            severity = float(rng.uniform(0.15, 0.40))

        elif anomaly_type == "production_disruption":
            duration = int(rng.integers(6, 73))
            severity = float(rng.uniform(0.30, 0.70))

        elif anomaly_type == "sensor_flatline":
            duration = int(rng.integers(4, 25))
            severity = None

        else:
            duration = int(rng.integers(1, 4))
            severity = float(rng.uniform(0.30, 0.80))

        # Avoid exceeding the target.
        # If the remaining budget is too small for a
        # particular type, try another event.
        if duration > remaining:
            continue

        # Choose one chronological split
        split_index = int(rng.integers(0, 3))
        split_start = boundaries[split_index]
        split_end = boundaries[split_index + 1]

        # Select candidate rows for this well/split
        candidate = df[
            (df["well_id"] == well_id)
            & (df["timestamp"] >= split_start)
            & (df["timestamp"] < split_end)
        ]

        if len(candidate) < duration:
            continue

        # Pick a valid starting position
        start_position = int(
            rng.integers(0, len(candidate) - duration + 1)
        )

        selected = candidate.iloc[
            start_position:start_position + duration
        ]

        # Reject any overlap
        if (selected["anomaly_label"] != 0).any():
            continue

        start_time = selected["timestamp"].iloc[0]
        end_time = start_time + pd.Timedelta(
            hours=duration
        )

        # Inject the chosen anomaly
        if anomaly_type == "pressure_loss":
            df = inject_pressure_loss(
                df,
                well_id=well_id,
                start_time=start_time,
                duration_hours=duration,
                severity=severity,
            )

        elif anomaly_type == "production_disruption":
            df = inject_production_disruption(
                df,
                well_id=well_id,
                start_time=start_time,
                duration_hours=duration,
                severity=severity,
            )

        elif anomaly_type == "sensor_flatline":
            df = inject_sensor_flatline(
                df,
                well_id=well_id,
                start_time=start_time,
                duration_hours=duration,
                sensor_column="tubing_pressure",
            )

        else:
            df = inject_sensor_spike(
                df,
                well_id=well_id,
                start_time=start_time,
                duration_hours=duration,
                sensor_column="tubing_pressure",
                severity=severity,
            )

        # Record event metadata
        event_log.append({
            "event_id": f"EVT_{event_id:04d}",
            "well_id": well_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_hours": duration,
            "anomaly_type": (
                "sensor_malfunction"
                if anomaly_type.startswith("sensor_")
                else anomaly_type
            ),
            "injection_method": anomaly_type,
            "severity": severity,
            "split": ["train", "val", "test"][split_index],
        })

        injected += duration
        event_id += 1

    return df, pd.DataFrame(event_log)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
    )

    df_anomaly, events = generate_anomaly_dataset(
        df,
        anomaly_fraction=0.02,
        seed=42,
    )

    df_anomaly.to_csv(OUTPUT_PATH, index=False)
    events.to_csv(EVENT_LOG_PATH, index=False)

    print("Dataset shape:", df_anomaly.shape)
    print("Anomalous observations:",
          df_anomaly["anomaly_label"].sum())

    print("\nAnomaly types:")
    print(df_anomaly["anomaly_type"].value_counts())

    print("\nTotal events:", len(events))
    print("\nEvents by type:")
    print(events["injection_method"].value_counts())

    print("\nFiles saved successfully!")
