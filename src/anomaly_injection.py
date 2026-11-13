
import numpy as np
import pandas as pd


def inject_pressure_loss(
    df,
    well_id,
    start_time,
    duration_hours=12,
    severity=0.25,
):
    """
    Inject a pressure-loss event into one well.

    severity=0.25 means a 25% pressure reduction.
    """

    df = df.copy()

    start_time = pd.Timestamp(start_time)

    # Match timezone with dataset timestamps
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    end_time = start_time + pd.Timedelta(
        hours=duration_hours
    )

    # Select observations affected by the anomaly
    mask = (
        (df["well_id"] == well_id)
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] < end_time)
    )

    # Prevent overlapping anomaly events
    if (df.loc[mask, "anomaly_label"] != 0).any():
        raise ValueError("Anomaly event overlaps an existing event.")

    if not mask.any():
        raise ValueError("No observations found for the event.")

    # Reduce tubing pressure
    df.loc[mask, "tubing_pressure"] *= (
        1 - severity
    )

    # Label anomalous observations
    df.loc[mask, "anomaly_label"] = 1
    df.loc[mask, "anomaly_type"] = "pressure_loss"

    return df

def inject_production_disruption(
    df,
    well_id,
    start_time,
    duration_hours=12,
    severity=0.60,
):
    """
    Simulate a production disruption.

    severity=0.60 means a 60% reduction in production.
    """

    # Validate inputs
    if not 0 < severity < 1:
        raise ValueError(
            "Severity must be between 0 and 1."
        )

    if duration_hours <= 0:
        raise ValueError(
            "Duration must be positive."
        )

    # Preserve the original DataFrame
    df = df.copy()

    # Convert event start to UTC
    start_time = pd.Timestamp(start_time)

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    end_time = start_time + pd.Timedelta(
        hours=duration_hours
    )

    # Select affected observations
    mask = (
        (df["well_id"] == well_id)
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] < end_time)
    )

    if not mask.any():
        raise ValueError(
            "No observations found for the event."
        )

    # Prevent overlapping anomalies
    if (df.loc[mask, "anomaly_label"] != 0).any():
        raise ValueError(
            "Anomaly event overlaps an existing event."
        )

    # Reduce all three production rates
    production_columns = [
        "oil_rate",
        "gas_rate",
        "water_rate",
    ]

    df.loc[mask, production_columns] *= (
        1 - severity
    )

    # Label affected observations
    df.loc[mask, "anomaly_label"] = 1

    df.loc[mask, "anomaly_type"] = (
        "production_disruption"
    )

    return df



def inject_sensor_flatline(
    df,
    well_id,
    start_time,
    duration_hours=12,
    sensor_column="tubing_pressure",
):
    """Simulate a sensor stuck at a constant reading."""

    allowed_sensors = [
        "tubing_pressure",
        "casing_pressure",
        "temperature",
    ]

    if sensor_column not in allowed_sensors:
        raise ValueError("Unsupported sensor column.")

    if duration_hours <= 0:
        raise ValueError("Duration must be positive.")

    df = df.copy()

    start_time = pd.Timestamp(start_time)

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    end_time = start_time + pd.Timedelta(
        hours=duration_hours
    )

    mask = (
        (df["well_id"] == well_id)
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] < end_time)
    )

    if mask.sum() != duration_hours:
        raise ValueError(
            "Event must contain exactly the requested "
            "number of hourly observations."
        )

    if (df.loc[mask, "anomaly_label"] != 0).any():
        raise ValueError("Overlapping anomaly detected.")

    # Get the first sensor value in the interval
    stuck_value = df.loc[mask, sensor_column].iloc[0]

    # Repeat the same value throughout the event
    df.loc[mask, sensor_column] = stuck_value

    df.loc[mask, "anomaly_label"] = 1
    df.loc[mask, "anomaly_type"] = "sensor_malfunction"

    return df


def inject_sensor_spike(
    df,
    well_id,
    start_time,
    duration_hours=1,
    sensor_column="tubing_pressure",
    severity=0.50,
):
    """Simulate abnormally high sensor readings."""

    allowed_sensors = [
        "tubing_pressure",
        "casing_pressure",
        "temperature",
    ]

    if sensor_column not in allowed_sensors:
        raise ValueError("Unsupported sensor column.")

    if duration_hours <= 0:
        raise ValueError("Duration must be positive.")

    if severity <= 0:
        raise ValueError("Severity must be positive.")

    df = df.copy()

    start_time = pd.Timestamp(start_time)

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    end_time = start_time + pd.Timedelta(
        hours=duration_hours
    )

    mask = (
        (df["well_id"] == well_id)
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] < end_time)
    )

    if mask.sum() != duration_hours:
        raise ValueError(
            "Event duration does not match available data."
        )

    if (df.loc[mask, "anomaly_label"] != 0).any():
        raise ValueError("Overlapping anomaly detected.")

    # Increase sensor readings by the severity factor
    df.loc[mask, sensor_column] *= (1 + severity)

    # Label affected observations
    df.loc[mask, "anomaly_label"] = 1
    df.loc[mask, "anomaly_type"] = "sensor_malfunction"

    return df
