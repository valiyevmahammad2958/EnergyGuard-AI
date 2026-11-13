
import pandas as pd


def create_features(df):
    """
    Create time-series features for each well.

    Uses only current and past measurements.
    """

    df = df.copy()

    # Sort chronologically within each well
    df = df.sort_values(
        ["well_id", "timestamp"]
    )

    # Group observations by well
    grouped = df.groupby("well_id")

    # 1. Hour-to-hour changes
    df["oil_rate_diff"] = grouped["oil_rate"].diff()

    df["pressure_diff"] = grouped[
        "tubing_pressure"
    ].diff()

    df["gas_rate_diff"] = grouped[
        "gas_rate"
    ].diff()

    # 2. Rolling statistics (6 hours)
    df["pressure_rolling_mean"] = (
        grouped["tubing_pressure"]
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=1
            ).mean()
        )
    )

    df["pressure_rolling_std"] = (
        grouped["tubing_pressure"]
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=2
            ).std()
        )
    )

    df["oil_rolling_mean"] = (
        grouped["oil_rate"]
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=1
            ).mean()
        )
    )

    # First observations have no previous hour
    # or sufficient history for std.
    # Use 0 as an initial placeholder.
    new_features = [
        "oil_rate_diff",
        "pressure_diff",
        "gas_rate_diff",
        "pressure_rolling_mean",
        "pressure_rolling_std",
        "oil_rolling_mean",
    ]

    df[new_features] = df[new_features].fillna(0)

    return df
