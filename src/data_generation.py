
import numpy as np
import pandas as pd



def generate_well_data(
    well_id="WELL_001",
    start_date="2025-01-01",
    days=365,
    initial_oil_rate=1000.0,
    decline_rate=0.0015,
    water_cut=0.15,
    gor=600.0,
    initial_tubing_pressure=1800.0,
    initial_casing_pressure=2100.0,
    initial_temperature=145.0,
    choke_size=65.0,
    seed=42,
):
    """Generate normal hourly operating data for one well."""

    rng = np.random.default_rng(seed)

    n_hours = days * 24

    timestamps = pd.date_range(
        start=start_date,
        periods=n_hours,
        freq="h",
        tz="UTC",
    )

    time_days = np.arange(n_hours) / 24.0

    # Exponential oil decline
    oil_trend = initial_oil_rate * np.exp(
        -decline_rate * time_days
    )

    oil_noise = rng.normal(0, 0.02, n_hours)

    oil_rate = oil_trend * (1 + oil_noise)
    oil_rate = np.clip(oil_rate, 0, None)

    # Gas and water rates
    gas_rate = oil_rate * gor / 1000.0

    water_rate = oil_rate * water_cut / (1 - water_cut)

    # Pressure trends with measurement noise
    tubing_pressure = (
        initial_tubing_pressure
        - 0.4 * time_days
        + rng.normal(0, 8, n_hours)
    )

    casing_pressure = (
        initial_casing_pressure
        - 0.2 * time_days
        + rng.normal(0, 6, n_hours)
    )

    # Temperature
    temperature = (
        initial_temperature
        - 0.01 * time_days
        + rng.normal(0, 1, n_hours)
    )

    # Choke opening
    choke = np.full(n_hours, choke_size)

    df = pd.DataFrame(
        {
            "well_id": well_id,
            "timestamp": timestamps,
            "oil_rate": oil_rate,
            "gas_rate": gas_rate,
            "water_rate": water_rate,
            "tubing_pressure": tubing_pressure,
            "casing_pressure": casing_pressure,
            "temperature": temperature,
            "choke_size": choke,
            "anomaly_label": 0,
            "anomaly_type": "normal",
        }
    )

    return df


def generate_multiple_wells(n_wells=20, seed=42):
    """Generate a combined dataset for multiple wells."""

    rng = np.random.default_rng(seed)

    all_wells = []

    for i in range(n_wells):

        well_id = f"WELL_{i + 1:03d}"

        # Randomize well characteristics
        initial_oil_rate = rng.uniform(500, 1500)

        decline_rate = rng.uniform(0.0005, 0.0025)

        water_cut = rng.uniform(0.05, 0.40)

        gor = rng.uniform(300, 1000)

        initial_tubing_pressure = rng.uniform(
            1400, 2400
        )

        initial_casing_pressure = (
            initial_tubing_pressure
            + rng.uniform(150, 400)
        )

        initial_temperature = rng.uniform(120, 180)

        choke_size = rng.uniform(40, 90)

        # Generate data for one well
        well_df = generate_well_data(
            well_id=well_id,
            initial_oil_rate=initial_oil_rate,
            decline_rate=decline_rate,
            water_cut=water_cut,
            gor=gor,
            initial_tubing_pressure=initial_tubing_pressure,
            initial_casing_pressure=initial_casing_pressure,
            initial_temperature=initial_temperature,
            choke_size=choke_size,
            seed=seed + i,
        )

        all_wells.append(well_df)

    # Combine all wells into one DataFrame
    combined_df = pd.concat(
        all_wells,
        ignore_index=True,
    )

    return combined_df


if __name__ == "__main__":

    df = generate_multiple_wells(
        n_wells=20,
        seed=42,
    )

    print("Dataset shape:", df.shape)

    print("Number of wells:", df["well_id"].nunique())

    print("\nObservations per well:")
    print(df.groupby("well_id").size())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Save combined dataset
    df.to_csv(
        "data/raw/wells_normal.csv",
        index=False,
    )

    print("\nDataset saved successfully!")
