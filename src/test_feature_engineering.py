
import pandas as pd

from feature_engineering import create_features

# Three consecutive observations from one well
df = pd.DataFrame({
    "well_id": ["WELL_001"] * 3,
    "timestamp": pd.date_range(
        "2025-01-01", periods=3, freq="h"
    ),
    "oil_rate": [100, 110, 105],
    "gas_rate": [200, 210, 205],
    "tubing_pressure": [3000, 2950, 2900],
})

result = create_features(df)

print(result[[
    "oil_rate_diff",
    "pressure_diff",
    "gas_rate_diff",
    "pressure_rolling_mean",
    "pressure_rolling_std",
    "oil_rolling_mean",
]])
