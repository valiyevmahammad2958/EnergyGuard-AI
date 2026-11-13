
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

# Load the fitted preprocessor once
preprocessor = joblib.load(PREPROCESSOR_PATH)

scaler = preprocessor["scaler"]
FEATURE_COLUMNS = preprocessor["feature_columns"]


def preprocess_features(features: dict) -> np.ndarray:
    """
    Convert engineered features into scaled model inputs.

    Parameters
    ----------
    features : dict
        Dictionary containing all 13 engineered features.

    Returns
    -------
    np.ndarray
        Scaled feature array with shape (1, 13).
    """

    # Check that all required features are present
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in features
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Preserve the exact training feature order
    values = np.array(
        [[features[col] for col in FEATURE_COLUMNS]],
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError("All features must be finite numbers")

    # Reuse the fitted training scaler
    scaled_values = scaler.transform(values)

    return scaled_values.astype(np.float32)
