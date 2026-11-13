
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Saved object type:", type(preprocessor))

if isinstance(preprocessor, dict):
    print("\nSaved keys:", list(preprocessor.keys()))

    for key, value in preprocessor.items():
        print(f"\n{key}:")
        print("Type:", type(value))

        if key == "feature_columns":
            print("Value:", value)

        elif hasattr(value, "n_features_in_"):
            print("Number of input features:", value.n_features_in_)

        else:
            print("Value:", str(value)[:300])
else:
    if hasattr(preprocessor, "n_features_in_"):
        print("Number of input features:",
              preprocessor.n_features_in_)
