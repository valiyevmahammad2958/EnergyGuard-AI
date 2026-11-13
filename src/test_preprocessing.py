
from preprocessing import preprocess_features, FEATURE_COLUMNS

# Example engineered features
sample = {
    column: 0.0
    for column in FEATURE_COLUMNS
}

scaled = preprocess_features(sample)

print("Scaled array shape:", scaled.shape)
print("Scaled values:", scaled)
