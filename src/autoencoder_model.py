
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load datasets
train_df = pd.read_csv(
    DATA_DIR / "train_model.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "val_model.csv"
)


# Load feature names
preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.joblib"
)

feature_columns = preprocessor["feature_columns"]


# Select normal training observations
normal_train = train_df[
    train_df["anomaly_label"] == 0
].copy()


# Extract input features
X_train = normal_train[
    feature_columns
].to_numpy(dtype=np.float32)

X_val = val_df[
    feature_columns
].to_numpy(dtype=np.float32)

y_val = val_df[
    "anomaly_label"
].to_numpy()


print("Normal training observations:", len(X_train))
print("Validation observations:", len(X_val))
print("Number of features:", X_train.shape[1])



class Autoencoder(nn.Module):

    def __init__(self, input_dim=13):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed


# Create model
model = Autoencoder(input_dim=13)

print("\nAutoencoder architecture:")
print(model)


# Convert training data to a PyTorch tensor
train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

# Create dataset
train_dataset = TensorDataset(train_tensor)

# Create batches
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

print("\nTraining batches:", len(train_loader))


# Reconstruction loss
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nLoss function and optimizer ready!")


# Take one batch
sample_batch = next(iter(train_loader))[0]

# Run one forward pass
with torch.no_grad():
    reconstructed = model(sample_batch)

print("\nInput batch shape:", sample_batch.shape)
print("Output batch shape:", reconstructed.shape)

assert sample_batch.shape == reconstructed.shape

print("\nAutoencoder architecture checks passed!")



# Number of training epochs
num_epochs = 30

# Store training loss
training_losses = []

# Training loop
for epoch in range(num_epochs):

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        # Extract input features
        X_batch = batch[0]

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(X_batch)

        # Calculate reconstruction loss
        loss = criterion(
            reconstructed,
            X_batch
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += (
            loss.item() * X_batch.size(0)
        )

    # Average loss across observations
    epoch_loss = total_loss / len(train_dataset)

    training_losses.append(epoch_loss)

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} "
        f"- Training Loss: {epoch_loss:.6f}"
    )


import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, num_epochs + 1),
    training_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Reconstruction Error")
plt.title("Autoencoder Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Switch to evaluation mode
model.eval()

# Convert validation features to tensor
val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

# Reconstruct validation observations
with torch.no_grad():
    val_reconstructed = model(val_tensor)

# Calculate MSE for each observation
val_errors = torch.mean(
    (val_tensor - val_reconstructed) ** 2,
    dim=1
).numpy()

print("\nValidation reconstruction errors:")
print(pd.Series(val_errors).describe())

assert len(val_errors) == len(X_val)
assert np.isfinite(val_errors).all()

print("\nValidation anomaly scores calculated!")


normal_errors = val_errors[y_val == 0]
anomaly_errors = val_errors[y_val == 1]

print("\nAverage reconstruction error:")

print(
    "Normal:",
    round(float(normal_errors.mean()), 6)
)

print(
    "Anomaly:",
    round(float(anomaly_errors.mean()), 6)
)


plt.figure(figsize=(10, 5))

plt.hist(
    normal_errors,
    bins=60,
    alpha=0.6,
    density=True,
    label="Normal"
)

plt.hist(
    anomaly_errors,
    bins=60,
    alpha=0.6,
    density=True,
    label="Anomaly"
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Autoencoder Validation Reconstruction Errors")

plt.legend()
plt.tight_layout()
plt.show()



# Save model weights and architecture metadata
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_columns),
        "feature_columns": feature_columns,
        "num_epochs": num_epochs,
        "training_losses": training_losses
    },
    MODEL_DIR / "autoencoder.pt"
)

print("\nAutoencoder saved successfully!")


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Calculate ranking metrics
roc_auc = roc_auc_score(
    y_val,
    val_errors
)

average_precision = average_precision_score(
    y_val,
    val_errors
)

print("\nAutoencoder Ranking Performance")
print("--------------------------------")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")


# Candidate percentiles
percentiles = np.linspace(
    50,
    99.9,
    201
)

# Convert percentiles into thresholds
thresholds = np.unique(
    np.percentile(
        val_errors,
        percentiles
    )
)

results = []

for threshold in thresholds:

    predictions = (
        val_errors > threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": int(predictions.sum())
    })

results_df = pd.DataFrame(results)

print("\nThreshold optimization completed!")
print("Thresholds evaluated:", len(results_df))


# Find best threshold
best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

best_precision = float(best_row["precision"])
best_recall = float(best_row["recall"])
best_f1 = float(best_row["f1"])

print("\nBest Autoencoder Results")
print("------------------------")
print(f"Threshold: {best_threshold:.6f}")
print(f"Precision: {best_precision:.3f}")
print(f"Recall:    {best_recall:.3f}")
print(f"F1-score:  {best_f1:.3f}")


# Generate final validation predictions
best_predictions = (
    val_errors > best_threshold
).astype(int)

# Confusion matrix
cm = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix")
print("----------------")
print(cm)

print(f"\nTrue negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")


comparison = pd.DataFrame({
    "Model": [
        "Z-score",
        "Isolation Forest",
        "Autoencoder"
    ],
    "Precision": [
        0.234,
        0.144,
        best_precision
    ],
    "Recall": [
        0.557,
        0.343,
        best_recall
    ],
    "F1": [
        0.330,
        0.203,
        best_f1
    ],
    "ROC-AUC": [
        0.792,
        0.740,
        roc_auc
    ],
    "Average Precision": [
        0.266,
        0.140,
        average_precision
    ]
})

print("\nModel Comparison")
print("----------------")
print(
    comparison.round(3).to_string(index=False)
)


plt.figure(figsize=(10, 5))

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1"],
    label="F1-score"
)

plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label=f"Best threshold = {best_threshold:.4f}"
)

plt.xlabel("Reconstruction Error Threshold")
plt.ylabel("Metric Value")
plt.title("Autoencoder Threshold Optimization")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


import json

results_summary = {
    "model": "PyTorchAutoencoder",
    "architecture": [13, 8, 4, 8, 13],
    "training_epochs": 30,
    "threshold": best_threshold,
    "validation_precision": best_precision,
    "validation_recall": best_recall,
    "validation_f1": best_f1,
    "validation_roc_auc": float(roc_auc),
    "validation_average_precision": float(
        average_precision
    ),
    "feature_columns": feature_columns
}

output_path = (
    MODEL_DIR / "autoencoder_results.json"
)

with open(output_path, "w") as f:
    json.dump(
        results_summary,
        f,
        indent=4
    )

print("\nAutoencoder results saved!")
print(output_path)
