
from pathlib import Path

import torch

from model import Autoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder.pt"

# Load saved checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

# Create model using saved input dimension
model = Autoencoder(
    input_dim=checkpoint["input_dim"]
)

# Load the trained weights
model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.eval()

print("Trained model loaded successfully!")
print("Input features:", checkpoint["input_dim"])
print("Training epochs:", checkpoint["num_epochs"])

# Test inference with random input
X = torch.randn(2, checkpoint["input_dim"])

with torch.no_grad():
    output = model(X)

print("Output shape:", output.shape)
