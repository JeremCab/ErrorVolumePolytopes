# ==================================================================== #
# Save correctly classified subset (Fashion-MNIST)                     #
# run: python -m src.models.evaluate_fashionMNIST --model <model_type> #
# ==================================================================== #

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import argparse

from src.models.networks import FashionMLP_Large, FashionCNN_Small

# ------------------------- #
# Argument parsing          #
# ------------------------- #
parser = argparse.ArgumentParser(description="Evaluate model and save correct subset")
parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp",
                    help="Model type: mlp or cnn")
args = parser.parse_args()

print(f"Using model: {args.model}")

# ------------ #
# Device setup #
# ------------ #
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------- #
# Data preparation #
# ---------------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=False)

# ---------- #
# Load model #
# ---------- #
checkpoint_path = f"./checkpoints/fashion_{args.model}_best.pth"

if args.model == "mlp":
    model = FashionMLP_Large().to(device)
elif args.model == "cnn":
    model = FashionCNN_Small().to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print(f"Loaded model from {checkpoint_path}")

# --------------------------------- #
# Find correctly classified indices #
# --------------------------------- #
correct_indices = []

with torch.no_grad():
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        preds = out.argmax(dim=1)

        batch_correct = (preds == y_batch).cpu()

        start_idx = batch_idx * train_loader.batch_size

        for i, is_correct in enumerate(batch_correct):
            if is_correct:
                correct_indices.append(start_idx + i)

accuracy = len(correct_indices) / len(train_dataset)
print(f"Train accuracy: {accuracy*100:.2f}%")

# ---------------------- #
# Create subset and save #
# ---------------------- #
correct_subset = Subset(train_dataset, correct_indices)

save_path = f"./data/fashionMNIST_correct_{args.model}.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(correct_subset, save_path)

print(f"Saved correctly classified subset to {save_path}")