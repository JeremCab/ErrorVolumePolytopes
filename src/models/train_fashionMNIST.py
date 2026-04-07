# ======================================================================== #
# Train model on Fashion-MNIST dataset                                     #
# run script: python -m src.models.train_fashionMNIST --model <model_type> #
# ======================================================================== #

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

from src.models.networks import FashionMLP_Large, FashionCNN_Small

# ---------------- #
# Argument parsing #
# ---------------- #
parser = argparse.ArgumentParser(description="Train model on Fashion-MNIST")
parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp",
                    help="Model type: mlp or cnn")
args = parser.parse_args()

print(f"Using model: {args.model}")

# ---------------- #
# Data preparation #
# ---------------- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Full training dataset
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split training into train + validation
val_size = 5000
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ------------------------- #
# Device setup              #
# ------------------------- #
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ------------------------- #
# Model, loss, optimizer    #
# ------------------------- #
if args.model == "mlp":
    model = FashionMLP_Large().to(device)
elif args.model == "cnn":
    model = FashionCNN_Small().to(device)
    
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)   # for MLP
optimizer = optim.Adam(model.parameters(), lr=5e-3)     # for CNN
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ------------------------- #
# Training setup            #
# ------------------------- #
num_epochs = 100
best_val_acc = 0.0

# Directory to save model
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(
    checkpoint_dir,
    f"fashion_{args.model}_best.pth"
    )

# ------------------------- #
# Training loop             #
# ------------------------- #
for epoch in range(1, num_epochs + 1):
    # Train
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    scheduler.step()
    train_loss = total_loss / len(train_loader.dataset)

    # Evaluate on validation set
    model.eval()
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()

    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Best model saved with val accuracy {best_val_acc*100:.2f}%")

# ------------------------- #
# Final evaluation on test  #
# ------------------------- #
model.load_state_dict(torch.load(best_model_path))
model.eval()
correct = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        preds = out.argmax(dim=1)
        correct += (preds == y_batch).sum().item()

test_acc = correct / len(test_loader.dataset)
print(f"\nFinal test accuracy: {test_acc*100:.2f}%")
print(f"Best model saved to {best_model_path}")
