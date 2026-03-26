"""MNIST dataset helpers for the project.

Provides simple helpers to build torchvision MNIST datasets and PyTorch DataLoaders
using the standard MNIST normalization and ToTensor transform.
"""
from typing import Tuple, Optional
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_mnist_transform() -> transforms.Compose:
    """Return the common transform used for MNIST in this project.

    Matches the user's requested normalization values.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform


def load_mnist_datasets(data_root: Optional[str] = None, download: bool = True) -> Tuple[MNIST, MNIST]:
    """Load and return (train_dataset, test_dataset).

    Args:
        data_root: directory to store/download MNIST. If None, uses ./data in repo root.
        download: whether to download the dataset if missing.
    """
    if data_root is None:
        data_root = str(Path.cwd() / "data")
    os.makedirs(data_root, exist_ok=True)

    transform = get_mnist_transform()

    train_dataset = MNIST(root=data_root, train=True, download=download, transform=transform)
    test_dataset = MNIST(root=data_root, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


def make_mnist_dataloaders(batch_size: int = 128,
                           num_workers: int = 2,
                           val_split: float = 0.1,
                           data_root: Optional[str] = None,
                           download: bool = True,
                           shuffle_train: bool = True,
                           generator_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for MNIST.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset, test_dataset = load_mnist_datasets(data_root=data_root, download=download)

    # split train -> train + val
    total_train = len(train_dataset)
    val_size = int(total_train * val_split) if val_split and 0.0 < val_split < 1.0 else 0
    train_size = total_train - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(generator_seed)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
    else:
        train_subset = train_dataset
        val_subset = None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # quick sanity check when run as a script
    tl, vl, tst = make_mnist_dataloaders(batch_size=64, num_workers=0)
    print('Train batches:', len(tl))
    if vl is not None:
        print('Val batches:', len(vl))
    print('Test batches:', len(tst))
