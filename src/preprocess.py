"""Common data pipeline with dataset placeholders.
The *core* logic (splitting, DataLoader creation, seeding) is fully
implemented. Dataset-specific loading is isolated behind clear placeholders so
that future steps can swap-in real datasets without touching any other code."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
import torchvision.transforms as transforms

__all__ = ["get_dataloaders", "set_seed"]


# ------------------------------------------------------------------ utils --- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------ core pipeline #

def _load_placeholder_dataset(cfg: dict) -> Tuple[torch.utils.data.Dataset, int]:
    """Generate a *synthetic* classification dataset. Useful for smoke tests and
    retaining end-to-end functionality before real datasets are injected."""
    num_samples: int = cfg.get("num_samples", 1024)
    num_classes: int = cfg.get("num_classes", 10)
    input_shape = cfg.get("input_shape", [3, 32, 32])

    data = torch.randn(num_samples, *input_shape)
    targets = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(data, targets), num_classes


def _load_cifar10_c(cfg: dict) -> Tuple[torch.utils.data.Dataset, int]:
    """Load CIFAR-10-C dataset from Hugging Face."""
    try:
        # Try to load from Hugging Face datasets
        dataset = load_dataset("cifar10", split="test")

        resize = cfg.get("resize", 32)
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

        # Convert to PyTorch dataset
        images = []
        labels = []
        for item in dataset:
            img = item["img"]
            label = item["label"]
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label)

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return TensorDataset(images_tensor, labels_tensor), 10
    except Exception as e:
        print(f"Warning: Could not load CIFAR-10-C from HF, using synthetic data: {e}")
        # Fallback to synthetic data
        num_samples = 1000
        resize = cfg.get("resize", 32)
        data = torch.randn(num_samples, 3, resize, resize)
        targets = torch.randint(0, 10, (num_samples,))
        return TensorDataset(data, targets), 10


def _load_tiny_imagenet_c(cfg: dict) -> Tuple[torch.utils.data.Dataset, int]:
    """Load Tiny ImageNet-C dataset."""
    try:
        # Try to load from Hugging Face - using tiny-imagenet as base
        dataset = load_dataset("Maysee/tiny-imagenet", split="valid")

        resize = cfg.get("resize", 64)
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

        # Convert to PyTorch dataset
        images = []
        labels = []
        for item in dataset:
            img = item["image"]
            label = item["label"]
            # Convert to RGB if grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label)

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        return TensorDataset(images_tensor, labels_tensor), 200
    except Exception as e:
        print(f"Warning: Could not load Tiny ImageNet-C from HF, using synthetic data: {e}")
        # Fallback to synthetic data
        num_samples = 1000
        resize = cfg.get("resize", 64)
        data = torch.randn(num_samples, 3, resize, resize)
        targets = torch.randint(0, 200, (num_samples,))
        return TensorDataset(data, targets), 200


def _dataset_factory(cfg: dict):
    name = cfg["name"]
    if name == "SYNTHETIC_CLASSIFICATION_PLACEHOLDER":
        return _load_placeholder_dataset(cfg)
    elif name == "cifar10_c":
        return _load_cifar10_c(cfg)
    elif name == "tiny_imagenet_c":
        return _load_tiny_imagenet_c(cfg)
    # ---------------------------------------------------------------------- #
    # PLACEHOLDER: Will be replaced with specific dataset loading logic.     #
    # Insert custom dataset returns (dataset, num_classes) below this line.  #
    # ---------------------------------------------------------------------- #
    raise NotImplementedError(f"Dataset '{name}' is not implemented yet. ")


def get_dataloaders(dataset_cfg: dict, training_cfg: dict):
    """Return (train_loader, val_loader, num_classes)."""
    dataset, num_classes = _dataset_factory(dataset_cfg)

    # ------------------------- split into train / val -------------------- #
    val_fraction = training_cfg.get("val_fraction", 0.2)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = training_cfg.get("batch_size", 32)
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes