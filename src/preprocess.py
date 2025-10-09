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


def _dataset_factory(cfg: dict):
    name = cfg["name"]
    if name == "SYNTHETIC_CLASSIFICATION_PLACEHOLDER":
        return _load_placeholder_dataset(cfg)
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