# src/preprocess.py
"""Dataset loading & preprocessing utilities."""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

CACHE_DIR = ".cache/datasets"

# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_transform_item(cfg: Dict):
    t_type = str(cfg["type"]).lower()
    if t_type == "resize":
        return T.Resize(cfg["size"], interpolation=T.InterpolationMode.BILINEAR)
    if t_type == "center_crop":
        return T.CenterCrop(cfg["size"])
    if t_type == "to_tensor":
        return T.ToTensor()
    if t_type == "normalize":
        mean = cfg.get("mean", 0.5)
        std = cfg.get("std", 0.5)
        # torchvision requires sequence – broadcast scalar if necessary
        if not isinstance(mean, (list, tuple)):
            mean = [mean] * 3
        if not isinstance(std, (list, tuple)):
            std = [std] * 3
        return T.Normalize(mean=mean, std=std)
    raise ValueError(f"Unsupported transform type: {t_type}")


def build_transforms(t_list: List[Dict]):
    return T.Compose([_build_transform_item(t) for t in t_list])


# ---------------------------------------------------------------------------
# Dataset / DataLoader builders
# ---------------------------------------------------------------------------

def build_dataset(cfg):
    name = str(cfg.name).lower()
    root_path = os.path.join(CACHE_DIR, name.replace("/", "_"))
    transform = build_transforms(cfg.transforms)

    if name == "cifar-10":
        train_flag = cfg.split.lower() in {"train", "training"}
        ds = datasets.CIFAR10(root=root_path, train=train_flag, download=True, transform=transform)
    elif name in {"imagenet64", "imagenet-64"}:
        img_root = os.environ.get("IMAGENET64_ROOT", os.path.join(CACHE_DIR, "imagenet64"))
        ds = datasets.ImageFolder(root=img_root, transform=transform)
    else:
        raise NotImplementedError(f"Dataset '{name}' not supported.")

    # Optional sub-sampling -------------------------------------------------
    if getattr(cfg, "subsample", None):
        random.seed(getattr(cfg, "seed", 42))
        indices = random.sample(range(len(ds)), cfg.subsample)
        ds = Subset(ds, indices)
    return ds


def build_dataloader(cfg, batch_size: int) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    ds = build_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 2),
        pin_memory=False,
    )
    return ds, loader