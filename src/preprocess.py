"""Dataset loading & preprocessing utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms

__all__ = ["get_dataloaders", "set_seed"]


# ---------------------------------------------------------------- utilities

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------- HF wrappers ---
class HFImageDataset(Dataset):
    def __init__(self, hf_ds, image_key: str = "image", label_key: str = "label",
                 img_size: int = 32, mean=None, std=None):
        self.ds = hf_ds
        self.img_key = image_key
        self.lbl_key = label_key
        mean = mean if mean else (0.5, 0.5, 0.5)
        std = std if std else (0.5, 0.5, 0.5)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = self.tf(item[self.img_key])
        label = int(item[self.lbl_key])
        return img, label


# ---------------------------------------------------------- loaders -------

def _load_synthetic(cfg) -> Tuple[Dataset, int]:
    n = cfg.get("num_samples", 1024)
    c = cfg.get("num_classes", 10)
    shape = cfg.get("input_shape", [3, 32, 32])
    x = torch.randn(n, *shape)
    y = torch.randint(0, c, (n,))
    return TensorDataset(x, y), c


def _load_cifar10_c(cfg):
    split = cfg.get("split", "train")
    hf_ds = load_dataset("robro/cifar10-c-parquet", split=split, trust_remote_code=False)
    dataset = HFImageDataset(hf_ds, img_size=32,
                             mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    return dataset, 10


def _load_cifar10_1(cfg):
    hf_ds = load_dataset("XiangPan/CIFAR10.1", split="test")
    dataset = HFImageDataset(hf_ds, img_size=32,
                             mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    return dataset, 10


def _load_tinyimagenet_c(cfg):
    hf_ds = load_dataset("randall-lab/tiny-imagenet-c", split="test", trust_remote_code=True)
    dataset = HFImageDataset(hf_ds, img_size=64,
                             mean=(0.4802, 0.4481, 0.3975),
                             std=(0.2302, 0.2265, 0.2262))
    return dataset, 200


_DATASET_TABLE = {
    "synthetic": _load_synthetic,
    "cifar10_c": _load_cifar10_c,
    "cifar10_1": _load_cifar10_1,
    "tinyimagenet_c": _load_tinyimagenet_c,
}


# ------------------------------------------------------ public interface ---

def _dataset_factory(cfg):
    name = cfg["name"].lower()
    if name not in _DATASET_TABLE:
        raise NotImplementedError(f"Unknown dataset: {name}")
    return _DATASET_TABLE[name](cfg)


def get_dataloaders(dataset_cfg: dict, train_cfg: dict):
    ds, num_classes = _dataset_factory(dataset_cfg)

    val_frac = train_cfg.get("val_fraction", 0.2)
    val_size = int(len(ds) * val_frac)
    train_size = len(ds) - val_size
    train_set, val_set = random_split(ds, [train_size, val_size])

    bs = train_cfg.get("batch_size", 64)
    nw = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    return train_loader, val_loader, num_classes