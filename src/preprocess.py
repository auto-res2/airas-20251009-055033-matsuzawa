"""Common data pipeline – now with real datasets for the experiments.

Supported dataset names (cfg["name"]):
1. synthetic_classification    – small Gaussian blobs for smoke tests.
2. tinyimagenet_c              – HugginFace randall-lab/tiny-imagenet-c.
3. speech_commands             – Google Speech Commands v2.

Each loader returns (torch.utils.data.Dataset, num_classes).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset, Audio
from torchvision import transforms
import torchaudio

__all__ = ["get_dataloaders", "set_seed"]


# ------------------------------------------------------------------ utils ----

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- datasets --

def _load_synthetic(cfg: dict):
    num_samples: int = cfg.get("num_samples", 1024)
    num_classes: int = cfg.get("num_classes", 10)
    input_shape = cfg.get("input_shape", [3, 32, 32])
    data = torch.randn(num_samples, *input_shape)
    targets = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(data, targets)
    return dataset, num_classes


# -----------------------------------------------------------------------------
class _VisionHFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[int(idx)]
        img = self.transform(sample["image"])
        label = int(sample["label"])
        return img, label


def _load_tinyimagenet_c(cfg: dict):
    split = cfg.get("split", "test")
    hf_ds = load_dataset("randall-lab/tiny-imagenet-c", split=split, trust_remote_code=True)

    # Down-scale to 64×64 as required by MCU budget
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # converts to [0,1]
        ]
    )
    dataset = _VisionHFDataset(hf_ds, transform)
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes


# -----------------------------------------------------------------------------
class _SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, n_mels: int = 40, max_frames: int = 128):
        self.ds = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels, n_fft=400, hop_length=160
        )
        self.max_frames = max_frames

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        wav = torch.tensor(item["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        spec = self.mel(wav).log()  # (1, n_mels, T)
        spec = spec.squeeze(0)  # (n_mels, T)
        # Pad / truncate to fixed length for batching simplicity
        if spec.size(1) < self.max_frames:
            pad = self.max_frames - spec.size(1)
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, : self.max_frames]
        # Normalise per-sample
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        label = int(item["label"])
        return spec, label


def _load_speech_commands(cfg: dict):
    split = cfg.get("split", "test")
    hf_ds = load_dataset("speech_commands", "v0.02", split=split)
    dataset = _SpeechCommandsDataset(hf_ds)
    num_classes = hf_ds.features["label"].num_classes
    return dataset, num_classes


# -----------------------------------------------------------------------------
_DATASET_FACTORY = {
    "synthetic_classification": _load_synthetic,
    "tinyimagenet_c": _load_tinyimagenet_c,
    "speech_commands": _load_speech_commands,
}


def _get_dataset(cfg: dict):
    name = cfg["name"].lower()
    if name not in _DATASET_FACTORY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_DATASET_FACTORY)}")
    dataset, num_classes = _DATASET_FACTORY[name](cfg)

    # Optional sub-sampling for quick experiments
    max_samples = cfg.get("max_samples")
    if max_samples is not None and len(dataset) > max_samples:
        indices = np.random.permutation(len(dataset))[: max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    return dataset, num_classes


# --------------------------------------------------------------- dataloaders --

def get_dataloaders(dataset_cfg: dict, training_cfg: dict):
    dataset, num_classes = _get_dataset(dataset_cfg)

    val_fraction = training_cfg.get("val_fraction", 0.2)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = training_cfg.get("batch_size", 32)
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes