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
import os
import tarfile
import urllib.request
from PIL import Image

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


class _TinyImageNetCDataset(torch.utils.data.Dataset):
    """Custom dataset for Tiny-ImageNet-C that downloads directly from Zenodo."""

    LABELS = [
        "n02124075", "n04067472", "n04540053", "n04099969", "n07749582", "n01641577", "n02802426", "n09246464",
        "n07920052", "n03970156", "n03891332", "n02106662", "n03201208", "n02279972", "n02132136", "n04146614",
        "n07873807", "n02364673", "n04507155", "n03854065", "n03838899", "n03733131", "n01443537", "n07875152",
        "n03544143", "n09428293", "n03085013", "n02437312", "n07614500", "n03804744", "n04265275", "n02963159",
        "n02486410", "n01944390", "n09256479", "n02058221", "n04275548", "n02321529", "n02769748", "n02099712",
        "n07695742", "n02056570", "n02281406", "n01774750", "n02509815", "n03983396", "n07753592", "n04254777",
        "n02233338", "n04008634", "n02823428", "n02236044", "n03393912", "n07583066", "n04074963", "n01629819",
        "n09332890", "n02481823", "n03902125", "n03404251", "n09193705", "n03637318", "n04456115", "n02666196",
        "n03796401", "n02795169", "n02123045", "n01855672", "n01882714", "n02917067", "n02988304", "n04398044",
        "n02843684", "n02423022", "n02669723", "n04465501", "n02165456", "n03770439", "n02099601", "n04486054",
        "n02950826", "n03814639", "n04259630", "n03424325", "n02948072", "n03179701", "n03400231", "n02206856",
        "n03160309", "n01984695", "n03977966", "n03584254", "n04023962", "n02814860", "n01910747", "n04596742",
        "n03992509", "n04133789", "n03937543", "n02927161", "n01945685", "n02395406", "n02125311", "n03126707",
        "n04532106", "n02268443", "n02977058", "n07734744", "n03599486", "n04562935", "n03014705", "n04251144",
        "n04356056", "n02190166", "n03670208", "n02002724", "n02074367", "n04285008", "n04560804", "n04366367",
        "n02403003", "n07615774", "n04501370", "n03026506", "n02906734", "n01770393", "n04597913", "n03930313",
        "n04118538", "n04179913", "n04311004", "n02123394", "n04070727", "n02793495", "n02730930", "n02094433",
        "n04371430", "n04328186", "n03649909", "n04417672", "n03388043", "n01774384", "n02837789", "n07579787",
        "n04399382", "n02791270", "n03089624", "n02814533", "n04149813", "n07747607", "n03355925", "n01983481",
        "n04487081", "n03250847", "n03255030", "n02892201", "n02883205", "n03100240", "n02415577", "n02480495",
        "n01698640", "n01784675", "n04376876", "n03444034", "n01917289", "n01950731", "n03042490", "n07711569",
        "n04532670", "n03763968", "n07768694", "n02999410", "n03617480", "n06596364", "n01768244", "n02410509",
        "n03976657", "n01742172", "n03980874", "n02808440", "n02226429", "n02231487", "n02085620", "n01644900",
        "n02129165", "n02699494", "n03837869", "n02815834", "n07720875", "n02788148", "n02909870", "n03706229",
        "n07871810", "n03447447", "n02113799", "n12267677", "n03662601", "n02841315", "n07715103", "n02504458",
    ]

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.label_to_idx = {label: idx for idx, label in enumerate(self.LABELS)}

        # Collect all image paths
        base_path = self.root_dir / "Tiny-ImageNet-C"
        if base_path.exists():
            for img_path in base_path.rglob("*.JPEG"):
                parts = img_path.parts
                # Extract label from path structure
                label_name = parts[-2]
                if label_name in self.label_to_idx:
                    label_idx = self.label_to_idx[label_name]
                    self.samples.append((str(img_path), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _load_tinyimagenet_c(cfg: dict):
    # Download and extract dataset if not already present
    cache_dir = Path.home() / ".cache" / "tiny-imagenet-c"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cache_dir / "Tiny-ImageNet-C"
    if not dataset_path.exists():
        print("Downloading Tiny-ImageNet-C dataset from Zenodo...")
        tar_path = cache_dir / "Tiny-ImageNet-C.tar"
        if not tar_path.exists():
            url = "https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1"
            urllib.request.urlretrieve(url, tar_path)

        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(cache_dir)

    # Down-scale to 64×64 as required by MCU budget
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),  # converts to [0,1]
        ]
    )

    dataset = _TinyImageNetCDataset(cache_dir, transform)
    num_classes = 200  # Tiny-ImageNet has 200 classes
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