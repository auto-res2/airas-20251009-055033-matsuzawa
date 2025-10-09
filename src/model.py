"""Model architectures & factory.
Only *core* logic is provided here. Dataset / modality specific models (e.g.
ResNet-20-GN, MobileNet-V2-GN, ViT-Tiny) will be injected later via the
placeholder mechanism."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["build_model"]


# ------------------------------------------------------------------ helpers #

def _make_norm(norm_type: str, num_features: int):
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features, affine=True)
    if norm_type == "group":
        return nn.GroupNorm(num_groups=4, num_channels=num_features, affine=True)
    if norm_type == "layer":
        return nn.GroupNorm(num_groups=1, num_channels=num_features, affine=True)
    raise ValueError(f"Unsupported norm type: {norm_type}")


# --------------------------------------------------------------  SIMPLE CNN #
class SimpleCNN(nn.Module):
    """Light-weight CNN with GroupNorm. Suitable for smoke tests and synthetic
    datasets. Real research models will be plugged-in later."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 norm: str = "group"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            _make_norm(norm, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            _make_norm(norm, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ------------------------------------------------------------------ factory #

def build_model(model_cfg: Dict):
    name = model_cfg.get("name", "simple_cnn").lower()

    if name in {"simple_cnn", "simple_cnn_placeholder"}:
        return SimpleCNN(num_classes=model_cfg.get("num_classes", 10))

    # ------------------------------------------------------------------ #
    # PLACEHOLDER: Additional models (ResNet-20-GN, ViT-Tiny, etc.) will be
    # registered here in subsequent experiment-specific steps.            #
    # ------------------------------------------------------------------ #
    raise NotImplementedError(f"Model '{name}' not yet implemented.")