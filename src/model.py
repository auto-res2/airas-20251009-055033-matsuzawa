"""Model architectures & factory registry."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

__all__ = ["build_model"]


# ----------------------------------------------------------------- utilities --

def _group_norm(channels: int):
    groups = 8 if channels >= 8 else 1
    return nn.GroupNorm(groups, channels, affine=True)


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# ------------------------------------------------------------- ResNet-20-GN ---
class _BasicBlockGN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.gn1 = _group_norm(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.gn2 = _group_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                _group_norm(planes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet20_GN(nn.Module):
    """ResNet-20 modified to use GroupNorm instead of BatchNorm (for micro-controllers)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(in_channels, 16)
        self.gn1 = _group_norm(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # ---------------------------------------------------------------------
    def _make_layer(self, planes, blocks, stride):
        layers = [_BasicBlockGN(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlockGN(self.in_planes, planes))
        return nn.Sequential(*layers)

    # ---------------------------------------------------------------------
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ----------------------------------------------------------- Speech CNN 1-D ---
class SpeechCNN1D(nn.Module):
    def __init__(self, in_channels: int = 40, num_classes: int = 35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            _group_norm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            _group_norm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            _group_norm(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, C, T)
        x = self.features(x)
        x = x.mean(dim=-1)  # global average pooling over time
        return self.classifier(x)


# -------------------------------------------------------------------- factory --

def build_model(model_cfg: Dict):
    name = model_cfg.get("name", "simple_cnn").lower()
    num_classes = model_cfg.get("num_classes", 10)

    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet20_gn":
        return ResNet20_GN(num_classes=num_classes)
    if name == "speech_cnn_1d":
        in_ch = model_cfg.get("in_channels", 40)
        return SpeechCNN1D(in_channels=in_ch, num_classes=num_classes)

    raise ValueError(f"Unknown model name '{name}'.")


# ------------------------- simple CNN (for smoke tests) ----------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            _group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            _group_norm(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)