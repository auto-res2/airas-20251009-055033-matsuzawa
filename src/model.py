"""Model zoo & builder for experiment variations."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["build_model"]

# ---------------------------------------------------------------- helpers --

def _make_norm(norm_type: str, channels: int):
    if norm_type == "group":
        groups = 4 if channels % 4 == 0 else 1
        return nn.GroupNorm(groups, channels, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm2d(channels, affine=True)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    raise ValueError(norm_type)

# --------------------------------------------------------- ResNet-20 GN ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, norm="group"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.norm1 = _make_norm(norm, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.norm2 = _make_norm(norm, out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                _make_norm(norm, out_c)
            )
    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, depth=20, num_classes=10, norm="group"):
        super().__init__()
        assert (depth - 2) % 6 == 0, "Depth should be 6n+2"
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.norm1 = _make_norm(norm, 16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, n, 1, norm)
        self.layer2 = self._make_layer(16, 32, n, 2, norm)
        self.layer3 = self._make_layer(32, 64, n, 2, norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _make_layer(self, in_c, out_c, blocks, stride, norm):
        layers = [BasicBlock(in_c, out_c, stride, norm)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, 1, norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ------------------------------------------------------- MobileNetV2 GN ----

def _replace_bn_with_gn(m: nn.Module):
    for name, module in m.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = _make_norm("group", module.num_features)
            setattr(m, name, gn)
        else:
            _replace_bn_with_gn(module)

class MobileNetV2GN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        _replace_bn_with_gn(base)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

# --------------------------------------------------------- ViT Tiny RMS ---
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim*mlp_ratio, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + h

class ViTTinyRMS(nn.Module):
    def __init__(self, img_size=32, patch=4, emb=192, depth=4, heads=3, num_classes=10):
        super().__init__()
        self.patch = patch
        self.emb = emb
        self.patch_embed = nn.Conv2d(3, emb, kernel_size=patch, stride=patch)
        num_patches = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, emb))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, emb) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(emb, heads) for _ in range(depth)])
        self.norm = RMSNorm(emb)
        self.head = nn.Linear(emb, num_classes)
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1,2)  # B, N, C
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], 1) + self.pos
        x = self.blocks(x)
        x = self.norm(x[:,0])
        return self.head(x)

# --------------------------------------------------------------- factory ---

def build_model(cfg: Dict):
    name = cfg.get("name", "simple_cnn").lower()
    nc = cfg.get("num_classes", 10)

    if name == "resnet20_gn":
        return ResNetCIFAR(depth=20, num_classes=nc, norm="group")
    if name == "mobilenetv2_gn":
        return MobileNetV2GN(num_classes=nc)
    if name == "vit_tiny_rms":
        img_size = cfg.get("img_size", 32)
        return ViTTinyRMS(img_size=img_size, num_classes=nc)

    # fallback simple CNN for smoke test
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)))
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.features(x); x = torch.flatten(x,1)
            return self.fc(x)
    return SimpleCNN(num_classes=nc)