"""Model architectures & factory.
Only *core* logic is provided here. Dataset / modality specific models (e.g.
ResNet-20-GN, MobileNet-V2-GN, ViT-Tiny) will be injected later via the
placeholder mechanism."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as models

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


# --------------------------------------------------------- MobileNet-V2-GN #
class MobileNetV2GN(nn.Module):
    """MobileNetV2 with GroupNorm instead of BatchNorm."""

    def __init__(self, num_classes: int = 200):
        super().__init__()
        # Use pretrained MobileNetV2 as base
        base_model = models.mobilenet_v2(pretrained=False)

        # Replace BatchNorm with GroupNorm in the features
        self.features = self._replace_bn_with_gn(base_model.features)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def _replace_bn_with_gn(self, module):
        """Recursively replace BatchNorm2d with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                # Use 4 groups for GroupNorm
                num_groups = min(4, num_channels)
                setattr(module, name, nn.GroupNorm(num_groups, num_channels, affine=True))
            else:
                self._replace_bn_with_gn(child)
        return module

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ------------------------------------------------------------ ViT-Tiny-RMS #
class ViTTinyRMS(nn.Module):
    """Vision Transformer Tiny with RMSNorm."""

    def __init__(self, num_classes: int = 10, img_size: int = 32):
        super().__init__()
        self.patch_size = 4
        self.num_patches = (img_size // self.patch_size) ** 2
        self.embed_dim = 192
        self.num_heads = 3
        self.depth = 12

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size,
                                      stride=self.patch_size)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim) * 0.02)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

        # Classifier head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classifier
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x


# ------------------------------------------------------------------ factory #

def build_model(model_cfg: Dict):
    name = model_cfg.get("name", "simple_cnn").lower()
    num_classes = model_cfg.get("num_classes", 10)

    if name in {"simple_cnn", "simple_cnn_placeholder"}:
        return SimpleCNN(num_classes=num_classes)
    elif name == "mobilenet_v2_gn":
        return MobileNetV2GN(num_classes=num_classes)
    elif name == "vit_tiny_rms":
        return ViTTinyRMS(num_classes=num_classes, img_size=model_cfg.get("img_size", 32))

    # ------------------------------------------------------------------ #
    # PLACEHOLDER: Additional models (ResNet-20-GN, ViT-Tiny, etc.) will be
    # registered here in subsequent experiment-specific steps.            #
    # ------------------------------------------------------------------ #
    raise NotImplementedError(f"Model '{name}' not yet implemented.")