"""
model.py
========
Deep Learning model architectures for road waterlogging detection.

Design Principles:
1. Transfer Learning with Pretrained Backbones:
   - MobileNetV2 (default baseline): High computational efficiency, low parameter count (~2.2M params),
     ideal for CPU inference, edge devices, and rapid GPU training.
   - ResNet18 / ResNet50 / EfficientNet-B0 support for comparative experiments.
2. Two-Stage Fine-Tuning Strategy:
   - Stage 1: Freeze pretrained backbone weights, train custom classification head.
   - Stage 2: Selectively unfreeze top convolutional blocks for domain adaptation with low learning rate.
3. Colab GPU & Local CPU Auto-Detection:
   - Transparent fallback to CPU without crashes when CUDA is unavailable.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, Optional


def get_device() -> torch.device:
    """Detect and return available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WaterloggingClassifier(nn.Module):
    """
    Transfer learning classifier for road waterlogging / flood presence.
    Supports MobileNetV2, ResNet18, ResNet50, and EfficientNet_B0 backbones.
    """

    SUPPORTED_BACKBONES = ["mobilenet_v2", "resnet18", "resnet50", "efficientnet_b0"]

    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            in_features = base.classifier[1].in_features
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(128, num_classes)
            )

        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            # Remove original fc
            modules = list(base.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.pool = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            modules = list(base.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.pool = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. Choose from: {self.SUPPORTED_BACKBONES}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat)
        if self.backbone_name in ["mobilenet_v2", "efficientnet_b0"]:
            flattened = torch.flatten(pooled, 1)
            out = self.classifier(flattened)
        else:
            out = self.classifier(pooled)
        return out

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the convolutional feature extractor."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone parameters for fine-tuning.
        If unfreeze_last_n_layers is specified, only that many child modules are unchilled.
        """
        if unfreeze_last_n_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            children = list(self.backbone.children())
            total = len(children)
            for i, child in enumerate(children):
                requires_grad = i >= (total - unfreeze_last_n_layers)
                for param in child.parameters():
                    param.requires_grad = requires_grad

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def create_model(
    backbone: str = "mobilenet_v2",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone_initial: bool = True
) -> WaterloggingClassifier:
    """Factory function to instantiate and configure model."""
    model = WaterloggingClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained
    )
    if freeze_backbone_initial:
        model.freeze_backbone()
    return model
